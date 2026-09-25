# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host disaggregated prefill/decode over PCIe P2P.

Prefill and decode are separate engines. Each engine's GPUs stay behind
one PLX switch, so tensor-parallel all-reduce never crosses the CPU root.
The KV handoff does: the decode worker maps the prefill allocation with
HIP IPC and copies blocks with SDMA (``hipMemcpy`` device-to-device).

Both processes must share one ``HIP_VISIBLE_DEVICES`` list so device
ordinals match, then pin ranks with ``--data-parallel`` or by setting
the current device to the GPUs on that switch.

.. code-block:: bash

    # Prefill, GPUs 0,1 on switch A
    --kv-transfer-config '{
      "kv_connector": "PcieP2pConnector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {
        "handshake_port": 19000,
        "handshake_host": "127.0.0.1"
      }
    }'

    # Decode, GPUs 2,3 on switch B
    --kv-transfer-config '{
      "kv_connector": "PcieP2pConnector",
      "kv_role": "kv_consumer",
      "kv_connector_extra_config": {
        "handshake_port": 19100,
        "peer_handshake_port": 19000,
        "peer_host": "127.0.0.1"
      }
    }'

The prefill request must arrive with ``do_remote_decode`` set. The
finished params use the same fields as NixlConnector so an existing
disagg proxy can hand them to decode as ``do_remote_prefill``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.pcie_p2p_sdma import (
    enable_peer_access,
    export_ipc_handle,
    memcpy_device_async,
    open_ipc_handle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.pcie_p2p_layout import (
    block_planes,
    finished_transfer_params,
    remote_prefill_token_count,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_HANDSHAKE = b"handles"
_ACK = b"ack"


@dataclass
class PcieP2pConnectorMetadata(KVConnectorMetadata):
    """Decode-side loads for this step. Prefill sends an empty metadata."""

    # req_id -> (local block ids per group, remote block ids per group)
    loads: dict[str, tuple[tuple[list[int], ...], tuple[list[int], ...]]] = field(
        default_factory=dict
    )


class PcieP2pConnector(KVConnectorBase_V1, SupportsHMA):
    """Pull KV blocks from a same-host prefill engine over PCIe SDMA."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._scheduler: _Scheduler | None = None
        self._worker: _Worker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = _Scheduler(vllm_config, kv_cache_config)
        else:
            self._worker = _Worker(vllm_config, kv_cache_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self._worker is not None
        meta = self._connector_metadata
        assert isinstance(meta, PcieP2pConnectorMetadata)
        self._worker.start_load_kv(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self._worker is not None
        return self._worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, (list(block_ids),))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, block_ids)


class _Scheduler:
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        transfer = vllm_config.kv_transfer_config
        assert transfer is not None
        extra = transfer.kv_connector_extra_config or {}
        self._role = transfer.kv_role
        self._engine_id = transfer.engine_id or "pcie-p2p"
        self._host = str(extra.get("handshake_host", "127.0.0.1"))
        self._port = int(extra["handshake_port"])
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._is_producer = self._role in ("kv_producer", "kv_both")
        self._pending: dict[
            str, tuple[tuple[list[int], ...], tuple[list[int], ...]]
        ] = {}
        del kv_cache_config

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if not params or not params.get("do_remote_prefill"):
            return 0, False
        prompt = request.prompt_token_ids or []
        count = remote_prefill_token_count(len(prompt), num_computed_tokens)
        if count <= 0 or not params.get("remote_block_ids"):
            return 0, False
        # Copies finish inside start_load_kv, before the forward reads them.
        return count, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        remote_prefill = bool(params and params.get("do_remote_prefill"))
        if num_external_tokens <= 0 or not remote_prefill:
            return
        local = blocks.get_block_ids()
        remote = tuple(list(group) for group in params["remote_block_ids"])
        self._pending[request.request_id] = (local, remote)
        params["do_remote_prefill"] = False

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> PcieP2pConnectorMetadata:
        del scheduler_output
        meta = PcieP2pConnectorMetadata(loads=dict(self._pending))
        self._pending.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if not self._is_producer or not params or not params.get("do_remote_decode"):
            return False, None
        if not any(group for group in block_ids):
            return False, None
        prompt = request.prompt_token_ids or []
        return True, finished_transfer_params(
            block_ids=tuple(list(group) for group in block_ids),
            engine_id=self._engine_id,
            request_id=request.request_id,
            handshake_host=self._host,
            handshake_port=self._port,
            tp_size=self._tp_size,
            remote_num_tokens=max(0, len(prompt) - 1),
        )


@dataclass
class _LayerMap:
    name: str
    local: torch.Tensor
    remote_ptr: int
    remote_device: int
    local_blocks: int
    remote_blocks: int
    planes: int
    elements_per_block: int
    group: int


class _Worker:
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        transfer = vllm_config.kv_transfer_config
        assert transfer is not None
        extra = transfer.kv_connector_extra_config or {}
        self._role = transfer.kv_role
        self._is_producer = self._role in ("kv_producer", "kv_both")
        self._host = str(extra.get("handshake_host", "127.0.0.1"))
        self._port = int(extra["handshake_port"])
        self._peer_host = str(extra.get("peer_host", "127.0.0.1"))
        self._peer_port = int(extra.get("peer_handshake_port", self._port))
        self._groups = kv_cache_config.kv_cache_groups
        self._configured_blocks = kv_cache_config.num_blocks
        self._layers: list[_LayerMap] = []
        self._acked: set[str] = set()
        self._loaded: set[str] = set()
        self._lock = threading.Lock()
        self._zmq_ctx = zmq.Context()
        self._thread: threading.Thread | None = None
        self._peer_ready = False
        self._stream: torch.cuda.Stream | None = None
        try:
            self._rank = get_tensor_model_parallel_rank()
        except Exception:  # noqa: BLE001 -- scheduler-less unit construction
            self._rank = 0

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        layer_group = {
            name: index
            for index, group in enumerate(self._groups)
            for name in group.layer_names
        }
        if self._is_producer:
            payload = []
            for name, tensor in kv_caches.items():
                if not tensor.is_cuda or not tensor.is_contiguous():
                    raise RuntimeError(
                        f"pcie_p2p requires a contiguous CUDA KV tensor for {name}"
                    )
                num_blocks = _block_count(tensor, self._configured_blocks)
                planes, elements = block_planes(tuple(tensor.shape), num_blocks)
                payload.append(
                    {
                        "name": name,
                        "group": layer_group.get(name, 0),
                        "handle": export_ipc_handle(tensor.data_ptr()).hex(),
                        "device": tensor.device.index,
                        "num_blocks": num_blocks,
                        "planes": planes,
                        "elements": elements,
                        "dtype": str(tensor.dtype),
                    }
                )
            self._serve(payload, kv_caches, layer_group)
            return
        self._local = kv_caches
        self._layer_group = layer_group

    def _serve(
        self,
        payload: list[dict[str, Any]],
        kv_caches: dict[str, torch.Tensor],
        layer_group: dict[str, int],
    ) -> None:
        self._payload = payload
        self._local_caches = kv_caches
        del layer_group
        port = self._port + self._rank
        path = make_zmq_path("tcp", self._host, port)

        def _loop() -> None:
            with make_zmq_socket(self._zmq_ctx, path, zmq.REP, bind=True) as sock:
                while True:
                    try:
                        message = sock.recv()
                    except zmq.ZMQError:
                        return
                    if message == _HANDSHAKE:
                        sock.send(msgspec.msgpack.encode(self._payload))
                        continue
                    if message.startswith(_ACK):
                        req_id = message[len(_ACK) :].decode()
                        with self._lock:
                            self._acked.add(req_id)
                        sock.send(b"ok")
                        continue
                    sock.send(b"err")

        self._thread = threading.Thread(
            target=_loop, name="pcie-p2p-handshake", daemon=True
        )
        self._thread.start()
        logger.info("pcie_p2p producer listening on %s (tp rank %s)", path, self._rank)

    def _ensure_peer(self) -> None:
        if self._peer_ready:
            return
        port = self._peer_port + self._rank
        path = make_zmq_path("tcp", self._peer_host, port)
        with make_zmq_socket(self._zmq_ctx, path, zmq.REQ, bind=False) as sock:
            sock.send(_HANDSHAKE)
            remote = msgspec.msgpack.decode(sock.recv())
        by_name = {item["name"]: item for item in remote}
        device = torch.cuda.current_device()
        peers: set[int] = set()
        for name, tensor in self._local.items():
            item = by_name.get(name)
            if item is None:
                raise RuntimeError(f"pcie_p2p peer has no KV tensor named {name}")
            if item["dtype"] != str(tensor.dtype) or item["elements"] <= 0:
                raise RuntimeError(f"pcie_p2p layout mismatch on {name}")
            remote_dev = int(item["device"])
            if remote_dev not in peers:
                enable_peer_access(remote_dev)
                peers.add(remote_dev)
            ptr = open_ipc_handle(bytes.fromhex(item["handle"]))
            num_blocks = _block_count(tensor, self._configured_blocks)
            planes, elements = block_planes(tuple(tensor.shape), num_blocks)
            self._layers.append(
                _LayerMap(
                    name=name,
                    local=tensor,
                    remote_ptr=ptr,
                    remote_device=remote_dev,
                    local_blocks=num_blocks,
                    remote_blocks=int(item["num_blocks"]),
                    planes=planes,
                    elements_per_block=elements,
                    group=self._layer_group.get(name, 0),
                )
            )
        self._stream = torch.cuda.Stream(device=device)
        self._peer_ready = True
        logger.info(
            "pcie_p2p decode mapped %d layers from %s", len(self._layers), path
        )

    def start_load_kv(self, meta: PcieP2pConnectorMetadata) -> None:
        if self._is_producer or not meta.loads:
            return
        self._ensure_peer()
        assert self._stream is not None
        stream_ptr = self._stream.cuda_stream
        copied: list[str] = []
        for req_id, (local_groups, remote_groups) in meta.loads.items():
            for layer in self._layers:
                group = layer.group
                if group >= len(local_groups) or group >= len(remote_groups):
                    continue
                local_ids = local_groups[group]
                remote_ids = remote_groups[group]
                count = min(len(local_ids), len(remote_ids))
                nbytes = layer.elements_per_block * layer.local.element_size()
                dst_base = layer.local.data_ptr()
                for index in range(count):
                    local_block = int(local_ids[index])
                    remote_block = int(remote_ids[index])
                    if local_block < 0 or remote_block < 0:
                        continue
                    for plane in range(layer.planes):
                        dst = dst_base + (
                            plane * layer.local_blocks + local_block
                        ) * nbytes
                        src = layer.remote_ptr + (
                            plane * layer.remote_blocks + remote_block
                        ) * nbytes
                        memcpy_device_async(dst, src, nbytes, stream_ptr)
            copied.append(req_id)
        # The ack frees prefill blocks. It must follow a finished DMA.
        self._stream.synchronize()
        for req_id in copied:
            self._send_ack(req_id)
            with self._lock:
                self._loaded.add(req_id)

    def _send_ack(self, req_id: str) -> None:
        port = self._peer_port + self._rank
        path = make_zmq_path("tcp", self._peer_host, port)
        with make_zmq_socket(self._zmq_ctx, path, zmq.REQ, bind=False) as sock:
            sock.send(_ACK + req_id.encode())
            sock.recv()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        with self._lock:
            if self._is_producer:
                done = set(self._acked)
                self._acked.clear()
                return done or None, None
            done_recv = set(self._loaded)
            self._loaded.clear()
            return None, done_recv or None


def _block_count(tensor: torch.Tensor, configured: int) -> int:
    """Prefer the engine block count when it is an axis of ``tensor``."""
    if configured in tensor.shape:
        return int(configured)
    return int(tensor.shape[0])
