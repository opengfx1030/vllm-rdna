# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

_KV_CONNECTOR_AGENT: KVConnectorBaseType | None = None

# Two-lane engine: one connector processes both lanes, but a connector binds
# one step's metadata at a time (`bind_connector_metadata`), and the attention
# layers read it through the bound global while a forward is in flight. Two
# lane threads forwarding concurrently would overwrite each other's step.
# Each lane thread therefore gets a view that keeps its own metadata and
# rebinds it under this lock around every forwarded call, so a connector
# method never observes the other lane's step. Flag off installs no view, so
# the single-threaded contract is unchanged.
_connector_lock = threading.RLock()
_lane_view = threading.local()


class LaneKVConnectorView:
    """One lane's view of the process-wide KV connector."""

    def __init__(self, connector: KVConnectorBaseType, lane: int) -> None:
        self._connector = connector
        self._lane = lane
        self._metadata: Any = None

    @property
    def lane(self) -> int:
        return self._lane

    def _rebind(self) -> None:
        if self._metadata is not None:
            self._connector.bind_connector_metadata(self._metadata)

    def bind_connector_metadata(self, metadata: Any) -> None:
        with _connector_lock:
            self._metadata = metadata
            self._connector.bind_connector_metadata(metadata)

    def clear_connector_metadata(self) -> None:
        with _connector_lock:
            self._metadata = None
            self._connector.clear_connector_metadata()

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._connector, name)
        if not callable(target):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            with _connector_lock:
                self._rebind()
                return target(*args, **kwargs)

        return call


def set_lane_connector_view(lane: int) -> None:
    """Route this thread's connector calls through a lane view (two-lane).

    Called by each lane thread when it starts. Idempotent: a thread that
    already holds a view keeps it.
    """
    if _KV_CONNECTOR_AGENT is None:
        _lane_view.view = None
        return
    _lane_view.view = LaneKVConnectorView(_KV_CONNECTOR_AGENT, lane)


def clear_lane_connector_view() -> None:
    _lane_view.view = None


def _in_lane_thread() -> bool:
    if _KV_CONNECTOR_AGENT is None:
        return False
    # Lazy import: the lane runtime imports this module's loader, not vice versa.
    from vllm.v1.lane import lane_mode_enabled
    from vllm.v1.worker.lane.runtime import in_lane_thread

    return lane_mode_enabled() and in_lane_thread()


def get_kv_transfer_group() -> KVConnectorBaseType:
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized"
    )
    view = getattr(_lane_view, "view", None)
    if view is not None and view._connector is _KV_CONNECTOR_AGENT:
        return cast(KVConnectorBaseType, view)
    if _in_lane_thread():
        # First connector call on this lane thread (the connector can be
        # initialized after the lane threads start).
        from vllm.v1.lane import current_lane

        view = LaneKVConnectorView(_KV_CONNECTOR_AGENT, int(current_lane()))
        _lane_view.view = view
        return cast(KVConnectorBaseType, view)
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    return _KV_CONNECTOR_AGENT is not None


def is_v1_kv_transfer_group(connector: KVConnectorBaseType | None = None) -> bool:
    """Check if the KV connector is the v1 connector.
    If the argument is None, it will check the global KV connector

    Args:
        connector: The KV connector to check. If None, it will check the
            global KV connector.

    Note:
        This function will no-longer be needed after the v1 KV connector
        becomes the default.

    """
    if connector is None:
        connector = _KV_CONNECTOR_AGENT
    elif isinstance(connector, LaneKVConnectorView):
        connector = connector._connector

    if connector is None:
        return False

    return isinstance(connector, KVConnectorBase_V1)


def _sync_engine_id_across_tp(vllm_config: "VllmConfig") -> None:
    """Broadcast engine_id from TP rank 0 so all workers in a
    multi-node TP group share the same value.

    When PP is enabled, also broadcast across PP ranks so all workers in
    the same model-parallel engine share the same value.
    """
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tp_group,
    )

    assert vllm_config.kv_transfer_config is not None
    synced_id = get_tp_group().broadcast_object(
        vllm_config.kv_transfer_config.engine_id, src=0
    )
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        synced_id = get_pp_group().broadcast_object(synced_id, src=0)
    vllm_config.kv_transfer_config.engine_id = synced_id


def ensure_kv_transfer_initialized(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
) -> None:
    """Initialize KV cache transfer parallel group."""
    global _KV_CONNECTOR_AGENT

    if _KV_CONNECTOR_AGENT is not None:
        return

    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance:
        _sync_engine_id_across_tp(vllm_config)

        _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
        )


def ensure_kv_transfer_shutdown() -> None:
    global _KV_CONNECTOR_AGENT
    if _KV_CONNECTOR_AGENT is not None:
        _KV_CONNECTOR_AGENT.shutdown()
        _KV_CONNECTOR_AGENT = None
        _lane_view.view = None
