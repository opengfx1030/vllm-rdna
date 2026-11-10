# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension of the G0a spike.

Loaded with ``--worker-extension-cls g0a_ext.G0aWorkerExtension``.

Methods are called through ``POST /collective_rpc`` (``VLLM_SERVER_DEV_MODE=1``);
every argument is a string. All run on the worker of one TP rank; the ``g0a_*``
methods need the NIXL connector worker (``get_kv_transfer_group().connector_worker``),
the ``g0r_*`` methods do not.

* ``g0a_regions()``: the transfer regions and cache groups this rank registered
  with NIXL, i.e. which model state is transferable and how large it is.
* ``g0a_dump(store_dir, tag, params_json)``: copy, for one finished producer
  request, every memory window a decode-side NIXL READ would address into
  ``store_dir/rank<r>/<tag>.{bin,json}``. The set is computed with the
  connector's own ``_compute_desc_ids`` over the producer's local descriptor
  list, so it is the descriptor set of a homogeneous-TP read of that request
  (``params_json`` is the ``kv_transfer_params`` the producer returned).
* ``g0r_probe()`` / ``g0r_begin(out_dir, tag, plan_json)`` / ``g0r_end()``: the
  per-step state tracer of ``g0r_trace.py`` (DisaggRework), independent of any
  connector.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import g0_rt
import numpy as np
from g0_desc import digest, resolve, row_starts

_KIND_BY_SPEC = {
    "FullAttentionSpec": "attention KV (QSA main + MTP layers)",
    "MLAAttentionSpec": "QSA compressed keys",
    "SlidingWindowMLASpec": "sliding-window compressed keys",
    "CircularBufferSpec": "QSA raw ring (circular buffer)",
    "MambaSpec": "GDN conv/ssm",
}


class G0aWorkerExtension:
    def _g0a_worker(self) -> Any:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        worker = get_kv_transfer_group().connector_worker
        assert worker is not None, "no NIXL connector worker on this rank"
        return worker

    def _g0a_group_kind(self, w: Any, g: int) -> str:
        spec_name = w._group_spec_types[g].__name__
        kind = _KIND_BY_SPEC.get(spec_name, spec_name)
        if spec_name == "MambaSpec":
            specs = w.kv_cache_config.transfer_groups[g].kv_cache_spec
            from vllm.v1.kv_cache_interface import iter_layer_specs

            if all(getattr(s, "tp_replicated", False) for s in iter_layer_specs(specs)):
                kind = "PLE short-conv"
        return kind

    def g0a_regions(self) -> dict[str, Any]:
        w = self._g0a_worker()
        groups = []
        for g, grp in enumerate(w.kv_cache_config.transfer_groups):
            groups.append(
                {
                    "group": g,
                    "spec": w._group_spec_types[g].__name__,
                    "kind": self._g0a_group_kind(w, g),
                    "num_layers": len(grp.layer_names),
                    "layers_first": list(grp.layer_names[:3]),
                    "layers_last": list(grp.layer_names[-2:]),
                    "mtp_layers": [n for n in grp.layer_names if "mtp" in n.lower()],
                }
            )
        regions = []
        for i in range(w.num_regions):
            regions.append(
                {
                    "region": i,
                    "name": w.region_names[i],
                    "group": int(w.region_group_ids[i]),
                    "block_len": int(w.block_len_per_layer[i]),
                    "block_stride": int(w.block_stride_per_layer[i]),
                    "num_blocks": int(w.region_num_blocks[i]),
                    "mem_type": w.region_mem_types[i],
                    "ssm": i in w._ssm_region_indices,
                    "scratch": i in w._scratch_region_indices,
                    "ple": i == w._ple_region_index,
                }
            )
        rows = w.src_blocks_data
        registered = getattr(w.nixl_wrapper, "registered", None)
        return {
            "rank": int(w.tp_rank),
            "device_id": int(w.device_id),
            "engine_id": w.engine_id,
            "block_size": int(w.block_size),
            "num_blocks": int(w.num_blocks),
            "physical_per_logical": int(w._physical_blocks_per_logical_kv_block),
            "use_host_buffer": bool(w.use_host_buffer),
            "kv_buffer_device": str(w.kv_buffer_device),
            "nixl_memory_type": w.nixl_memory_type,
            "has_mamba": bool(w._has_mamba),
            "is_csa_linear": bool(w._is_csa_linear),
            "ple_region": w._ple_region_index,
            "ple_block_len": w._ple_block_len,
            "ssm_regions": len(w._ssm_region_indices),
            "scratch_regions": list(w._scratch_region_indices),
            "num_regions": int(w.num_regions),
            "num_desc_rows": int(rows.shape[0]),
            "num_descs": int(w.num_descs),
            "groups": groups,
            "regions": regions,
            "registered_ranges": registered,
            "registered_bytes": (
                sum(r[1] for r in registered) if registered is not None else None
            ),
        }

    def g0a_dump(self, store_dir: str, tag: str, params_json: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        w = self._g0a_worker()
        params = json.loads(params_json)
        ratio = w._physical_blocks_per_logical_kv_block
        phys = w._logical_to_kernel_block_ids(params["remote_block_ids"], ratio)
        n_groups = len(phys)
        assert n_groups == len(w._group_spec_types), (
            n_groups,
            len(w._group_spec_types),
        )
        rows = w.src_blocks_data
        starts = row_starts(rows)
        n_fa_rows = int(w.num_regions)
        engine = w.engine_id

        windows: dict[tuple[int, int], int] = {}
        per_group: dict[int, dict[str, Any]] = {}
        mtp_bytes = 0
        for g in range(n_groups):
            kind = self._g0a_group_kind(w, g)
            stats = per_group.setdefault(
                g,
                {
                    "spec": w._group_spec_types[g].__name__,
                    "kind": kind,
                    "blocks": len(phys[g]),
                    "windows": 0,
                    "bytes": 0,
                },
            )
            if len(phys[g]) == 0:
                continue
            select = [list(phys[h]) if h == g else [] for h in range(n_groups)]
            ids = w._compute_desc_ids(
                select,
                w.dst_num_blocks[engine],
                None,
                ratio,
                region_num_blocks=w.dst_region_num_blocks.get(engine) or None,
                region_group_ids=w.dst_region_group_ids.get(engine) or None,
                uses_region_group_mapping=w.dst_uses_region_group_mapping[engine],
            )
            addr, length, row = resolve(rows, starts, ids)
            for a, n, r in zip(addr.tolist(), length.tolist(), row.tolist()):
                stats["windows"] += 1
                stats["bytes"] += n
                if r < n_fa_rows and "mtp" in str(w.region_names[r]).lower():
                    mtp_bytes += n
                windows.setdefault((a, n), g)

        keys = list(windows)
        total = sum(n for _, n in keys)
        rank_dir = os.path.join(store_dir, f"rank{w.tp_rank}")
        os.makedirs(rank_dir, exist_ok=True)
        buf = np.empty(total, dtype=np.uint8)
        g0_rt.set_device(int(w.device_id))
        g0_rt.sync()
        metas = []
        off = 0
        for addr_i, n in keys:
            g0_rt.d2h(buf[off : off + n], addr_i, n)
            metas.append(
                {
                    "addr": addr_i,
                    "len": n,
                    "off": off,
                    "h": digest(buf[off : off + n]),
                    "g": windows[(addr_i, n)],
                }
            )
            off += n
        with open(os.path.join(rank_dir, f"{tag}.bin"), "wb") as f:
            f.write(memoryview(buf))
        meta = {
            "tag": tag,
            "rank": int(w.tp_rank),
            "engine_id": engine,
            "request_id": params.get("remote_request_id"),
            "bytes": total,
            "windows": metas,
        }
        with open(os.path.join(rank_dir, f"{tag}.json"), "w") as f:
            json.dump(meta, f)
        return {
            "rank": int(w.tp_rank),
            "tag": tag,
            "windows": len(keys),
            "bytes": total,
            "mtp_attention_bytes": mtp_bytes,
            "per_group": {str(g): v for g, v in sorted(per_group.items())},
            "seconds": round(time.perf_counter() - t0, 3),
        }

    def _g0r_tracer(self) -> Any:
        import g0r_trace

        tracer = getattr(self, "_g0r", None)
        if tracer is None:
            tracer = g0r_trace.StateTracer(self)
            self._g0r = tracer
        return tracer

    def g0r_probe(self) -> dict[str, Any]:
        """Cache layout the tracer addresses (groups, kinds, layers, page sizes)."""
        return self._g0r_tracer().probe()

    def g0r_begin(self, out_dir: str, tag: str, plan_json: str) -> dict[str, Any]:
        """Arm the state tracer for the next request (g0r_trace.py): ``plan`` has
        ``raw`` (save state pages byte for byte) and ``raw_at`` (token position of
        the ``post`` snapshot to save)."""
        return self._g0r_tracer().begin(out_dir, tag, json.loads(plan_json))

    def g0r_end(self) -> dict[str, Any]:
        """Disarm the tracer and write ``rank<r>/<tag>/steps.json``."""
        return self._g0r_tracer().end()

    def g0a_exec(self, code: str) -> str:
        """Dev-only (``VLLM_SERVER_DEV_MODE=1``): run ``code`` in this worker.

        ``self`` is the extension and ``w`` the NIXL connector worker; assign
        ``result`` to return its repr. Lets a live server be inspected without a
        reboot (a boot is minutes; an RPC is milliseconds).
        """
        import torch

        scope: dict[str, Any] = {"self": self, "w": self._g0a_worker(), "torch": torch}
        exec(code, scope)  # noqa: S102 - dev-mode spike hook
        return repr(scope.get("result"))
