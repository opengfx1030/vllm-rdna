# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the RidgeFill cache dump (PR 155). No GPU.

The comparison checks need torch only; the probe checks also need the vllm
package and skip without it. Run with pytest, or directly::

    python3 tests/model_executor/test_ridgefill_dump.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "docs" / "explore" / "ridgefill" / "cache_dump.py"


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _driver():
    spec = importlib.util.spec_from_file_location("ridgefill_cache_dump", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ridgefill_cache_dump"] = module
    spec.loader.exec_module(module)
    return module


def _record(torch, *, layer: int, consumed: bool, cmp_noise: float) -> dict:
    """One synthetic probe record: 8 filled rows, 2 KV heads, rotary dim 4."""
    gen = torch.Generator().manual_seed(layer)
    rows, heads, dim, kdim = 8, 2, 8, 4
    positions = torch.arange(3 * rows, dtype=torch.int64).reshape(3, rows)
    ring_rows = torch.tensor([4, 5, 6, 7])
    group_rows = torch.tensor([3, 7])

    def snap(rope_noise: float, cmp: float) -> dict:
        k = base_k.clone()
        k[..., :4] += rope_noise
        return {
            "main_k": k,
            "main_v": base_v.clone(),
            "raw_k": base_raw.clone(),
            "raw_pos": positions.t().index_select(0, ring_rows).reshape(-1, 1, 3),
            "compressed_k": base_cmp + cmp,
        }

    base_k = torch.randn(rows, heads, dim, generator=gen)
    base_v = torch.randn(rows, heads, dim, generator=gen)
    base_raw = torch.randn(4, 1, kdim, generator=gen)
    base_cmp = torch.randn(2, 1, kdim, generator=gen)
    stock = torch.full((rows,), 2.0)
    return {
        "forward": 0,
        "layer_idx": layer,
        "consumed": consumed,
        "fallback": False,
        "rotary_dim": 4,
        "positions": positions,
        "slot_rows": {
            "main": torch.arange(rows),
            "raw": ring_rows,
            "compressed": group_rows,
        },
        # A: a rounding-level RoPE difference and a compressed-key offset.
        "A": snap(1e-4, cmp_noise),
        "B": snap(0.0, 0.0),
        "B2": snap(0.0, 0.0),
        "out": {
            "a_zero": True,
            "block_rms": torch.full((rows,), 4.0),
            "stock_rms": stock,
            "floor_rms": torch.zeros(rows),
        },
    }


def test_compare_flags_a_writer_difference_above_the_floor() -> None:
    torch = _torch()
    if torch is None:
        return
    driver = _driver()
    rows = driver.compare_record(
        _record(torch, layer=3, consumed=True, cmp_noise=0.5), factor=10.0
    )
    by = {r["cache"]: r for r in rows}
    # The K split puts the whole RoPE difference in the rotary dims.
    assert by["main_k.rope"]["ab_rms"] > 0 and not by["main_k.rope"]["flag"]
    assert by["main_k.pass"]["ab_equal"] and by["main_k.pass"]["ab_rms"] == 0
    assert "main_k[h0]" in by and "main_k[h1]" in by and "main_v[h1]" in by
    assert by["main_v"]["ab_equal"] and by["raw_k"]["ab_equal"]
    assert by["compressed_k"]["flag"] and by["compressed_k"]["bb_equal"]
    assert by["raw_pos"]["ab_equal"] and by["raw_pos"]["a_rows_equal"]
    assert not by["raw_pos"]["flag"]
    # A consumed layer's zeroed output is the cache-only drop, flagged.
    assert by["attn_out"]["flag"] and abs(by["attn_out"]["out_vs_block"] - 0.5) < 1e-9


def test_compare_does_not_flag_a_predicted_layer_output() -> None:
    torch = _torch()
    if torch is None:
        return
    driver = _driver()
    rows = driver.compare_record(
        _record(torch, layer=27, consumed=False, cmp_noise=0.0), factor=10.0
    )
    by = {r["cache"]: r for r in rows}
    assert not by["attn_out"]["flag"] and by["attn_out"]["ab_rel"] == 1.0
    assert not any(r["flag"] for r in rows)


def test_compare_flags_wrong_ring_positions() -> None:
    torch = _torch()
    if torch is None:
        return
    driver = _driver()
    rec = _record(torch, layer=31, consumed=False, cmp_noise=0.0)
    rec["A"]["raw_pos"] = rec["A"]["raw_pos"] + 1
    by = {r["cache"]: r for r in driver.compare_record(rec)}
    assert by["raw_pos"]["flag"] and not by["raw_pos"]["a_rows_equal"]
    assert by["raw_pos"]["b_rows_equal"]


def test_compare_cli_writes_json_and_table() -> None:
    torch = _torch()
    if torch is None:
        return
    import json

    driver = _driver()
    with tempfile.TemporaryDirectory() as tmp:
        records = [
            _record(torch, layer=3, consumed=True, cmp_noise=0.0),
            _record(torch, layer=27, consumed=False, cmp_noise=0.0),
        ]
        for rank in (0, 1):
            torch.save({"rank": rank, "records": records}, Path(tmp) / f"rank{rank}.pt")
        assert driver.main(["compare", tmp]) == 0
        result = json.loads((Path(tmp) / "compare.json").read_text())
    assert {r["rank"] for r in result["rows"]} == {0, 1}
    flagged = {(r["layer"], r["cache"]) for r in result["flagged"]}
    assert flagged == {(3, "attn_out")}
    table = driver.format_table(result)
    assert "attn_out" in table and "positions equal" in table


def _has_vllm() -> bool:
    try:
        import vllm.forward_context  # noqa: F401
        from vllm.model_executor.layers.ridgefill import dump_probe  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


def _fake_attn(torch):
    """A QSA layer double: paged caches and both writers on the CPU."""
    blocks, block_size, head_dim = 2, 8, 4
    ring, key_width = 4, 4

    class Attn:
        layer_name = "fake.attn"

        def __init__(self) -> None:
            self.head_dim = head_dim
            self.rotary_emb = SimpleNamespace(rotary_dim=2)
            self.kv_cache = torch.zeros(
                blocks, 1, block_size, 2 * head_dim, dtype=torch.float16
            )
            self.indexer = SimpleNamespace(
                raw_key_cache=SimpleNamespace(
                    prefix="fake.raw",
                    kv_cache=torch.zeros(blocks, ring, 1, key_width + 12).half(),
                    key_head_size=key_width,
                    cache_rope_positions=True,
                    rope_position_offset=key_width,
                ),
                compressed_key_cache=SimpleNamespace(
                    prefix="fake.cmp",
                    kv_cache=torch.zeros(blocks, block_size // 4, 1, key_width),
                ),
            )
            self.ring_seen: list = []
            self.calls: list[str] = []

        def _write(self, hidden, positions, slots, k_offset: float) -> None:
            from vllm.forward_context import get_forward_context
            from vllm.model_executor.layers.ridgefill import dump_probe

            md: Any = get_forward_context().attn_metadata
            key_cache, value_cache = self.kv_cache.transpose(1, 2).split(
                head_dim, dim=-1
            )
            main = md[self.layer_name].slot_mapping.index_select(0, slots)
            k = (hidden[:, None, :head_dim] + k_offset).half()
            dump_probe.scatter_slots(key_cache, main, k)
            dump_probe.scatter_slots(
                value_cache, main, hidden[:, None, :head_dim].half()
            )
            raw = self.indexer.raw_key_cache
            ring_slots = md[raw.prefix].slot_mapping.index_select(0, slots)
            keep = (ring_slots >= 0).nonzero().squeeze(1)
            self.ring_seen.append(raw.kv_cache.clone())
            rows = torch.zeros(keep.numel(), 1, key_width + 12).half()
            rows[:, 0, :key_width] = hidden.index_select(0, keep)[:, :key_width].half()
            pos = positions.t().index_select(0, keep).contiguous()
            rows[:, 0, key_width:] = pos.view(torch.float16)
            dump_probe.scatter_slots(
                raw.kv_cache, ring_slots.index_select(0, keep), rows
            )

        def cache_only_forward(self, hidden_states, positions, row_index) -> None:
            self.calls.append("A")
            self._write(hidden_states, positions, row_index, k_offset=0.5)

        def __call__(self, hidden_states, positions):
            self.calls.append("B")
            n = hidden_states.shape[0]
            self._write(hidden_states, positions, torch.arange(n), k_offset=0.0)
            return hidden_states * 2

    return Attn()


def test_dump_probe_runs_a_then_stock_twice_and_restores_the_ring() -> None:
    torch = _torch()
    if torch is None or not _has_vllm():
        return
    from vllm.forward_context import override_forward_context
    from vllm.model_executor.layers.ridgefill import dump_probe
    from vllm.model_executor.layers.ridgefill import ops as rf_ops

    attn = _fake_attn(torch)
    rf_ops.register_layer(attn.layer_name, SimpleNamespace(self_attn=attn, layer_idx=3))
    rf_ops.set_session(SimpleNamespace(cfg=SimpleNamespace(split=24)))
    # 8 filled rows of one request on block 1; the last 4 own the ring.
    metadata = {
        "fake.attn": SimpleNamespace(slot_mapping=torch.arange(8, 16)),
        "fake.raw": SimpleNamespace(
            slot_mapping=torch.tensor([-1, -1, -1, -1, 4, 5, 6, 7])
        ),
        "fake.cmp": SimpleNamespace(slot_mapping=torch.tensor([-1, -1, -1, 2] * 2)),
    }
    attn.indexer.raw_key_cache.kv_cache[1] = 3.0
    ring_before = attn.indexer.raw_key_cache.kv_cache.clone()
    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    positions = torch.arange(24, dtype=torch.int64).reshape(3, 8)
    filled = torch.arange(8)
    exact = torch.arange(0)

    def original(*args):
        raise AssertionError("captured forward must not run the original op")

    state = dump_probe._DumpState("unused", forwards=1, skip=0, rank=0)
    try:
        with override_forward_context(SimpleNamespace(attn_metadata=metadata)):
            out = state.call(original, x, positions, filled, exact, attn.layer_name)
        # A second forward is past the capture window: the original op runs.
        seen: list = []
        with override_forward_context(SimpleNamespace(attn_metadata=dict(metadata))):
            state.call(lambda *a: seen.append(a), x, positions, filled, exact, "x")
    finally:
        rf_ops._LAYERS.pop(attn.layer_name, None)
        rf_ops.set_session(None)
    assert attn.calls == ["A", "B", "B"] and len(seen) == 1
    # The forward continues as the legacy fill: the stock output.
    assert torch.equal(out, x * 2)
    # Both stock calls saw the ring as it was before A wrote it.
    assert torch.equal(attn.ring_seen[1], ring_before)
    assert torch.equal(attn.ring_seen[2], ring_before)
    (rec,) = state.records
    assert rec["consumed"] and rec["layer_idx"] == 3 and rec["rotary_dim"] == 2
    assert torch.equal(rec["A"]["main_k"] - 0.5, rec["B"]["main_k"])
    assert torch.equal(rec["B"]["main_v"], rec["B2"]["main_v"])
    assert torch.equal(rec["slot_rows"]["raw"], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(
        rec["B"]["raw_pos"].reshape(-1, 3), positions.t()[4:].contiguous()
    )
    assert rec["B"]["compressed_k"].shape == (2, 1, 4)
    assert torch.equal(rec["out"]["stock_rms"], (x * 2).pow(2).mean(-1).sqrt())


def test_dump_probe_arm_swaps_the_op_and_disarm_restores_it() -> None:
    torch = _torch()
    if torch is None or not _has_vllm():
        return
    from vllm.model_executor.layers.ridgefill import dump_probe

    class Worker(dump_probe.RidgeFillDumpProbe):
        rank = 0

    packet = torch.ops.vllm.ridgefill_cache_only
    worker = Worker()
    worker.ridgefill_dump_arm("unused", 1, 0)
    try:
        assert torch.ops.vllm.ridgefill_cache_only is not packet
    finally:
        worker.ridgefill_dump_disarm()
    assert torch.ops.vllm.ridgefill_cache_only is packet


def _main() -> None:
    tests = [
        test_compare_flags_a_writer_difference_above_the_floor,
        test_compare_does_not_flag_a_predicted_layer_output,
        test_compare_flags_wrong_ring_positions,
        test_compare_cli_writes_json_and_table,
        test_dump_probe_runs_a_then_stock_twice_and_restores_the_ring,
        test_dump_probe_arm_swaps_the_op_and_disarm_restores_it,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _main()
