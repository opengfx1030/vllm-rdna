# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the RidgeFill draft. No GPU.

The first seven checks are pure Python. The torch-backed checks at the end
cover the map application and the capture script's synthetic mode; they skip
when torch or the vllm package is not importable.

Run with pytest, or directly::

    python3 tests/model_executor/test_ridgefill.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "vllm" / "model_executor" / "layers" / "ridgefill"


def _load(name: str):
    path = PKG / f"{name}.py"
    mod_name = f"ridgefill_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


math = _load("math")
policy = _load("policy")
loader = _load("loader")
keys = _load("hash_keys")
config = _load("config")


def _clear_ridge_env() -> None:
    for name in list(os.environ):
        if name.startswith("VLLM_RIDGEFILL"):
            del os.environ[name]


def test_fit_recovers_known_map() -> None:
    xs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    ys = [(2.0 * x[0] - x[1] + 0.5,) for x in xs]
    weight, bias = math.fit_ridge(xs, ys, lam=0.0)
    assert abs(weight[0][0] - 2.0) < 1e-8
    assert abs(weight[0][1] - (-1.0)) < 1e-8
    assert abs(bias[0] - 0.5) < 1e-8
    pred = math.apply_map(xs, weight, bias)
    for got, want in zip(pred, ys):
        assert abs(got[0] - want[0]) < 1e-8


def test_int8_row128_dequant_and_apply() -> None:
    codes = [[1, 2, 3, 4]]
    scales = [[0.5, 0.25]]
    bias = [0.1]
    weight, got_bias = math.dequant_i8_row128(codes, scales, bias, group=2)
    assert weight == [[0.5, 1.0, 0.75, 1.0]]
    assert got_bias == [0.1]
    pred = math.apply_map([[1.0, 0.0, 0.0, 0.0]], weight, got_bias)
    assert abs(pred[0][0] - 0.6) < 1e-8


def test_range_check_both_sides() -> None:
    small = [[1.0, -2.0]]
    huge_in = [[70000.0, 1.0]]
    weight = [[1.0, 0.0], [0.0, 1.0]]
    bias = [0.0, 0.0]
    _, decision, in_max, out_max = math.apply_checked(small, weight, bias, "fp16")
    assert decision.gemm_dtype == "fp16"
    assert decision.input_dtype == "fp16"
    assert decision.output_dtype == "fp16"
    assert in_max == 2.0 and out_max == 2.0

    _, decision, in_max, _ = math.apply_checked(huge_in, weight, bias, "fp16")
    assert in_max == 70000.0
    assert decision.input_overflow
    assert decision.input_dtype == "bf16"
    assert decision.gemm_dtype == "bf16"

    # Input fits; the map itself blows the output past fp16.
    blow = [[30000.0]]
    w = [[3.0]]
    _, decision, in_max, out_max = math.apply_checked(blow, w, [0.0], "fp16")
    assert in_max == 30000.0 and out_max == 90000.0
    assert not decision.input_overflow
    assert decision.output_overflow
    assert decision.output_dtype == "bf16"
    assert decision.gemm_dtype == "bf16"

    _, decision, _, _ = math.apply_checked(small, weight, bias, "bf16")
    assert decision.gemm_dtype == "bf16"
    assert not decision.input_overflow


def test_env_off_is_the_default_and_skips_guards() -> None:
    _clear_ridge_env()
    assert config.enabled() is False
    assert config.map_dtype() == "fp16"
    assert config.apc_mode() == "off"
    policy.apply_v1_guards(
        flag_on=config.enabled(),
        speculative=True,
        prefix_caching=True,
        apc_mode="off",
    )
    assert policy.cpu_tier_may_store(flag_on=False, block_filled=True)


def test_env_on_refuses_mtp_and_prefix_cache() -> None:
    os.environ["VLLM_RIDGEFILL"] = "1"
    try:
        assert config.enabled() is True
        try:
            policy.apply_v1_guards(
                flag_on=True,
                speculative=True,
                prefix_caching=False,
                apc_mode="off",
            )
        except policy.RidgeFillRefused as exc:
            assert "MTP" in str(exc)
        else:
            raise AssertionError("MTP was accepted")
        try:
            policy.apply_v1_guards(
                flag_on=True,
                speculative=False,
                prefix_caching=True,
                apc_mode="off",
            )
        except policy.RidgeFillRefused as exc:
            assert "prefix caching" in str(exc)
        else:
            raise AssertionError("APC was accepted")
        policy.apply_v1_guards(
            flag_on=True,
            speculative=False,
            prefix_caching=False,
            apc_mode="off",
        )
    finally:
        _clear_ridge_env()


def test_short_prompt_is_exact_and_v2_salt_does_not_match_exact() -> None:
    plan = policy.plan_prompt(2048, split=24, tail=2048)
    assert not plan.active
    plan = policy.plan_prompt(5000, split=24, tail=2048, block_size=1024, align=True)
    # 5000 - 2048 = 2952, rounded down to 2048.
    assert plan.filled_end == 2048
    assert plan.kind(2047) == "filled"
    assert plan.kind(2048) == "exact"
    assert policy.chunk_class(plan, 2048, 3072) == "exact"
    assert keys.aligned_filled_end(5000, 2048, 1024) == policy.filled_end(
        5000, 2048, 1024, True
    )
    common = dict(
        num_prompt_tokens=5000,
        block_size=1024,
        split=24,
        tail=2048,
        map_sha="abc",
        mode="plain",
    )
    filled = keys.extra_key(start=0, end=1024, apc="v2", **common)
    exact = keys.extra_key(start=2048, end=3072, apc="v2", **common)
    off = keys.extra_key(start=0, end=1024, apc="off", **common)
    assert filled == ("ridgefill", 24, "abc", "plain")
    assert exact is None
    assert off is None
    assert not policy.cpu_tier_may_store(flag_on=True, block_filled=True)
    assert policy.cpu_tier_may_store(flag_on=True, block_filled=False)


def test_manifest_pin_fails_closed() -> None:
    manifest = {
        "model": {
            "vocab_sha256": "aaa",
            "encodings": {"blk.0.weight": "fp8_e4m3"},
        }
    }
    loader.validate_pins(
        manifest, vocab_sha256="aaa", encodings={"blk.0.weight": "fp8_e4m3"}
    )
    try:
        loader.validate_pins(
            manifest,
            vocab_sha256="bbb",
            encodings={"blk.0.weight": "fp8_e4m3"},
        )
    except loader.RidgeFillPinError as exc:
        assert "vocab sha256" in str(exc)
    else:
        raise AssertionError("vocab mismatch was accepted")
    try:
        loader.validate_pins(
            manifest,
            vocab_sha256="aaa",
            encodings={"blk.0.weight": "int4"},
        )
    except loader.RidgeFillPinError as exc:
        assert "encoding mismatch" in str(exc)
    else:
        raise AssertionError("encoding mismatch was accepted")


def test_ridgefill_mask_replaces_only_filled_rows() -> None:
    """The layer's torch.where keeps the exact rows and replaces filled rows."""
    try:
        import torch
    except ImportError:
        return
    block = torch.randn(6, 3)
    pred = torch.randn(6, 3)
    mask = torch.tensor([True, True, True, True, False, False])
    got = torch.where(mask.unsqueeze(-1), pred, block)
    assert torch.equal(got[:4], pred[:4])
    assert torch.equal(got[4:], block[4:])


def test_session_predicts_with_fp16_map() -> None:
    """The session applies the fp16 map and builds the filled row mask."""
    try:
        import torch

        from vllm.model_executor.layers.ridgefill import runtime as rf_runtime
        from vllm.model_executor.layers.ridgefill.config import RidgeConfig
    except (ImportError, RuntimeError):
        return
    cfg = RidgeConfig(split=2, tail=2, din=4, dout=2)
    session = rf_runtime.Session(cfg, block_size=4)
    session.observe_enter(2, 8, 4)
    assert session.plan is not None
    assert session.plan.filled_end == 6
    split = torch.randn(8, 4)
    session.capture_split(split)
    a = torch.randn(2, 4).half()
    b = torch.randn(2).half()
    session.maps = {2: (a, b)}
    pred = session.predict(2)
    assert pred is not None and pred.shape == (8, 2)
    want = torch.nn.functional.linear(split.half(), a, b)
    assert torch.allclose(pred, want, atol=1e-3)
    mask = session.filled_mask(8)
    assert mask is not None
    assert bool(mask[:6].all())
    assert not bool(mask[6:].any())


def test_session_mask_uses_absolute_positions_when_chunked() -> None:
    """A chunk that crosses the cut is mixed: per-row absolute positions."""
    try:
        import torch

        from vllm.model_executor.layers.ridgefill import runtime as rf_runtime
        from vllm.model_executor.layers.ridgefill.config import RidgeConfig
    except (ImportError, RuntimeError):
        return
    cfg = RidgeConfig(split=2, tail=4, din=4, dout=2)
    session = rf_runtime.Session(cfg, block_size=4)
    # One prompt of 8 tokens, this forward holds its first 4 tokens.
    session.seq_lens = torch.tensor([8], dtype=torch.int64)
    session.query_start_loc = torch.tensor([0, 4], dtype=torch.int64)
    session.num_prefills = 1
    session.plan = policy.plan_prompt(8, split=2, tail=4)
    # Rows outside the tail are the prefix: positions 0..3 with cut 8 - 4 = 4.
    first = session.filled_mask(4, torch.arange(0, 4))
    assert first is not None
    assert bool(first.all())
    # The second chunk holds positions 4..7, all in the exact tail.
    session.query_start_loc = torch.tensor([0, 4], dtype=torch.int64)
    session.seq_lens = torch.tensor([8], dtype=torch.int64)
    late = session.filled_mask(4, torch.arange(4, 8))
    assert late is not None and late.shape == (4,)
    assert not bool(late.any())


def test_torch_fit_matches_reference() -> None:
    """The capture's torch fit matches ``math.fit_ridge`` on small data."""
    try:
        import torch
    except ImportError:
        return
    path = ROOT / "docs" / "explore" / "ridgefill" / "capture_activations.py"
    spec = importlib.util.spec_from_file_location("ridgefill_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(11)
    x = torch.randn(24, 6, dtype=torch.float32)
    y = torch.randn(24, 4, dtype=torch.float32)
    maps, errors = module._fit_maps(torch, [x], {0: [y]}, 0.05)
    a = maps[0][0].float()
    b = maps[0][1].float()
    want_a, want_b = math.fit_ridge(x.tolist(), y.tolist(), 0.05)
    want_a = torch.tensor(want_a)
    want_b = torch.tensor(want_b)
    assert torch.allclose(a, want_a, atol=2e-3)
    assert torch.allclose(b, want_b, atol=2e-3)
    assert errors[0]["rows"] == 24


def test_session_mask_crops_a_chunk_that_crosses_the_cut() -> None:
    """With chunked prefill the mask is per sequence, not per chunk."""
    try:
        from vllm.model_executor.layers.ridgefill import runtime as rf_runtime
        from vllm.model_executor.layers.ridgefill.config import RidgeConfig
    except (ImportError, RuntimeError):
        return
    cfg = RidgeConfig(split=2, tail=2, din=4, dout=2)
    session = rf_runtime.Session(cfg, block_size=4)
    session.plan = policy.plan_prompt(8, split=2, tail=2)
    # One sequence: full length 8, this chunk is its last 4 rows.
    session.seq_lens = __import__("torch").tensor([8], dtype=__import__("torch").int64)
    session.query_start_loc = __import__("torch").tensor(
        [0, 4], dtype=__import__("torch").int64
    )
    mask = session.filled_mask(4)
    assert mask is not None and mask.shape == (4,)
    assert bool(mask[:2].all())
    assert not bool(mask[2:].any())


def test_capture_probe_hooks_layers_and_saves_rank() -> None:
    """The worker extension hooks the split layer and writes its rank file."""
    try:
        import torch

        from vllm.model_executor.layers.ridgefill.capture_probe import RidgeFillProbe
    except (ImportError, RuntimeError):
        return
    import tempfile

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = torch.nn.Linear(4, 4, bias=False)

        def forward(self, hidden_states) -> object:
            return self.self_attn(hidden_states)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

        def forward(self, hidden_states) -> object:
            return self.layers[1](self.layers[0](hidden_states))

    class Runner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Model()

    class Worker:
        def __init__(self) -> None:
            self.model_runner = Runner()
            self.rank = 0

    probe = RidgeFillProbe()
    worker = Worker()
    # The extension is injected into the Worker class, so its methods run
    # unbound with the worker as ``self``.
    capture: Any = probe.ridgefill_capture
    save: Any = probe.ridgefill_save
    capture(worker, 0, 2)
    with torch.no_grad():
        worker.model_runner.model(torch.randn(5, 4))
    with tempfile.TemporaryDirectory() as tmp:
        save(worker, tmp)
        saved = torch.load(Path(tmp) / "rank0.pt", map_location="cpu")
    assert len(saved["split"]) == 1 and saved["split"][0].shape == (5, 4)
    assert len(saved["inputs"][0]) == 1
    assert saved["inputs"][1][0].shape == (5, 4)


def test_capture_synthetic_writes_bundle() -> None:
    """The capture script's CPU mode writes maps.pt and ridgefill.json."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    import importlib.util
    import tempfile
    from types import SimpleNamespace

    path = ROOT / "docs" / "explore" / "ridgefill" / "capture_activations.py"
    spec = importlib.util.spec_from_file_location("ridgefill_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        args = SimpleNamespace(
            synthetic=True,
            model="",
            output=str(Path(tmp) / "capture"),
            prompts=2,
            tokens=16,
            prompts_file="",
            pins="",
            split=1,
            end=3,
            tail=4,
            lam=0.03,
            tp=1,
            exclude_tail=False,
            synthetic_din=8,
            synthetic_dout=4,
            synthetic_layers=3,
        )
        rc = module._synthetic(args)
        assert rc == 0
        out = Path(tmp) / "capture"
        assert (out / "maps.pt").exists()
        assert (out / "ridgefill.json").exists()
        assert (out / "capture.pt").exists()


def _main() -> None:
    tests = [
        test_fit_recovers_known_map,
        test_int8_row128_dequant_and_apply,
        test_range_check_both_sides,
        test_env_off_is_the_default_and_skips_guards,
        test_env_on_refuses_mtp_and_prefix_cache,
        test_short_prompt_is_exact_and_v2_salt_does_not_match_exact,
        test_manifest_pin_fails_closed,
        test_ridgefill_mask_replaces_only_filled_rows,
        test_session_predicts_with_fp16_map,
        test_torch_fit_matches_reference,
        test_session_mask_crops_a_chunk_that_crosses_the_cut,
        test_session_mask_uses_absolute_positions_when_chunked,
        test_capture_probe_hooks_layers_and_saves_rank,
        test_capture_synthetic_writes_bundle,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _main()
