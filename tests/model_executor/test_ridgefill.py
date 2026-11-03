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
        if name.startswith("VLLM_RDNA_RIDGEFILL"):
            del os.environ[name]


def _has_vllm_envs() -> bool:
    """The knobs live in ``vllm.envs``; the pure-Python host has no vllm."""
    try:
        import vllm.envs  # noqa: F401
    except ImportError:
        return False
    return True


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


def test_fused_map_is_opt_in_and_inert_without_maps() -> None:
    """``..._MAP_FUSED`` defaults off and the wide launch needs maps.

    The fused launch is one HIP op for all late layers, so it stays opt-in
    until the A/B measures it. With no fitted maps it returns None and the
    per-layer path is untouched.
    """
    if not _has_vllm_envs():
        return
    _clear_ridge_env()
    import vllm.envs as vllm_envs

    assert vllm_envs.VLLM_RDNA_RIDGEFILL_MAP_FUSED is False
    cfg = config.load_config()
    assert cfg.map_fused is False
    runtime = _load("runtime")
    session = runtime.Session(cfg, block_size=4)
    assert session._predict_fused(object(), cfg.dout) is None


def test_env_off_is_the_default_and_skips_guards() -> None:
    if not _has_vllm_envs():
        return
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


def test_flag_off_leaves_the_default_path_untouched() -> None:
    """Unset flag: no extra-key salt, no session, optional forward args.

    The non-RidgeFill serve must be byte-identical to the base tree. The
    block-hash salt returns before importing the draft, and the model's
    RidgeFill arguments stay optional with the flag-off defaults.
    """
    if not _has_vllm_envs():
        return
    _clear_ridge_env()
    import inspect

    from vllm.models.qwen4_exp.amd import model as flashnext_model
    from vllm.v1.core import kv_cache_utils

    assert config.enabled() is False
    assert kv_cache_utils._ridgefill_block_extra_key(object(), 0, 16) is None
    sig = inspect.signature(flashnext_model.Qwen4ExpModel.forward)
    assert sig.parameters["ridgefill_filled"].default is None
    assert sig.parameters["ridgefill_exact"].default is None
    assert sig.parameters["ridgefill_has_exact"].default is False


def test_env_on_refuses_mtp_and_prefix_cache() -> None:
    if not _has_vllm_envs():
        return
    os.environ["VLLM_RDNA_RIDGEFILL"] = "1"
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
    session.begin_forward(torch.arange(8), 8)
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


def test_session_mask_accepts_mrope_positions() -> None:
    """Bench sends M-RoPE positions (3 x tokens); do not compare a list."""
    try:
        import torch

        from vllm.model_executor.layers.ridgefill import runtime as rf_runtime
        from vllm.model_executor.layers.ridgefill.config import RidgeConfig
    except (ImportError, RuntimeError):
        return
    cfg = RidgeConfig(split=2, tail=4, din=4, dout=2)
    session = rf_runtime.Session(cfg, block_size=4)
    session.seq_lens = torch.tensor([8], dtype=torch.int64)
    session.query_start_loc = torch.tensor([0, 4], dtype=torch.int64)
    session.plan = policy.plan_prompt(8, split=2, tail=4)
    positions = torch.arange(0, 4, dtype=torch.int64).repeat(3, 1)
    mask = session.filled_mask(4, positions)
    assert mask is not None and mask.shape == (4,)
    assert bool(mask.all())


def test_model_ridgefill_steps_are_opaque_ops() -> None:
    """The fill steps are registered custom ops, not a Dynamo graph break.

    ``fullgraph`` refuses a ``torch._dynamo.disable`` split point, so the
    model must reach the fill work through ``torch.ops.vllm.ridgefill_*``.
    """
    try:
        import torch

        from vllm.model_executor.layers.ridgefill import ops as rf_ops
        from vllm.models.qwen4_exp.amd import model as flashnext_model
    except (ImportError, RuntimeError):
        return
    assert not hasattr(flashnext_model, "_ridgefill_apply")
    for name in (
        "ridgefill_capture",
        "ridgefill_predict",
        "ridgefill_replace",
        "ridgefill_cache_only",
    ):
        assert hasattr(torch.ops.vllm, name), name
    # The fake impls must produce the shapes the graph is traced with.
    x_s = torch.empty(8, 4)
    filled = torch.tensor([0, 1, 2])
    assert rf_ops._capture_fake(x_s).shape == x_s.shape
    assert rf_ops._predict_fake(x_s, filled, 2, 2).shape == (3, 2)
    assert rf_ops._cache_only_fake(x_s, None, filled, filled, "l").shape == x_s.shape


def test_cache_only_op_writes_then_attends_the_exact_rows() -> None:
    """The cache-only op writes filled K/V first, then attends exact rows.

    Filled rows precede the cut, so every exact row's causal window sees the
    filled K/V. The op must keep that order and leave the filled rows' output
    at zero: their residual is unused.
    """
    try:
        import torch

        from vllm.model_executor.layers.ridgefill import ops as rf_ops
    except (ImportError, RuntimeError):
        return

    class FakeAttn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[int] | None]] = []

        def cache_only_forward(self, hidden_states, positions, row_index) -> None:
            self.calls.append(("cache", row_index.tolist()))

        def __call__(self, hidden_states, positions, row_index=None):
            self.calls.append(
                ("attn", None if row_index is None else row_index.tolist())
            )
            return hidden_states * 2

    class FakeLayer:
        def __init__(self) -> None:
            self.self_attn = FakeAttn()

    layer = FakeLayer()
    rf_ops.register_layer("fake", layer)
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    filled = torch.tensor([0, 1, 2])
    exact = torch.tensor([3, 4, 5])
    out = rf_ops.ridgefill_cache_only(x, None, filled, exact, "fake")
    assert layer.self_attn.calls == [("cache", [0, 1, 2]), ("attn", [3, 4, 5])]
    assert torch.equal(out[exact], x[exact] * 2)
    assert torch.equal(out[filled], torch.zeros(3, 4))


def test_compiled_fill_branch_is_opaque() -> None:
    """A ``fullgraph`` compile accepts the fill branch: the ops are opaque.

    ``fullgraph`` refuses a ``torch._dynamo.disable`` split point, which is
    why the flag-on boot failed. With the fill steps as custom ops the same
    branch compiles, and a forward with no filled rows traces the flag-off
    graph because the guard is a Python ``None`` check.
    """
    try:
        import torch

        from vllm.model_executor.layers.ridgefill import ops as rf_ops
    except (ImportError, RuntimeError):
        return

    class FakeSession:
        def __init__(self) -> None:
            self.split_state = None

        def capture_split(self, hidden_states) -> None:
            self.split_state = hidden_states.detach()

        def predict_rows(self, x_s, filled_index, layer_idx, dout):
            return x_s.index_select(0, filled_index)

    rf_ops.set_session(FakeSession())

    def step(x, filled):
        if filled is None:
            return x + 1
        x_s = torch.ops.vllm.ridgefill_capture(x)
        pred = torch.ops.vllm.ridgefill_predict(x_s, filled, 0, x.shape[1])
        torch.ops.vllm.ridgefill_replace(x, pred, filled)
        return x + 1

    compiled = torch.compile(step, fullgraph=True, backend="eager")
    x = torch.randn(6, 4)
    assert torch.allclose(compiled(x, None), x + 1)
    filled = torch.tensor([0, 1, 2])
    assert torch.allclose(compiled(x.clone(), filled), x + 1)


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


# --------------------------------------------------------------------------
# Row split and cache-write ledger (items 1-2). Pure Python first, then the
# torch-backed session checks. These are the CPU proofs the GPU A/B relies on.
# --------------------------------------------------------------------------

rows = _load("rows")
cost = _load("cost")


def _load_runtime():
    """Load ``runtime`` without importing vllm's compiled package.

    On the box the real package imports; on a bare CPU host it does not. The
    fallback registers bare namespace modules and loads each file by path.
    """
    try:
        from vllm.model_executor.layers.ridgefill import runtime as real

        return real
    except Exception:
        pass
    import importlib.util
    import types

    full = "vllm.model_executor.layers.ridgefill"
    if f"{full}.runtime" in sys.modules:
        return sys.modules[f"{full}.runtime"]
    # Drop a partially imported real package so the bare namespace wins.
    for key in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
        del sys.modules[key]
    for name, path in (
        ("vllm", ROOT / "vllm"),
        ("vllm.model_executor", ROOT / "vllm" / "model_executor"),
        ("vllm.model_executor.layers", ROOT / "vllm" / "model_executor" / "layers"),
        (full, PKG),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    for short in ("config", "loader", "policy", "rows", "cost", "runtime"):
        name = f"{full}.{short}"
        spec = importlib.util.spec_from_file_location(name, PKG / f"{short}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{full}.runtime"]


def _metadata():
    """A packed two-sequence forward that crosses the cut in sequence 0."""
    seq_lens = [12, 12]
    qsl = [0, 8, 12]
    cuts = [8, 8]
    return seq_lens, qsl, cuts


def _packed_positions():
    """Absolute positions matching ``_metadata``: seq0 starts at 4, seq1 at 8."""
    import torch

    return torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11])


def test_row_split_partitions_every_row() -> None:
    seq_lens, qsl, cuts = _metadata()
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts)
    rows.assert_partition(split)
    assert split.num_tokens == 12
    assert split.filled_idx == (0, 1, 2, 3)
    assert split.exact_idx == (4, 5, 6, 7, 8, 9, 10, 11)
    assert split.per_seq_filled == (4, 0)
    assert split.per_seq_exact == (4, 4)
    # Order is preserved inside every sequence and across the pack.
    assert list(split.filled_idx) == sorted(split.filled_idx)
    assert list(split.exact_idx) == sorted(split.exact_idx)


def test_row_split_arithmetic_matches_positions() -> None:
    """The metadata arithmetic and the positions authority agree."""
    seq_lens, qsl, cuts = _metadata()
    positions: list[int] = []
    for seq, length in enumerate(seq_lens):
        start, end = qsl[seq], qsl[seq + 1]
        first = length - (end - start)
        positions.extend(first + row for row in range(end - start))
    arithmetic = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts)
    from_positions = rows.build_row_split_from_positions(
        seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts, positions=positions
    )
    assert arithmetic.filled_idx == from_positions.filled_idx
    assert arithmetic.exact_idx == from_positions.exact_idx


def test_row_split_prefix_and_decode() -> None:
    whole = rows.prefix_split(8, 6)
    assert whole.filled_idx == (0, 1, 2, 3, 4, 5)
    assert whole.exact_idx == (6, 7)
    decode = rows.prefix_split(1, 0)
    assert not decode.active
    assert decode.exact_idx == (0,)


def test_subset_query_start_loc_keeps_sequence_groups() -> None:
    _, qsl, _ = _metadata()
    # Exact rows of the packed forward: 4..11.
    subset = rows.subset_query_start_loc(qsl, tuple(range(4, 12)))
    assert subset == (0, 4, 8)
    # Filled rows of sequence 0 only.
    subset = rows.subset_query_start_loc(qsl, (0, 1, 2, 3))
    assert subset == (0, 4, 4)


def test_cache_streams_write_every_row_once() -> None:
    seq_lens, qsl, cuts = _metadata()
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts)
    slots = list(range(100, 112))
    positions: list[int] = []
    for seq, length in enumerate(seq_lens):
        start, end = qsl[seq], qsl[seq + 1]
        first = length - (end - start)
        positions.extend(first + row for row in range(end - start))
    for name in ("main_kv", "indexer_raw", "indexer_compressed"):
        stream = rows.plan_cache_stream(
            name,
            split,
            slots=slots,
            positions=positions,
            ring=name.endswith("compressed"),
        )
        assert not stream.overlapping(), name
        writes = stream.writes
        assert len(writes) == split.num_tokens
        assert sorted(w.row for w in writes) == list(range(split.num_tokens))
        # Each row keeps the slot and absolute position the full path used.
        for write in writes:
            assert write.slot == slots[write.row]
            assert write.position == positions[write.row]


def test_compressed_group_straddle_is_detected_and_alignment_removes_it() -> None:
    seq_lens, qsl, cuts = _metadata()
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts)
    straddles = rows.straddling_groups(
        split, seq_lens=seq_lens, query_start_loc=qsl, compress_ratio=4
    )
    # Sequence 0's cut at 8 sits mid-group 4..7 -> 8..11? No: cut 8 is a group
    # boundary for ratio 4, so only sequence 0's first chunk crossing matters.
    assert straddles == ()
    # Move the cut to 6: group 4..7 now spans the split in sequence 0.
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=[6, 8])
    straddles = rows.straddling_groups(
        split, seq_lens=seq_lens, query_start_loc=qsl, compress_ratio=4
    )
    assert straddles == ((0, 4),)
    # Rounding the cut down to the group unit removes the straddle.
    aligned = rows.align_cut(6, block_size=2, compress_ratio=4)
    assert aligned == 0
    split = rows.build_row_split(
        seq_lens=seq_lens, query_start_loc=qsl, cuts=[aligned, 8]
    )
    assert (
        rows.straddling_groups(
            split, seq_lens=seq_lens, query_start_loc=qsl, compress_ratio=4
        )
        == ()
    )


def test_cuts_from_lengths_aligns_to_block_and_ratio() -> None:
    cuts = rows.cuts_from_lengths([4096, 4097], tail=2048, block_size=16, align=True)
    assert cuts == (2048, 2048)
    cuts = rows.cuts_from_lengths(
        [4096], tail=2048, block_size=16, align=True, compress_ratio=4
    )
    # 64-token unit: 2048 is already a multiple.
    assert cuts == (2048,)
    cuts = rows.cuts_from_lengths(
        [4100], tail=2048, block_size=16, align=True, compress_ratio=4
    )
    assert cuts == (2048,)


def test_row_split_rejects_bad_metadata() -> None:
    for kwargs in (
        dict(seq_lens=[4], query_start_loc=[1, 4], cuts=[2]),
        dict(seq_lens=[4], query_start_loc=[0], cuts=[2]),
        dict(seq_lens=[2], query_start_loc=[0, 4], cuts=[2]),
        dict(seq_lens=[4], query_start_loc=[0, 4], cuts=[]),
    ):
        try:
            rows.build_row_split(**kwargs)
        except rows.RowSplitError:
            continue
        raise AssertionError(f"accepted bad metadata: {kwargs}")


def test_forward_cost_table_drops_per_layer_syncs() -> None:
    """Before: three syncs per late layer. After: five per forward."""
    layers, filled = 24, 12_000
    before = cost.cost_before(layers=layers, filled_rows=filled)
    after = cost.cost_after(layers=layers, filled_rows=filled)
    assert before.syncs == 4 + 3 * layers
    assert after.syncs == 5
    assert after.launches < before.launches // 100
    # Decode reads ``num_prefills`` (a plain int) and never syncs.
    assert cost.cost_decode() == cost.ForwardCost(0, 0, 0)
    columns = cost.table(layers=layers, filled_rows=filled, cache_only=True)
    assert set(columns) == {"before", "after", "cache_only"}
    assert columns["cache_only"].syncs == 0
    # The cache-only extra is per QSA layer, and defaults to the upper bound.
    per_qsa = cost.cost_cache_only(layers=layers, filled_rows=filled, qsa_layers=6)
    assert per_qsa.launches == 36
    assert columns["cache_only"].launches == 6 * layers


def test_session_builds_the_row_plan_once_per_forward() -> None:
    """Four late layers reuse one plan and one set of device tensors."""
    try:
        import torch
    except ImportError:
        return
    try:
        runtime = _load_runtime()
    except Exception:
        return
    cfg = runtime.RidgeConfig(split=2, tail=4, din=4, dout=2)
    session = runtime.Session(cfg, block_size=4)
    session.seq_lens = torch.tensor([12, 12], dtype=torch.int64)
    session.query_start_loc = torch.tensor([0, 8, 12], dtype=torch.int64)
    session.num_prefills = 2
    calls: list[int] = []
    original = session._build_rows

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    session._build_rows = counted
    positions = _packed_positions()
    session.begin_forward(
        positions,
        12,
        metadata=(
            torch.tensor([12, 12], dtype=torch.int64),
            torch.tensor([0, 8, 12], dtype=torch.int64),
            2,
        ),
    )
    assert len(calls) == 1
    assert session.rows is not None
    assert session.rows.filled_idx == (0, 1, 2, 3)
    split_state = torch.randn(12, 4)
    session.capture_split(split_state)
    # The mask is cached: returning it does not rebuild or read the host.
    first = session.filled_mask(12, positions)
    second = session.filled_mask(12, positions)
    assert first is second is session.fill_mask_t
    assert bool(first[:4].all()) and not bool(first[4:].any())


def test_filled_prediction_is_the_full_prediction_rows() -> None:
    """The F-row map GEMM equals the matching rows of the T-row GEMM."""
    try:
        import torch
    except ImportError:
        return
    try:
        runtime = _load_runtime()
    except Exception:
        return
    cfg = runtime.RidgeConfig(split=2, tail=4, din=4, dout=2)
    session = runtime.Session(cfg, block_size=4)
    session.seq_lens = torch.tensor([12], dtype=torch.int64)
    session.query_start_loc = torch.tensor([0, 12], dtype=torch.int64)
    session.num_prefills = 1
    session.begin_forward(
        torch.arange(12),
        12,
        metadata=(
            torch.tensor([12], dtype=torch.int64),
            torch.tensor([0, 12], dtype=torch.int64),
            1,
        ),
    )
    torch.manual_seed(7)
    split_state = torch.randn(12, 4)
    session.capture_split(split_state)
    a = torch.randn(2, 4).half()
    b = torch.randn(2).half()
    session.maps = {2: (a, b)}
    full = session.predict(2)
    part = session.predict(2, filled_only=True)
    assert full.shape == (12, 2)
    assert part.shape == (session.rows.filled_count, 2)
    want = full.index_select(0, session.filled_index)
    assert torch.allclose(part.float(), want.float(), atol=1e-6)


def test_fill_tensors_partition_the_forward() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        runtime = _load_runtime()
    except Exception:
        return
    cfg = runtime.RidgeConfig(split=2, tail=4, din=4, dout=2)
    session = runtime.Session(cfg, block_size=4)
    session.seq_lens = torch.tensor([12, 12], dtype=torch.int64)
    session.query_start_loc = torch.tensor([0, 8, 12], dtype=torch.int64)
    session.num_prefills = 2
    session.begin_forward(
        _packed_positions(),
        12,
        metadata=(
            torch.tensor([12, 12], dtype=torch.int64),
            torch.tensor([0, 8, 12], dtype=torch.int64),
            2,
        ),
    )
    session.capture_split(torch.randn(12, 4))
    mask, filled, exact = session.fill_tensors()
    assert mask is not None and filled is not None and exact is not None
    assert int(mask.sum()) == filled.numel()
    assert filled.numel() + exact.numel() == 12
    assert torch.equal(
        mask, torch.zeros(12, dtype=torch.bool).index_fill_(0, filled, True)
    )
    for index in (filled, exact):
        assert torch.equal(index, index.sort().values)
        assert index.unique().numel() == index.numel()


def test_session_cache_streams_cover_the_forward() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        runtime = _load_runtime()
    except Exception:
        return
    cfg = runtime.RidgeConfig(split=2, tail=4, din=4, dout=2)
    session = runtime.Session(cfg, block_size=4)
    session.seq_lens = torch.tensor([12, 12], dtype=torch.int64)
    session.query_start_loc = torch.tensor([0, 8, 12], dtype=torch.int64)
    session.num_prefills = 2
    session.begin_forward(
        _packed_positions(),
        12,
        metadata=(
            torch.tensor([12, 12], dtype=torch.int64),
            torch.tensor([0, 8, 12], dtype=torch.int64),
            2,
        ),
    )
    slots = list(range(50, 62))
    positions = list(range(12))
    streams = session.cache_streams(slots=slots, positions=positions)
    assert [s.name for s in streams] == [
        "main_kv",
        "indexer_raw",
        "indexer_compressed",
    ]
    for stream in streams:
        assert not stream.overlapping()
        assert len(stream.writes) == 12
        assert len(stream.filled) == 4


def test_row_kind_matches_the_index_sets() -> None:
    seq_lens, qsl, cuts = _metadata()
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts)
    for row in split.filled_idx:
        assert split.kind(row) == "filled"
    for row in split.exact_idx:
        assert split.kind(row) == "exact"


def test_index_copy_replaces_the_same_rows_as_where() -> None:
    """The cached-index replacement is the bool-mask replacement, row for row."""
    try:
        import torch
    except ImportError:
        return
    seq_lens, qsl, cuts = _metadata()
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts)
    torch.manual_seed(3)
    block = torch.randn(12, 4)
    filled_index = torch.tensor(split.filled_idx, dtype=torch.long)
    mask = torch.tensor(split.filled_mask())
    pred_filled = torch.randn(len(split.filled_idx), 4)
    pred_full = torch.zeros(12, 4).index_copy(0, filled_index, pred_filled)
    got = block.index_copy(0, filled_index, pred_filled)
    want = torch.where(mask.unsqueeze(-1), pred_full, block)
    assert torch.equal(got, want)


# --- cache-write equivalence: one full pass vs filled then exact -----------


def _fake_row_value(row: int, width: int, dtype):
    """A deterministic per-row vector: same row, same bytes, every pass."""
    import torch

    base = torch.arange(width, dtype=torch.float32) * 0.5
    return (base + float(row)).to(dtype)


def _write_cache(cache, slots, rows, width):
    """Write each row's K/V at its own slot. One launch per row, like the ledger."""
    order = []
    for row in rows:
        cache[slots[row]] = _fake_row_value(row, width, cache.dtype)
        order.append((row, slots[row]))
    return order


def _cache_only_equivalence(split, width: int):
    import torch

    nrows = split.num_tokens
    slots = [101 + 3 * row for row in range(nrows)]
    dtype = torch.float16
    # One pass over every row, the way the layer runs without the split.
    full = torch.zeros(max(slots) + 1, width, dtype=dtype)
    _write_cache(full, slots, list(range(nrows)), width)
    # Two passes: filled rows first, then the exact rows.
    split_cache = torch.zeros(max(slots) + 1, width, dtype=dtype)
    filled_order = _write_cache(split_cache, slots, split.filled_idx, width)
    exact_order = _write_cache(split_cache, slots, split.exact_idx, width)
    assert torch.equal(full, split_cache)
    # Order inside each pass is packed row order; every row appears once.
    written = [row for row, _ in filled_order] + [row for row, _ in exact_order]
    assert sorted(written) == list(range(nrows))
    assert len(written) == len(set(written))
    assert written == list(split.filled_idx) + list(split.exact_idx)
    # Slots are the row's own slot in both passes.
    for row, slot in filled_order + exact_order:
        assert slot == slots[row]


def test_cache_write_equivalence_full_vs_split() -> None:
    """The two-pass cache-only write leaves the cache byte-identical."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    seq_lens, qsl, cuts = _metadata()
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=cuts)
    assert split.filled_idx == (0, 1, 2, 3)
    _cache_only_equivalence(split, 8)


def test_cache_write_equivalence_when_the_split_is_mid_group() -> None:
    """A cut that lands mid compressed group still writes every row once."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    seq_lens, qsl, _ = _metadata()
    # Cut at 6: group 4..7 straddles, so the alignment matters for the ring.
    split = rows.build_row_split(seq_lens=seq_lens, query_start_loc=qsl, cuts=[6, 8])
    _cache_only_equivalence(split, 8)
    straddles = rows.straddling_groups(
        split, seq_lens=seq_lens, query_start_loc=qsl, compress_ratio=4
    )
    assert straddles == ((0, 4),)
    # The ring writer writes a group once, in group units. Aligning the cut
    # removes the straddle; a straddling group would be written twice.
    aligned = rows.align_cut(6, block_size=2, compress_ratio=4)
    fixed = rows.build_row_split(
        seq_lens=seq_lens, query_start_loc=qsl, cuts=[aligned, 8]
    )
    assert (
        rows.straddling_groups(
            fixed, seq_lens=seq_lens, query_start_loc=qsl, compress_ratio=4
        )
        == ()
    )
    _cache_only_equivalence(fixed, 8)


def _row_view():
    """Load ``common/qsa_row_view.py`` without importing the vllm package."""
    import importlib.util
    import sys

    name = "rf_qsa_row_view"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "vllm/models/qwen4_exp/common/qsa_row_view.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fake_metadata(rows_present: int = 12):
    """A dataclass shaped like QSAForwardMetadata's row-indexed fields."""
    from dataclasses import dataclass

    import torch

    @dataclass
    class Fake:
        block_table: torch.Tensor
        seq_lens: torch.Tensor
        slot_mapping: torch.Tensor
        query_start_loc: torch.Tensor
        token_to_req: torch.Tensor
        logical_positions: torch.Tensor
        visible_blocks: torch.Tensor
        k_work_metadata: torch.Tensor
        num_actual_tokens: int
        num_decode_tokens: int
        num_prefill_tokens: int
        num_prefills: int

    n = rows_present
    return Fake(
        block_table=torch.tensor([[0, 1], [2, 3]]),
        seq_lens=torch.tensor([12, 12]),
        slot_mapping=torch.arange(n) + 100,
        query_start_loc=torch.tensor([0, 8, 12]),
        token_to_req=torch.tensor([0] * 8 + [1] * 4),
        logical_positions=torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11]),
        visible_blocks=torch.tensor([8] * 8 + [4] * 4),
        k_work_metadata=torch.tensor([[0, 3], [1, 2]]),
        num_actual_tokens=n,
        num_decode_tokens=0,
        num_prefill_tokens=n,
        num_prefills=2,
    )


def test_row_view_gathers_rows_and_keeps_requests() -> None:
    try:
        import torch
    except ImportError:
        return
    view_mod = _row_view()
    metadata = _fake_metadata()
    index = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
    view = view_mod.row_view(metadata, index)
    # Row-indexed fields follow the rows.
    assert torch.equal(view.slot_mapping, metadata.slot_mapping.index_select(0, index))
    assert torch.equal(view.token_to_req, metadata.token_to_req.index_select(0, index))
    assert torch.equal(
        view.logical_positions, metadata.logical_positions.index_select(0, index)
    )
    assert torch.equal(
        view.visible_blocks, metadata.visible_blocks.index_select(0, index)
    )
    # Request-indexed fields stay.
    assert view.block_table is metadata.block_table
    assert view.seq_lens is metadata.seq_lens
    assert view.k_work_metadata is metadata.k_work_metadata
    # Counts follow the rows.
    assert view.num_actual_tokens == 8
    assert view.num_prefill_tokens == 8
    # The subset cumulative counts match the pure-Python authority.
    assert view.query_start_loc.tolist() == list(
        rows.subset_query_start_loc([0, 8, 12], tuple(range(4, 12)))
    )
    # The original is not mutated.
    assert metadata.num_actual_tokens == 12


def test_row_view_is_identity_without_an_index_and_caches_per_object() -> None:
    try:
        import torch
    except ImportError:
        return
    view_mod = _row_view()
    metadata = _fake_metadata()
    assert view_mod.row_view(metadata, None) is metadata
    index = torch.tensor([0, 1, 2, 3])
    first = view_mod.row_view(metadata, index)
    second = view_mod.row_view(metadata, index)
    assert first is second
    # A different row set builds a different view, and the first still stands.
    other = view_mod.row_view(metadata, torch.tensor([8, 9, 10, 11]))
    assert other is not first
    assert first.num_actual_tokens == 4 and other.num_actual_tokens == 4
    assert first.slot_mapping.tolist() == [100, 101, 102, 103]
    assert other.slot_mapping.tolist() == [108, 109, 110, 111]
    # A non-dataclass is refused rather than silently mishandled.
    try:
        view_mod.row_view(object(), index)
    except TypeError:
        pass
    else:
        raise AssertionError("row_view accepted a non-dataclass")


def test_build_rows_prefers_positions_and_matches_arithmetic_without_them() -> None:
    """Positions are the authority; the metadata arithmetic is equivalent."""
    try:
        import torch
    except ImportError:
        return
    try:
        runtime = _load_runtime()
    except Exception:
        return

    class NoRead:
        def detach(self):
            raise AssertionError("row split skipped the positions authority")

    cfg = runtime.RidgeConfig(split=2, tail=4, din=4, dout=2)
    session = runtime.Session(cfg, block_size=4)
    session.seq_lens = torch.tensor([12, 12], dtype=torch.int64)
    session.query_start_loc = torch.tensor([0, 8, 12], dtype=torch.int64)
    session.num_prefills = 2
    session.plan = runtime.plan_prompt(12, split=2, tail=4)
    # Positions present: they are read and decide the split.
    try:
        session._build_rows(12, NoRead())
    except AssertionError:
        pass
    else:
        raise AssertionError("positions were not read")
    with_positions = session._build_rows(12, _packed_positions())
    without = session._build_rows(12, None)
    assert with_positions.filled_idx == without.filled_idx == (0, 1, 2, 3)
    assert with_positions.exact_idx == without.exact_idx


def _main() -> None:
    tests = [
        test_fit_recovers_known_map,
        test_int8_row128_dequant_and_apply,
        test_range_check_both_sides,
        test_env_off_is_the_default_and_skips_guards,
        test_fused_map_is_opt_in_and_inert_without_maps,
        test_flag_off_leaves_the_default_path_untouched,
        test_env_on_refuses_mtp_and_prefix_cache,
        test_short_prompt_is_exact_and_v2_salt_does_not_match_exact,
        test_manifest_pin_fails_closed,
        test_ridgefill_mask_replaces_only_filled_rows,
        test_session_predicts_with_fp16_map,
        test_torch_fit_matches_reference,
        test_session_mask_crops_a_chunk_that_crosses_the_cut,
        test_session_mask_uses_absolute_positions_when_chunked,
        test_session_mask_accepts_mrope_positions,
        test_model_ridgefill_steps_are_opaque_ops,
        test_cache_only_op_writes_then_attends_the_exact_rows,
        test_compiled_fill_branch_is_opaque,
        test_capture_probe_hooks_layers_and_saves_rank,
        test_capture_synthetic_writes_bundle,
        test_row_split_partitions_every_row,
        test_row_split_arithmetic_matches_positions,
        test_row_split_prefix_and_decode,
        test_subset_query_start_loc_keeps_sequence_groups,
        test_cache_streams_write_every_row_once,
        test_compressed_group_straddle_is_detected_and_alignment_removes_it,
        test_cuts_from_lengths_aligns_to_block_and_ratio,
        test_row_split_rejects_bad_metadata,
        test_forward_cost_table_drops_per_layer_syncs,
        test_session_builds_the_row_plan_once_per_forward,
        test_filled_prediction_is_the_full_prediction_rows,
        test_fill_tensors_partition_the_forward,
        test_session_cache_streams_cover_the_forward,
        test_row_kind_matches_the_index_sets,
        test_index_copy_replaces_the_same_rows_as_where,
        test_cache_write_equivalence_full_vs_split,
        test_cache_write_equivalence_when_the_split_is_mid_group,
        test_row_view_gathers_rows_and_keeps_requests,
        test_row_view_is_identity_without_an_index_and_caches_per_object,
        test_build_rows_prefers_positions_and_matches_arithmetic_without_them,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _main()
