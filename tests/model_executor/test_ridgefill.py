# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the RidgeFill draft. No torch and no vLLM import.

Run with pytest, or directly::

    python3 tests/model_executor/test_ridgefill.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

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


def _main() -> None:
    tests = [
        test_fit_recovers_known_map,
        test_int8_row128_dequant_and_apply,
        test_range_check_both_sides,
        test_env_off_is_the_default_and_skips_guards,
        test_env_on_refuses_mtp_and_prefix_cache,
        test_short_prompt_is_exact_and_v2_salt_does_not_match_exact,
        test_manifest_pin_fails_closed,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _main()
