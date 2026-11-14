# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of benchmarks/replayssm/e2e_decode_speedup.py.

The script runs a GPU benchmark. These tests cover what needs no GPU: argument
parsing, the LLM(...) keyword arguments and the environment of each arm, the
report, and --dry-run (which must import neither torch nor vllm). The existing
arms (standard, replayssm) must behave as they did before the LeapQuant arm
was added.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "replayssm"
    / "e2e_decode_speedup.py"
)
RETIRED_ENV = "VLLM_RDNA_GDN_LEAPQUANT"
STATE_DTYPE = "replayssm_state_dtype"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("e2e_decode_speedup", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# LLM(...) arguments of the two original arms with default options, in the order
# the script built them before the LeapQuant arm existed.
BASELINE_KWARGS = {
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "tensor_parallel_size": 1,
    "dtype": "bfloat16",
    "max_model_len": 1256,
    "trust_remote_code": True,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 16384,
    "enforce_eager": False,
    "disable_log_stats": True,
    "gpu_memory_utilization": 0.9,
    "mamba_ssm_cache_dtype": "auto",
    "enable_flashinfer_autotune": False,
}


def test_default_arms_and_llm_kwargs_are_unchanged(bench):
    args = bench.parse_args([])
    assert args.modes == ["standard", "replayssm"]
    standard = bench.build_llm_kwargs(args, "standard")
    assert standard == BASELINE_KWARGS
    assert list(standard) == list(BASELINE_KWARGS)
    replayssm = bench.build_llm_kwargs(args, "replayssm")
    assert replayssm == {
        **BASELINE_KWARGS,
        "use_replayssm": True,
        "replayssm_buffer_len": 16,
    }
    assert list(replayssm) == [
        *BASELINE_KWARGS,
        "use_replayssm",
        "replayssm_buffer_len",
    ]


def test_default_environment_is_the_callers_environment(bench):
    args = bench.parse_args([])
    environ = {"PATH": "/bin", "HF_HOME": "/hf"}
    for mode in ("standard", "replayssm"):
        assert bench.arm_env(args, mode, environ) == environ


def test_default_report_is_unchanged(bench, monkeypatch, capsys):
    """Same table and the same speedup line as before for the two old arms."""
    fake = {
        "standard": {"mode": "standard", "elapsed_s": 2.0, "tok_s": 128000.0,
                     "per_step_ms": 2.0},
        "replayssm": {"mode": "replayssm", "elapsed_s": 1.0, "tok_s": 256000.0,
                      "per_step_ms": 1.0},
    }  # fmt: skip
    monkeypatch.setattr(bench, "run_one_mode", lambda args, mode: fake[mode])
    bench.main([])
    header = f"{'mode':<10}{'ms/step':>12}{'tok/s':>16}{'wall (s)':>12}"
    assert capsys.readouterr().out == (
        "model=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16  batch_size=256  "
        "steps=1000  buffer_len=16  dtype=bfloat16\n"
        "\n"
        f"{header}\n"
        f"{'-' * len(header)}\n"
        f"{'standard':<10}{2.0:>12.3f}{128000.0:>16,.0f}{2.0:>12.3f}\n"
        f"{'ReplaySSM':<10}{1.0:>12.3f}{256000.0:>16,.0f}{1.0:>12.3f}\n"
        f"{'-' * len(header)}\n"
        "speedup (standard / ReplaySSM, per step): 2.000x\n"
    )


def test_leapquant_arm_differs_from_the_baseline_by_the_state_dtype_only(bench):
    args = bench.parse_args(["--modes", "standard,leapquant"])
    standard = bench.build_llm_kwargs(args, "standard")
    assert STATE_DTYPE not in standard
    assert bench.build_llm_kwargs(args, "leapquant") == {
        **standard,
        STATE_DTYPE: "leapquant",
    }
    # No environment switch any more: both arms run in the caller's environment.
    environ = {"PATH": "/bin"}
    assert bench.arm_env(args, "standard", environ) == environ
    assert bench.arm_env(args, "leapquant", environ) == environ


def test_report_has_one_speedup_line_per_arm_after_the_baseline(
    bench, monkeypatch, capsys
):
    fake = {
        mode: {"mode": mode, "elapsed_s": 1.0, "tok_s": 1.0, "per_step_ms": ms}
        for mode, ms in (("standard", 4.0), ("leapquant", 2.0), ("replayssm", 8.0))
    }
    monkeypatch.setattr(bench, "run_one_mode", lambda args, mode: fake[mode])
    bench.main(["--modes", "standard,leapquant,replayssm"])
    out = capsys.readouterr().out
    assert "speedup (standard / LeapQuant, per step): 2.000x\n" in out
    assert "speedup (standard / ReplaySSM, per step): 0.500x\n" in out


def test_the_retired_environment_variable_stops_every_arm(bench):
    args = bench.parse_args(["--modes", "standard,leapquant"])
    for mode in ("standard", "replayssm", "leapquant"):
        with pytest.raises(SystemExit, match=RETIRED_ENV):
            bench.arm_env(args, mode, {RETIRED_ENV: "1"})
    # The engine ignores the variable, so only "1" (the launcher's test) stops.
    assert bench.arm_env(args, "standard", {RETIRED_ENV: "0"}) == {RETIRED_ENV: "0"}


def test_env_option_reaches_every_arm_but_cannot_set_the_retired_variable(bench):
    args = bench.parse_args(
        ["--modes", "standard,leapquant", "--env", "A=1", "--env", "B=x=y"]
    )
    for mode in ("standard", "leapquant"):
        env = bench.arm_env(args, mode, {})
        assert env["A"] == "1" and env["B"] == "x=y"
    with pytest.raises(SystemExit):
        bench.parse_args(["--env", f"{RETIRED_ENV}=1"])


def test_llm_kwarg_cannot_set_the_state_dtype(bench):
    """A baseline must not run LeapQuant by a stray --llm-kwarg, and the
    leapquant arm must not be overridden."""
    with pytest.raises(SystemExit):
        bench.parse_args(["--llm-kwarg", f'{STATE_DTYPE}="leapquant"'])


def test_rdna_options_reach_the_llm_kwargs_and_survive_the_worker_command(bench):
    argv = [
        "--modes", "standard,leapquant", "--num-speculative-tokens", "2",
        "--monokernel", "--tensor-parallel-size", "4",
        "--llm-kwarg", 'quantization="awq"',
        "--llm-kwarg", 'compilation_config={"cudagraph_capture_sizes": [3, 6]}',
        "--dtype", "float16", "--mamba-ssm-cache-dtype", "float32",
        "--no-disable-flashinfer-autotune",
    ]  # fmt: skip
    args = bench.parse_args(argv)
    kwargs = bench.build_llm_kwargs(args, "leapquant")
    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["dtype"] == "float16"
    assert kwargs["mamba_ssm_cache_dtype"] == "float32"
    assert "enable_flashinfer_autotune" not in kwargs
    assert kwargs["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 2,
        "use_local_argmax_reduction": True,
    }
    assert kwargs["monokernel"] is True
    assert kwargs["quantization"] == "awq"
    assert kwargs["compilation_config"] == {"cudagraph_capture_sizes": [3, 6]}
    # The worker is a fresh process: its command must rebuild the same kwargs.
    cmd = bench.worker_command(args, "leapquant")
    assert cmd[2:4] == ["--worker", "leapquant"]
    worker_args = bench.parse_args(cmd[2:])
    assert bench.build_llm_kwargs(worker_args, "leapquant") == kwargs


@pytest.mark.parametrize(
    "argv",
    [
        ["--modes", "standard,nope"],
        ["--modes", "standard,standard"],
        ["--modes", ""],
        ["--num-speculative-tokens", "-1"],
        ["--llm-kwarg", "quantization=awq"],  # not JSON
        ["--llm-kwarg", "novalue"],
        ["--env", "novalue"],
    ],
)
def test_bad_arguments_are_rejected(bench, argv):
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(argv)
    assert exc.value.code == 2


def _clean_env():
    return {k: v for k, v in os.environ.items() if k != RETIRED_ENV}


def test_help_lists_the_leapquant_arm():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env=_clean_env(),
        check=True,
    ).stdout
    assert "--modes" in out and "leapquant" in out and "--dry-run" in out


def test_dry_run_prints_every_arm_and_imports_neither_torch_nor_vllm():
    # -S: no site-packages, so an import of torch or vllm would fail the run.
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{str(SCRIPT)!r}, '--dry-run', '--modes', 'standard,leapquant',"
        " '--num-speculative-tokens', '2', '--monokernel']\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
        "loaded = {'torch', 'vllm'} & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    out = subprocess.run(
        [sys.executable, "-S", "-c", code],
        capture_output=True,
        text=True,
        env=_clean_env(),
        check=True,
    ).stdout
    lines = out.splitlines()
    assert lines[0] == "arm standard (standard)"
    assert "arm leapquant (LeapQuant)" in lines
    assert lines.count("  env +{}") == 2  # no arm adds an environment variable
    llm_lines = [line for line in lines if line.startswith("  LLM ")]
    kwargs_by_arm = [json.loads(line[len("  LLM ") :]) for line in llm_lines]
    assert [STATE_DTYPE in kwargs for kwargs in kwargs_by_arm] == [False, True]
    assert kwargs_by_arm[1][STATE_DTYPE] == "leapquant"
    assert len(llm_lines) == 2
    for line in llm_lines:
        kwargs = json.loads(line[len("  LLM ") :])
        assert kwargs["speculative_config"]["num_speculative_tokens"] == 2
        assert kwargs["monokernel"] is True
