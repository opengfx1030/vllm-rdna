# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling-2-Mini serving recipes (print mode).

``serve_rdna.sh PRINT=1`` needs only a VENV directory (the stub python below cannot
import torch, nothing touches a GPU). Four recipes cover TP=1, TP=2 and TP=4:
plain TP=1 for full 180k single-card context, TP=1 DSpark, and unified TP=2 / TP=4
recipes with DSpark ON by default plus ``ADAPTIVE=1``, ``DRAFT_VOCAB=...`` and
``DSPARK=0`` switches.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
DROPPED = (
    "HIP_VISIBLE_DEVICES",
    "GPUIDS",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "PCI_BUSES",
    "VENV",
    "MODEL",
    "VIRTUAL_ENV",
    "ADAPTIVE",
    "DSPARK",
    "DRAFT_VOCAB",
)

# recipe -> (tp, port, max_model_len, max_num_seqs, max_num_batched_tokens, spec)
RECIPES = {
    "inkling2-mini-tp1": (1, 18221, 180000, 16, 1024, None),
    "inkling2-mini-tp1-dspark": (
        1,
        18221,
        115000,
        16,
        1024,
        (
            '{"method":"dspark","num_speculative_tokens":7,'
            '"num_speculative_tokens_per_batch_size":[[1,2,2],[3,16,0]]}'
        ),
    ),
    "inkling2-mini-tp2": (
        2,
        18222,
        720000,
        8,
        1024,
        (
            '{"method":"dspark","num_speculative_tokens":7,'
            '"num_speculative_tokens_per_batch_size":[[1,2,2],[3,8,0]]}'
        ),
    ),
    "inkling2-mini-tp4": (
        4,
        18223,
        1048576,
        8,
        4096,
        (
            '{"method":"dspark","num_speculative_tokens":7,'
            '"num_speculative_tokens_per_batch_size":[[1,2,2],[3,8,0]]}'
        ),
    ),
}


@pytest.fixture()
def venv(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    return tmp_path / "venv"


def plan(venv, *args, **env):
    full = {k: v for k, v in os.environ.items() if k not in DROPPED}
    full.update(VENV=str(venv), MODEL="/nonexistent/model", PRINT="1", **env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=60,
    )


def flag(out: str, name: str, value: str) -> bool:
    return f"'{name}' '{value}'" in out


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_plan(venv, recipe):
    tp, port, max_len, seqs, max_batched, spec = RECIPES[recipe]
    result = plan(venv, f"RECIPE={recipe}")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert flag(out, "--served-model-name", "inkling2-mini")
    assert flag(out, "--port", str(port))
    assert flag(out, "--tensor-parallel-size", str(tp))
    assert flag(out, "--max-model-len", str(max_len))
    assert flag(out, "--max-num-seqs", str(seqs))
    assert flag(out, "--max-num-batched-tokens", str(max_batched))
    assert flag(out, "--dtype", "float16")
    assert flag(out, "--gpu-memory-utilization", "0.92")
    assert flag(out, "--limit-mm-per-prompt", '{"image":1}')
    assert "'--enable-prefix-caching'" in out
    assert "'--enable-prompt-tokens-details'" in out
    assert "--kv-cache-memory-bytes" not in out
    assert "--attention-backend" not in out
    if spec is None:
        assert "--speculative-config" not in out
    else:
        assert flag(out, "--speculative-config", spec)


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_captures_graphs_on_the_default_grid(venv, recipe):
    out = plan(venv, f"RECIPE={recipe}").stdout
    assert flag(out, "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE"}')
    assert "--enforce-eager" not in out
    assert "cudagraph_capture_sizes" not in out


@pytest.mark.parametrize("recipe", RECIPES)
def test_eager_override_drops_the_graphs(venv, recipe):
    out = plan(venv, f"RECIPE={recipe}", "EAGER=1").stdout
    assert "'--enforce-eager'" in out
    assert "--compilation-config" not in out


@pytest.mark.parametrize("recipe", RECIPES)
def test_all_reduce_environment(venv, recipe):
    tp = RECIPES[recipe][0]
    out = plan(venv, f"RECIPE={recipe}").stdout
    assert f"export VLLM_RDNA_AR='{1 if tp > 1 else 0}'" in out
    if tp > 1:
        assert "export VLLM_RDNA_AR_MAX_KB='256'" in out
        assert "export VLLM_RDNA_AR_ONESHOT_KB='256'" in out
        assert "export VLLM_FORCE_CUSTOM_ALL_REDUCE='0'" in out


def test_tp2_lowers_the_kv_margin_floor_and_the_others_keep_it(venv):
    tp2 = plan(venv, "RECIPE=inkling2-mini-tp2").stdout
    assert "export VLLM_KV_CACHE_MARGIN_FLOOR_MIB='2048'" in tp2
    restored = plan(venv, "RECIPE=inkling2-mini-tp2", "VLLM_KV_CACHE_MARGIN_FLOOR_MIB=")
    assert "export VLLM_KV_CACHE_MARGIN_FLOOR_MIB=''" in restored.stdout
    for recipe in (
        "inkling2-mini-tp1",
        "inkling2-mini-tp1-dspark",
        "inkling2-mini-tp4",
    ):
        assert (
            "VLLM_KV_CACHE_MARGIN_FLOOR_MIB"
            not in plan(venv, f"RECIPE={recipe}").stdout
        )


def test_no_rocblas_tunableop_rows(venv):
    for recipe in RECIPES:
        out = plan(venv, f"RECIPE={recipe}").stdout
        assert "# tunableop: disabled (TUNABLEOP=0)" in out
        assert "PYTORCH_TUNABLEOP_ENABLED" not in out


def test_ports_by_tp_are_distinct():
    ports = {(RECIPES[r][0], RECIPES[r][1]) for r in RECIPES}
    assert len(ports) == 3


@pytest.mark.parametrize(
    ("recipe", "max_len", "schedule"),
    [
        ("inkling2-mini-tp1-dspark", 95000, "[[1,1,7],[2,2,2],[3,16,0]]"),
        ("inkling2-mini-tp2", 660000, "[[1,2,7],[3,8,0]]"),
        ("inkling2-mini-tp4", 1048576, "[[1,2,7],[3,8,0]]"),
    ],
)
def test_adaptive_switch_and_optional_draft_vocab(
    venv, tmp_path, recipe, max_len, schedule
):
    out = plan(venv, f"RECIPE={recipe}", "ADAPTIVE=1").stdout
    assert flag(out, "--max-model-len", str(max_len))
    assert flag(
        out,
        "--speculative-config",
        (
            '{"method":"dspark","num_speculative_tokens":7,'
            f'"num_speculative_tokens_per_batch_size":{schedule},'
            '"speculator_options":{"adaptive_k":{}}}'
        ),
    )

    ids = tmp_path / "ids.npy"
    for out_v in (
        plan(venv, f"RECIPE={recipe}", "ADAPTIVE=1", f"DRAFT_VOCAB={ids}").stdout,
        plan(venv, f"RECIPE={recipe}", ADAPTIVE="1", DRAFT_VOCAB=str(ids)).stdout,
    ):
        assert flag(out_v, "--max-model-len", str(max_len))
        assert flag(
            out_v,
            "--speculative-config",
            (
                '{"method":"dspark","num_speculative_tokens":7,'
                f'"num_speculative_tokens_per_batch_size":{schedule},'
                f'"speculator_options":{{"adaptive_k":{{}},"draft_vocab":"{ids}"}}}}'
            ),
        )


def test_dspark_zero_restores_plain_tp2_and_tp4(venv):
    tp2 = plan(venv, "RECIPE=inkling2-mini-tp2", "DSPARK=0").stdout
    assert "--speculative-config" not in tp2
    assert flag(tp2, "--max-model-len", "1048576")
    assert flag(tp2, "--max-num-batched-tokens", "320")

    tp4 = plan(venv, "RECIPE=inkling2-mini-tp4", "DSPARK=0").stdout
    assert "--speculative-config" not in tp4
    assert flag(tp4, "--max-model-len", "1048576")
    assert flag(tp4, "--max-num-batched-tokens", "4096")


@pytest.mark.parametrize(
    "recipe",
    ["inkling2-mini-tp1-dspark", "inkling2-mini-tp2", "inkling2-mini-tp4"],
)
def test_caller_overrides_win(venv, recipe):
    out = plan(venv, f"RECIPE={recipe}", PORT="19999", SEQS="4").stdout
    assert flag(out, "--port", "19999")
    assert flag(out, "--max-num-seqs", "4")
    out2 = plan(venv, f"RECIPE={recipe}", "MAXLEN=90000", "SPEC_CONFIG=").stdout
    assert flag(out2, "--max-model-len", "90000")
    assert "--speculative-config" not in out2
