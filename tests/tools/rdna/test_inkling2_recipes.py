# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling-2-Mini serving recipes (print mode).

``serve_rdna.sh PRINT=1`` needs only a VENV directory (the stub python below cannot
import torch, nothing touches a GPU). The three recipes are the configurations the box
runs of docs/rdna2/inkling2-serving.md validated: FULL_AND_PIECEWISE HIP graphs on the
model's default capture grid, prefix caching, one image per prompt, the one-shot HIP
all-reduce on a tensor-parallel group, and the model's length: the measured fit at
TP=1, 1,048,576 at TP=2 and TP=4.
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
)

# recipe -> (tp, port, max_model_len, max_num_seqs, max_num_batched_tokens)
RECIPES = {
    "inkling2-mini-tp1": (1, 18221, 180000, 16, 1024),
    "inkling2-mini-tp2": (2, 18222, 1048576, 8, 320),
    "inkling2-mini-tp4": (4, 18223, 1048576, 8, 4096),
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
    tp, port, max_len, seqs, max_batched = RECIPES[recipe]
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
    # The cache is sized from the memory measured after warmup and graph capture.
    assert "--kv-cache-memory-bytes" not in out
    # No attention backend flag: the model's own HIP attention.
    assert "--attention-backend" not in out
    assert "--speculative-config" not in out


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_captures_graphs_on_the_default_grid(venv, recipe):
    out = plan(venv, f"RECIPE={recipe}").stdout
    # FULL_AND_PIECEWISE with no explicit sizes: the engine's grid for this architecture
    # (decode graphs for every request count up to min(16, SEQS), vllm/config/vllm.py).
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
    # At TP=2 the full context plus the graphs does not fit within the engine's 3 GiB
    # margin floor (docs/rdna2/inkling2-serving.md, "Long context"); the recipe lowers
    # it to 2 GiB and says so. An empty value restores the engine default.
    tp2 = plan(venv, "RECIPE=inkling2-mini-tp2").stdout
    assert "export VLLM_KV_CACHE_MARGIN_FLOOR_MIB='2048'" in tp2
    restored = plan(venv, "RECIPE=inkling2-mini-tp2", "VLLM_KV_CACHE_MARGIN_FLOOR_MIB=")
    assert "export VLLM_KV_CACHE_MARGIN_FLOOR_MIB=''" in restored.stdout
    for recipe in ("inkling2-mini-tp1", "inkling2-mini-tp4"):
        assert (
            "VLLM_KV_CACHE_MARGIN_FLOOR_MIB"
            not in plan(venv, f"RECIPE={recipe}").stdout
        )


def test_no_rocblas_tunableop_rows(venv):
    # Nothing of the serving path goes through rocBLAS: no TunableOp profile is chosen.
    for recipe in RECIPES:
        out = plan(venv, f"RECIPE={recipe}").stdout
        assert "# tunableop: disabled (TUNABLEOP=0)" in out
        assert "PYTORCH_TUNABLEOP_ENABLED" not in out


def test_ports_are_distinct():
    assert len({port for _, port, *_ in RECIPES.values()}) == len(RECIPES)


# DSpark recipe -> (base, max_len, seqs, max_batched_override, spec)
DSPARK = {
    "inkling2-mini-tp1-dspark": (
        "inkling2-mini-tp1",
        115000,
        16,
        None,
        (
            '{"method":"dspark","num_speculative_tokens":7,'
            '"num_speculative_tokens_per_batch_size":[[1,2,2],[3,16,0]]}'
        ),
    ),
    "inkling2-mini-tp2-dspark": (
        "inkling2-mini-tp2",
        720000,
        8,
        1024,
        (
            '{"method":"dspark","num_speculative_tokens":7,'
            '"num_speculative_tokens_per_batch_size":[[1,2,2],[3,8,0]]}'
        ),
    ),
    "inkling2-mini-tp4-dspark": (
        "inkling2-mini-tp4",
        1048576,
        8,
        None,
        (
            '{"method":"dspark","num_speculative_tokens":7,'
            '"num_speculative_tokens_per_batch_size":[[1,2,2],[3,8,0]]}'
        ),
    ),
}


@pytest.mark.parametrize("recipe", DSPARK)
def test_dspark_recipe_is_its_base_recipe_plus_the_draft(venv, recipe):
    base, max_len, seqs, max_bat, spec = DSPARK[recipe]
    result = plan(venv, f"RECIPE={recipe}")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert flag(out, "--speculative-config", spec)
    assert flag(out, "--max-model-len", str(max_len))
    assert flag(out, "--max-num-seqs", str(seqs))
    if max_bat is not None:
        assert flag(out, "--max-num-batched-tokens", str(max_bat))
    # Everything else is the base recipe's plan, flag for flag and variable for
    # variable: the launcher's own keys on the command line give the same plan.
    overrides = [
        f"RECIPE={base}",
        f"SEQS={seqs}",
        f"MAXLEN={max_len}",
        f"SPEC_CONFIG={spec}",
    ]
    if max_bat is not None:
        overrides.append(f"MAXBAT={max_bat}")
    ref = plan(venv, *overrides)
    assert ref.returncode == 0, ref.stderr
    assert out.replace(recipe, base) == ref.stdout


@pytest.mark.parametrize("recipe", DSPARK)
def test_dspark_recipe_keys_follow_the_caller(venv, recipe):
    _, max_len, _, _, spec = DSPARK[recipe]
    # The caller's environment wins over the recipe and over the recipe it builds on
    # (PORT is set only by the latter).
    out = plan(venv, f"RECIPE={recipe}", PORT="19999", SEQS="4").stdout
    assert flag(out, "--port", "19999")
    assert flag(out, "--max-num-seqs", "4")
    assert flag(out, "--max-model-len", str(max_len))
    assert flag(out, "--speculative-config", spec)
    # KEY=value after RECIPE= wins over both; an empty SPEC_CONFIG drops the draft.
    out = plan(venv, f"RECIPE={recipe}", "MAXLEN=90000", "SPEC_CONFIG=").stdout
    assert flag(out, "--max-model-len", "90000")
    assert "--speculative-config" not in out


ADAPTIVE = "inkling2-mini-tp1-dspark-adaptive"


def adaptive_spec(ids) -> str:
    return (
        '{"method":"dspark","num_speculative_tokens":7,'
        '"num_speculative_tokens_per_batch_size":[[1,2,7],[3,16,0]],'
        f'"speculator_options":{{"adaptive_k":{{}},"draft_vocab":"{ids}"}}}}'
    )


def test_adaptive_recipe_names_the_ids_file(venv, tmp_path):
    ids = tmp_path / "ids.npy"
    # as a launcher argument and as an environment variable
    for out in (
        plan(venv, f"RECIPE={ADAPTIVE}", f"DRAFT_VOCAB={ids}").stdout,
        plan(venv, f"RECIPE={ADAPTIVE}", DRAFT_VOCAB=str(ids)).stdout,
    ):
        assert flag(out, "--speculative-config", adaptive_spec(ids))
        assert flag(out, "--max-num-seqs", "16")
        assert flag(out, "--max-model-len", "95000")
        assert flag(out, "--tensor-parallel-size", "1")
    default = plan(venv, f"RECIPE={ADAPTIVE}").stdout
    assert "/.cache/rdna/inkling2-mini-draft-vocab-32768.npy" in default


def test_adaptive_recipe_keys_of_every_level_follow_the_caller(venv):
    # PORT is set by inkling2-mini-tp1, two recipes below this one.
    out = plan(venv, f"RECIPE={ADAPTIVE}", PORT="19999", DRAFT_VOCAB="x.npy").stdout
    assert flag(out, "--port", "19999")
    assert flag(out, "--gpu-memory-utilization", "0.92")
