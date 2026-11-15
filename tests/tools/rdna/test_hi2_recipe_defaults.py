# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recipes 27b-hi2-tp2 and 27b-hi2-tp2-mtp2: HI2 switch defaults, opt-outs, MTP delta.

The "Switches" table of docs/rdna2/hi2-27b-tp2-serving.md is the contract: one row per
switch, the recipe default in the first cell and the one-line opt-out in the last. The
code defaults stay off; only these two recipes turn the switches on.

``serve_rdna.sh PRINT=1`` resolves a recipe the way a launch does (a stub venv keeps
it off torch and the GPUs), so these tests read what a server would be started with,
not what the file text says. ``TUNABLEOP_PREFER`` is a launcher variable and never an
export: it is compared on the sourced recipe here, and its effect on the TunableOp
profile is tested in test_hi2_tunableop_profile.py.
"""

import os
import subprocess
from pathlib import Path

import pytest
import regex as re

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
RECIPES = REPO / "tools/rdna/recipes"
DOC = REPO / "docs/rdna2/hi2-27b-tp2-serving.md"
HI2 = ("27b-hi2-tp2", "27b-hi2-tp2-mtp2")
# The validated switches; the table may carry more (rows still pending), never fewer.
VALIDATED = {
    "RDNA_HINT_FLEX_GEMV",
    "VLLM_FA_RDNA2_UNFUSED_PREFILL",
    "VLLM_RDNA_GEMMA_RESIDUAL_DIRECT",
    "VLLM_RDNA_GDN_DECODE_INPLACE",
    "RDNA_HINT_FLEX_FUSE_SILU",
    "VLLM_RDNA_LM_HEAD_GEMV_MAX_M",
    "TUNABLEOP_PREFER",
}
NOT_EXPORTED = {"TUNABLEOP_PREFER"}  # read by the launcher, not by the server
DROPPED = (
    "HIP_VISIBLE_DEVICES",
    "GPUIDS",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "PCI_BUSES",
    "VENV",
    "MODEL",
    "VIRTUAL_ENV",
    "TUNABLEOP_PROFILE",
)
CELL = re.compile(r"`([A-Z][A-Z0-9_]*)=([^`]*)`")
ASSIGNMENT = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)=", re.MULTILINE)
EXPORT = re.compile(r"^export ([A-Za-z0-9_]+)='(.*)'$", re.MULTILINE)


def switch_table():
    """Doc table: key -> (recipe default, opt-out); "" default = not set by recipes."""
    text = DOC.read_text()
    assert "\n## Switches\n" in text, "the serving doc lost its Switches section"
    section = text.split("\n## Switches\n", 1)[1].split("\n## ", 1)[0]
    table = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 4, f"switch row needs 4 cells: {line}"
        default, off = CELL.match(cells[0]), CELL.search(cells[3])
        assert default and off and default[1] == off[1], line
        table[default[1]] = (default[2], off[2])
    return table


def recipe_keys(name):
    return set(ASSIGNMENT.findall((RECIPES / f"{name}.env").read_text()))


def recipe_values(name):
    """KEY -> value of each assignment a recipe makes, sourced as the launcher does."""
    keys = sorted(recipe_keys(name))
    script = 'source "$1"; shift; for k; do printf "%s=%s\\0" "$k" "${!k}"; done'
    out = subprocess.run(
        ["bash", "-c", script, "bash", str(RECIPES / f"{name}.env"), *keys],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ["PATH"]},
    ).stdout
    return dict(item.split("=", 1) for item in out.split("\0") if item)


def clean_env():
    """The caller's environment minus all that a recipe or the table could pre-empt."""
    own = set(DROPPED) | set(switch_table())
    for path in RECIPES.glob("*.env"):
        own |= recipe_keys(path.stem)
    return {k: v for k, v in os.environ.items() if k not in own}


@pytest.fixture(scope="module")
def venv(tmp_path_factory):
    root = tmp_path_factory.mktemp("hi2-venv")
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    return root / "venv"


def plan(venv, *args, **env):
    full = clean_env()
    full.update(VENV=str(venv), MODEL="/nonexistent/model", PRINT="1", **env)
    result = subprocess.run(
        ["bash", str(LAUNCHER), *args],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def exports(stdout):
    return dict(EXPORT.findall(stdout))


def exported_switches():
    return {k: v for k, v in switch_table().items() if k not in NOT_EXPORTED}


def test_table_carries_the_validated_switches_and_a_real_opt_out():
    table = switch_table()
    assert table.keys() >= VALIDATED
    for key, (default, off) in table.items():
        assert default != off, f"{key}: the opt-out equals the default"


@pytest.mark.parametrize("name", HI2)
def test_recipe_sources_to_the_table_defaults(name):
    values = recipe_values(name)
    for key, (default, _) in switch_table().items():
        if default:
            assert values.get(key) == default, (name, key)
        else:
            assert key not in values, f"{name} sets {key}, the table says not a default"


@pytest.mark.parametrize("name", HI2)
def test_plan_exports_the_table_defaults(venv, name):
    resolved = exports(plan(venv, f"RECIPE={name}"))
    for key, (default, _) in exported_switches().items():
        if default:
            assert resolved.get(key) == default, (name, key)
        else:
            assert key not in resolved, (name, key)


@pytest.fixture(scope="module")
def baseline(venv):
    return {name: exports(plan(venv, f"RECIPE={name}")) for name in HI2}


@pytest.mark.parametrize("name", HI2)
@pytest.mark.parametrize("key", sorted(VALIDATED - NOT_EXPORTED))
def test_trailing_opt_out_wins_and_touches_nothing_else(venv, baseline, name, key):
    before = baseline[name]
    off = switch_table()[key][1]
    after = exports(plan(venv, f"RECIPE={name}", f"{key}={off}"))
    assert after[key] == off
    assert {k: v for k, v in after.items() if k != key} == {
        k: v for k, v in before.items() if k != key
    }


@pytest.mark.parametrize("name", HI2)
def test_caller_environment_opt_out_wins(venv, name):
    off = {k: v[1] for k, v in exported_switches().items() if v[0]}
    resolved = exports(plan(venv, f"RECIPE={name}", **off))
    assert {k: resolved[k] for k in off} == off


@pytest.mark.parametrize("name", HI2)
def test_trailing_argument_wins_over_the_caller_environment(venv, name):
    resolved = exports(
        plan(
            venv,
            f"RECIPE={name}",
            "VLLM_RDNA_GDN_DECODE_INPLACE=1",
            VLLM_RDNA_GDN_DECODE_INPLACE="0",
        )
    )
    assert resolved["VLLM_RDNA_GDN_DECODE_INPLACE"] == "1"


@pytest.mark.parametrize("name", HI2)
def test_gdn_in_place_keeps_its_prerequisite(name):
    """The in-place GDN decode needs the conv1d update kernel off, else it warns and
    keeps the per-layer arenas."""
    assert recipe_values(name)["VLLM_CAUSAL_CONV1D_RDNA2_UPDATE"] == "0"


def code_lines(name):
    return [
        line
        for line in (RECIPES / f"{name}.env").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_two_recipes_differ_only_in_mtp():
    base, mtp2 = (code_lines(name) for name in HI2)
    assert [x for x in base if x != "MTP=0"] == [x for x in mtp2 if x != "MTP=2"]
    assert "MTP=0" in base and "MTP=2" in mtp2


def test_mtp0_recipe_plan(venv):
    out = plan(venv, "RECIPE=27b-hi2-tp2")
    assert '"cudagraph_capture_sizes":[1,2,4,8]' in out
    assert "--speculative-config" not in out


def test_mtp2_recipe_plan(venv):
    out = plan(venv, "RECIPE=27b-hi2-tp2-mtp2")
    assert '"cudagraph_capture_sizes":[3,6,12,24]' in out
    assert '"num_speculative_tokens":2' in out


@pytest.mark.parametrize("name", HI2)
def test_recipe_serving_shape(venv, name):
    out = plan(venv, f"RECIPE={name}")
    for fragment in (
        "'--tensor-parallel-size' '2'",
        "'--served-model-name' 'q27hi2'",
        "'--port' '18211'",
        "'--kv-cache-memory-bytes' '17179869184'",
        "'--max-num-seqs' '8'",
        "'--max-num-batched-tokens' '2048'",
        '"cudagraph_mode":"FULL_AND_PIECEWISE"',
    ):
        assert fragment in out, (name, fragment)


def test_prefill_gemm_recipe_line_follows_the_table():
    """The recipes set PREFILL_GEMM as the doc table says, else a commented line."""
    default = switch_table()["RDNA_HINT_FLEX_PREFILL_GEMM"][0]
    for name in HI2:
        text = (RECIPES / f"{name}.env").read_text()
        if default:
            assert f"\nRDNA_HINT_FLEX_PREFILL_GEMM={default}\n" in text, name
        else:
            assert "\n# RDNA_HINT_FLEX_PREFILL_GEMM=hip\n" in text, name


def test_no_other_recipe_sets_a_switch():
    keys = set(switch_table())
    for path in sorted(RECIPES.glob("*.env")):
        if path.stem in HI2:
            continue
        assert not keys & recipe_keys(path.stem), path.name
