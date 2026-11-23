# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/rdna/recipes: the README index matches the folder, headers cite real files.

The recipes are a flat folder looked up by bare name (``serve_rdna.sh RECIPE=<name>``),
so the README inventory is the one place that lists them. These tests keep that table
honest: a recipe without a row, a row without a recipe, a header that cites a doc or a
test that is not in the tree, and a ``source`` line that names a missing recipe all
fail. ``serve_rdna.sh PRINT=1`` resolves each recipe (a stub venv keeps it off torch
and the GPUs) and its TP, PP and port are compared with the table.
"""

import os
import subprocess
from pathlib import Path

import pytest
import regex as re

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
RECIPES = REPO / "tools/rdna/recipes"
README = RECIPES / "README.md"
NAMES = sorted(p.stem for p in RECIPES.glob("*.env"))
ROW = re.compile(r"^\| `([a-z0-9][a-z0-9-]*)` \|(.*)\|\s*$", re.MULTILINE)
CITED = re.compile(r"\b((?:docs|tests|tools|tunableop)/[A-Za-z0-9_./-]*[A-Za-z0-9_])")
SOURCE = re.compile(
    r'^\s*source\s+"\$RDSE_RECIPES/([A-Za-z0-9_-]+)\.env"', re.MULTILINE
)
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


def inventory():
    """README inventory table: recipe -> (tp, pp, cards, port) as written there."""
    text = README.read_text()
    section = text.split("\n## Inventory\n", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for name, rest in ROW.findall(section):
        cells = [c.strip() for c in rest.split("|")]
        parallel, cards, port = cells[1], cells[2], cells[3]
        tp, pp = re.match(r"(\d+) x (\d+)", parallel).groups()
        rows[name] = (tp, pp, cards, port)
    return rows


def plan(tmp_path, recipe):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in DROPPED}
    env.update(
        VENV=str(tmp_path / "venv"),
        MODEL="/nonexistent/model",
        GGUF_DIR="/nonexistent/gguf",
        PRINT="1",
    )
    return subprocess.run(
        ["bash", str(LAUNCHER), f"RECIPE={recipe}"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=60,
    )


def test_readme_lists_every_recipe_once():
    text = README.read_text()
    section = text.split("\n## Inventory\n", 1)[1].split("\n## ", 1)[0]
    listed = [name for name, _ in ROW.findall(section)]
    assert sorted(listed) == NAMES
    assert len(listed) == len(set(listed))


@pytest.mark.parametrize("recipe", NAMES)
def test_header_cites_existing_files(recipe):
    text = (RECIPES / f"{recipe}.env").read_text()
    comments = "\n".join(
        line for line in text.splitlines() if line.lstrip().startswith("#")
    )
    missing = [
        path
        for path in CITED.findall(comments)
        if not (REPO / path).exists() and "*" not in path
    ]
    assert not missing, f"{recipe}.env cites files that are not in the tree: {missing}"


@pytest.mark.parametrize("recipe", NAMES)
def test_source_chain_resolves(recipe):
    text = (RECIPES / f"{recipe}.env").read_text()
    for base in SOURCE.findall(text):
        assert (RECIPES / f"{base}.env").is_file(), f"{recipe}.env sources {base}"


@pytest.mark.parametrize("recipe", NAMES)
def test_table_matches_the_resolved_plan(tmp_path, recipe):
    tp, pp, cards, port = inventory()[recipe]
    assert int(tp) * int(pp) == int(cards)
    result = plan(tmp_path, recipe)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert f"'--tensor-parallel-size' '{tp}'" in out
    if int(pp) > 1:
        assert f"'--pipeline-parallel-size' '{pp}'" in out
    else:
        assert "'--pipeline-parallel-size'" not in out
    assert f"'--port' '{port}'" in out


def test_unknown_recipe_lists_the_folder(tmp_path):
    result = plan(tmp_path, "no-such-recipe")
    assert result.returncode == 2
    assert 'unknown recipe "no-such-recipe"' in result.stderr
    for name in NAMES:
        assert f"  - {name}\n" in result.stderr
