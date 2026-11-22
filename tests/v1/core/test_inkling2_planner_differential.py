# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``_get_packed_kv_cache_groups`` against its pre-Inkling-2 version (CPU only).

The Inkling-2 merge changed how a bucket's repeats are built: from the longest
page-size list (``zip`` truncated to the shortest) and only up to the number of
repeats (no empty group). The claim is that every input the old function handled
correctly gives the identical groups, and that only inputs it handled wrongly
(raised, dropped layers, or emitted an empty group) change.

The old function is the one in ``vllm/v1/core/kv_cache_planning.py`` at git revision
``KV_PLANNER_BASE_REV`` (default ``703349599a``, the merge base with
``rdna_extras``), loaded with ``git show``; the tests skip when git or the revision
is unavailable. ``KV_PLANNER_BASE_FILE`` names that file directly (a ``git archive``
export has no history).
"""

import os
import random
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.ops.leapquant_gdn import LEAPQUANT_FORMAT
from vllm.v1.core import kv_cache_planning as head_planning
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test

BASE_REV = os.environ.get("KV_PLANNER_BASE_REV", "703349599a")
PLANNING_PATH = "vllm/v1/core/kv_cache_planning.py"


def _load_base_planning() -> types.ModuleType:
    root = Path(__file__).resolve().parents[3]
    base_file = os.environ.get("KV_PLANNER_BASE_FILE")
    try:
        if base_file:
            source = Path(base_file).read_text()
        else:
            source = subprocess.run(
                ["git", "-C", str(root), "show", f"{BASE_REV}:{PLANNING_PATH}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git revision {BASE_REV} is not available")
    name = "vllm.v1.core._kv_cache_planning_base"
    module = types.ModuleType(name)
    module.__file__ = base_file or f"{BASE_REV}:{PLANNING_PATH}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def base_planning():
    module = _load_base_planning()
    yield module
    sys.modules.pop(module.__name__, None)


def _config(layout: str = "BLNHC"):
    config = MagicMock()
    config.cache_config = CacheConfig()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = layout
    config.attention_config.hisparse_config = None
    config.speculative_config = None
    return config


def _mla(head_size: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=head_size, dtype=torch.uint8
    )


def _full(num_kv_heads: int = 2, head_size: int = 64) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )


def _swa(num_kv_heads: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        dtype=torch.float16,
        sliding_window=64,
    )


def _mamba(width: int, state_format: str | None = None) -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((width,),),
        dtypes=(torch.bfloat16,),
        state_format=state_format,
    )


def _ring(head_size: int) -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=head_size, dtype=torch.float16
    )


def _family(rng: random.Random, kind: str, count: int, name: str, sizes: list[int]):
    """``count`` layers of one spec family; ``sizes`` indexes the page sizes."""
    makers = {
        "mla": lambda s: _mla(64 * (s + 1)),
        "full": lambda s: _full(head_size=32 * (s + 1)),
        "swa": lambda s: _swa(2 * (s + 1)),
        "mamba": lambda s: _mamba(256 * (s + 1)),
        "lq_mamba": lambda s: _mamba(256 * (s + 1), f"{LEAPQUANT_FORMAT}/x"),
        "ring": lambda s: _ring(16 * (s + 1)),
    }
    make = makers[kind]
    return {f"{name}.{i}": make(rng.choice(sizes)) for i in range(count)}


def _random_specs(rng: random.Random, shape: str) -> dict[str, KVCacheSpec]:
    """A random layer set. ``shape`` steers the buckets it contains:
    ``balanced`` (equal layer counts per page size), ``single`` (one page size per
    bucket) or ``any`` (page sizes in unequal counts included)."""
    specs: dict[str, KVCacheSpec] = {}
    kinds = ["mla", "full", "swa", "mamba", "lq_mamba", "ring"]
    # One family per kind: two families of a kind share a bucket, which would make a
    # "balanced" or "single" set unbalanced.
    for fam, kind in enumerate(rng.sample(kinds, rng.randint(1, 4))):
        name = f"{kind}{fam}"
        if shape == "single":
            sizes = [rng.randrange(3)]
            count = rng.randint(1, 24)
            specs.update(_family(rng, kind, count, name, sizes))
        elif shape == "balanced":
            pages = rng.randint(1, 3)
            repeats = rng.randint(1, 12)
            for p in range(pages):
                specs.update(_family(rng, kind, repeats, f"{name}.p{p}", [p]))
        else:
            for p in range(rng.randint(1, 3)):
                specs.update(
                    _family(rng, kind, rng.randint(1, 14), f"{name}.p{p}", [p])
                )
    names = list(specs)
    rng.shuffle(names)
    return {n: specs[n] for n in names}


def _signature(groups) -> list[tuple[list[str], dict[str, KVCacheSpec], int]]:
    return [
        (
            list(g.layer_names),
            dict(g.kv_cache_spec.kv_cache_specs),
            g.kv_cache_spec.block_size,
        )
        for g in groups
    ]


def _run(planning, config, specs):
    """(status, groups-or-exception). status: ok / raised."""
    try:
        return "ok", planning._get_packed_kv_cache_groups(config, specs)
    except BaseException as exc:  # noqa: BLE001 - StopIteration is one of them
        return "raised", exc


def _base_status(outcome, specs) -> str:
    """How the old function did: raised, none, clean, dropped or empty."""
    status, groups = outcome
    if status == "raised":
        return "raised"
    if groups is None:
        return "none"
    if any(not g.layer_names for g in groups):
        return "empty"
    if Counter(n for g in groups for n in g.layer_names) != Counter(list(specs)):
        return "dropped"
    return "clean"


def _assert_valid_partition(groups, specs):
    assert groups, "no groups"
    assert all(g.layer_names for g in groups), "an empty group"
    assert Counter(n for g in groups for n in g.layer_names) == Counter(list(specs)), (
        "layers dropped or duplicated"
    )
    for g in groups:
        assert set(g.kv_cache_spec.kv_cache_specs) == set(g.layer_names)


SHAPES = ["balanced", "single", "any"]
LAYOUTS = ["BLNHC", "LBNHC"]


@pytest.mark.parametrize("pack_flag", [None, "1"])
@pytest.mark.parametrize("shape", SHAPES)
def test_head_equals_base_wherever_base_was_correct(
    base_planning, monkeypatch, shape, pack_flag
):
    if pack_flag is None:
        monkeypatch.delenv("VLLM_RDNA_GDN_LEAPQUANT_PACK", raising=False)
    else:
        monkeypatch.setenv("VLLM_RDNA_GDN_LEAPQUANT_PACK", pack_flag)
    rng = random.Random(f"planner-differential/{shape}/{pack_flag}")
    seen: Counter[str] = Counter()
    for _ in range(1500):
        specs = _random_specs(rng, shape)
        config = _config(rng.choice(LAYOUTS))
        old = _run(base_planning, config, specs)
        new = _run(head_planning, config, specs)
        status = _base_status(old, specs)
        seen[status] += 1
        assert new[0] == "ok", f"head raised {new[1]!r} (base: {status})"
        if new[1] is not None:
            _assert_valid_partition(new[1], specs)
        if shape in ("balanced", "single") and status == "raised":
            assert isinstance(old[1], StopIteration), old[1]
        if status == "none":
            assert new[1] is None
        elif status == "clean":
            assert _signature(new[1]) == _signature(old[1]), (
                f"head differs from a correct base result: {list(specs)}"
            )
    # The sample must exercise the claim, not skirt it.
    assert seen["clean"] + seen["none"] >= 100, seen
    if shape in ("balanced", "single"):
        # Balanced and single-page-size buckets never lose layers in the old
        # function; its only failure there is the StopIteration of a state bucket
        # asked for more groups than it has repeats (see the next test).
        assert seen["dropped"] == seen["empty"] == 0, seen
        assert seen["clean"] >= 300, seen


def test_balanced_and_single_size_buckets_never_change(base_planning):
    """The claim itself: on balanced and single-page-size inputs the old function
    either returned groups (the new ones are identical), returned None (so does the
    new one) or raised StopIteration (the new one returns valid groups)."""
    outcomes: Counter[str] = Counter()
    mismatches = []
    for shape in ("balanced", "single"):
        rng = random.Random(f"planner-differential/pinned/{shape}")
        for _ in range(3000):
            specs = _random_specs(rng, shape)
            config = _config(rng.choice(LAYOUTS))
            old = _run(base_planning, config, specs)
            new = _run(head_planning, config, specs)
            status = _base_status(old, specs)
            outcomes[f"{shape}/{status}"] += 1
            if new[0] != "ok":
                mismatches.append((shape, "head raised", list(specs)))
            elif status == "raised":
                if not isinstance(old[1], StopIteration):
                    mismatches.append((shape, "other exception", repr(old[1])))
                _assert_valid_partition(new[1], specs)
            elif status == "none":
                if new[1] is not None:
                    mismatches.append((shape, "head packs, base did not", list(specs)))
            elif status == "clean":
                if _signature(old[1]) != _signature(new[1]):
                    mismatches.append((shape, "groups differ", list(specs)))
            else:
                mismatches.append((shape, f"base {status}", list(specs)))
    print(f"outcomes of the old function: {dict(outcomes)}")
    assert not mismatches, mismatches[:3]
    assert outcomes["balanced/clean"] >= 500 and outcomes["single/clean"] >= 500


@pytest.mark.parametrize(
    ("heads", "group_sizes"),
    [
        ([4] * 6 + [8] * 2, [2, 3, 3]),
        ([8, 8, 4], [1, 2]),
    ],
)
def test_unbalanced_sliding_window_bucket_old_failure_new_success(
    base_planning, heads, group_sizes
):
    """Inkling-2 with a replicated DSpark draft at TP=2: one sliding-window
    bucket with two page sizes in unequal counts. The old function dropped the
    longer list's surplus layers (or raised); the new one keeps every layer."""
    specs: dict[str, KVCacheSpec] = {"full": _full()}
    for i, num_kv_heads in enumerate(heads):
        specs[f"swa.{i}"] = _swa(num_kv_heads)
    config = _config()
    old = _run(base_planning, config, specs)
    assert _base_status(old, specs) in ("raised", "dropped", "empty"), old
    new = _run(head_planning, config, specs)
    assert new[0] == "ok", new[1]
    _assert_valid_partition(new[1], specs)
    swa_groups = [g for g in new[1] if g.layer_names[0].startswith("swa.")]
    assert sorted(len(g.layer_names) for g in swa_groups) == group_sizes


def test_base_failures_are_recorded(base_planning):
    """Over the ``any`` shape, list what the old function did wrong (the numbers
    printed with ``-s``); the new function is valid on every one."""
    rng = random.Random("planner-differential/failures")
    seen: Counter[str] = Counter()
    raised: Counter[str] = Counter()
    for _ in range(3000):
        specs = _random_specs(rng, "any")
        config = _config(rng.choice(LAYOUTS))
        old = _run(base_planning, config, specs)
        status = _base_status(old, specs)
        seen[status] += 1
        if status == "raised":
            raised[type(old[1]).__name__] += 1
        new = _run(head_planning, config, specs)
        assert new[0] == "ok", new[1]
        if new[1] is not None:
            _assert_valid_partition(new[1], specs)
    print(f"base outcomes over 3000 random layer sets: {dict(seen)}")
    print(f"base exceptions: {dict(raised)}")
    assert seen["dropped"] > 0 and seen["raised"] > 0, seen
