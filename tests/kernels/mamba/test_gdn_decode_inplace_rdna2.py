# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_RDNA_GDN_DECODE_INPLACE=1 against the state-arena decode (gfx1030).

The arena path copies each decode row's conv/SSM state from the paged cache
into a private per-layer arena, decodes there and copies it back. The in-place
path (``gdn_decode_inplace_enabled``) decodes on the paged cache itself through
slot indices staged in the builder's static buffers, and drops the copies the
arenas needed (the ``core_attn_out`` fill, the ``b``/``a`` and ``z`` copies).

Every case runs the real ``QwenGatedDeltaNetAttention`` decode methods and the
real ``GDNAttentionMetadataBuilder`` through ``gdn_glue_harness`` at the HI2 TP2
per-rank dimensions and checks, bit for bit, that the layer output and every
byte of the paged cache (all pages, including the NULL page and the unreferenced
ones) match the arena path: eager, over a chain of steps, with NULL padding, at
the page boundaries, behind a 2^31-element offset and inside a replayed FULL
CUDA graph.

Needs a gfx10x GPU and the ROCm extension; skips elsewhere.

Run `pytest tests/kernels/mamba/test_gdn_decode_inplace_rdna2.py`.
"""

import random

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 GDN decode is ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 GDN decode is gfx10x-only", allow_module_level=True)

if not (
    hasattr(torch.ops, "_rocm_C")
    and hasattr(torch.ops._rocm_C, "gdn_decode_rdna2")
    and hasattr(torch.ops._rocm_C, "gated_rms_norm")
):
    pytest.skip(
        "gdn_decode_rdna2 / gated_rms_norm are not built", allow_module_level=True
    )

from tests.kernels.mamba import gdn_glue_harness as hz  # noqa: E402
from tests.kernels.mamba.gdn_glue_harness import GdnDecodeRig  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

DEVICE = "cuda"
NUM_BLOCKS = 64


def _rigs(
    *,
    cache_mode: str = "align",
    num_blocks: int = NUM_BLOCKS,
    num_layers: int = 1,
    seed: int = 0,
    **kwargs,
) -> tuple[GdnDecodeRig, GdnDecodeRig]:
    """(arena, in-place): the same weights and the same random initial state."""
    arena = GdnDecodeRig(
        inplace=False,
        num_blocks=num_blocks,
        num_layers=num_layers,
        mamba_cache_mode=cache_mode,
        seed=seed,
        **kwargs,
    )
    inplace = GdnDecodeRig(
        inplace=True,
        num_blocks=num_blocks,
        num_layers=num_layers,
        mamba_cache_mode=cache_mode,
        seed=seed,
        **kwargs,
    )
    assert not arena.builder_inplace and inplace.builder_inplace
    for ref, new in zip(arena.raws, inplace.raws):
        assert torch.equal(ref, new)
    return arena, inplace


def _assert_same_state(
    arena: GdnDecodeRig, inplace: GdnDecodeRig, what: str, *, null_page: bool = True
) -> None:
    """Every cache byte matches. ``null_page=False`` leaves page 0 out: the
    builder counts the zero-length padding requests of a FULL decode batch as
    decodes, so the arena path decodes them against page 0 (gathers it into an
    arena row, scatters the result back), while the in-place path never
    touches the NULL page."""
    first = 0 if null_page else 1
    for i, (ref, new) in enumerate(zip(_pages(arena), _pages(inplace))):
        if not torch.equal(ref[first:], new[first:]):
            bad = (ref != new).any(dim=1).nonzero().flatten().tolist()
            raise AssertionError(f"{what}: layer {i} cache differs in pages {bad}")


def _step(
    arena: GdnDecodeRig,
    inplace: GdnDecodeRig,
    slots: list[int],
    hidden: torch.Tensor,
    *,
    pad_to: int | None = None,
    what: str = "step",
) -> torch.Tensor:
    """One step on both rigs; the rows of the real requests must be equal. The
    padded rows are the in-place path's zero rows and the arena path's decode
    of page 0 (discarded by the runner), so they are not compared."""
    real = len(slots)
    out_a = arena.forward(hidden, arena.build_md(slots, pad_to=pad_to))
    out_i = inplace.forward(hidden, inplace.build_md(slots, pad_to=pad_to))
    torch.accelerator.synchronize()
    assert torch.equal(out_a[:real], out_i[:real]), (
        f"{what}: output differs, max |d| "
        f"{(out_a[:real].float() - out_i[:real].float()).abs().max().item():.3g}"
    )
    assert not out_i[real:].any(), f"{what}: NULL rows must be zero rows"
    return out_i


def _slots(c: int, num_blocks: int, seed: int) -> list[int]:
    """C distinct slots in 1..num_blocks-1 that always include both page
    boundaries (the first and the last real page) and a mid-pool page."""
    rng = random.Random(seed)
    must = [1, num_blocks - 1, num_blocks // 2][:c]
    rest = [s for s in range(1, num_blocks) if s not in must]
    picked = must + rng.sample(rest, c - len(must))
    rng.shuffle(picked)
    return picked


def _pages(rig: GdnDecodeRig) -> list[torch.Tensor]:
    return [raw.view(-1, hz.PAGE_BYTES) for raw in rig.raws]


def _index_ptr(md) -> int:
    indices = md.non_spec_state_indices_tensor
    assert indices is not None
    return indices.data_ptr()


@pytest.mark.parametrize("cache_mode", ["none", "align"])
@pytest.mark.parametrize("c", [1, 2, 4, 8])
def test_decode_step_bit_exact_vs_arena(c: int, cache_mode: str):
    """One step from a random state: output and every cache byte match."""
    arena, inplace = _rigs(cache_mode=cache_mode, num_layers=2)
    gen = torch.Generator(DEVICE).manual_seed(11)
    slots = _slots(c, NUM_BLOCKS, seed=c)
    before = [p.clone() for p in _pages(inplace)]
    _step(arena, inplace, slots, inplace.random_hidden(c, gen), what=f"c={c}")
    _assert_same_state(arena, inplace, f"c={c}")
    touched = set(slots)
    for layer, (page, old) in enumerate(zip(_pages(inplace), before)):
        for blk in range(NUM_BLOCKS):
            same = torch.equal(page[blk], old[blk])
            assert same == (blk not in touched), (
                f"layer {layer} page {blk}: {'changed' if not same else 'unchanged'}"
                f" (decoded slots {sorted(touched)})"
            )


@pytest.mark.parametrize("c,pad_to", [(1, 4), (1, 8), (2, 4), (3, 8), (4, 8)])
def test_null_slot_padding(c: int, pad_to: int):
    """The V2 runner pads a FULL decode batch with zero-length requests whose
    block-table rows are 0 (the NULL block): the kernels write a zero row for
    them and never touch page 0, which here holds random state. The real rows
    and all other pages match the arena path bit for bit."""
    arena, inplace = _rigs(cache_mode="align")
    gen = torch.Generator(DEVICE).manual_seed(5)
    slots = _slots(c, NUM_BLOCKS, seed=100 + c)
    page0 = _pages(inplace)[0][0].clone()
    for step in range(5):
        _step(
            arena,
            inplace,
            slots,
            inplace.random_hidden(pad_to, gen),
            pad_to=pad_to,
            what=f"c={c} padded to {pad_to}, step {step}",
        )
    assert torch.equal(_pages(inplace)[0][0], page0), "NULL page was written"
    _assert_same_state(arena, inplace, f"c={c} padded to {pad_to}", null_page=False)


def test_poisoned_scratch_is_not_read():
    """``core_attn_out`` is ``torch.empty`` in the in-place step: fill every
    ``torch.empty``/``empty_like`` result with NaN. A row the kernels do not
    write (a NULL row, a skipped copy) would reach the output."""
    arena, inplace = _rigs(cache_mode="align")
    gen = torch.Generator(DEVICE).manual_seed(7)
    real_empty, real_empty_like = torch.empty, torch.empty_like
    poisoned_shapes: list[tuple[int, ...]] = []

    def poisoned(real):
        def wrapper(*args, **kwargs):
            t = real(*args, **kwargs)
            if t.is_floating_point():
                t.fill_(float("nan"))
                poisoned_shapes.append(tuple(t.shape))
            return t

        return wrapper

    slots = _slots(3, NUM_BLOCKS, seed=3)
    page0 = _pages(inplace)[0][0].clone()
    for step in range(3):
        hidden = inplace.random_hidden(8, gen)
        out_a = arena.forward(hidden, arena.build_md(slots, pad_to=8))
        torch.empty, torch.empty_like = poisoned(real_empty), poisoned(real_empty_like)
        try:
            out_i = inplace.forward(hidden, inplace.build_md(slots, pad_to=8))
        finally:
            torch.empty, torch.empty_like = real_empty, real_empty_like
        torch.accelerator.synchronize()
        assert (8, hz.HV, hz.V) in poisoned_shapes, "core_attn_out was not poisoned"
        assert not torch.isnan(out_i).any(), f"step {step}: NaN reached the output"
        assert torch.equal(out_a[:3], out_i[:3])
        assert not out_i[3:].any()
    assert torch.equal(_pages(inplace)[0][0], page0)
    _assert_same_state(arena, inplace, "poisoned scratch", null_page=False)


@pytest.mark.parametrize("c", [1, 4, 8])
def test_chained_steps_bit_exact(c: int):
    """200 chained decode steps; the slots change on some of them, so the
    builder's static index buffers are rewritten between steps like the engine
    does. Outputs match every step, every cache byte at the end."""
    arena, inplace = _rigs(cache_mode="align", num_layers=2, seed=2)
    gen = torch.Generator(DEVICE).manual_seed(13)
    rng = random.Random(c)
    slots = _slots(c, NUM_BLOCKS, seed=c)
    for step in range(200):
        if step % 7 == 6:
            slots = rng.sample(range(1, NUM_BLOCKS), c)
        _step(
            arena,
            inplace,
            slots,
            inplace.random_hidden(c, gen),
            what=f"c={c} step {step}",
        )
        if step % 50 == 49:
            _assert_same_state(arena, inplace, f"c={c} after step {step}")
    _assert_same_state(arena, inplace, f"c={c} after 200 steps")


def test_decode_is_in_place_on_the_paged_state():
    """The state the step reads is the paged state: a slot written between two
    steps changes the next step's output (an arena that kept its own copy would
    not notice)."""
    _, inplace = _rigs(cache_mode="align")
    gen = torch.Generator(DEVICE).manual_seed(17)
    hidden = inplace.random_hidden(1, gen)
    md = inplace.build_md([9])
    snap = inplace.snapshot()
    first = inplace.forward(hidden, md).clone()
    inplace.restore(snap)
    hz.randomize_state(inplace.convs[0], inplace.ssms[0], blocks=[9], seed=999)
    second = inplace.forward(hidden, md)
    torch.accelerator.synchronize()
    assert not torch.equal(first, second)


def test_no_arenas_allocated_in_place():
    arena, inplace = _rigs()
    assert all(layer._conv_state_arena is not None for layer in arena.layers)
    assert all(layer._conv_state_arena is None for layer in inplace.layers)
    assert all(layer._ssm_state_arena is None for layer in inplace.layers)
    assert not arena.builder_inplace and inplace.builder_inplace


@pytest.mark.parametrize("c,pad_to", [(1, None), (4, None), (2, 4)])
def test_full_graph_capture_and_replay(c: int, pad_to: int | None):
    """Capture the in-place decode of a layer stack into a FULL CUDA graph with
    a first set of slots, then replay it with other slots and a different
    number of real requests (NULL padding up to the captured batch). The
    builder rewrites its static buffers on every build; each replay matches
    the eager arena path, output and every cache byte."""
    batch = pad_to or c
    arena, inplace = _rigs(cache_mode="align", num_layers=3, seed=4)
    gen = torch.Generator(DEVICE).manual_seed(19)
    rng = random.Random(c)
    slots0 = _slots(c, NUM_BLOCKS, seed=50 + c)
    md0 = inplace.build_md(slots0, pad_to=pad_to)
    hidden = inplace.random_hidden(batch, gen)
    snap = inplace.snapshot()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # JIT and the strided-z probe before capture
        for _ in range(3):
            inplace.forward(hidden, md0)
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = inplace.forward(hidden, md0)
    inplace.restore(snap)
    torch.accelerator.synchronize()
    index_ptr = _index_ptr(md0)
    page0s = [pages[0].clone() for pages in _pages(inplace)]

    for replay in range(12):
        n_real = c if replay % 3 == 0 else rng.randint(1, c)
        slots = rng.sample(range(1, NUM_BLOCKS), n_real)
        md = inplace.build_md(slots, pad_to=batch)
        assert _index_ptr(md) == index_ptr
        step_hidden = inplace.random_hidden(batch, gen)
        hidden.copy_(step_hidden)
        out_ref = arena.forward(step_hidden, arena.build_md(slots, pad_to=batch))
        graph.replay()
        torch.accelerator.synchronize()
        got, want = out_graph[:n_real], out_ref[:n_real]
        assert torch.equal(got, want), (
            f"replay {replay} ({n_real} real of {batch}): output differs, max |d| "
            f"{(got.float() - want.float()).abs().max().item():.3g}"
        )
        assert not out_graph[n_real:].any(), f"replay {replay}: NULL rows not zero"
        _assert_same_state(arena, inplace, f"replay {replay}", null_page=False)
    for pages, page0 in zip(_pages(inplace), page0s):
        assert torch.equal(pages[0], page0), "NULL page was written"


def test_strided_z_gated_norm_bit_exact():
    """The decode step hands the gated norm z at its token stride (a view of
    the qkvz projection); the result is the flattened-z path's, bit for bit."""
    _, inplace = _rigs()
    layer = inplace.layers[0]
    gen = torch.Generator(DEVICE).manual_seed(23)
    for t in (1, 2, 3, 8):
        x = torch.randn(t, hz.HV, hz.V, device=DEVICE, generator=gen).to(torch.float16)
        qkvz = torch.randn(t, hz.QKVZ_DIM, device=DEVICE, generator=gen).to(
            torch.float16
        )
        z = qkvz[:, hz.CONV_DIM :].view(t, hz.HV, hz.V)
        if t > 1:  # a size-1 dim has no meaningful stride
            assert z.stride(0) == hz.QKVZ_DIM
        want = layer.norm(x, z)
        got = layer._gated_norm_strided_z(x, z)
        torch.accelerator.synchronize()
        assert got.shape == want.shape and torch.equal(got, want), f"T={t}"


def test_decode_step_in_place_predicate():
    """Only the pure non-spec decode over exactly ``num_tokens`` rows takes the
    no-fill path; anything else keeps the zero fill and the flattened z."""
    arena, inplace = _rigs()
    layer = inplace.layers[0]
    md = inplace.build_md([5, 9], pad_to=4)

    def predicate(rig, layer, md, num_tokens):
        with (
            rig._env(),
            set_forward_context({layer.prefix: md}, rig.vllm_config, num_tokens=4),
        ):
            return layer._decode_step_in_place(num_tokens, torch.float16)

    assert predicate(inplace, layer, md, 4) is True
    assert predicate(inplace, layer, md, 5) is False  # more rows than metadata
    assert predicate(inplace, layer, md, 3) is False
    md_a = arena.build_md([5, 9], pad_to=4)
    assert predicate(arena, arena.layers[0], md_a, 4) is False  # flag off


@pytest.mark.parametrize("c", [1, 8])
def test_launches_per_layer(c: int):
    """The point of the change, as a regression guard: one layer is the 3
    stand-in projection launches plus 3 work kernels (conv update, recurrence,
    gated norm), where the arena path adds 10 (c=1) or 13 (c>=2) glue launches."""
    from torch.profiler import ProfilerActivity, profile

    arena, inplace = _rigs()
    gen = torch.Generator(DEVICE).manual_seed(29)
    slots = _slots(c, NUM_BLOCKS, seed=c)
    counts = {}
    for name, rig in (("arena", arena), ("inplace", inplace)):
        md = rig.build_md(slots)
        hidden = rig.random_hidden(c, gen)
        for _ in range(3):
            rig.forward(hidden, md)
        torch.accelerator.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            rig.forward(hidden, md)
            torch.accelerator.synchronize()
        counts[name] = sum(
            1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        )
    assert counts["inplace"] == 6, counts
    assert counts["arena"] - counts["inplace"] >= 10, counts


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.accelerator.get_memory_info()[0] < (13 << 30),
    reason="needs ~13 GiB free for a pool past 2^31 elements",
)
def test_state_offset_past_2_pow_31_elements():
    """Slots whose fp32 element offset exceeds 2^31 (page 4096 of a 2 MiB-page
    pool): the in-place kernels must index with 64-bit offsets. The same decode
    on a small pool (state copied there) is the reference."""
    big_blocks = 4200
    big = GdnDecodeRig(
        inplace=True,
        num_blocks=big_blocks,
        mamba_cache_mode="align",
        seed=6,
        init_state=False,
    )
    small = GdnDecodeRig(
        inplace=True, num_blocks=16, mamba_cache_mode="align", seed=6, init_state=False
    )
    slots_big, slots_small = [4097, big_blocks - 1, 7], [3, 4, 5]
    assert slots_big[0] * hz.PAGE_BYTES // 4 > 2**31
    hz.randomize_state(big.convs[0], big.ssms[0], blocks=slots_big, seed=31)
    big_pages, small_pages = _pages(big)[0], _pages(small)[0]
    for b, s in zip(slots_big, slots_small):
        small_pages[s].copy_(big_pages[b])
    gen = torch.Generator(DEVICE).manual_seed(37)
    for step in range(10):
        hidden = big.random_hidden(3, gen)
        out_big = big.forward(hidden, big.build_md(slots_big))
        out_small = small.forward(hidden, small.build_md(slots_small))
        torch.accelerator.synchronize()
        assert torch.equal(out_big, out_small), f"step {step}"
    for b, s in zip(slots_big, slots_small):
        assert torch.equal(big_pages[b], small_pages[s])
