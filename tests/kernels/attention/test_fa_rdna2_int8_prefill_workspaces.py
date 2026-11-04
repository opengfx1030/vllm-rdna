# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The int8 split-K paged prefill keeps its own partial workspaces.

``fa_rdna2_prefill_paged_varlen_int8`` and ``fa_rdna2_prefill_paged_varlen_splitk``
stage their O/M/L split-K partials in grow-only persistent buffers
(``csrc/rocm/rdna2_scratch.cuh``). Both kernel families, and the reduce kernels
they share, index the partials as ``[N, H_q, kv_splits, D]``. The int8 wrapper
used to allocate them as ``[N, H_q, BR_PREFILL, kv_splits, D]`` (16x) out of
the fp16 path's slots, so every int8 call zeroed, regrew and retired the fp16
workspaces. ``fa_rdna2_prefill_splitk_workspaces(int8)`` returns the buffers a
path allocates from.

Run ``pytest tests/kernels/attention/test_fa_rdna2_int8_prefill_workspaces.py``.
"""

import math
from types import SimpleNamespace

import pytest
import torch


def _on_gfx103() -> bool:
    return (
        torch.cuda.is_available()
        and "gfx103" in torch.cuda.get_device_properties(0).gcnArchName
    )


pytestmark = pytest.mark.skipif(
    not _on_gfx103(), reason="Requires AMD RDNA2 (gfx1030) GPU"
)

if _on_gfx103():
    import vllm._rocm_C  # noqa: F401

BS = 16  # tokens per KV block
X_INT8 = 16  # 16-byte x-groups of the 5-D paged layout
X_FP16 = 8
# The largest int8 call of this module, so the int8 workspaces end up sized
# exactly for it whatever the test order.
BIG_SEQS = [(300, 300), (77, 77)]
BIG_HEADS = (8, 2)
BIG_D = 256
BIG_SPLITS = 8
SENTINEL = 1234.5


def _store(cache: torch.Tensor, table_row: torch.Tensor, values: torch.Tensor):
    """Write values [seq_len, H_kv, D] into a [nb, H_kv, D/x, BS, x] cache."""
    seq_len, num_kv_heads, head_dim = values.shape
    x = cache.shape[-1]
    device = cache.device
    pos = torch.arange(seq_len, device=device)
    block = table_row[pos // BS].long()[:, None, None]
    slot = (pos % BS)[:, None, None]
    head = torch.arange(num_kv_heads, device=device)[None, :, None]
    group = torch.arange(head_dim // x, device=device)[None, None, :]
    cache[block, head, group, slot] = values.reshape(
        seq_len, num_kv_heads, head_dim // x, x
    )


def _inputs(
    seqs: list[tuple[int, int]], num_heads: int, num_kv_heads: int, head_dim: int
) -> SimpleNamespace:
    """Paged int8 K/V with per-(position, head) fp32 scales, the same K/V
    dequantized into an fp16 cache, and the fp32 reference output.

    ``seqs`` holds (query_len, seq_len): each sequence's queries are its last
    query_len positions. The int8 kernels read the scale of KV position p at
    row p for every sequence, so the scale rows are shared across sequences.
    """
    g = torch.Generator(device="cuda").manual_seed(sum(sl for _, sl in seqs))
    blocks = [-(-sl // BS) for _, sl in seqs]
    num_blocks = sum(blocks) + 1  # block 0 stays unused, like the null block
    order = torch.randperm(num_blocks - 1, generator=torch.Generator().manual_seed(3))
    table = torch.zeros(len(seqs), max(blocks), dtype=torch.int32)
    start = 0
    for s, nb in enumerate(blocks):
        table[s, :nb] = order[start : start + nb] + 1
        start += nb
    table = table.cuda()

    max_len = max(sl for _, sl in seqs)
    k_scale = torch.rand(max_len, num_kv_heads, generator=g, device="cuda") * 0.01
    v_scale = torch.rand(max_len, num_kv_heads, generator=g, device="cuda") * 0.01
    k_scale += 0.005
    v_scale += 0.005
    shape8 = (num_blocks, num_kv_heads, head_dim // X_INT8, BS, X_INT8)
    shape16 = (num_blocks, num_kv_heads, head_dim // X_FP16, BS, X_FP16)
    k8 = torch.zeros(shape8, dtype=torch.int8, device="cuda")
    v8 = torch.zeros(shape8, dtype=torch.int8, device="cuda")
    k16 = torch.zeros(shape16, dtype=torch.float16, device="cuda")
    v16 = torch.zeros(shape16, dtype=torch.float16, device="cuda")

    num_tokens = sum(q for q, _ in seqs)
    q_all = torch.randn(
        num_tokens, num_heads, head_dim, generator=g, device="cuda"
    ).half()
    ref = torch.empty(num_tokens, num_heads, head_dim, device="cuda")
    group = num_heads // num_kv_heads
    start = 0
    for s, (q, sl) in enumerate(seqs):
        kq, vq = (
            torch.randint(
                -127, 128, (sl, num_kv_heads, head_dim), generator=g, device="cuda"
            ).to(torch.int8)
            for _ in range(2)
        )
        kf = kq.float() * k_scale[:sl, :, None]
        vf = vq.float() * v_scale[:sl, :, None]
        for cache, values in ((k8, kq), (v8, vq), (k16, kf.half()), (v16, vf.half())):
            _store(cache, table[s], values)
        k = kf.repeat_interleave(group, dim=1)
        v = vf.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_all[start : start + q].float(), k)
        scores /= math.sqrt(head_dim)
        q_pos = torch.arange(sl - q, sl, device="cuda")
        future = torch.arange(sl, device="cuda")[None, :] > q_pos[:, None]
        scores.masked_fill_(future[None], float("-inf"))
        ref[start : start + q] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        start += q

    cu = torch.zeros(len(seqs) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor([q for q, _ in seqs]).cumsum(0)
    return SimpleNamespace(
        q=q_all,
        k8=k8,
        v8=v8,
        k_scale=k_scale,
        v_scale=v_scale,
        k16=k16,
        v16=v16,
        table=table,
        cu=cu.cuda(),
        seq_lens=torch.tensor([sl for _, sl in seqs], dtype=torch.int32).cuda(),
        ref=ref,
    )


def _int8(x: SimpleNamespace, kv_splits: int) -> torch.Tensor:
    return torch.ops._rocm_C.fa_rdna2_prefill_paged_varlen_int8(
        x.q, x.k8, x.v8, x.table, x.cu, x.seq_lens, BS, 1, 0, kv_splits,
        x.k_scale, x.v_scale,
    )  # fmt: skip


def _fp16_splitk(x: SimpleNamespace, kv_splits: int, out: torch.Tensor) -> None:
    torch.ops._rocm_C.fa_rdna2_prefill_paged_varlen_splitk(
        x.q, x.k16, x.v16, x.table, x.cu, x.seq_lens, BS, 1, kv_splits, 0,
        x.q.shape[-1] ** -0.5, out,
    )  # fmt: skip


def _workspaces(int8: bool) -> list[torch.Tensor]:
    return list(torch.ops._rocm_C.fa_rdna2_prefill_splitk_workspaces(int8))


def _ptr_numel(tensors: list[torch.Tensor]) -> list[tuple[int, int]]:
    return [(t.data_ptr(), t.numel()) for t in tensors]


def test_int8_prefill_keeps_fp16_splitk_workspaces() -> None:
    """An int8 call must neither move nor write the fp16 split-K partials,
    and must size its own as [N, H_q, kv_splits, D]."""
    x = _inputs(BIG_SEQS, *BIG_HEADS, BIG_D)
    _fp16_splitk(x, BIG_SPLITS, torch.empty_like(x.q))
    fp16_ws = _workspaces(False)
    before = _ptr_numel(fp16_ws)
    for t in fp16_ws:
        t.fill_(SENTINEL)

    out = _int8(x, BIG_SPLITS)
    torch.accelerator.synchronize()

    assert _ptr_numel(_workspaces(False)) == before, (
        "the int8 call regrew the fp16 split-K workspaces"
    )
    for t in fp16_ws:
        assert bool((t == SENTINEL).all()), (
            "the int8 call wrote into the fp16 split-K workspaces"
        )
    rows = x.q.shape[0] * x.q.shape[1] * BIG_SPLITS
    int8_ws = _workspaces(True)
    assert [t.numel() for t in int8_ws] == [rows * BIG_D, rows, rows], (
        "int8 split-K workspaces are not [N, H_q, kv_splits, D]"
    )
    fp16_spans = [(p, p + 4 * n) for p, n in before]
    for t in int8_ws:
        lo, hi = t.data_ptr(), t.data_ptr() + 4 * t.numel()
        assert all(hi <= a or b <= lo for a, b in fp16_spans), (
            "int8 and fp16 split-K workspaces overlap"
        )
    torch.testing.assert_close(out.float(), x.ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("heads", [(6, 1), (8, 2)], ids=["6q1kv", "8q2kv"])
@pytest.mark.parametrize("kv_splits", [1, 3])
@pytest.mark.parametrize(
    "seqs",
    [[(37, 37)], [(16, 16), (50, 120), (1, 33)]],
    ids=["one-prompt", "prompts-and-chunks"],
)
def test_int8_prefill_matches_reference(
    head_dim: int,
    heads: tuple[int, int],
    kv_splits: int,
    seqs: list[tuple[int, int]],
) -> None:
    """The [N, H_q, kv_splits, D] workspaces hold every partial the kernels
    write: ragged q-blocks, chunked prefill and GQA agree with fp32."""
    x = _inputs(seqs, *heads, head_dim)
    out = _int8(x, kv_splits)
    torch.testing.assert_close(out.float(), x.ref, atol=2e-2, rtol=2e-2)


def test_fp16_splitk_graph_replays_after_int8_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A captured fp16 split-K graph keeps its workspace pointer; an int8 call
    in between (which used to regrow and retire that buffer) must not change
    what the graph computes."""
    seqs, splits = [(130, 600), (45, 45)], 4
    x = _inputs(seqs, 6, 1, BIG_D)
    out = torch.empty_like(x.q)
    _fp16_splitk(x, splits, out)  # allocate the workspaces outside the capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _fp16_splitk(x, splits, out)
    captured = _ptr_numel(_workspaces(False))

    _int8(_inputs(BIG_SEQS, *BIG_HEADS, BIG_D), BIG_SPLITS)
    moved = _ptr_numel(_workspaces(False)) != captured

    x.q.copy_(torch.randn(x.q.shape, device="cuda").half())
    graph.replay()
    eager = torch.empty_like(x.q)
    _fp16_splitk(x, splits, eager)
    torch.accelerator.synchronize()
    with capsys.disabled():
        print(f"\nfp16 split-K workspaces moved by the int8 call: {moved}")
    assert torch.equal(out, eager), "graph replay differs from the eager call"
