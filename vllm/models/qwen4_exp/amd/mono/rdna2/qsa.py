# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel QSA (gfx1030, was E3): the QSA decode path, all exact.

* Bounded indexer: decode scoring is bounded to the live context (QSA runs as
  an eager break in every cudagraph mode, so the host bound is valid per
  step) and scoring tiles past the live context exit early (also valid inside
  CUDA graphs); all-padding tiles are skipped in the Triton sparse attention.
  Those are constexpr switches in ``ops/qsa.py`` keyed on ``ops.qsa.MONO_QSA``.
* HIP sparse decode: one ``_rocm_C::mono_qsa_decode_rdna2`` launch (split +
  merge) for decode / MTP-verify rows from ``MIN_ROWS`` rows (``sparse_decode``).
* Fused norm + RoPE: the indexer's GemmaRMSNorm + RoPE / interleaved MRoPE in
  one Triton launch (``norm_rope``).
* 3-row store: the indexer's three per-token cache stores in one launch
  (``store_rows3``).
* Indexer epilogue: q norm+RoPE, group pooling, k norm+RoPE and the three
  stores in one HIP launch (``epilogue``), on decode / MTP-verify steps.
* HIP indexer scoring: ``_rocm_C::mono_qsa_mqa_rdna2`` (``mqa_paged``), with
  the early exit; the kernel shape is picked per call from the rows, the live
  score width and the decode width (``mqa_config``).
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

MIN_ROWS = 3  # HIP (plain-load merge) wins from 3 rows at long context
DECODE_WIDTH = 1  # rows per request in a decode step (1 + speculative tokens)
# Indexer key rows per layer above which a decode step streams K from DRAM
# (the QSA layers' keys of a step no longer fit the 128 MB Infinity Cache):
# 32768 rows = 8 MB per layer.
DRAM_KEYS = 32768
# (group, waves, tile slots, lanes per key, unroll[, q in LDS]) forced for
# every call; benches and tests set it, None = mqa_config's choice.
FORCE: tuple | None = None
_COUNTERS: dict[int, torch.Tensor] = {}
_LOGGED: set = set()


def _log_once(what: str, rows: int, taken: bool) -> None:
    """One line per (part, decode width); one for every wider batch."""
    key = (what, rows if rows <= 64 else -1, taken)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "mono qsa_rdna2: %s %s -> %s",
            what,
            f"{rows} rows" if rows <= 64 else "> 64 rows",
            "mono" if taken else "today's",
        )


def _hip_ok(q, k_cache, v_cache, group_size) -> bool:
    # The kernel reads K and V as FP16 words. A packed or quantized cache (the
    # int4_fp16_block record layout, fp8, ...) has a different layout and would
    # be read as garbage, so the gate requires the dtype it was written for
    # rather than trusting the caller.
    return (
        MIN_ROWS <= q.shape[0] <= 64
        and q.dtype == torch.float16
        and k_cache.dtype == torch.float16
        and v_cache.dtype == torch.float16
        and q.shape[2] == 256
        and group_size <= 8
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and all(st % 8 == 0 for st in (*k_cache.stride()[:3], *v_cache.stride()[:3]))
        and q.stride(0) % 8 == 0
        and q.stride(1) % 8 == 0
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "mono_qsa_decode_rdna2")
    )


def sparse_decode(
    q, k_cache, v_cache, group_size, indices, block_table, token_to_req, out
) -> bool:
    """One-launch HIP sparse decode attention into ``out``; False (nothing
    done) when the batch is not one the kernel covers."""
    if not _hip_ok(q, k_cache, v_cache, group_size):
        _log_once("sparse decode", q.shape[0], False)
        return False
    _log_once("sparse decode", q.shape[0], True)
    dev = (
        q.device.index
        if q.device.index is not None
        else torch.accelerator.current_device_index()
    )
    counters = _COUNTERS.get(dev)
    if counters is None:  # QSA runs as an eager break: never allocated under capture
        counters = torch.zeros(64 * 64, dtype=torch.int32, device=q.device)
        _COUNTERS[dev] = counters
    rows, heads, kvh = q.shape[0], q.shape[1], k_cache.shape[2]
    splits = int(
        torch.ops._rocm_C.mono_qsa_decode_rdna2_splits(rows, kvh, indices.shape[1])
    )
    scratch = torch.empty(
        splits * rows * heads * 257, dtype=torch.float32, device=q.device
    )
    torch.ops._rocm_C.mono_qsa_decode_rdna2(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        token_to_req,
        out,
        scratch,
        counters,
        splits,
    )
    return True


def norm_rope(
    norm: nn.Module,
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor | None:
    """GemmaRMSNorm + the main attention's RoPE / interleaved MRoPE in one
    Triton launch (the fused q/k norm-rope kernel with no q-gate heads),
    instead of rms_norm + cos/sin gather + copies + mrope. Returns None when
    the rotary layout is not one that kernel covers."""
    num_tokens, num_heads, head_dim = tensor.shape
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    if positions.ndim == 2:
        if not (
            mrope_section
            and len(mrope_section) == 3
            and getattr(rotary_emb, "mrope_interleaved", False)
        ):
            return None
    elif positions.ndim != 1:
        return None
    if not getattr(rotary_emb, "is_neox_style", False) or tensor.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        return None
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate

    flat = tensor.reshape(num_tokens, num_heads * head_dim)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    _, out, _ = fused_qk_rmsnorm_rope_gate(
        flat,  # no q-gate heads: only the "k" side runs
        flat,
        norm.weight,
        norm.weight,
        cache,
        positions,
        norm.variance_epsilon,
        0,
        num_heads,
        head_dim,
        rotary_emb.rotary_dim,
        mrope_section=mrope_section if positions.ndim == 2 else None,
        norm_beta=1.0,  # GemmaRMSNorm scales by (1 + w)
    )
    _log_once(f"norm+rope ({num_heads} heads)", num_tokens, True)
    return out.view(num_tokens, num_heads, head_dim)


def _has_hip_mqa() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "mono_qsa_mqa_rdna2"
    )


def mqa_config(rows: int, columns: int) -> tuple | None:
    """Kernel shape (group, waves, tile slots, lanes per key, unroll) for one
    indexer call, from host-known sizes (graph-safe: QSA runs as an eager
    break, so ``columns`` is the step's live bound), or None where the Triton
    kernel is faster (tools/rdna/mono/qsa_mqa_bench.py, --bounded, warm and
    --cold):

    * cache-resident steps: 4 lanes per key, one step per tile, a tile slot
      count that shrinks with the rows (1.04-1.30x Triton+E3);
    * DRAM-bound decode, one row per request: 16 lanes per key (each load
      covers whole key rows), 8 waves, one tile per block (at the roofline,
      1.00-1.10x);
    * DRAM-bound decode with speculative rows: Triton (its rows of a request
      share K through L2; the group kernel loses 2-8 % there);
    * prefill chunks (> 64 rows): groups of 3 adjacent rows from 64 rows at
      >= 16k context (each key row read once per group, 1.3-2.2x)."""
    if FORCE is not None:
        return FORCE
    small = (1, 4, max(8, 1024 // max(1, rows)), 4, 1)
    if rows > 64:
        return (3, 4, 0, 4, 1) if columns >= 4096 else small
    width = max(1, int(DECODE_WIDTH))
    if -(-rows // width) * columns <= DRAM_KEYS:
        return small
    if width == 1:
        return (1, 8, 0, 16, 2)
    return None


def mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    score_divisor: float,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
) -> bool:
    """The indexer scoring with the early exit (columns [0, visible) only, as
    the decode top-k reads them) in one HIP launch; False when the layout is
    not one the kernel covers (the caller then runs the Triton kernel)."""
    if not (
        _has_hip_mqa()
        and q.dtype == torch.float16
        and k_cache.dtype == torch.float16
        and q.shape[1:] == (4, 128)
        and k_cache.shape[2:] == (1, 128)
        and q.stride(2) == 1
        and k_cache.stride(3) == 1
        and q.stride(0) % 8 == 0
        and q.stride(1) % 8 == 0
        and k_cache.stride(0) % 8 == 0
        and k_cache.stride(1) % 8 == 0
        and q.data_ptr() % 16 == 0
        and k_cache.data_ptr() % 16 == 0
        and page_table.dtype == torch.int32
        and token_to_req.dtype == torch.int32
        and token_to_req.is_contiguous()
        and query_positions.is_contiguous()
        and sequence_lengths.is_contiguous()
        and query_positions.dtype in (torch.int32, torch.int64)
        and sequence_lengths.dtype in (torch.int32, torch.int64)
    ):
        return False
    cfg = mqa_config(q.shape[0], logits.shape[1])
    if cfg is None:
        return False
    group, waves, tiles, lanes, unroll, *rest = cfg
    torch.ops._rocm_C.mono_qsa_mqa_rdna2(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        logits,
        visible_blocks,
        int(compress_ratio),
        float(score_divisor),
        group,
        waves,
        tiles,
        lanes,
        unroll,
        bool(rest and rest[0]),
    )
    _log_once("indexer scoring (HIP)", q.shape[0], True)
    return True


def _has_hip_epilogue() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "mono_qsa_epilogue_rdna2"
    )


def epilogue(
    q_norm: nn.Module,
    k_norm: nn.Module,
    rotary_emb: nn.Module,
    qk: torch.Tensor,
    positions: torch.Tensor,
    raw_metadata,
    compressed_metadata,
    compress_ratio: int,
    c_cache: torch.Tensor,
    r_cache: torch.Tensor,
    p_cache: torch.Tensor | None,
) -> torch.Tensor | None:
    """The indexer's per-token epilogue after the qk projection in one HIP
    launch: q norm+RoPE (returned as [n, 4, 128]), the group pooling of
    qsa_compress_groups, k norm+RoPE into the compressed cache and the raw-key
    / RoPE-position ring rows. None when the layout or the step is not one
    the kernel covers (the caller runs the separate launches)."""
    if p_cache is None or not _has_hip_epilogue():
        return None
    n = qk.shape[0]
    ring = r_cache.shape[1]
    # ring reads (members before the chunk) and ring writes (the chunk's
    # rows) must not share a slot: at most ring - ratio + 1 rows per request
    if n == 0 or raw_metadata.max_query_len > ring - compress_ratio + 1:
        return None
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    if mrope_section:
        if not (
            len(mrope_section) == 3
            and getattr(rotary_emb, "mrope_interleaved", False)
            and positions.ndim == 2
            and positions.shape[0] == 3
        ):
            return None
        sec_h, sec_w = int(mrope_section[1]), int(mrope_section[2])
    else:
        if positions.ndim != 1:
            return None
        sec_h = sec_w = 0
    cache = rotary_emb._match_cos_sin_cache_dtype(qk)  # noqa: SLF001
    if (
        not getattr(rotary_emb, "is_neox_style", False)
        or qk.dtype != torch.float16
        or qk.shape[1] != 5 * 128
        or qk.stride(1) != 1
        or q_norm.weight.dtype != torch.float16
        or k_norm.weight.dtype != torch.float16
        or cache.dtype != torch.float16
        or cache.shape[1] != rotary_emb.rotary_dim
        or c_cache.dtype != torch.float16
        or c_cache.shape[2:] != (1, 128)
        or r_cache.shape[2:] != (1, 128)
        or compressed_metadata.slot_mapping.shape[0] < n
        or raw_metadata.slot_mapping.shape[0] < n
    ):
        return None
    q = torch.empty((n, 4, 128), dtype=qk.dtype, device=qk.device)
    torch.ops._rocm_C.mono_qsa_epilogue_rdna2(
        qk,
        q,
        q_norm.weight,
        k_norm.weight,
        float(q_norm.variance_epsilon),
        float(k_norm.variance_epsilon),
        1.0,  # GemmaRMSNorm scales by (1 + w)
        cache,
        positions,
        int(rotary_emb.rotary_dim),
        sec_h,
        sec_w,
        raw_metadata.block_table,
        raw_metadata.token_to_req,
        raw_metadata.query_start_loc,
        raw_metadata.logical_positions,
        int(compress_ratio),
        c_cache,
        compressed_metadata.slot_mapping,
        r_cache,
        raw_metadata.slot_mapping,
        p_cache,
    )
    _log_once("indexer epilogue (HIP)", n, True)
    return q


def _has_hip_k_store() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "mono_qsa_k_store_rdna2"
    )


def norm_rope_store3(
    norm: nn.Module,
    rotary_emb: nn.Module,
    first_positions: torch.Tensor,
    pooled: torch.Tensor,
    c_cache: torch.Tensor,
    c_slots: torch.Tensor,
    r_cache: torch.Tensor,
    r_slots: torch.Tensor,
    r_rows: torch.Tensor,
    p_cache: torch.Tensor,
    p_rows: torch.Tensor,
) -> bool:
    """The compressed-key epilogue (norm_rope) fused with the three indexer
    stores (store_rows3) in one HIP launch; False when the layout is not one
    the kernel covers (the caller then runs norm_rope + store_rows3)."""
    if not _has_hip_k_store():
        return False
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    if mrope_section:
        if not (
            len(mrope_section) == 3 and getattr(rotary_emb, "mrope_interleaved", False)
        ):
            return False
        positions = first_positions.transpose(0, 1)
        sec_h, sec_w = int(mrope_section[1]), int(mrope_section[2])
    else:
        positions = first_positions[:, 0]
        sec_h = sec_w = 0
    if not getattr(rotary_emb, "is_neox_style", False):
        return False
    pooled = pooled.reshape(-1, pooled.shape[-1])
    r_rows = r_rows[:, 0] if r_rows.ndim == 3 else r_rows
    p_rows = p_rows[:, 0] if p_rows.ndim == 3 else p_rows
    weight = norm.weight
    cache = rotary_emb._match_cos_sin_cache_dtype(pooled)  # noqa: SLF001
    if (
        pooled.dtype != torch.float16
        or pooled.shape[1] != 128
        or pooled.stride(1) != 1
        or weight.dtype != torch.float16
        or cache.dtype != torch.float16
        or c_cache.ndim != 4
        or c_cache.shape[2] != 1
        or c_cache.shape[3] != 128
        or cache.shape[1] != rotary_emb.rotary_dim
    ):
        return False
    torch.ops._rocm_C.mono_qsa_k_store_rdna2(
        pooled,
        weight,
        cache,
        positions,
        float(norm.variance_epsilon),
        1.0,  # GemmaRMSNorm scales by (1 + w)
        int(rotary_emb.rotary_dim),
        sec_h,
        sec_w,
        c_cache,
        c_slots,
        r_cache,
        r_slots,
        r_rows,
        p_cache,
        p_rows,
    )
    _log_once("norm+rope+store (HIP)", pooled.shape[0], True)
    return True


@triton.jit
def _store_rows3_kernel(
    c_cache_ptr,
    c_slots_ptr,
    c_rows_ptr,  # compressed keys (compressed slots)
    r_cache_ptr,
    r_slots_ptr,
    r_rows_ptr,  # raw keys (raw slots)
    p_cache_ptr,
    p_rows_ptr,  # rope positions (raw slots)
    c_sb,
    c_st,
    c_sd,
    c_rr,
    c_rd,
    c_nb,
    r_sb,
    r_st,
    r_sd,
    r_rr,
    r_rd,
    r_nb,
    p_sb,
    p_st,
    p_sd,
    p_rr,
    p_rd,
    p_nb,
    num_rows,
    C_PAGE: tl.constexpr,
    C_WIDTH: tl.constexpr,
    C_BLOCK: tl.constexpr,
    R_PAGE: tl.constexpr,
    R_WIDTH: tl.constexpr,
    R_BLOCK: tl.constexpr,
    P_PAGE: tl.constexpr,
    P_WIDTH: tl.constexpr,
    P_BLOCK: tl.constexpr,
) -> None:
    """The indexer's three per-token cache stores in one launch (same masking
    as ops/qsa.py _store_qsa_rows_kernel, applied to each store)."""
    row = tl.program_id(0)
    # compressed key row
    dims = tl.arange(0, C_BLOCK)
    slot = tl.load(c_slots_ptr + row)
    ok = (row < num_rows) & (slot >= 0) & (slot < c_nb * C_PAGE)
    blk = tl.maximum(slot, 0) // C_PAGE
    tok = tl.maximum(slot, 0) % C_PAGE
    v = tl.load(
        c_rows_ptr + row * c_rr + dims * c_rd, mask=ok & (dims < C_WIDTH), other=0
    )
    tl.store(
        c_cache_ptr + blk * c_sb + tok * c_st + dims * c_sd,
        v,
        mask=ok & (dims < C_WIDTH),
    )
    # raw key row and its rope positions share the raw slot
    slot = tl.load(r_slots_ptr + row)
    dims = tl.arange(0, R_BLOCK)
    ok = (row < num_rows) & (slot >= 0) & (slot < r_nb * R_PAGE)
    blk = tl.maximum(slot, 0) // R_PAGE
    tok = tl.maximum(slot, 0) % R_PAGE
    v = tl.load(
        r_rows_ptr + row * r_rr + dims * r_rd, mask=ok & (dims < R_WIDTH), other=0
    )
    tl.store(
        r_cache_ptr + blk * r_sb + tok * r_st + dims * r_sd,
        v,
        mask=ok & (dims < R_WIDTH),
    )
    pdims = tl.arange(0, P_BLOCK)
    pok = (row < num_rows) & (slot >= 0) & (slot < p_nb * P_PAGE)
    pblk = tl.maximum(slot, 0) // P_PAGE
    ptok = tl.maximum(slot, 0) % P_PAGE
    pv = tl.load(
        p_rows_ptr + row * p_rr + pdims * p_rd, mask=pok & (pdims < P_WIDTH), other=0
    )
    tl.store(
        p_cache_ptr + pblk * p_sb + ptok * p_st + pdims * p_sd,
        pv,
        mask=pok & (pdims < P_WIDTH),
    )


def _has_hip_store3() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "mono_qsa_store3_rdna2"
    )


def store_rows3(
    c_cache: torch.Tensor,
    c_slots: torch.Tensor,
    c_rows: torch.Tensor,
    r_cache: torch.Tensor,
    r_slots: torch.Tensor,
    r_rows: torch.Tensor,
    p_cache: torch.Tensor,
    p_rows: torch.Tensor,
) -> bool:
    """One launch for the three indexer stores; False when the shapes are not
    the one-head, same-row-count case (the caller then stores one by one).

    HIP (``_rocm_C::mono_qsa_store3_rdna2``) when built, else the Triton
    kernel below."""
    hip = _has_hip_store3()
    if not hip and not HAS_TRITON:
        return False
    rows = []
    for cache, slots, r in (
        (c_cache, c_slots, c_rows),
        (r_cache, r_slots, r_rows),
        (p_cache, r_slots, p_rows),
    ):
        if cache.ndim != 4 or cache.shape[2] != 1 or not all(cache.shape):
            return False
        if r.ndim == 3:
            if r.shape[1] != 1:
                return False
            r = r[:, 0]
        if r.shape != (slots.numel(), cache.shape[3]):
            return False
        rows.append(r)
    n = rows[0].shape[0]
    if rows[1].shape[0] != n or rows[2].shape[0] != n:
        return False
    if n == 0:
        return True
    cr, rr, pr = rows
    if hip:
        torch.ops._rocm_C.mono_qsa_store3_rdna2(
            c_cache, c_slots, cr, r_cache, r_slots, rr, p_cache, pr
        )
        _log_once("indexer store (HIP)", n, True)
        return True
    _store_rows3_kernel[(n,)](
        c_cache,
        c_slots,
        cr,
        r_cache,
        r_slots,
        rr,
        p_cache,
        pr,
        c_cache.stride(0),
        c_cache.stride(1),
        c_cache.stride(3),
        cr.stride(0),
        cr.stride(1),
        c_cache.shape[0],
        r_cache.stride(0),
        r_cache.stride(1),
        r_cache.stride(3),
        rr.stride(0),
        rr.stride(1),
        r_cache.shape[0],
        p_cache.stride(0),
        p_cache.stride(1),
        p_cache.stride(3),
        pr.stride(0),
        pr.stride(1),
        p_cache.shape[0],
        n,
        C_PAGE=c_cache.shape[1],
        C_WIDTH=c_cache.shape[3],
        C_BLOCK=triton.next_power_of_2(c_cache.shape[3]),
        R_PAGE=r_cache.shape[1],
        R_WIDTH=r_cache.shape[3],
        R_BLOCK=triton.next_power_of_2(r_cache.shape[3]),
        P_PAGE=p_cache.shape[1],
        P_WIDTH=p_cache.shape[3],
        P_BLOCK=triton.next_power_of_2(p_cache.shape[3]),
        num_warps=4,
    )
    _log_once("indexer store", n, True)
    return True
