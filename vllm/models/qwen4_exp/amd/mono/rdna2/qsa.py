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
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

from ...ops import qsa_rdna2

logger = init_logger(__name__)

MIN_ROWS = 3  # HIP (plain-load merge) wins from 3 rows at long context
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
    return (
        MIN_ROWS <= q.shape[0] <= 64
        and q.dtype == torch.float16
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

    Unlike the fused sparse decode, this store is a Triton kernel, so with the
    HIP QSA path on it stands down: the caller's three qsa_store_cache_rows
    calls are HIP and keep the store on the all-HIP default.
    """
    if qsa_rdna2.qsa_use_rdna2() or not HAS_TRITON:
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
