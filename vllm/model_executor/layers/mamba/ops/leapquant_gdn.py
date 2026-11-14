# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Algorithm: LeapQuant, Y. Pan, H. Xi, K. Zhu, X. Li, Y. Wu, M. Mishra, H. Zhang,
# W. X. Zheng, B. Kasikci, S. Han, K. Keutzer, R. Iyer, I. Stoica (arXiv 2609.38166).
# The record ring, cursor and per-window flush follow ReplaySSM (Dao AI Lab and
# NVIDIA; vllm-project/vllm#48018, #62004). The FP32 oracle below and the RDNA2
# HIP kernels are original code written for this fork.
"""LeapQuant recurrent-state storage for Gated DeltaNet (arXiv 2609.38166).

The FP32 GDN state ``S`` of one value head (``[V, K]``, the row-major layout of
the ROCm decode kernels) is stored as one byte blob per (cache block, head):

* an INT8 residual ``Z`` with per-value-row scales ``B`` and per-key-column
  smoothing ``C`` (dequant: ``Z[v, k] * B[v] * C[k]``),
* ``r`` FP16 compensator tokens ``S ~ U~ K~^T`` (the top-``r`` subspace of the
  state at the last window boundary),
* a ring of FP16 per-token records ``(k_j, u_j, g_j)`` for the tokens since the
  boundary, where ``S_j = exp(g_j) S_{j-1} + u_j k_j^T``,
* an int32 header ``[cursor, pending, 0, 0]``.

The state is ``replay(cursor records, Z + U~ K~^T)``. A decode step appends its
records at ``cursor`` and sets ``pending`` to the number of tokens it ran.
Pending records play the role of the per-token state slots of the FP32 layout:
slot ``t`` of the home block is "commit ``t + 1`` records".
``leapquant_normalize`` (run first in the next step) commits the accepted ones
(``cursor += num_accepted``, or one record for a reader of slot 0) and, once
``cursor >= window``, folds the records into a new quantized boundary state.
Rejected speculative tokens are dropped by never committing their records, so
MTP verify needs no per-token state copies. The align-mode prefix-cache copies
(``v1_mamba_rdna2.cu``) apply the same rule: copying slot ``bias`` copies the
home blob and commits ``bias + 1`` records in the copy's header. An all-zero
blob is the zero state.

The fold must never produce NaN: ``_orthonormalize`` scales every column by its
largest absolute value and sweeps twice (see there), and the smoothing factor
divides by ``V`` before it sums. So zero, tiny, huge (up to the float limit)
and exactly low-rank states fold to finite blobs with orthonormal
compensators. A block that a request has not written yet holds the leftovers
of another request, so the state is a function of ``z``, ``bscale``,
``csmooth``, the compensators, the first ``cursor`` records and the header,
nothing else. Header words outside the range that ``_sane_header`` checks read
as an empty ring.

This module holds the layout, the opt-in switch and an FP32 PyTorch reference
used as the oracle for the ROCm kernels; the reference is not on the serving
path.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

LEAPQUANT_ALIGN = 16
# The int32 header [cursor, pending, 0, 0] is the last section of every blob.
LEAPQUANT_HEADER_BYTES = 16
LEAPQUANT_WINDOW = 32
LEAPQUANT_NUM_COMP = 4
LEAPQUANT_POWER_ITERS = 2
INT8_MAX = 127.0
SMOOTH_FLOOR = 1e-6
FP32_MAX = 3.4028234663852886e38
ORTH_DEPENDENT = 0.5
FP16_MAX = 65504.0


@functools.cache
def leapquant_gdn_enabled() -> bool:
    """Opt-in LeapQuant GDN state storage (``VLLM_RDNA_GDN_LEAPQUANT=1``).

    The kernels are RDNA2 HIP only, so the flag fails loudly elsewhere instead
    of silently serving FP32 state."""
    if os.environ.get("VLLM_RDNA_GDN_LEAPQUANT", "0") != "1":
        return False
    from vllm.platforms.rdna import on_rdna2_hip

    if not on_rdna2_hip():
        raise ValueError(
            "VLLM_RDNA_GDN_LEAPQUANT=1 needs gfx10x: the LeapQuant GDN "
            "kernels are RDNA2 HIP only"
        )
    return True


@functools.cache
def leapquant_gdn_pack_enabled() -> bool:
    """Opt-in KV-cache packing of LeapQuant GDN layers
    (``VLLM_RDNA_GDN_LEAPQUANT_PACK=1``, on top of ``VLLM_RDNA_GDN_LEAPQUANT=1``).

    A LeapQuant page is smaller than the attention page it is padded to. With
    the flag, GDN pages keep their natural size and the KV-cache planner puts
    as many GDN layers into one KV-cache group as one block holds, so fewer
    groups (block tables) are resident per request."""
    if os.environ.get("VLLM_RDNA_GDN_LEAPQUANT_PACK", "0") != "1":
        return False
    if not leapquant_gdn_enabled():
        raise ValueError(
            "VLLM_RDNA_GDN_LEAPQUANT_PACK=1 needs VLLM_RDNA_GDN_LEAPQUANT=1"
        )
    return True


def leapquant_gdn_layout(
    head_k_dim: int, head_v_dim: int, num_spec: int
) -> LeapQuantLayout:
    """Serving layout: window 32, r = 4, ring sized for one
    verify step of ``num_spec + 1`` tokens."""
    if head_k_dim != 128 or head_v_dim != 128:
        raise ValueError(
            "VLLM_RDNA_GDN_LEAPQUANT=1 supports head_k_dim == head_v_dim == 128, "
            f"got {head_k_dim} / {head_v_dim}"
        )
    return LeapQuantLayout(
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        window=LEAPQUANT_WINDOW,
        num_comp=LEAPQUANT_NUM_COMP,
        max_query_len=num_spec + 1,
        power_iters=LEAPQUANT_POWER_ITERS,
    )


def _align(n: int) -> int:
    return (n + LEAPQUANT_ALIGN - 1) // LEAPQUANT_ALIGN * LEAPQUANT_ALIGN


@dataclass(frozen=True)
class LeapQuantLayout:
    """Byte layout of one (block, value head) LeapQuant blob."""

    head_k_dim: int
    head_v_dim: int
    window: int = 16
    num_comp: int = 4
    max_query_len: int = 1  # 1 + num_speculative_tokens
    power_iters: int = 2

    @property
    def capacity(self) -> int:
        # normalize leaves cursor <= window - 1; one step appends at most
        # max_query_len records.
        return self.window + self.max_query_len - 1

    def _sections(self) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
        K, V, r, cap = self.head_k_dim, self.head_v_dim, self.num_comp, self.capacity
        return [
            ("z", torch.int8, (V, K)),
            ("bscale", torch.float32, (V,)),
            ("csmooth", torch.float32, (K,)),
            ("comp_k", torch.float16, (r, K)),
            ("comp_u", torch.float16, (r, V)),
            ("rec_k", torch.float16, (cap, K)),
            ("rec_u", torch.float16, (cap, V)),
            ("rec_g", torch.float32, (cap,)),
            ("header", torch.int32, (4,)),
        ]

    @property
    def offsets(self) -> dict[str, int]:
        out, off = {}, 0
        for name, dtype, shape in self._sections():
            out[name] = off
            n = dtype.itemsize
            for s in shape:
                n *= s
            off = _align(off + n)
        out["_total"] = off
        return out

    @property
    def nbytes(self) -> int:
        return self.offsets["_total"]

    def views(self, blob: torch.Tensor) -> dict[str, torch.Tensor]:
        """Typed views of ``blob`` (uint8, ``[..., nbytes]``, last dim dense)."""
        assert blob.dtype == torch.uint8 and blob.shape[-1] == self.nbytes
        assert blob.stride(-1) == 1
        offs = self.offsets
        lead = blob.shape[:-1]
        out = {}
        for name, dtype, shape in self._sections():
            n = dtype.itemsize
            for s in shape:
                n *= s
            sl = blob[..., offs[name] : offs[name] + n]
            out[name] = sl.view(dtype).view(*lead, *shape)
        return out


def gdn_gates(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log decay ``g`` and write strength ``beta`` (fp32), matching
    ``gdn_spec_verify_rdna2`` / ``gdn_lq_decode_rdna2``."""
    x = a.float() + dt_bias.float()
    softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g = -torch.exp(A_log.float()) * softplus
    beta = torch.sigmoid(b.float())
    return g, beta


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn_step_fp32(
    S: torch.Tensor,  # [..., V, K] fp32
    q: torch.Tensor,  # [..., K] l2-normalized and scaled
    k: torch.Tensor,  # [..., K] l2-normalized
    v: torch.Tensor,  # [..., V]
    g: torch.Tensor,  # [...]
    beta: torch.Tensor,  # [...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FP32 GDN token: returns (new S, output, record u)."""
    S = S * torch.exp(g)[..., None, None]
    acc = (S * k[..., None, :]).sum(-1)
    u = (v.float() - acc) * beta[..., None]
    S = S + u[..., :, None] * k[..., None, :]
    o = (S * q[..., None, :]).sum(-1)
    return S, o, u


def _boundary_state(views: dict[str, torch.Tensor]) -> torch.Tensor:
    z = views["z"].float()
    S = z * views["bscale"][..., :, None] * views["csmooth"][..., None, :]
    cu = views["comp_u"].float()  # [..., r, V]
    ck = views["comp_k"].float()  # [..., r, K]
    return S + torch.einsum("...rv,...rk->...vk", cu, ck)


def _sane_header(
    layout: LeapQuantLayout, header: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cursor, pending)`` as the kernels read them (``lq_header_ok``).

    Header words that an earlier user of the block left outside ``0 <= cursor``,
    ``0 <= pending`` and ``cursor + pending <= capacity`` read as an empty ring,
    so no record index is ever out of bounds."""
    cursor, pending = header[..., 0].long(), header[..., 1].long()
    ok = (
        (cursor >= 0)
        & (pending >= 0)
        & (cursor <= layout.capacity)
        & (pending <= layout.capacity - cursor)
    )
    zero = torch.zeros_like(cursor)
    return torch.where(ok, cursor, zero), torch.where(ok, pending, zero)


def leapquant_materialize(layout: LeapQuantLayout, blob: torch.Tensor) -> torch.Tensor:
    """FP32 state ``[..., V, K]`` of normalized blobs (pending must be 0)."""
    vw = layout.views(blob)
    cursor, pending = _sane_header(layout, vw["header"])
    assert bool((pending == 0).all()), "materialize needs pending == 0"
    S = _boundary_state(vw)
    rk, ru, rg = vw["rec_k"].float(), vw["rec_u"].float(), vw["rec_g"]
    for j in range(int(cursor.max().item()) if cursor.numel() else 0):
        live = (j < cursor)[..., None, None]
        upd = (
            S * torch.exp(rg[..., j])[..., None, None]
            + ru[..., j, :, None] * rk[..., j, None, :]
        )
        S = torch.where(live, upd, S)
    return S


def _sweep(y: torch.Tensor, cols: list[torch.Tensor]) -> torch.Tensor:
    """One modified Gram-Schmidt pass of ``y`` against the unit columns ``cols``."""
    for c in cols:
        y = y - (c * y).sum(-1, keepdim=True) * c
    return y


def _orthonormalize(Y: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt over the last dim's columns: Y [..., N, r] -> Q, same shape.

    For every finite Y each column of Q is zero or has unit norm, and the
    non-zero columns are orthogonal to fp32 precision:

    * A column is scaled by its largest absolute value before the sweeps and
      again after them, so no square under- or overflows (the rule of the
      LeapQuant RFC, vllm-project/vllm#64725, section 5). A column that is zero
      or not finite is returned as zero.
    * Two sweeps against the earlier columns ("twice is enough", W. Kahan; see
      Parlett, The Symmetric Eigenvalue Problem, 1980). One sweep is not enough
      for a state of rank below ``r``: its columns lie in the span of the
      earlier ones, the sweep leaves rounding noise, and that noise is partly
      parallel to the earlier columns. The dominant direction would be counted
      twice and the INT8 residual would carry a state-sized error.
    * The dependence test of Daniel, Gragg, Kaufman and Stewart (Math. Comp. 30,
      1976). When the second sweep removes more than half of the largest entry
      (``m2 <= ORTH_DEPENDENT * m1``), the column was rounding noise along the
      earlier columns, and no further sweep can make it orthogonal to them. The
      column is returned as zero. The next fold restarts a zero column from the
      fixed block (``leapquant_fold_parts``).
    """
    cols: list[torch.Tensor] = []
    zero = torch.zeros_like(Y[..., 0])
    for i in range(Y.shape[-1]):
        y = Y[..., i]
        m = y.abs().amax(-1, keepdim=True)
        live = (m > 0) & (m <= FP32_MAX)
        y = _sweep(torch.where(live, y / torch.where(live, m, 1.0), zero), cols)
        m1 = y.abs().amax(-1, keepdim=True)
        y = _sweep(y, cols)
        m2 = y.abs().amax(-1, keepdim=True)
        live = live & (m2 > ORTH_DEPENDENT * m1)
        y = y / torch.where(live, m2, 1.0)
        n = torch.sqrt((y * y).sum(-1, keepdim=True))  # 1 <= n <= sqrt(N) if live
        cols.append(torch.where(live, y / n.clamp_min(1.0), zero))
    return torch.stack(cols, dim=-1)


def _power_init(layout: LeapQuantLayout, device) -> torch.Tensor:
    """Deterministic start block X0 [K, r] (integers in [-8, 8])."""
    K, r = layout.head_k_dim, layout.num_comp
    k = torch.arange(K, device=device)[:, None]
    i = torch.arange(r, device=device)[None, :]
    return (((k * (2 * i + 3) + 7 * i + 1) % 17) - 8).float()


def leapquant_fold_parts(
    layout: LeapQuantLayout, S: torch.Tensor, prev_comp_k: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """Quantize FP32 states ``S [..., V, K]`` into boundary parts.

    Compensators come from subspace iteration warm-started at the previous
    boundary's ``comp_k``. A column of it that is all zero (a fresh block, a
    zero state, a direction the last fold found dependent) restarts from the
    fixed block instead of staying dead. The residual is taken against the
    fp16-rounded compensators, smoothed per key column, and quantized per value
    row to symmetric INT8.
    """
    S = S.float()
    X0 = _power_init(layout, S.device).expand(*S.shape[:-2], -1, -1)
    if prev_comp_k is not None:
        pk = prev_comp_k.float().transpose(-1, -2)  # [..., K, r]
        alive = pk.abs().sum(-2, keepdim=True) > 0  # [..., 1, r]
        X0 = torch.where(alive, pk, X0)
    Q = _orthonormalize(S @ X0)  # [..., V, r]
    for _ in range(layout.power_iters):
        Q = _orthonormalize(S @ (S.transpose(-1, -2) @ Q))
    Kt = S.transpose(-1, -2) @ Q  # [..., K, r]
    comp_u = Q.transpose(-1, -2).clamp(-FP16_MAX, FP16_MAX).half()  # [..., r, V]
    comp_k = Kt.transpose(-1, -2).clamp(-FP16_MAX, FP16_MAX).half()  # [..., r, K]
    R = S - torch.einsum("...rv,...rk->...vk", comp_u.float(), comp_k.float())
    # Divide before the sum: the sum of |R| over V overflows for |R| of about
    # 1e37 and more, and an inf factor makes z * bscale * csmooth 0 * inf = NaN.
    # For V a power of two this equals mean() bit for bit.
    csmooth = torch.sqrt((R.abs() / R.shape[-2]).sum(-2)).clamp_min(SMOOTH_FLOOR)
    Rs = R / csmooth[..., None, :]
    bscale = Rs.abs().amax(-1) / INT8_MAX  # [..., V]
    safe = torch.where(bscale > 0, bscale, torch.ones_like(bscale))
    z = torch.round(Rs / safe[..., None]).clamp(-INT8_MAX, INT8_MAX)
    z = torch.where((bscale > 0)[..., None], z, torch.zeros_like(z))
    return {
        "z": z.to(torch.int8),
        "bscale": bscale,
        "csmooth": csmooth,
        "comp_k": comp_k,
        "comp_u": comp_u,
    }


def leapquant_store_state(
    layout: LeapQuantLayout, blob: torch.Tensor, S: torch.Tensor, warm: bool = False
) -> None:
    """Fold FP32 states into ``blob`` in place (cursor = pending = 0).

    ``warm`` starts the subspace iteration from the blob's previous
    compensators (a live window boundary); a freshly allocated block holds
    stale bytes, so prefill stores start from the fixed block."""
    vw = layout.views(blob)
    parts = leapquant_fold_parts(layout, S, vw["comp_k"] if warm else None)
    for name, val in parts.items():
        vw[name].copy_(val)
    vw["header"].zero_()


def leapquant_normalize(
    layout: LeapQuantLayout,
    blob: torch.Tensor,  # [N, HV, nbytes]
    num_accepted: torch.Tensor | None,  # [N] int, None = slot-0 reader
) -> None:
    """Commit accepted records and fold full windows, in place.

    ``num_accepted`` is the per-request count a speculative verify reads the
    state at (slot ``num_accepted - 1``); ``None`` is a reader of slot 0
    (prefill, non-spec decode), which commits one pending record. A header
    outside the range of ``_sane_header`` reads as empty and is rewritten."""
    vw = layout.views(blob)
    hdr = vw["header"]
    cursor, pending = _sane_header(layout, hdr)
    if num_accepted is None:
        acc = torch.clamp(pending, max=1)
    else:
        acc = torch.minimum(
            num_accepted.long()[:, None].expand_as(pending), pending
        ).clamp_min(0)
    hdr[..., 0] = (cursor + acc).to(torch.int32)
    hdr[..., 1] = 0
    full = hdr[..., 0] >= layout.window  # [N, HV]
    if bool(full.any()):
        idx = full.nonzero(as_tuple=True)
        sub = blob[idx]
        S = leapquant_materialize(layout, sub)
        leapquant_store_state(layout, sub, S, warm=True)
        blob[idx] = sub


def leapquant_copy_slot(
    layout: LeapQuantLayout, src: torch.Tensor, token_bias: int
) -> torch.Tensor:
    """Align-mode copy of per-token slot ``token_bias`` from a home blob
    ``[HV, nbytes]``: the copy commits ``token_bias + 1`` pending records
    (mirrors the LeapQuant branch of ``copy_mamba_state_block``)."""
    dst = src.clone()
    hdr = layout.views(dst)["header"]
    pending = hdr[..., 1]
    commit = torch.where(
        pending > 0, torch.clamp(pending, max=max(token_bias, 0) + 1), 0
    )
    hdr[..., 0] += commit
    hdr[..., 1] = 0
    return dst


def leapquant_decode(
    layout: LeapQuantLayout,
    blob: torch.Tensor,  # [N, HV, nbytes], normalized
    q: torch.Tensor,  # [N, T, H, K] raw
    k: torch.Tensor,  # [N, T, H, K] raw
    v: torch.Tensor,  # [N, T, HV, V]
    a: torch.Tensor,  # [N, T, HV]
    b: torch.Tensor,  # [N, T, HV]
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run T tokens per request on normalized blobs; returns o [N, T, HV, V].

    Appends T records at ``cursor`` and sets ``pending = T``; the next
    ``leapquant_normalize`` commits ``num_accepted`` of them.
    """
    N, T, H, K = q.shape
    HV = v.shape[2]
    rep = HV // H
    vw = layout.views(blob)
    S = leapquant_materialize(layout, blob)  # [N, HV, V, K]
    qn = l2norm(q).repeat_interleave(rep, dim=2) * scale
    kn = l2norm(k).repeat_interleave(rep, dim=2)
    g, beta = gdn_gates(a, b, A_log, dt_bias)
    cursor, _ = _sane_header(layout, vw["header"])  # [N, HV]
    outs = []
    n_idx = torch.arange(N, device=blob.device)[:, None]
    h_idx = torch.arange(HV, device=blob.device)[None, :]
    for t in range(T):
        S, o, u = gdn_step_fp32(S, qn[:, t], kn[:, t], v[:, t], g[:, t], beta[:, t])
        outs.append(o)
        pos = cursor + t
        vw["rec_k"][n_idx, h_idx, pos] = kn[:, t].half()
        vw["rec_u"][n_idx, h_idx, pos] = u.half()
        vw["rec_g"][n_idx, h_idx, pos] = g[:, t].float()
    vw["header"][..., 1] = T
    return torch.stack(outs, dim=1)
