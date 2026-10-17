#!/usr/bin/env python
"""GDN prefill chain: HIP vs Triton/FLA, whole chain and per kernel.

Two things are measured, both at the TP4 production geometry
(linear_num_value_heads=48 / linear_num_key_heads=16 / 4 -> H=12, Hg=4,
K=V=128, BT=64):

  chain  _gdn_prefill_chain_rdna2 (cumsum + kkt + solve_wy + delta_h + o)
         vs chunk_local_cumsum + chunk_gated_delta_rule
  per-kernel  each HIP op vs its Triton counterpart, with output buffers
         preallocated so the number is the kernel, not the allocator.

    HIP_VISIBLE_DEVICES=1 .venv/bin/python tools/rdna2/bench_gdn_prefill_chain_rdna2.py
"""

import json
import sys

import torch

from vllm import _rocm_C  # noqa: F401  (registers torch.ops._rocm_C.*)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _gdn_prefill_chain_rdna2,
)
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chain,
)
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

DEV = "cuda"
BT = FLA_CHUNK_SIZE
K = V = 128
WARMUP, ITERS, REPS = 5, 30, 5
WARM_S = 0.03  # sustained-load seconds before each timing


def warm_clocks(seconds=2.0):
    """The GPU idles at 500 MHz and only boosts after ~10 ms of sustained load.
    A short kernel (prep is 23-410 us) never ramps the clocks on its own, so the
    first arm timed in a fresh process can read up to 4x high. Warm up first."""
    import time

    a = torch.randn(4096, 4096, device=DEV, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=DEV, dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def _one(fn):
    # Warm up by DURATION, not by call count. A count-based warmup gives a 23 us
    # kernel 30x less sustained load than a 700 us one, and the GPU only boosts
    # after ~10 ms of load: prep's 1k row read 72 us that way instead of 24 us.
    # The Triton arms are CPU-issue-bound, so they let the clocks drop again for
    # whatever is timed next -- which is why the warmup has to be per-timing.
    import time

    t0 = time.time()
    while time.time() - t0 < WARM_S:
        for _ in range(WARMUP):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS * 1000.0


def timeit(fn, reps=REPS):
    """Minimum of `reps` batched measurements. Minimum, not mean: the only
    variation is clock ramp and machine noise, both one-sided."""
    return min(_one(fn) for _ in range(reps))

def make_inputs(T, H=12, Hg=4, seed=7, n_seq=1):
    """Production-geometry varlen inputs; B=1 flattened, n_seq sequences."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    if n_seq == 1:
        lens = [T]
    else:
        lens = [T // n_seq] * n_seq
        lens[-1] = T - sum(lens[:-1])
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device=DEV
    )
    total = int(cu_seqlens[-1])
    q = torch.nn.functional.normalize(
        torch.randn(1, total, Hg, K, device=DEV, generator=gen), dim=-1
    ).half()
    k = torch.nn.functional.normalize(
        torch.randn(1, total, Hg, K, device=DEV, generator=gen), dim=-1
    ).half()
    v = (torch.randn(1, total, H, V, device=DEV, generator=gen) * 0.5).half()
    beta = (torch.rand(1, total, H, device=DEV, generator=gen) * 0.9 + 0.05).float()
    g_raw = (-0.05 * torch.rand(1, total, H, device=DEV, generator=gen)).float()
    initial_state = torch.zeros(
        len(lens), H, V, K, dtype=torch.float32, device=DEV
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT).to(torch.int32)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT).to(torch.int32)
    # The HIP chain consumes the per-chunk inclusive cumsum.
    g_hip = g_raw.clone()
    for c0 in range(0, total, BT):
        g_hip[:, c0 : c0 + BT] = g_raw[:, c0 : c0 + BT].cumsum(dim=1)
    return dict(
        q=q, k=k, v=v, beta=beta, g_raw=g_raw, g_hip=g_hip.float(),
        initial_state=initial_state, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, scale=K**-0.5,
    )


def chain_arms(d):
    def hip():
        g = chunk_local_cumsum(
            d["g_raw"], chunk_size=BT, reverse=False,
            cu_seqlens=d["cu_seqlens"], chunk_indices=d["chunk_indices"],
            head_first=False, output_dtype=torch.float32,
        ).contiguous()
        return _gdn_prefill_chain_rdna2(
            q=d["q"], k=d["k"], v=d["v"], g_cumsum=g.float(), beta=d["beta"],
            initial_state=d["initial_state"], scale=d["scale"],
            cu_seqlens=d["cu_seqlens"], chunk_indices=d["chunk_indices"],
            chunk_offsets=d["chunk_offsets"],
        )

    def triton():
        return fla_chain(
            q=d["q"], k=d["k"], v=d["v"], g=d["g_raw"], beta=d["beta"],
            initial_state=d["initial_state"], output_final_state=True,
            cu_seqlens=d["cu_seqlens"], chunk_indices=d["chunk_indices"],
            chunk_offsets=d["chunk_offsets"], use_qk_l2norm_in_kernel=False,
        )

    return triton, hip


def prep_arms(d, H=12, Hg=4, seed=11):
    """fused_post_conv_prep + chunk_local_cumsum vs gdn_prefill_prep_rdna2."""
    total = d["q"].shape[1]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    mixed_qkv = torch.randn(
        total, 2 * Hg * K + H * V, device=DEV, dtype=torch.float16, generator=gen
    )
    a = torch.randn(total, H, device=DEV, dtype=torch.float16, generator=gen)
    b = torch.randn(total, H, device=DEV, dtype=torch.float16, generator=gen)
    A_log = torch.randn(H, device=DEV, dtype=torch.float32, generator=gen)
    dt_bias = torch.randn(H, device=DEV, dtype=torch.float32, generator=gen)
    cu, ci = d["cu_seqlens"], d["chunk_indices"]
    q_out = torch.empty(total, Hg, K, dtype=torch.float16, device=DEV)
    k_out = torch.empty(total, Hg, K, dtype=torch.float16, device=DEV)
    v_out = torch.empty(total, H, V, dtype=torch.float16, device=DEV)
    g_out = torch.empty(total, H, dtype=torch.float32, device=DEV)
    beta_out = torch.empty(total, H, dtype=torch.float32, device=DEV)

    def ref():
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            num_k_heads=Hg, head_k_dim=K, head_v_dim=V,
            apply_l2norm=True, output_g_exp=False,
        )
        return chunk_local_cumsum(
            g.unsqueeze(0), chunk_size=BT, reverse=False, cu_seqlens=cu,
            chunk_indices=ci, head_first=False, output_dtype=torch.float32,
        )

    def hip():
        return torch.ops._rocm_C.gdn_prefill_prep_rdna2(
            mixed_qkv, a, b, A_log, dt_bias, q_out, k_out, v_out, g_out,
            beta_out, cu, ci,
        )

    return {"prep": (ref, hip)}


def kernel_arms(d):
    B, T, Hg, _ = d["q"].shape
    H, cu, ci, co = d["v"].shape[2], d["cu_seqlens"], d["chunk_indices"], d["chunk_offsets"]
    NT = ci.shape[0]
    A = torch.zeros(B, T, H, BT, dtype=torch.float32, device=DEV)
    A_inv = torch.zeros(B, T, H, BT, dtype=d["q"].dtype, device=DEV)
    w = torch.zeros(B, T, H, K, dtype=d["q"].dtype, device=DEV)
    u = torch.zeros_like(d["v"])
    h = torch.zeros(B, NT, H, V, K, dtype=d["q"].dtype, device=DEV)
    v_new = torch.zeros_like(d["v"])
    final_state = torch.zeros_like(d["initial_state"])
    o = torch.empty_like(d["v"])
    ops = torch.ops._rocm_C

    def kkt_hip():
        ops.gdn_prefill_kkt_rdna2(d["k"], d["beta"], d["g_hip"], A, cu, ci)

    def kkt_ref():
        return chunk_scaled_dot_kkt_fwd(
            k=d["k"], g=d["g_hip"], beta=d["beta"], cu_seqlens=cu,
            chunk_indices=ci, chunk_size=BT, output_dtype=torch.float32,
        )

    def wy_hip():
        ops.gdn_prefill_solve_wy_rdna2(
            A, d["k"], d["v"], d["beta"], d["g_hip"], A_inv, w, u, cu, ci
        )

    def wy_ref():
        ai = solve_tril(A=A, cu_seqlens=cu, chunk_indices=ci,
                        output_dtype=d["k"].dtype)
        return recompute_w_u_fwd(k=d["k"], v=d["v"], beta=d["beta"],
                                 g_cumsum=d["g_hip"], A=ai, cu_seqlens=cu,
                                 chunk_indices=ci)

    def dh_hip():
        ops.gdn_prefill_delta_h_rdna2(
            d["k"], u, w, d["g_hip"], h, v_new, d["initial_state"],
            final_state, cu, co, BT
        )

    def dh_ref():
        return chunk_gated_delta_rule_fwd_h(
            k=d["k"], w=w, u=u, g=d["g_hip"], initial_state=d["initial_state"],
            output_final_state=True, chunk_size=BT, save_new_value=True,
            cu_seqlens=cu,
        )

    def o_hip():
        ops.gdn_prefill_o_rdna2(
            d["q"], d["k"], v_new, h, d["g_hip"], o, d["scale"], cu, co
        )

    def o_ref():
        return chunk_fwd_o(q=d["q"], k=d["k"], v=v_new, h=h, g=d["g_hip"],
                           scale=d["scale"], cu_seqlens=cu, chunk_indices=ci)

    return {
        "kkt": (kkt_ref, kkt_hip),
        "solve_wy": (wy_ref, wy_hip),
        "delta_h": (dh_ref, dh_hip),
        "o": (o_ref, o_hip),
    }


def main():
    out = {"device": torch.cuda.get_device_name(0)}
    warm_clocks()
    # 1024/2048 are the served prefill chunks (--max-num-batched-tokens 2048);
    # 8k/16k are the long-context chunked-prefill sizes.
    for label, T, n_seq in (("prefill_1k", 1024, 1),
                            ("prefill_2k", 2048, 1),
                            ("prefill_8k", 8192, 1),
                            ("prefill_16k", 16384, 1),
                            ("prefill_16k_x2", 16384, 2)):
        d = make_inputs(T, n_seq=n_seq)
        ref, hip = chain_arms(d)
        r, h = timeit(ref), timeit(hip)
        entry = {"chain": {"triton_us": r, "hip_us": h,
                           "delta_pct": (h - r) / r * 100}}
        arms = prep_arms(d)
        arms.update(kernel_arms(d))
        for name, (kr, kh) in arms.items():
            a, b = timeit(kr), timeit(kh)
            entry[name] = {"triton_us": a, "hip_us": b,
                           "delta_pct": (b - a) / a * 100}
        out[label] = entry
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
