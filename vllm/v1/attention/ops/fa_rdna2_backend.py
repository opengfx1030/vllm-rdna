"""FA RDNA2 attention backend for gfx1030.

This module loads the FA RDNA2 kernels (paged decode + paged prefill)
via torch.utils.cpp_extension.load_inline and exposes Python-callable
functions. It is opt-in via VLLM_USE_RDNA2_FA=1.

The kernel is loaded lazily on first call to avoid compilation
cost when the backend is not selected.

Usage:
    from vllm.attention.ops.fa_rdna2_backend import fa_rdna2_decode_paged
    out = fa_rdna2_decode_paged(Q, key_cache, value_cache, block_table,
                                seq_lens, block_size, kv_splits=8)
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

_KERNEL_SRC = None
_EXT = None


def _load_kernel():
    """Compile and load the FA RDNA2 kernel via load_inline."""
    global _KERNEL_SRC, _EXT
    if _EXT is not None:
        return _EXT

    import pathlib
    kernel_path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "csrc" / "rocm" / "fa_rdna2.cu"
    )
    if not kernel_path.is_file():
        raise FileNotFoundError(
            f"fa_rdna2.cu not found at {kernel_path}. "
            "Ensure csrc/rocm/fa_rdna2.cu is in the vLLM source tree.")

    with open(kernel_path) as f:
        _KERNEL_SRC = f.read()

    cpp_src = """
torch::Tensor fa_rdna2_decode_paged(torch::Tensor Q,
                                    torch::Tensor key_cache,
                                    torch::Tensor value_cache,
                                    torch::Tensor block_table,
                                    torch::Tensor seq_lens,
                                    int64_t block_size, int64_t kv_splits,
                                    int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen(torch::Tensor Q,
                                            torch::Tensor key_cache,
                                            torch::Tensor value_cache,
                                            torch::Tensor block_table,
                                            torch::Tensor cu_query_lens,
                                            torch::Tensor seq_lens,
                                            int64_t block_size,
                                            int64_t causal,
                                            int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen_short(torch::Tensor Q,
                                                  torch::Tensor key_cache,
                                                  torch::Tensor value_cache,
                                                  torch::Tensor block_table,
                                                  torch::Tensor cu_query_lens,
                                                  torch::Tensor seq_lens,
                                                  int64_t block_size,
                                                  int64_t causal,
                                                  int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen_splitk(torch::Tensor Q,
                                                  torch::Tensor key_cache,
                                                  torch::Tensor value_cache,
                                                  torch::Tensor block_table,
                                                  torch::Tensor cu_query_lens,
                                                  torch::Tensor seq_lens,
                                                  int64_t block_size,
                                                  int64_t causal,
                                                  int64_t kv_splits,
                                                  int64_t sliding_window);
"""

    _EXT = load_inline(
        name="fa_rdna2_backend",
        cpp_sources=[cpp_src],
        cuda_sources=[_KERNEL_SRC],
        functions=["fa_rdna2_decode_paged",
                   "fa_rdna2_prefill_paged_varlen",
                   "fa_rdna2_prefill_paged_varlen_short",
                   "fa_rdna2_prefill_paged_varlen_splitk"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx1030"],
        verbose=False,
    )
    return _EXT


def is_available() -> bool:
    """Check if FA RDNA2 backend is available.

    Returns True only when:
    - VLLM_USE_RDNA2_FA=1 is set
    - torch CUDA is available (HIP)
    - The GPU is gfx1030 (RDNA2)
    """
    if os.environ.get("VLLM_USE_RDNA2_FA") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    # Check for gfx1030
    try:
        props = torch.cuda.get_device_properties(0)
        if "gfx103" not in props.gcnArchName:
            return False
    except Exception:
        return False
    return True




def fa_rdna2_decode_paged(Q: torch.Tensor,
                          key_cache: torch.Tensor,
                          value_cache: torch.Tensor,
                          block_table: torch.Tensor,
                          seq_lens: torch.Tensor,
                          block_size: int = 16,
                          kv_splits: int = 8,
                          sliding_window: int = 0) -> torch.Tensor:
    """FA2 decode kernel for gfx1030 reading from paged KV cache.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged V cache
        block_table: [num_tokens, max_blocks] int32 per-query block indices
        seq_lens: [num_tokens] int32 per-query KV length
        block_size: physical block size (16, 32, etc.)
        kv_splits: number of CTAs per head (1..16)
        sliding_window: sliding window size (0 = no window)

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_decode_paged(
        Q, key_cache, value_cache, block_table, seq_lens,
        block_size, kv_splits, sliding_window)






def fa_rdna2_prefill_paged_varlen(Q: torch.Tensor,
                                   key_cache: torch.Tensor,
                                   value_cache: torch.Tensor,
                                   block_table: torch.Tensor,
                                   cu_query_lens: torch.Tensor,
                                   seq_lens: torch.Tensor,
                                   block_size: int = 16,
                                   causal: bool = True,
                                   sliding_window: int = 0) -> torch.Tensor:
    """FA2 paged prefill kernel for gfx1030 with varlen (multiple sequences).

    Gap 2: supports chunked prefill with multiple sequences per launch.
    Each query block of BR_PREFILL tokens may span a different sequence.
    The kernel uses cu_query_lens to determine which sequence each block
    belongs to, then reads that sequence's seq_lens and block_table slice.

    Supports HEAD_DIM=128 and HEAD_DIM=256.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (16, 32, 784, etc.)
        causal: whether to apply causal masking (per-sequence)

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_prefill_paged_varlen(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), sliding_window)


def fa_rdna2_prefill_paged_varlen_short(Q: torch.Tensor,
                                        key_cache: torch.Tensor,
                                        value_cache: torch.Tensor,
                                        block_table: torch.Tensor,
                                        cu_query_lens: torch.Tensor,
                                        seq_lens: torch.Tensor,
                                        block_size: int = 16,
                                        causal: bool = True,
                                        sliding_window: int = 0) -> torch.Tensor:
    """Sub-4096 optimized FA2 paged prefill kernel for gfx1030 (HEAD_DIM=128).

    Uses BR_PREFILL=32, THREADS_PREFILL=256 for better grid utilization
    at short sequence lengths (< 4096 tokens). Larger BR processes more
    query tokens per CTA, more threads improve warp utilization.

    Only valid for HEAD_DIM=128. For HEAD_DIM=256, use
    fa_rdna2_prefill_paged_varlen (which is gated to max_seq_len >= 4096).

    Args:
        Q: [num_tokens, H_q, 128] fp16 query tensor
        key_cache: [num_blocks, H_kv, 128/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, 128/x, block_size, x] fp16 paged V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (16, 32, 784, etc.)
        causal: whether to apply causal masking (per-sequence)
        sliding_window: sliding window size (0 = no window)

    Returns:
        O: [num_tokens, H_q, 128] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_prefill_paged_varlen_short(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), sliding_window)


def fa_rdna2_prefill_paged_varlen_splitk(Q: torch.Tensor,
                                         key_cache: torch.Tensor,
                                         value_cache: torch.Tensor,
                                         block_table: torch.Tensor,
                                         cu_query_lens: torch.Tensor,
                                         seq_lens: torch.Tensor,
                                         block_size: int = 16,
                                         causal: bool = True,
                                         kv_splits: int = 4) -> torch.Tensor:
    """FA2 paged prefill with split-K varlen for better grid utilization.

    Partitions the KV sequence across kv_splits CTAs per (q_block, h_q).
    Each split produces partial O/M/L; a reduction kernel merges them.
    Use for large seq_len where H_q CTAs per q_block underutilize the GPU.

    Args:
        Q: [num_tokens, H_q, D] fp16 query tensor
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache
        value_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged V cache
        block_table: [num_seqs, max_blocks] int32 per-sequence block indices
        cu_query_lens: [num_seqs + 1] int32 cumulative query counts
        seq_lens: [num_seqs] int32 per-sequence KV length
        block_size: physical block size (16, 32, 784, etc.)
        causal: whether to apply causal masking (per-sequence)
        kv_splits: number of KV splits (1..16). 1 = no split.
        sliding_window: sliding window size (0 = no window)

    Returns:
        O: [num_tokens, H_q, D] fp16 attention output
    """
    ext = _load_kernel()
    return ext.fa_rdna2_prefill_paged_varlen_splitk(
        Q, key_cache, value_cache, block_table, cu_query_lens, seq_lens,
        block_size, int(causal), int(kv_splits), sliding_window)
