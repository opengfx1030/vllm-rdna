# SPDX-License-Identifier: Apache-2.0
"""Inverse tensor transforms for Qwen GDN tensors stored in GGUF order."""

import re

import torch

NUM_K_HEADS = 16
NUM_V_HEADS = 48
NUM_V_PER_K = NUM_V_HEADS // NUM_K_HEADS
HEAD_DIM = 128


def inverse_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """Undo converter grouped-K to tiled-V head ordering along ``dim``."""
    ndim = tensor.ndim
    if ndim == 0 or not -ndim <= dim < ndim:
        raise ValueError(f"invalid dimension {dim} for rank-{ndim} tensor")
    dim %= ndim
    expected = num_k_heads * num_v_per_k * head_dim
    if min(num_k_heads, num_v_per_k, head_dim) <= 0 or tensor.shape[dim] != expected:
        raise ValueError(
            f"dimension {dim} must have length {expected}, got {tensor.shape[dim]}"
        )
    shape = list(tensor.shape)
    grouped = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1 :]
    view = tensor.reshape(grouped)
    permutation = list(range(len(grouped)))
    permutation[dim], permutation[dim + 1] = permutation[dim + 1], permutation[dim]
    return view.permute(*permutation).contiguous().reshape(shape)


def inverse_gdn_tensor(
    name,
    tensor,
    *,
    num_k_heads: int = NUM_K_HEADS,
    num_v_heads: int = NUM_V_HEADS,
    head_k_dim: int = HEAD_DIM,
    head_v_dim: int = HEAD_DIM,
    conv_kernel: int = 4,
    num_layers: int = 48,
    full_attention_interval: int = 4,
):
    """Restore one GGUF GDN tensor suffix to native flat Qwen4 layout."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        raise ValueError("invalid GDN head configuration")
    num_v_per_k = num_v_heads // num_k_heads
    roles = {
        "attn_qkv.weight": "attn_qkv",
        "attn_gate.weight": "attn_gate",
        "ssm_conv1d.weight": "ssm_conv1d",
        "ssm_alpha.weight": "ssm_alpha",
        "ssm_beta.weight": "ssm_beta",
        "ssm_dt.bias": "ssm_dt.bias",
        "ssm_a": "ssm_a",
        "ssm_out.weight": "ssm_out",
        "ssm_norm.weight": "ssm_norm",
    }
    match = (
        re.fullmatch(r"blk\.(0|[1-9][0-9]*)\.(.+)", name)
        if isinstance(name, str)
        else None
    )
    if (
        match is None
        or int(match[1]) >= num_layers
        or (
            full_attention_interval > 0
            and int(match[1]) % full_attention_interval == full_attention_interval - 1
        )
        or match[2] not in roles
    ):
        raise ValueError(f"invalid GDN tensor name: {name!r}")
    suffix = roles[match[2]]
    qk_rows = 2 * num_k_heads * head_k_dim
    rows = num_v_heads * head_v_dim
    if suffix == "attn_qkv":
        if tensor.ndim != 2 or tensor.shape[0] != qk_rows + rows:
            raise ValueError(
                "attn_qkv must be rank 2 with QKV rows followed by V heads"
            )
        return torch.cat(
            (
                tensor[:qk_rows],
                inverse_v_heads(
                    tensor[qk_rows:], 0, num_k_heads, num_v_per_k, head_v_dim
                ),
            ),
            dim=0,
        )
    if suffix == "ssm_conv1d":
        if tensor.ndim != 2 or tensor.shape != (qk_rows + rows, conv_kernel):
            raise ValueError(
                f"ssm_conv1d must have shape ({qk_rows + rows}, {conv_kernel})"
            )
        restored = torch.cat(
            (
                tensor[:qk_rows],
                inverse_v_heads(
                    tensor[qk_rows:], 0, num_k_heads, num_v_per_k, head_v_dim
                ),
            ),
            dim=0,
        )
        return restored.unsqueeze(1)
    if suffix == "attn_gate":
        if tensor.ndim != 2 or tensor.shape[0] != rows:
            raise ValueError(f"{suffix} must be rank 2 with {rows} rows")
        result = inverse_v_heads(tensor, 0, num_k_heads, num_v_per_k, head_v_dim)
        return result
    if suffix in ("ssm_alpha", "ssm_beta"):
        if tensor.ndim != 2 or tensor.shape[0] != num_v_heads:
            raise ValueError(f"{suffix} must be a matrix with {num_v_heads} rows")
        return inverse_v_heads(tensor, 0, num_k_heads, num_v_per_k, 1)
    if suffix == "ssm_dt.bias":
        if tensor.ndim != 1 or tensor.shape[0] != num_v_heads:
            raise ValueError(f"{suffix} must be a vector of length {num_v_heads}")
        return inverse_v_heads(tensor, 0, num_k_heads, num_v_per_k, 1)
    if suffix == "ssm_a":
        if tensor.ndim != 1 or tensor.shape[0] != num_v_heads:
            raise ValueError(f"ssm_a must be a vector of length {num_v_heads}")
        if not torch.isfinite(tensor).all() or (tensor >= 0).any():
            raise ValueError("GGUF ssm_a values must be finite and negative")
        reordered = inverse_v_heads(tensor, 0, num_k_heads, num_v_per_k, 1)
        return torch.log(-reordered.float())
    if suffix == "ssm_out":
        if tensor.ndim != 2 or tensor.shape[1] != rows:
            raise ValueError(f"ssm_out must have {rows} input columns")
        return inverse_v_heads(tensor, 1, num_k_heads, num_v_per_k, head_v_dim)
    if suffix == "ssm_norm":
        return tensor
    raise ValueError(f"unsupported GDN tensor suffix: {suffix}")
