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


def inverse_gdn_tensor(name, tensor):
    """Restore one GGUF GDN tensor suffix to native flat Qwen4 layout."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
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
        or int(match[1]) >= 48
        or int(match[1]) % 4 == 3
        or match[2] not in roles
    ):
        raise ValueError(f"invalid GDN tensor name: {name!r}")
    suffix = roles[match[2]]
    rows = NUM_V_HEADS * HEAD_DIM
    if suffix == "attn_qkv":
        if tensor.ndim != 2 or tensor.shape[0] != 4096 + rows:
            raise ValueError(
                "attn_qkv must be rank 2 with QKV rows followed by 48 V heads"
            )
        return torch.cat(
            (
                tensor[:4096],
                inverse_v_heads(tensor[4096:], 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_DIM),
            ),
            dim=0,
        )
    if suffix == "ssm_conv1d":
        if tensor.ndim != 2 or tensor.shape != (4096 + rows, 4):
            raise ValueError("ssm_conv1d must have shape (10240, 4)")
        restored = torch.cat(
            (
                tensor[:4096],
                inverse_v_heads(tensor[4096:], 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_DIM),
            ),
            dim=0,
        )
        return restored.unsqueeze(1)
    if suffix == "attn_gate":
        if tensor.ndim != 2 or tensor.shape[0] != rows:
            raise ValueError(f"{suffix} must be rank 2 with {rows} rows")
        result = inverse_v_heads(tensor, 0, NUM_K_HEADS, NUM_V_PER_K, HEAD_DIM)
        return result
    if suffix in ("ssm_alpha", "ssm_beta"):
        if tensor.ndim != 2 or tensor.shape[0] != NUM_V_HEADS:
            raise ValueError(f"{suffix} must be a matrix with {NUM_V_HEADS} rows")
        return inverse_v_heads(tensor, 0, NUM_K_HEADS, NUM_V_PER_K, 1)
    if suffix == "ssm_dt.bias":
        if tensor.ndim != 1 or tensor.shape[0] != NUM_V_HEADS:
            raise ValueError(f"{suffix} must be a vector of length {NUM_V_HEADS}")
        return inverse_v_heads(tensor, 0, NUM_K_HEADS, NUM_V_PER_K, 1)
    if suffix == "ssm_a":
        if tensor.ndim != 1 or tensor.shape[0] != NUM_V_HEADS:
            raise ValueError(f"ssm_a must be a vector of length {NUM_V_HEADS}")
        if not torch.isfinite(tensor).all() or (tensor >= 0).any():
            raise ValueError("GGUF ssm_a values must be finite and negative")
        reordered = inverse_v_heads(tensor, 0, NUM_K_HEADS, NUM_V_PER_K, 1)
        return torch.log(-reordered.float())
    if suffix == "ssm_out":
        if tensor.ndim != 2 or tensor.shape[1] != rows:
            raise ValueError(f"ssm_out must have {rows} input columns")
        return inverse_v_heads(tensor, 1, NUM_K_HEADS, NUM_V_PER_K, HEAD_DIM)
    if suffix == "ssm_norm":
        return tensor
    raise ValueError(f"unsupported GDN tensor suffix: {suffix}")
