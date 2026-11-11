# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import envs, ir
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms.rdna import on_rdna_family
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)
            # Dummy weights keep norms neutral: random ones would shrink every
            # activation and skew data-dependent kernels (e.g. sparse top-k).
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})

        # When has_weight=False, pass weight=None so implementations that
        # support a weightless path can skip the per-channel multiply.
        # Implementations that require weight (e.g. oink) fall back via IR
        # op priority when weight=None is unsupported.
        self.pass_weight = self.has_weight
        self.pass_weight_add = self.has_weight

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
            return ir.ops.fused_add_rms_norm.maybe_inplace(
                x,
                residual,
                self.weight.data if self.pass_weight_add else None,
                self.variance_epsilon,
                self.variance_size_override,
            )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if on_rdna_family() and x.dtype == torch.bfloat16:
            # The RDNA AOT RMSNorm kernels only accept FP16. DFlash2 keeps
            # its BF16 range even when its target is served in FP16.
            assert self.variance_size_override is None
            if residual is not None:
                residual = x + residual
            normalized = F.rms_norm(
                residual if residual is not None else x,
                (self.hidden_size,),
                self.weight.data if self.has_weight else None,
                self.variance_epsilon,
            )
            return (normalized, residual) if residual is not None else normalized
        if envs.VLLM_BATCH_INVARIANT:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            pass_weight = (
                self.pass_weight_add if residual is not None else self.pass_weight
            )
            return rms_norm_batch_invariant(
                x,
                self.weight.data if pass_weight else None,
                self.variance_epsilon,
                residual=residual,
            )

        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_BATCH_INVARIANT and residual is not None:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            weight = self.weight.data if self.pass_weight_add else None
            return ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
                x, residual, weight, self.variance_epsilon
            )
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# --8<-- [start:gemma_rms_norm]
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    # --8<-- [end:gemma_rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        set_weight_attrs(self.weight, {"dummy_weight_value": 0.0})
        self.variance_epsilon = eps
        self._opaque_rdna = on_rdna_family()

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if not (self._opaque_rdna and x.is_cuda):
            weight = self.weight.float() + 1.0
            if residual is None:
                return ir.ops.rms_norm(x, weight, self.variance_epsilon)
            return ir.ops.fused_add_rms_norm(x, residual, weight, self.variance_epsilon)
        # RDNA: opaque custom ops so inductor cannot lower Gemma's (1+w) RMS
        # (that lowering produces garbage greedy decode on Qwen3.5 hybrid).
        if residual is None:
            return torch.ops.vllm.gemma_rms_norm(
                x, self.weight.data, self.variance_epsilon
            )
        return torch.ops.vllm.gemma_fused_add_rms_norm(
            x, residual, self.weight.data, self.variance_epsilon
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_BATCH_INVARIANT:
            if residual is None:
                return self.forward_cuda(x, residual)
            return ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
                x,
                residual,
                self.weight.float() + 1.0,
                self.variance_epsilon,
            )
        import vllm._xpu_ops  # noqa: F401 registers torch.ops.vllm.xpu_gemma_rms_norm

        # Fall back to the native path if the fused gemma kernels are not
        # available in the installed vllm-xpu-kernels package.
        if not hasattr(torch.ops._C, "gemma_rms_norm"):
            return self.forward_native(x, residual)

        # Pass the raw (bf16/fp16) weight; the +1 offset and the fp32 multiply
        # are folded into the kernel (matches forward_native numerics).
        if residual is not None:
            torch.ops.vllm.xpu_fused_add_gemma_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        # empty_like preserves x's strides, but the kernel requires a
        # contiguous out (unlike x, which it can handle non-contiguous).
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        torch.ops.vllm.xpu_gemma_rms_norm(
            out, x, self.weight.data, self.variance_epsilon
        )
        return out


@CustomOp.register("layer_norm")
class LayerNorm(CustomOp):
    """Standard (mean-centered) LayerNorm.

    Drop-in for a bare `nn.LayerNorm` that dispatches to a fused XPU kernel
    when one is available, and to the native implementation otherwise.

    Normalization runs in the wider of the input and parameter dtypes and the
    result is cast back to the input dtype, so `dtype=torch.float32` gives an
    fp32 reduction for a lower-precision input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = eps
        weight_dtype = dtype or torch.get_default_dtype()
        self.weight: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=weight_dtype))
                set_weight_attrs(self.bias, {"dummy_weight_value": 0.0})

    def _compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.weight is None:
            return x.dtype
        return torch.promote_types(x.dtype, self.weight.dtype)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._compute_dtype(x)
        if compute_dtype != x.dtype:
            return F.layer_norm(
                x.to(compute_dtype),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).type_as(x)
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        import vllm._xpu_ops  # noqa: F401 registers torch.ops.vllm.xpu_layer_norm

        if (
            self.weight is None
            or self.bias is None
            or not hasattr(torch.ops._C, "layer_norm")
            # The kernel reduces in the input dtype, so a widened reduction
            # takes the native path.
            or self._compute_dtype(x) != x.dtype
        ):
            return self.forward_native(x)
        # empty_like preserves x's strides, but the kernel requires a
        # contiguous out (unlike x, which it can handle non-contiguous).
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        torch.ops.vllm.xpu_layer_norm(out, x, self.weight, self.bias, self.eps)
        return out

    def extra_repr(self) -> str:
        return f"hidden_size={self.normalized_shape[0]}, eps={self.eps}"


# --8<-- [start:rms_norm_gated]
@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    # --8<-- [end:rms_norm_gated]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            device: Device to create parameters on
            dtype: Data type for parameters
            activation: Activation function name for gating

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        z: torch.Tensor | None,
        weight: torch.Tensor,
        epsilon: float,
        orig_dtype: torch.dtype,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
    ) -> torch.Tensor:
        """Pure-PyTorch RMS normalization with optional gating.

        This static method contains the full native logic so that both
        ``forward_native`` and ``MatcherRMSNormGated`` (used by the
        compilation pattern matcher) can share the same implementation.

        If *z* is not None and *norm_before_gate* is True:
            ``out = rms_norm(x) * act(z)``
        If *z* is not None and *norm_before_gate* is False:
            ``out = rms_norm(x * act(z))``
        """
        x = x.float()
        weight = weight.float()
        if z is not None:
            z = z.float()

        assert activation in ["silu", "sigmoid", "swish"]
        act_fn = F.sigmoid if activation == "sigmoid" else F.silu

        if z is not None and not norm_before_gate:
            x = x * act_fn(z)

        if group_size is None:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + epsilon)
            out = x_normed * weight
        else:
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + epsilon)
            out = rearrange(x_normed, "... g d -> ... (g d)") * weight

        if z is not None and norm_before_gate:
            out = out * act_fn(z)

        return out.to(orig_dtype)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(
            x,
            z,
            self.weight,
            self.eps,
            x.dtype,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.ops.vllm.fla_rms_norm_gated(
            x,
            z,
            self.weight,
            self.eps,
            self.group_size,
            self.norm_before_gate,
            self.activation,
        )

    @classmethod
    def enabled(cls) -> bool:
        # Inductor defaults custom_ops to 'none', which leaves the gated
        # norm on the eager decomposed path (~9 kernels/call). The HIP AOT
        # kernel covers Qwen3.x GDN (group=None, norm_before_gate, fp16);
        # anything else falls back to forward_native inside forward_hip.
        if (
            on_rdna_family()
            and hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gated_rms_norm")
        ):
            return True
        return super().enabled()

    def forward_hip(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (
            z is not None
            and self.group_size is None
            and self.norm_before_gate
            and x.dtype == torch.float16
            and z.dtype == torch.float16
            and self.activation in ("silu", "swish", "sigmoid")
            and x.dim() >= 2
            and z.shape == x.shape
        ):
            # The kernel takes 2D rows; GDN passes [tokens, heads, head_dim]
            # and normalizes per head, i.e. over the last dim. Released builds
            # of _rocm_C.gated_rms_norm still declare 2D inputs, so a 3D z is
            # always flattened: passing a heads-contiguous 3D z in place
            # raised "input and z must be 2D [M, N]" at the first TP2 profile
            # run (model_runner.profile_run -> _output_projection -> forward_hip).
            act = 1 if self.activation == "sigmoid" else 0
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            out = torch.empty_like(x2)
            z_arg = z.reshape(-1, z.shape[-1]).contiguous()
            torch.ops._rocm_C.gated_rms_norm(
                out,
                x2,
                z_arg,
                self.weight.data,
                self.eps,
                act,
            )
            return out.view(x.shape)
        return self.forward_native(x, z)

    def forward_xpu(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_cuda(x, z)


@functools.cache
def _gemma_norm_hip() -> bool:
    """One-kernel RDNA Gemma norm available (VLLM_RDNA_GEMMA_NORM_HIP=0 off)."""
    if os.environ.get("VLLM_RDNA_GEMMA_NORM_HIP", "1") == "0":
        return False
    return (
        torch.version.hip is not None
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gemma_rms_norm")
    )


def _norm_rows(x: torch.Tensor) -> torch.Tensor | None:
    """Return x as a [rows, N] view with unit column stride and any row stride.

    None when the leading dims do not merge into one row stride (a [T, heads, D]
    slice of a wider tensor); the caller then keeps the eager path, since the
    fused kernel would need a copy.
    """
    n = x.shape[-1]
    if x.stride(-1) != 1:
        return None
    try:
        rows = x.view(-1, n)
    except RuntimeError:
        return None
    return rows if rows.shape[0] < 2 or rows.stride(0) >= n else None


def gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Gemma RMSNorm: x * (1+w) / rms. Opaque to inductor."""
    if (
        x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.is_cuda
        and weight.is_contiguous()
        and _gemma_norm_hip()
        and (rows := _norm_rows(x)) is not None
    ):
        # One kernel instead of the cast, add, cast and norm launches below
        # (28 calls per Flash-Next decode step), reading row-strided views
        # (pooled keys cut out of a wider buffer) in place.
        from vllm import _custom_ops as ops

        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.gemma_rms_norm(out.view(rows.shape), rows, weight, epsilon)
        return out
    # Fold in x.dtype: an fp32 scale fails the vllm_c dtype-match guard and
    # silently falls back to the decomposed native path (~5 kernels/call).
    scale = (weight.float() + 1.0).to(x.dtype)
    return ir.ops.rms_norm(x, scale, epsilon)


def gemma_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    # The real op preserves x's stride; at runtime x is contiguous
    # (e.g. (2048,6,256) with stride (1536,256,1)). torch.empty_like on a
    # non-contiguous traced fake returns a transposed stride (256,524288,1),
    # so inductor asserts a stride the runtime output lacks under piecewise
    # compile. Return a contiguous tensor to match the real op's output.
    return torch.empty(x.size(), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="gemma_rms_norm",
    op_func=gemma_rms_norm,
    fake_impl=gemma_rms_norm_fake,
)


def gemma_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma fused residual+RMSNorm. Opaque to inductor."""
    scale = (weight.float() + 1.0).to(x.dtype)
    return ir.ops.fused_add_rms_norm(x, residual, scale, epsilon)


def gemma_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # See gemma_rms_norm_fake: the real op preserves the (contiguous at
    # runtime) input stride; empty_like on a transposed traced fake makes
    # inductor assert the wrong stride under piecewise compile.
    return (
        torch.empty(x.size(), dtype=x.dtype, device=x.device),
        torch.empty(residual.size(), dtype=residual.dtype, device=residual.device),
    )


direct_register_custom_op(
    op_name="gemma_fused_add_rms_norm",
    op_func=gemma_fused_add_rms_norm,
    fake_impl=gemma_fused_add_rms_norm_fake,
)


def _fla_rms_norm_gated(
    x: torch.Tensor,
    z: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    group_size: int | None,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        rmsnorm_fn,
    )

    return rmsnorm_fn(
        x,
        weight,
        None,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )


def _fla_rms_norm_gated_fake(
    x: torch.Tensor,
    z: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    group_size: int | None,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    return x.new_empty(x.shape)


direct_register_custom_op(
    op_name="fla_rms_norm_gated",
    op_func=_fla_rms_norm_gated,
    fake_impl=_fla_rms_norm_gated_fake,
)
