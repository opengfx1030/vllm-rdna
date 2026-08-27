# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (QTIP-style bitshift trellis) quantization method for RDNA (gfx1030/gfx1100).

Loads EXL3 checkpoints produced by ExLlamaV3:
  - ``{prefix}.trellis``  (k/16, n/16, 256*bits/16) int16 — packed tail-biting stream
  - ``{prefix}.suh``      (k,)    fp16 — A-side scales + Hadamard flips
  - ``{prefix}.svh``      (n,)    fp16 — C-side scales + Hadamard flips

The forward path (wiki kernels/exl3.md: Hadamard OUTSIDE the K-dot):
  1. xh = exl3_hadamard_128(x, pre=suh)            # A-side H, 1/sqrt(128)
  2. mid = exl3_gemm_rdna2(xh, trellis)            # raw decode: xh @ W_hat
  3. out = exl3_hadamard_128(mid, post=svh)        # C-side H

RDNA-only kernels (V_DOT2_F32_F16 + Wave32). On non-RDNA the method falls
back to UnquantizedLinearMethod (no EXL3 support elsewhere).
"""

from typing import TYPE_CHECKING, Any, List, Optional

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _rdna_exl3_available() -> bool:
    """The HIP kernels are registered by the C++ extension (RDNA build)."""
    return (
        hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "exl3_gemm_rdna2")
        and hasattr(torch.ops._rocm_C, "exl3_hadamard_128")
    )


class Exl3Config(QuantizationConfig):
    """Quantization config for EXL3 (bitshift trellis) checkpoints."""

    def __init__(
        self,
        bits_per_weight: float,
        head_bits: int,
        codebook: str,
        calibration: dict[str, int] | None = None,
        out_scales: str | None = None,
        ignored_layers: list[str] | None = None,
        original_quantization_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits_per_weight = bits_per_weight
        self.head_bits = head_bits
        self.codebook = codebook
        self.calibration = calibration
        self.out_scales = out_scales
        self.ignored_layers = ignored_layers
        self.original_quantization_config = original_quantization_config

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm gates via on_gfx10x()/on_gfx1x() at load time.
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantization_config.json", "config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        """Build from the parsed quantization_config.json dict."""
        hf_cfg = config if isinstance(config, dict) else {}
        bits = hf_cfg.get("bits", 3.0)
        head_bits = hf_cfg.get("head_bits", 6)
        codebook = hf_cfg.get("codebook", "3inst")
        calibration = hf_cfg.get("calibration")
        out_scales = hf_cfg.get("out_scales")
        ignored_layers = hf_cfg.get("ignored_layers")
        original = hf_cfg.get("original_quantization_config")
        cu = cls(
            bits_per_weight=float(bits),
            head_bits=int(head_bits),
            codebook=codebook,
            calibration=calibration,
            out_scales=out_scales,
            ignored_layers=ignored_layers,
            original_quantization_config=original,
        )
        if not _rdna_exl3_available():
            logger.warning_once(
                "EXL3 kernels not registered in torch.ops._rocm_C; "
                "EXL3 layers will fall back to unquantized.",
            )
        return cu

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """vLLM entry: map a module to a quant method. Only dense LinearBase
        layers are handled; embeddings/attention etc. fall through."""
        if "embed_tokens" in prefix:
            return None
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not _rdna_exl3_available():
                logger.warning_once(
                    "EXL3 kernels not available on this build; "
                    "falling back to unquantized for %s",
                    prefix,
                )
                return UnquantizedLinearMethod()
            return Exl3LinearMethod()
        return None


class Exl3LinearMethod(LinearMethodBase):
    """Linear method for EXL3 (trellis/suh/svh) layers."""

    def __init__(self) -> None:
        super().__init__()
        self.bits = 3
        self.cb = 0  # 3inst

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Register trellis / suh / svh parameters, named exactly as the
        checkpoint keys so the default loader maps them by suffix."""
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                f"EXL3 requires input (K) divisible by 16, got "
                f"{input_size_per_partition}"
            )
        if output_size_per_partition % 16 != 0:
            raise ValueError(
                f"EXL3 requires output (N) divisible by 16, got "
                f"{output_size_per_partition}"
            )
        k_tiles = input_size_per_partition // 16
        n_tiles = output_size_per_partition // 16
        trellis_words = 256 * self.bits // 16  # 48
        trellis = torch.nn.Parameter(
            torch.empty(
                k_tiles,
                n_tiles,
                trellis_words,
                dtype=torch.int16,
                device="cuda",
            ),
            requires_grad=False,
        )
        layer.register_parameter("trellis", trellis)
        set_weight_attrs(trellis, extra_weight_attrs)
        suh = torch.nn.Parameter(
            torch.empty(
                input_size_per_partition, dtype=params_dtype, device="cuda"
            ),
            requires_grad=False,
        )
        layer.register_parameter("suh", suh)
        set_weight_attrs(suh, extra_weight_attrs)
        svh = torch.nn.Parameter(
            torch.empty(
                output_size_per_partition,
                dtype=params_dtype,
                device="cuda",
            ),
            requires_grad=False,
        )
        layer.register_parameter("svh", svh)
        set_weight_attrs(svh, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the 3-op EXL3 pipeline (Hadamard outside the K-dot)."""
        del bias  # EXL3 dense layers in this checkpoint carry no bias
        trellis: torch.Tensor = layer.trellis
        suh: torch.Tensor = layer.suh
        svh: torch.Tensor = layer.svh

        if (
            not x.is_cuda
            or not _rdna_exl3_available()
            or x.dtype not in (torch.half, torch.bfloat16)
        ):
            raise RuntimeError(
                "EXL3 linear requires RDNA + the HIP kernels; layer has no "
                "unquantized fallback weight."
            )

        x = x.to(torch.half) if x.dtype == torch.bfloat16 else x
        M, K = x.shape
        N = trellis.shape[1] * 16
        bits = self.bits
        cb = self.cb

        # 1. A-side Hadamard + input scale
        xh = torch.empty_like(x)
        ops.exl3_hadamard_128(x, xh, suh, None, 1.0)
        # 2. raw decode GEMM
        mid = torch.empty(M, N, dtype=torch.half, device=x.device)
        ops.exl3_gemm_rdna2(xh, mid, trellis, M, N, K, bits, cb)
        # 3. C-side Hadamard + output scale
        out = torch.empty_like(mid)
        ops.exl3_hadamard_128(mid, out, None, svh, 1.0)
        return out