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

import os
import re

import torch

_call_idx = [0]
_exl3_log_fh = [None]


@torch._dynamo.allow_in_graph
def _exl3_log(msg):
    if _exl3_log_fh[0] is None:
        _exl3_log_fh[0] = open("/tmp/exl3_apply_path.log", "a")
    _exl3_log_fh[0].write(msg + "\n")
    _exl3_log_fh[0].flush()


@torch._dynamo.allow_in_graph
def _exl3_hadamard(x, xh, suh, svh, scale):
    return torch.ops._rocm_C.exl3_hadamard_128(x, xh, suh, svh, scale)


@torch._dynamo.allow_in_graph
def _exl3_gemm(a, c, b_q_weight, bits, cb):
    return torch.ops._rocm_C.exl3_gemm_rdna2(a, c, b_q_weight, bits, cb)
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
    """Quantization config for EXL3 (bitshift trellis) checkpoints.

    Only the trellis-quantized projections use Exl3LinearMethod; the rest of
    the architecture (embeddings, norms, GDN in_proj_a/b, rmsnorm etc.) stays
    fp16.
    """

    # exllamav3 stores codebook marker scalars on some layers (lm_head.mul1 /
    # .mcg). vLLM has no such weights; ignore them on load.
    _ignore_unexpected_suffixes = (
        ".bias",
        ".mul1",
        ".mcg",
        ".q_scale",
        ".k_scale",
        ".v_scale",
    )

    # GDN/mamba projections that are NOT trellis-quantized in this checkpoint
    # family (stored as plain fp16 ``.weight``). Prefix-matched.
    _DEFAULT_IGNORED = [
        "in_proj_a",
        "in_proj_b",
        "in_proj_ba",
        "conv1d",
        "dt_bias",
        "A_log",
        "norm.weight",
    ]

    # Filled during weight loading by utils.AutoWeightsLoader._capture_marker_
    # names: checkpoint modules whose training used the mul1/mcg codebook
    # (marker tensors like ``...layers.17.mlp.gate_proj.mul1``).
    # The decoder (exllamav3_ext.reconstruct_had_slice) requires the right
    # codebook per module; the default is the 3inst codebook.
    _exl3_mul1_marks: set[str] = set()
    _exl3_mcg_marks: set[str] = set()
    _capture_marker_prefixes = (".mul1", ".mcg")

    def _capture_marker_names(self, name: str) -> bool:
        if name.endswith(self._capture_marker_prefixes):
            container = re.sub(r"^.*?layers\.", "layers.", name)
            container = container.rsplit(".", 2)[0]
            if name.endswith(".mul1"):
                self._exl3_mul1_marks.add(container)
            else:
                self._exl3_mcg_marks.add(container)
        return False

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
        # Build the set of module prefixes the checkpoint actually quantized
        # (from tensor_storage quant_format=exl3). get_quant_method only
        # applies Exl3LinearMethod to these.
        storage = hf_cfg.get("tensor_storage", {})
        def _norm(m):
            m = m.replace("model.language_model.", "").replace(
                "language_model.", "")
            return m
        cu._exl3_storage = {
            k: v for k, v in storage.items()
            if v.get("quant_format") == "exl3"
        }
        cu._exl3_suffixes = [_norm(m) for m in cu._exl3_storage]
        if not _rdna_exl3_available():
            logger.warning_once(
                "EXL3 kernels not registered in torch.ops._rocm_C; "
                "EXL3 layers will fall back to unquantized.",
            )
        return cu

    # Checkpoint tensors that exist for the exllamav3 runtime but are not
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """vLLM entry: map a module to a quant method. Dense LinearBase
        layers and the (untied) ParallelLMHead get Exl3LinearMethod; fp16-only
        modules (embeddings, norms, GDN in_proj_a/b) are skipped."""
        if "embed_tokens" in prefix:
            return None
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        is_linear = isinstance(layer, LinearBase) or isinstance(
            layer, VocabParallelEmbedding
        )
        if not is_linear:
            return None
        if any(pat in prefix for pat in self._DEFAULT_IGNORED):
            return UnquantizedLinearMethod()
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
        # Only EXL3-quantized modules (from tensor_storage, when available) get
        # the method; fp16-only modules fall through unquantized. Storage may
        # be absent (config-embedded quant metadata), so fall back to
        # known-projection names.
        known = {"gate_proj", "up_proj", "down_proj", "out_proj", "q_proj",
                 "k_proj", "v_proj", "o_proj", "gate_up_proj", "qkv_proj",
                 "in_proj_qkv", "in_proj_z", "in_proj_qkvz", "lm_head"}
        pn = prefix.split(".")[-1]
        if pn not in known:
            import os as _os
            if _os.environ.get("VLLM_EXL3_DEBUG") == "1":
                print(f"[exl3] UNQUANTIZED prefix={prefix} pn={pn}", flush=True)
            return UnquantizedLinearMethod()
        if getattr(self, "_exl3_suffixes", None):
            if not any(s.endswith(pn) for s in self._exl3_suffixes):
                return UnquantizedLinearMethod()
        head_bits = getattr(self, "head_bits", 6)
        is_head = prefix.endswith("lm_head")
        if is_head and int(head_bits) <= 0:
            # head_bits=0: head left unquantized (dense fp16). Cheap
            # fallback for large-vocab models whose 6bpw head trellis
            # search is memory-heavy; the head never uses the trellis
            # kernel (always the dequant path) so kernel coverage is
            # unchanged.
            return UnquantizedLinearMethod()
        return Exl3LinearMethod(bits=head_bits if is_head else 3)


class Exl3LinearMethod(LinearMethodBase):
    """Linear method for EXL3 (trellis/suh/svh) layers.

    ``bits`` is the bpw used to size the trellis storage: 3 for the body
    projections, ``head_bits`` (6) for the lm_head (ExLlamaV3 quantizes the
    head at a higher bpw than the body).
    """

    def __init__(self, bits: int = 3) -> None:
        super().__init__()
        self.bits = bits
        self.cb = 0  # 3inst

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Register trellis / suh / svh parameters, named exactly as the
        checkpoint keys so the loader maps them by suffix.

        ``output_partition_sizes`` are the per-shard N widths (single-width
        for a plain linear, or [key,key,value,value] for the fused
        in_proj_qkvz). The trellis N-dim is in 16-element tiles, so shard
        widths are divided by 16 when the loader places each part.
        """
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
        layer._exl3_part_sizes = list(output_partition_sizes)
        suh_parts: list[torch.Tensor] = []
        layer._exl3_suh_parts = suh_parts

        # The trellis param is the checkpoint's tile layout [K/16, N/16, W].
        # No output_dim is set: vLLM's generic merged-module shard slicing
        # operates on flat element dims, which do not apply here, so the
        # loader handles all shard placement itself via output_partition_sizes
        # (each shard is N/16-aligned).
        def _trellis_loader(param, loaded_weight, shard_id=None):
            if os.environ.get("VLLM_EXL3_DEBUG") == "1":
                print(f"[exl3] {getattr(layer, 'prefix', '?'):80s} "
                      f"trellis shard={shard_id} w_shape={tuple(loaded_weight.shape)}",
                      flush=True)
            if shard_id is None:
                if param.data.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                    return
                # TP>1: framework passed no shard_id; contiguous N-tile slice.
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_world_size() > 1:
                    rank = dist.get_rank()
                    n_tiles_shard = param.data.shape[1]
                    nt_off = rank * n_tiles_shard
                    param.data[:, :, :] = loaded_weight[
                        :, nt_off:nt_off + n_tiles_shard, :]
                    return
            if shard_id is not None:
                off, width = _shard_range(shard_id)
                n_t = width // 16
                assert loaded_weight.shape[1] == n_t
                param.data[
                    :, off // 16: off // 16 + n_t, :] = loaded_weight
                return
            raise AssertionError(
                f"trellis shape mismatch param={tuple(param.data.shape)} "
                f"loaded={tuple(loaded_weight.shape)}")

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
        set_weight_attrs(trellis, {
            "weight_loader": _trellis_loader,
            **{k: v for k, v in extra_weight_attrs.items()
               if k != "weight_loader"},
        })

        def _shard_range(shard_id):
            """(offset, width) of a shard in N elements. shard_id may be
            ints (output index), strings (q/k/v), or a tuple/list of them."""
            def _idx(i):
                if isinstance(i, str):
                    name = {"q": 0, "k": 1, "v": 2, "up": 0, "gate": 1}.get(
                        i.lower(), 0)
                    return name
                return int(i)

            idxs = [_idx(i) for i in shard_id] if isinstance(
                shard_id, (tuple, list)) else [_idx(shard_id)]
            off = sum(output_partition_sizes[: idxs[0]])
            width = sum(output_partition_sizes[i] for i in idxs)
            return off, width

        def _suh_loader(param, loaded_weight, shard_id=None):
            if os.environ.get("VLLM_EXL3_DEBUG") == "1":
                print(f"[exl3] {getattr(layer, 'prefix', '?'):80s} "
                      f"suh shard={shard_id} w_shape={tuple(loaded_weight.shape)}",
                      flush=True)
            # suh is input-side (K) — never sharded across N. Fused
            # multi-output layers ship one (submodule-specific) vector per
            # shard; keep every part with its N-range for the per-submodule
            # dequant.
            param.data.copy_(loaded_weight)
            if shard_id is not None:
                rng = _shard_range(shard_id)
            else:
                # TP>1: framework passed no shard_id; use contiguous N-range.
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_world_size() > 1:
                    rank = dist.get_rank()
                    tp_size = dist.get_world_size()
                    total_n = sum(output_partition_sizes)
                    shard_n = total_n // tp_size
                    rng = (rank * shard_n, (rank + 1) * shard_n)
                else:
                    rng = (0, output_size_per_partition)
            suh_parts.append((loaded_weight.detach().to(
                param.device).clone(), rng[0], rng[1]))

        suh = torch.nn.Parameter(
            torch.empty(
                input_size_per_partition, dtype=params_dtype, device="cuda"
            ),
            requires_grad=False,
        )
        setattr(suh, "output_dim", 0)
        layer.register_parameter("suh", suh)
        set_weight_attrs(suh, {
            "weight_loader": _suh_loader,
            **{k: v for k, v in extra_weight_attrs.items()
               if k != "weight_loader"},
        })

        def _svh_loader(param, loaded_weight, shard_id=None):
            if shard_id is None:
                if param.data.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)
                    return
                # TP>1: framework passed no shard_id; contiguous slice by rank.
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_world_size() > 1:
                    rank = dist.get_rank()
                    shard_size = param.data.shape[0]
                    param.data.copy_(
                        loaded_weight[rank * shard_size:
                                      (rank + 1) * shard_size])
                    return
            if shard_id is not None:
                off, width = _shard_range(shard_id)
                param.data[off:off + width] = loaded_weight
                return
            raise AssertionError(
                f"svh shape mismatch param={tuple(param.data.shape)} "
                f"loaded={tuple(loaded_weight.shape)}")

        svh = torch.nn.Parameter(
            torch.empty(
                output_size_per_partition,
                dtype=params_dtype,
                device="cuda",
            ),
            requires_grad=False,
        )
        setattr(svh, "output_dim", 0)
        layer.register_parameter("svh", svh)
        set_weight_attrs(svh, {
            "weight_loader": _svh_loader,
            **{k: v for k, v in extra_weight_attrs.items()
               if k != "weight_loader"},
        })

        # vLLM's fused-linear loader always looks for a `.weight` param; the
        # EXL3 checkpoint stores trellis/suh/svh instead. Replace the
        # layer's original fp16 weight (which can be 18 GB on a 9B model)
        # with a 1×1 dummy + noop loader so `getattr(layer, "weight")`
        # resolves (and synthetic .weight loads are ignored) instead of
        # falling back to the module and failing on `.data`.
        if hasattr(layer, "weight"):
            layer._parameters.pop("weight", None)
        weight = torch.nn.Parameter(
            torch.empty(1, 1, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(weight, {
            "weight_loader": lambda p, w, shard_id=None: None,
        })
        layer.register_parameter("weight", weight)

        # Path B: register _w_fp16 for layers that need the dense dequant
        # path (MergedLinear with >1 partition, or bits=6 lm_head). When
        # the checkpoint has `{prefix}._w_fp16` embedded, the loader fills
        # it and process_weights_after_loading skips the exllamav3 import.
        # Single-shard body layers (bits 2/3/4, 1 partition, unmarked) use
        # the kernel directly and don't need _w_fp16 — skip the allocation
        # to save ~50 MB / layer * ~200 layers = ~10 GB.
        needs_fp16_dequant = (
            len(output_partition_sizes) > 1 or self.bits == 6
        )
        if needs_fp16_dequant:
            def _w_fp16_loader(param, loaded_weight, shard_id=None):
                layer._w_fp16_loaded = True
                if shard_id is not None:
                    off, width = _shard_range(shard_id)
                    param.data[off:off + width] = loaded_weight.to(
                        param.data.device)
                    return
                if param.data.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight.to(param.data.device))
                    return
                # TP>1 contiguous slice by rank (N is sharded dim 0).
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_world_size() > 1:
                    rank = dist.get_rank()
                    shard_n = param.data.shape[0]
                    param.data.copy_(
                        loaded_weight[rank * shard_n:(rank + 1) * shard_n]
                        .to(param.data.device))
                    return
                raise AssertionError(
                    f"_w_fp16 shape mismatch param={tuple(param.data.shape)}"
                    f" loaded={tuple(loaded_weight.shape)}")

            w_fp16 = torch.nn.Parameter(
                torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=params_dtype,
                    device="cuda",
                ),
                requires_grad=False,
            )
            setattr(w_fp16, "output_dim", 0)
            layer.register_parameter("_w_fp16", w_fp16)
            set_weight_attrs(w_fp16, {
                "weight_loader": _w_fp16_loader,
                **{k: v for k, v in extra_weight_attrs.items()
                   if k != "weight_loader"},
            })

        # Cudagraph-friendly buffers for the per-partition runtime GEMM path.
        # Without these, apply() allocates fresh out/xh_i/trellis_i/mid_i/out_i
        # each call. The captured cudagraph would replay with the warmup-time
        # tensor pointers, but subsequent calls allocate at different addresses
        # -> kernel reads stale memory -> NaN. Slicing pre-allocated buffers
        # gives stable data ptrs so cudagraph replay always hits the same memory.
        # M_MAX covers the cudagraph capture sizes used at decode time. Larger
        # M (chunked prefill above this) falls back to dynamic allocation in
        # apply(); per-call allocations there don't replay cleanly under
        # captured graphs, so the fallback is eager-only. With M_MAX=64 we
        # keep buffer memory to ~1/8 of M_MAX=512 (8 GiB total on 9B models)
        # while still covering decode (M=1) and short prompts.
        M_MAX = 64  # CG-PATH active; trellis copy in process_weights_after_loading
        layer._exl3_M_MAX = M_MAX
        layer._exl3_part_widths = list(output_partition_sizes)
        layer._exl3_buf_out = torch.empty(
            M_MAX, sum(output_partition_sizes),
            dtype=params_dtype, device="cuda",
        )
        if os.environ.get("VLLM_EXL3_DEBUG") == "1":
            _exl3_log("[exl3] CW %s part_widths=%s sum=%d buf_out=%s"
                      % (str(getattr(layer, "prefix", "?")),
                         layer._exl3_part_widths,
                         sum(output_partition_sizes),
                         tuple(layer._exl3_buf_out.shape)))
        layer._exl3_bufs_xh = [
            torch.empty(M_MAX, input_size_per_partition,
                       dtype=params_dtype, device="cuda")
            for _ in output_partition_sizes
        ]
        layer._exl3_bufs_mid = [
            torch.empty(M_MAX, w, dtype=params_dtype, device="cuda")
            for w in output_partition_sizes
        ]
        layer._exl3_bufs_out_part = [
            torch.empty(M_MAX, w, dtype=params_dtype, device="cuda")
            for w in output_partition_sizes
        ]
        # Per-partition contiguous trellis slices so apply() doesn't need a
        # .contiguous() allocation each call.
        layer._exl3_bufs_trellis = [
            torch.empty(
                input_size_per_partition // 16, w // 16, 48,
                dtype=torch.int16, device="cuda",
            )
            for w in output_partition_sizes
        ]
        layer._exl3_bufs_svh = [
            torch.empty(w, dtype=params_dtype, device="cuda")
            for w in output_partition_sizes
        ]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Bits 2/3/4 single-shard layers run the RDNA2 trellis kernel;
        fused layers (per-shard suh differs), mul1/mcg-marked layers and
        bits 6 (lm_head) are dequantized once here and run as dense fp16
        GEMMs — same weight-prep trick as the RDNA2 W4A16 dense kernel."""
        if hasattr(layer, "_exl3_bufs_trellis"):
            for i, width in enumerate(layer._exl3_part_widths):
                off = sum(layer._exl3_part_widths[:i])
                layer._exl3_bufs_trellis[i].copy_(
                    layer.trellis[:, off // 16:(off + width) // 16, :]
                )
                layer._exl3_bufs_svh[i].copy_(layer.svh[off:off + width])
        # Free the original fp16 weight the framework may have loaded into
        # layer.weight after create_weights ran. On a 9B model that's
        # ~18 GB sitting unused alongside the trellis.
        if hasattr(layer, "weight") and layer.weight is not None:
            w = layer.weight
            if w.numel() > 1:
                layer._parameters.pop("weight", None)
                del w
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        part_sizes = list(getattr(layer, "_exl3_part_sizes", []))
        suh_parts = list(getattr(layer, "_exl3_suh_parts", []))
        fused = len(suh_parts) > 1
        prefix = getattr(layer, "prefix", "") or ""
        m = re.search(r"(layers\.\d+\.\w+)", prefix)
        container = "lm_head" if self.bits == 6 else (
            m.group(1) if m else "")
        qc = getattr(layer, "quant_config", None)
        marked = False
        mul1 = mcg = False
        if qc is not None and container:
            mul1 = container in qc._exl3_mul1_marks
            mcg = container in qc._exl3_mcg_marks
            marked = mul1 or mcg
        if container == "lm_head" and qc is not None and not mul1 and not mcg:
            mul1 = True  # ExLlamaV3 encodes 6bpw heads in the mul1 codebook
        if mul1:
            self.cb = 2
        elif mcg:
            self.cb = 1
        if not fused and not marked and int(self.bits) in (2, 3, 4) and not (
                os.environ.get("VLLM_EXL3_DEQUANT_ALL") == "1"):
            return
        if self.bits not in (2, 3, 4, 6):
            raise NotImplementedError(
                f"EXL3: unsupported bits={self.bits} "
                "(kernel: 2/3/4/6)")
        if self.bits == 6:
            # bits=6 lm_head: kernel runtime GEMM path produces wrong output
            # (exl3_window_pos<6> K-3 fallback). Dequant to fp16 and fold
            # suh/svh on GPU via PyTorch; forward becomes a plain rocBLAS GEMM.
            K_tile, N_tile, _ = layer.trellis.shape
            K, N = K_tile * 16, N_tile * 16
            device = layer.trellis.device
            out = torch.empty(K, N, dtype=torch.half, device=device)
            ops.exl3_dequant_bits6_mul1(layer.trellis, out)
            suh = layer.suh.to(device=device, dtype=torch.half)
            svh = layer.svh.to(device=device, dtype=torch.half)
            r_scale = 1.0 / 12.649110640673516
            out = out * suh.view(K, 1)
            out = out.view(K // 128, 128, N)
            h = out
            for _ in range(7):
                h = h.view(K // 128, 2, 64, N)
                a, b = h[:, 0], h[:, 1]
                h = torch.stack([a + b, a - b], dim=1).view(K // 128, 128, N)
            out = h.view(K, N) * r_scale
            out = out.view(K, N // 128, 128).transpose(0, 1)
            h = out
            for _ in range(7):
                h = h.view(N // 128, K, 2, 64)
                a, b = h[:, :, 0], h[:, :, 1]
                h = torch.stack([a + b, a - b], dim=2).view(N // 128, K, 128)
            out = h.transpose(0, 1).reshape(K, N) * svh.view(1, N) * r_scale
            layer._w_fp16.data.copy_(out.t().contiguous())
            layer._w_fp16_loaded = True
        # Path B: prefer pre-folded weight from safetensors (no exllamav3).
        # The loader fills layer._w_fp16 only when the checkpoint embeds it
        # (repack_with_folded.py). Older checkpoints without it fall
        # through to the runtime reconstruct_had_slice path.
        if (hasattr(layer, "_w_fp16") and layer._w_fp16 is not None
                and getattr(layer, "_w_fp16_loaded", False)
                and layer._w_fp16.numel() > 1):
            self._w_fp16 = layer._w_fp16
            return
        trellis: torch.Tensor = layer.trellis
        svh: torch.Tensor = layer.svh
        K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
        cache_dir = os.environ.get("VLLM_EXL3_FOLDED_CACHE")
        if cache_dir and os.environ.get("VLLM_EXL3_DEBUG") == "1":
            print(f"[exl3] {getattr(layer, 'prefix', '?'):80s} cache_dir={cache_dir}",
                  flush=True)
        cache_key = None
        if cache_dir:
            cache_key = re.sub(r"[^A-Za-z0-9_.-]", "_",
                               getattr(layer, "prefix", "root") or "root")
            cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                self._w_fp16 = torch.load(
                    cache_path, map_location=trellis.device, weights_only=True)
                return
        # apply() uses per-partition runtime GEMM (exl3_hadamard_128 +
        # exl3_gemm_rdna2 per MergedLinear sub-slice). exllamav3 dequant
        # was removed: can't JIT-build on ROCm, and we don't need it.
        if os.environ.get("VLLM_EXL3_DEBUG") == "1":
            print(f"[exl3] {getattr(layer, 'prefix', '?'):80s} "
                  "no _w_fp16 / no cache - per-partition runtime GEMM in apply()",
                  flush=True)
        return

    @torch._dynamo.disable
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

        _exl3_dbg_apply = os.environ.get("VLLM_EXL3_APPLY_DBG") == "1"
        if _exl3_dbg_apply:
            print(f"[exl3_apply] x.shape={tuple(x.shape)} x.device={x.device} "
                  f"x.is_cuda={x.is_cuda} x.data_ptr={x.data_ptr()} "
                  f"prefix={getattr(layer, 'prefix', '?')}",
                  flush=True)

        if (
            not x.is_cuda
            or not _rdna_exl3_available()
            or x.dtype not in (torch.half, torch.bfloat16)
        ):
            raise RuntimeError(
                "EXL3 linear requires RDNA + the HIP kernels; layer has no "
                "unquantized fallback weight."
            )

        # Defensive: vLLM pre-allocates activation buffers sized to
        # max_num_batched_tokens * num_layers (e.g., 8192 = 4 * 2048 for a
        # 4-prompt probe). The tensor shape reflects the buffer size, not the
        # actual batch. If the underlying storage is smaller than the shape
        # implies, the HIP kernel reads past valid memory and faults at a
        # bogus GPU address. Slice x to what the storage actually holds.
        _elem_size = x.element_size()
        _storage_nbytes = x.untyped_storage().nbytes()
        _implied_nbytes = x.numel() * _elem_size
        if _implied_nbytes > _storage_nbytes:
            _actual_rows = _storage_nbytes // (x.shape[1] * _elem_size)
            if _exl3_dbg_apply:
                print(f"[exl3_apply] x.shape={tuple(x.shape)} -> [{_actual_rows}, {x.shape[1]}] "
                      f"(storage={_storage_nbytes} < implied={_implied_nbytes})",
                      flush=True)
            x = x[:_actual_rows]
        # vLLM pre-allocates the activation buffer to
        # max_num_batched_tokens * num_sequences (e.g. 2048 * 4 = 8192 for a
        # 4-prompt probe), but the actual batch is much smaller (1024 here).
        # Even though the storage is properly sized for the full shape, the
        # padding pages are not committed (torch.empty maps virtual pages
        # but doesn't commit physical pages until written). The HIP kernel
        # reads from uncommitted pages and faults at a bogus GPU address.
        # Hard-limit M to max_num_batched_tokens (default 2048) to avoid the
        # uncommitted padding pages.
        _max_batched = int(os.environ.get("VLLM_EXL3_MAX_BATCH", "2048"))
        _original_M = x.shape[0]
        if x.shape[0] > _max_batched:
            if _exl3_dbg_apply:
                print(f"[exl3_apply] x.shape={tuple(x.shape)} -> [{_max_batched}, {x.shape[1]}] "
                      f"(hard cap to max_num_batched_tokens={_max_batched})",
                      flush=True)
            x = x[:_max_batched]
        # Commit GPU pages: vLLM allocates the activation buffer with
        # torch.empty (uncommitted virtual pages). The HIP kernel reads
        # from these pages and faults at a bogus GPU address. The * 1.0
        # operation allocates a new tensor and writes the result into
        # it, which commits the physical pages. The .contiguous() ensures
        # the result is contiguous in case the slice broke contiguity.
        x = (x * 1.0).contiguous()

        x = x.to(torch.half) if x.dtype == torch.bfloat16 else x
        _exl3_dbg = os.environ.get("VLLM_EXL3_DEBUG") == "1"
        folded = getattr(self, "_w_fp16", None)
        if folded is not None:
            # Was the reference's own folded dequant
            # (reconstruct_had_slice: suh/svh + both Hadamards inside the
            # weight), so the forward is a plain GEMM.
            return torch.nn.functional.linear(x, folded)
        M, K = x.shape
        bits = self.bits
        cb = self.cb

        # Cudagraph-friendly path: slice the pre-allocated buffers set up
        # in create_weights. Data ptrs are stable so the captured graph
        # replay hits the same memory each call.
        if (hasattr(layer, "_exl3_bufs_xh")
                and M <= layer._exl3_M_MAX):
            if _exl3_dbg and _call_idx[0] < 8:
                _call_idx[0] += 1
                _exl3_log("[exl3] CG-PATH %s M=%s buf_xh_ptr=%s buf_mid_ptr=%s"
                          % (str(getattr(layer, "prefix", "?")),
                             M,
                             hex(layer._exl3_bufs_xh[0].data_ptr()),
                             hex(layer._exl3_bufs_mid[0].data_ptr())))
            suh_parts_raw = getattr(layer, "_exl3_suh_parts", [])
            suh_tensors = ([sp[0] for sp in suh_parts_raw]
                           if len(suh_parts_raw) == len(layer._exl3_part_widths)
                           else None)
            suh_parts = [
                ((suh_tensors[i] if suh_tensors else suh),
                 sum(layer._exl3_part_widths[:i]), w)
                for i, w in enumerate(layer._exl3_part_widths)
            ]
            buf_out = layer._exl3_buf_out
            for i, (suh_i, off, width) in enumerate(suh_parts):
                xh_i = layer._exl3_bufs_xh[i][:M]
                mid_i = layer._exl3_bufs_mid[i][:M]
                out_i = layer._exl3_bufs_out_part[i][:M]
                trellis_i = layer._exl3_bufs_trellis[i]
                if _exl3_dbg and _call_idx[0] < 12:
                    _exl3_log("[exl3] LOOP %s i=%d off=%d width=%d out_i=%s buf_out_slice=%s part_widths=%s"
                              % (str(getattr(layer, "prefix", "?")),
                                 i, off, width,
                                 tuple(out_i.shape),
                                 tuple(buf_out[:M, off:off + width].shape)
                                 if off + width <= buf_out.shape[1] else "OOB",
                                 layer._exl3_part_widths))
                svh_i = layer._exl3_bufs_svh[i]
                mid_i.zero_()
                _exl3_hadamard(x, xh_i, suh_i, None, 1.0)
                _exl3_gemm(xh_i, mid_i, trellis_i, bits, cb)
                _exl3_hadamard(mid_i, out_i, None, svh_i, 1.0)
                buf_out[:M, off:off + width] = out_i
            # Pad output back to _original_M so downstream shape assertions
            # (e.g., GDN's assert z.shape == x_shape_og) pass. First M rows
            # are kernel output; remaining rows are zeros (padding that
            # downstream doesn't use).
            if _original_M > M and _original_M <= buf_out.shape[0]:
                return buf_out[:_original_M]
            if _original_M > M:
                padded = torch.zeros(_original_M, buf_out.shape[1],
                                     dtype=buf_out.dtype, device=buf_out.device)
                padded[:M] = buf_out[:M]
                return padded
            return buf_out[:M]

        # Fallback (eager / M > M_MAX): dynamic allocation. Not
        # cudagraph-safe — re-introduces per-call allocations that produce
        # stale-pointer NaN under captured graphs.
        if _exl3_dbg and _call_idx[0] < 8:
            _call_idx[0] += 1
            _exl3_log("[exl3] FB-PATH %s M=%s (M_MAX=%s)"
                      % (str(getattr(layer, "prefix", "?")),
                         M,
                         getattr(layer, "_exl3_M_MAX", "?")))
        N = trellis.shape[1] * 16
        # Allocate with _original_M rows (not x.shape[0]) so downstream
        # shape assertions (e.g., GDN's assert z.shape == x_shape_og) pass.
        # torch.zeros (not torch.empty) commits the padding pages; the
        # kernel only writes the first M rows, rest stay zero.
        out = torch.zeros(_original_M, N, dtype=torch.half, device=x.device)
        suh_parts = getattr(layer, "_exl3_suh_parts", [])
        if not suh_parts:
            suh_parts = [(layer.suh, 0, N)]
        for i, (suh_i, off, width) in enumerate(suh_parts):
            xh_i = torch.empty_like(x)
            _exl3_hadamard(x, xh_i, suh_i, None, 1.0)
            trellis_i = trellis[:, off // 16:(off + width) // 16, :].contiguous()
            mid_i = torch.zeros(x.shape[0], width, dtype=torch.half, device=x.device)
            _exl3_gemm(xh_i, mid_i, trellis_i, bits, cb)
            svh_i = layer._exl3_bufs_svh[i] if hasattr(layer, "_exl3_bufs_svh") else svh[off:off + width].to(x.device)
            out_i = torch.zeros(_original_M, width, dtype=torch.half, device=x.device)
            _exl3_hadamard(mid_i, out_i[:M], None, svh_i, 1.0)
            out[:, off:off + width] = out_i
        return out