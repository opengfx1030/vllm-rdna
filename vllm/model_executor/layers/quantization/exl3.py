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


# torch.library custom op: opaque to dynamo, so the M dispatch runs at
# execution time. Plain Python branches (even inside allow_in_graph
# wrappers) are baked at trace time with the warmup M, which sends M=1
# decode down the decode-trellis + rocBLAS path (~2.5x slower). Fused
# kernel re-decodes each codebook tile M/8 times and loses at large M;
# decode-trellis rewrites the full fp16 weight per call and loses at
# small M. Crossover measured on gfx1030: M > 64 prefers decode-trellis.
@torch.library.custom_op("vllm::exl3_mid_rdna2", mutates_args=("w_raw",))
def _exl3_mid(xh_i: torch.Tensor, trellis_i: torch.Tensor,
              w_raw: torch.Tensor, bits: int, cb: int) -> torch.Tensor:
    if xh_i.size(0) > 64:
        torch.ops._rocm_C.exl3_decode_trellis_rdna2(trellis_i, w_raw,
                                                      bits, cb)
        return torch.nn.functional.linear(xh_i, w_raw.t())
    mid = torch.zeros(xh_i.size(0), w_raw.shape[1], dtype=xh_i.dtype,
                      device=xh_i.device)
    torch.ops._rocm_C.exl3_gemm_rdna2(xh_i, mid, trellis_i, bits, cb)
    return mid


@_exl3_mid.register_fake
def _exl3_mid_fake(xh_i: torch.Tensor, trellis_i: torch.Tensor,
                   w_raw: torch.Tensor, bits: int, cb: int) -> torch.Tensor:
    return torch.empty(xh_i.size(0), w_raw.shape[1], dtype=xh_i.dtype,
                       device=xh_i.device)


_prefill_scratch: dict = {}


def _get_prefill_scratch(K, width, device):
    key = (K, width, device)
    buf = _prefill_scratch.get(key)
    if buf is None:
        buf = torch.zeros(K, width, dtype=torch.half, device=device)
        _prefill_scratch[key] = buf
    return buf
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
        # VLLM_EXL3_MEMORY_MODE: 'full' = int16 trellis (current/default),
        # 'packed' = uint8 packed (kernel decodes in registers, ~2x trellis
        # VRAM saving). Stub: the dispatcher branch is wired but the packed
        # kernel is not yet implemented; both modes call exl3_gemm_rdna2 today.
        self.memory_mode = os.environ.get("VLLM_EXL3_MEMORY_MODE", "full")
        if self.memory_mode not in ("full", "packed"):
            raise ValueError(
                f"VLLM_EXL3_MEMORY_MODE must be 'full' or 'packed', "
                f"got {self.memory_mode!r}")

    def _exl3_gemm_dispatch(self, a, c, trellis, bits, cb):
        """Select GEMM kernel by memory_mode. Stub: both modes call the
        same int16-trellis kernel; the packed branch lands with the
        exl3_gemm_rdna2_packed kernel in a follow-up commit."""
        if self.memory_mode == "packed":
            return _exl3_gemm(a, c, trellis, bits, cb)
        return _exl3_gemm(a, c, trellis, bits, cb)

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
        # VLLM_EXL3_M_MAX=0 disables CG-PATH (forces the eager-only FB-PATH,
        # used to bisect captured-graph regressions).
        M_MAX = int(os.environ.get("VLLM_EXL3_M_MAX", "64"))
        layer._exl3_M_MAX = M_MAX
        layer._exl3_part_widths = list(output_partition_sizes)
        # torch.zeros (not torch.empty) commits all GPU pages at allocation
        # time. On RDNA2, torch.empty returns virtual address space with
        # uncommitted physical pages; the HIP kernel reads from these and
        # faults at a bogus GPU address. zeros commits every page so the
        # kernel can read without faulting.
        layer._exl3_buf_out = torch.zeros(
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
            torch.zeros(M_MAX, input_size_per_partition,
                        dtype=params_dtype, device="cuda")
            for _ in output_partition_sizes
        ]
        layer._exl3_bufs_mid = [
            torch.zeros(M_MAX, w, dtype=params_dtype, device="cuda")
            for w in output_partition_sizes
        ]
        layer._exl3_bufs_out_part = [
            torch.zeros(M_MAX, w, dtype=params_dtype, device="cuda")
            for w in output_partition_sizes
        ]
        # Per-partition contiguous trellis slices so apply() doesn't need a
        # .contiguous() allocation each call. The copy in
        # process_weights_after_loading overwrites these with real weights,
        # but torch.zeros ensures the pages are committed even before the
        # copy (avoids uncommitted-page read faults if a kernel ever reads
        # the buffer before the copy).
        layer._exl3_bufs_trellis = [
            torch.zeros(
                input_size_per_partition // 16, w // 16, 48,
                dtype=torch.int16, device="cuda",
            )
            for w in output_partition_sizes
        ]
        layer._exl3_bufs_svh = [
            torch.zeros(w, dtype=params_dtype, device="cuda")
            for w in output_partition_sizes
        ]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Bits 2/3/4 single-shard layers run the RDNA2 trellis kernel;
        fused layers (per-shard suh differs), mul1/mcg-marked layers and
        bits 6 (lm_head) are dequantized once here and run as dense fp16
        GEMMs — same weight-prep trick as the RDNA2 W4A16 dense kernel."""
        if hasattr(layer, "_exl3_bufs_trellis"):
            # Per-partition trellis slices. The pre-allocated buffers from
            # create_weights assume bits=3 (inner dim 48 = 256*3/16). For
            # bits=6 layers (mul1-marked) the inner dim is 96 = 256*6/16 —
            # reallocate from the actual trellis shape so the copy below
            # doesn't trip a size mismatch (RDNA2 uncommitted-page guard
            # stays intact because torch.zeros still commits pages).
            actual_inner = layer.trellis.shape[2]
            k_tiles = layer.trellis.shape[0]
            for i, width in enumerate(layer._exl3_part_widths):
                off = sum(layer._exl3_part_widths[:i])
                if (layer._exl3_bufs_trellis[i].shape[0] != k_tiles
                        or layer._exl3_bufs_trellis[i].shape[2] != actual_inner):
                    layer._exl3_bufs_trellis[i] = torch.zeros(
                        k_tiles, width // 16, actual_inner,
                        dtype=torch.int16, device="cuda",
                    )
                layer._exl3_bufs_trellis[i].copy_(
                    layer.trellis[:, off // 16:(off + width) // 16, :]
                )
                layer._exl3_bufs_svh[i].copy_(layer.svh[off:off + width])
        part_sizes = list(getattr(layer, "_exl3_part_sizes", []))
        suh_parts = list(getattr(layer, "_exl3_suh_parts", []))
        # Pre-populate the prefill decode scratch here (load time, eager):
        # apply()'s FB path must only READ the pool — a store to the global
        # dict inside dynamo-traced code decompiles to dict.update, which
        # trips vLLM's cudagraph bytecode_hook ("update" in co_names).
        # Per-call allocation instead would churn ~100 MB per partition per
        # chunk and measurably degrades decode (cudagraph pool lands in a
        # churned memory layout at capture time).
        if int(self.bits) in (2, 3, 4) and hasattr(layer, "trellis"):
            K = layer.trellis.shape[0] * 16
            widths = (list(layer._exl3_part_widths)
                      if hasattr(layer, "_exl3_part_widths")
                      else [layer.trellis.shape[1] * 16])
            for width in widths:
                _get_prefill_scratch(K, width, layer.trellis.device)
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
        if os.environ.get("VLLM_EXL3_MARKER_DBG") == "1":
            layer17_markers = [m for m in qc._exl3_mul1_marks if "layers.17" in m] if qc else []
            suh_norm = layer.suh.float().norm().item() if hasattr(layer, 'suh') else 0.0
            svh_norm = layer.svh.float().norm().item() if hasattr(layer, 'svh') else 0.0
            print(f"[exl3_marker_dbg] prefix={prefix} container={container!r} "
                  f"mul1={mul1} mcg={mcg} marked={marked} cb={self.cb} "
                  f"total_mul1_marks={len(qc._exl3_mul1_marks) if qc else 0} "
                  f"layer17_mul1_marks={layer17_markers} "
                  f"suh_norm={suh_norm:.4f} svh_norm={svh_norm:.4f}",
                  flush=True)
        if not fused and not marked and int(self.bits) in (2, 3, 4) and not (
                os.environ.get("VLLM_EXL3_DEQUANT_ALL") == "1"):
            return
        if self.bits not in (2, 3, 4, 6):
            raise NotImplementedError(
                f"EXL3: unsupported bits={self.bits} "
                "(kernel: 2/3/4/6)")
        if self.bits == 6:
            print(f"[exl3_dbg] process_weights_after_loading bits=6 "
                  f"prefix={getattr(layer, 'prefix', '?')} "
                  f"hasattr_w_fp16={hasattr(layer, '_w_fp16')} "
                  f"w_fp16_numel={layer._w_fp16.numel() if hasattr(layer, '_w_fp16') and layer._w_fp16 is not None else 'N/A'} "
                  f"trellis_shape={tuple(layer.trellis.shape) if hasattr(layer, 'trellis') else 'N/A'}",
                  flush=True)
            # bits=6 lm_head: kernel runtime GEMM path produces wrong output
            # (exl3_window_pos<6> K-3 fallback). Dequant to fp16 and fold
            # suh/svh on GPU via PyTorch; forward becomes a plain rocBLAS GEMM.
            K_tile, N_tile, _ = layer.trellis.shape
            K, N = K_tile * 16, N_tile * 16
            device = layer.trellis.device
            # torch.zeros (not torch.empty) commits all GPU pages at
            # allocation time. On RDNA2, torch.empty returns virtual
            # address space with uncommitted physical pages; if the
            # kernel doesn't write to every byte, the uninitialized
            # regions read as garbage/NaN, which propagates to the
            # lm_head logits. This is the same RDNA2 page-commit guard
            # the CG-PATH uses for its pre-allocated buffers (see
            # create_weights comments).
            out = torch.zeros(K, N, dtype=torch.half, device=device)
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
            print(f"[exl3_dbg] lm_head dequant stats prefix={getattr(layer, 'prefix', '?')} "
                  f"out_norm_pre_copy={out.float().norm().item():.4f} "
                  f"out_max_pre_copy={out.float().abs().max().item():.6f} "
                  f"w_fp16_norm_post_copy={layer._w_fp16.float().norm().item():.4f} "
                  f"w_fp16_max_post_copy={layer._w_fp16.float().abs().max().item():.6f}",
                  flush=True)
        if (int(self.bits) in (2, 3, 4)
                and (mul1 or os.environ.get("VLLM_EXL3_DEQUANT_ALL") == "1")
                and not getattr(layer, "_w_fp16_loaded", False)):
            # Mul1 (cb=2) layers overflow fp16 in the runtime kernel's
            # Hadamard intermediates on gfx1030 (wide decode range + large
            # activation outliers). Fold to dense fp16 once at load; apply()
            # then runs one fp32-accumulated rocBLAS GEMM. The fold pushes
            # identity chunks through the kernel GEMM (raw decode) and
            # applies the suh/svh Hadamards in PyTorch — exact, since the
            # pipeline is linear in x.
            K_tile, N_tile, _ = layer.trellis.shape
            K, N = K_tile * 16, N_tile * 16
            device = layer.trellis.device
            suh_parts_fold = suh_parts if suh_parts else [(layer.suh, 0, N)]
            if (K % 128 or N % 128
                    or any(w % 128 for _, _, w in suh_parts_fold)):
                logger.warning(
                    "EXL3: %s K=%d N=%d not 128-divisible; marked layer "
                    "stays on the runtime kernel path", prefix, K, N)
            else:
                raw = torch.zeros(K, N, dtype=torch.half, device=device)
                for r0 in range(0, K, 1024):
                    r1 = min(r0 + 1024, K)
                    eye = torch.eye(r1 - r0, K, dtype=torch.half,
                                    device=device)
                    _exl3_gemm(eye, raw[r0:r1], layer.trellis,
                               int(self.bits), self.cb)
                    del eye
                r_scale = 1.0 / 12.649110640673516
                out = torch.empty(K, N, dtype=torch.half, device=device)
                for suh_i, off, width in suh_parts_fold:
                    w = raw[:, off:off + width] * suh_i.view(K, 1)
                    h = w.view(K // 128, 128, width)
                    for _ in range(7):
                        h = h.view(K // 128, 2, 64, width)
                        a, b = h[:, 0], h[:, 1]
                        h = torch.stack([a + b, a - b], dim=1).view(
                            K // 128, 128, width)
                    w = h.view(K, width) * r_scale
                    h = w.view(K, width // 128, 128).transpose(0, 1)
                    for _ in range(7):
                        h = h.view(width // 128, K, 2, 64)
                        a, b = h[:, :, 0], h[:, :, 1]
                        h = torch.stack([a + b, a - b], dim=2).view(
                            width // 128, K, 128)
                    out[:, off:off + width] = (
                        h.transpose(0, 1).reshape(K, width)
                        * layer.svh[off:off + width].view(1, width)
                        * r_scale)
                w_nk = out.t().contiguous()
                del raw, out
                if (hasattr(layer, "_w_fp16") and layer._w_fp16 is not None
                        and tuple(layer._w_fp16.shape) == tuple(w_nk.shape)):
                    layer._w_fp16.data.copy_(w_nk)
                    del w_nk
                else:
                    layer._w_fp16 = w_nk
                layer._w_fp16_loaded = True
                if os.environ.get("VLLM_EXL3_DEBUG") == "1":
                    w_ref = layer._w_fp16
                    print(f"[exl3] {prefix:80s} mul1 fold cb={self.cb} "
                          f"w_fp16={tuple(w_ref.shape)} "
                          f"norm={w_ref.float().norm().item():.4f} "
                          f"max={w_ref.float().abs().max().item():.6f}",
                          flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # The folded lm_head weight lives in layer._w_fp16 (set above in
        # the bits==6 branch) and the logits processor reads
        # lm_head.weight directly via torch.mm (see
        # vllm/model_executor/layers/logits_processor.py:161). Swap
        # layer.weight to alias _w_fp16 so the sampler reads the real
        # dequantized weight instead of the 1x1 dummy created in
        # create_weights (without this swap, lm_head logits are all
        # zeros because torch.mm(flat, dummy.t()) produces a 1xV result).
        if (hasattr(layer, "_w_fp16")
                and layer._w_fp16 is not None
                and getattr(layer, "_w_fp16_loaded", False)):
            # nn.Module.__getattr__ checks _parameters before instance
            # attributes, so just rebinding layer.weight to _w_fp16 isn't
            # enough. The dummy 1x1 Parameter registered by create_weights
            # has a different shape than the dequantized _w_fp16, so
            # in-place copy_ also fails (shape mismatch). Pop the dummy
            # Parameter from _parameters so the instance attribute lookup
            # for lm_head.weight falls through to the dequantized _w_fp16.
            print(f"[exl3_dbg] lm_head swap BEFORE prefix={getattr(layer, 'prefix', '?')} "
                  f"_params_keys={list(layer._parameters.keys())} "
                  f"w_fp16_shape={tuple(layer._w_fp16.shape)} "
                  f"_w_fp16_in_params={'weight' in layer._parameters}",
                  flush=True)
            layer._parameters.pop("weight", None)
            layer.weight = layer._w_fp16
            print(f"[exl3_dbg] lm_head swap AFTER prefix={getattr(layer, 'prefix', '?')} "
                  f"_params_keys={list(layer._parameters.keys())} "
                  f"layer.weight_shape={tuple(layer.weight.shape) if layer.weight is not None else None}",
                  flush=True)
        else:
            # Single-shard body layers use the trellis kernel in apply(),
            # so layer.weight is unused at forward time — free the dummy
            # to reclaim GPU memory.
            if hasattr(layer, "weight") and layer.weight is not None:
                w = layer.weight
                if w.numel() > 1:
                    layer._parameters.pop("weight", None)
                    del w
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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
        # Free the _w_fp16 buffer allocated in create_weights — it will never
        # be read (apply() checks self._w_fp16, not layer._w_fp16, and we
        # never set self._w_fp16 on this path). Without this free, ~17 GB of
        # allocated-but-unwritten VRAM sits idle for merged layers
        # (gate_up, qkv, in_proj_qkvz) on 9B-class models.
        if hasattr(layer, "_w_fp16") and layer._w_fp16 is not None:
            w = layer._w_fp16
            sz_bytes = w.numel() * w.element_size()
            if os.environ.get("VLLM_EXL3_DEBUG") == "1":
                print(f"[exl3_free] {getattr(layer, 'prefix', '?'):80s} "
                      f"freeing _w_fp16 shape={tuple(w.shape)} "
                      f"size={sz_bytes/1e6:.1f} MB",
                      flush=True)
            layer._parameters.pop("_w_fp16", None)
            del w
        # Return the freed pages to the GPU driver, not just the allocator.
        # Calling empty_cache once per layer is wasteful but matches the
        # existing pattern for layer.weight freeing (line ~611) and
        # guarantees VRAM shrinks instead of staying in the allocator pool.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

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

        if (os.environ.get("VLLM_EXL3_INPUT_NAN_DBG") == "1"
                and ".layers.17." in getattr(layer, "prefix", "")):
            print(f"[exl3_nan_dbg] apply INPUT prefix={getattr(layer, 'prefix', '?')} "
                  f"x_norm={x.float().norm().item():.4f} "
                  f"x_max={x.float().abs().max().item():.6f} "
                  f"has_nan={torch.isnan(x.float()).any().item()}",
                  flush=True)

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

        _original_M = x.size(0)

        # Commit GPU pages: vLLM allocates the activation buffer with
        # torch.empty (uncommitted virtual pages on RDNA2). The HIP kernel
        # reads from these pages and faults. Use .clone() to force a new
        # allocation + memcpy, which writes to every page and commits them.
        x = x.contiguous().clone()

        x = x.to(torch.half) if x.dtype == torch.bfloat16 else x
        _exl3_dbg = os.environ.get("VLLM_EXL3_DEBUG") == "1"
        folded = getattr(self, "_w_fp16", None)
        if folded is not None:
            # Was the reference's own folded dequant
            # (reconstruct_had_slice: suh/svh + both Hadamards inside the
            # weight), so the forward is a plain GEMM.
            return torch.nn.functional.linear(x, folded)
        bits = self.bits
        cb = self.cb

        # Cudagraph-friendly path: slice the pre-allocated buffers set up
        # in create_weights. Data ptrs are stable so the captured graph
        # replay hits the same memory each call. Buffers are allocated with
        # torch.zeros (see create_weights) so all pages are committed and
        # the kernel can read without faulting on uncommitted virtual pages.
        # The M <= M_MAX branch compares a SymInt to a Python int — dynamo
        # tracks the comparison symbolically and routes to CG-PATH only when
        # the dynamic M actually fits.
        if (hasattr(layer, "_exl3_bufs_xh")
                and x.size(0) <= layer._exl3_M_MAX):
            # Guard the debug log with is_compiling() — `%s % x.size(0)`
            # materializes the dynamic SymInt at trace time and triggers
            # dynamo ConstraintViolationError when input_ids.size()[0] is
            # also marked dynamic. The log is for human debugging only.
            if _exl3_dbg and _call_idx[0] < 8 \
                    and not torch._dynamo.is_compiling():
                _call_idx[0] += 1
                _exl3_log("[exl3] CG-PATH %s M=%s buf_xh_ptr=%s buf_mid_ptr=%s"
                          % (str(getattr(layer, "prefix", "?")),
                             x.size(0),
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
                xh_i = layer._exl3_bufs_xh[i][:x.size(0)]
                mid_i = layer._exl3_bufs_mid[i][:x.size(0)]
                out_i = layer._exl3_bufs_out_part[i][:x.size(0)]
                trellis_i = layer._exl3_bufs_trellis[i]
                if _exl3_dbg and _call_idx[0] < 12 \
                        and not torch._dynamo.is_compiling():
                    _exl3_log("[exl3] LOOP %s i=%d off=%d width=%d out_i=%s buf_out_slice=%s part_widths=%s"
                              % (str(getattr(layer, "prefix", "?")),
                                 i, off, width,
                                 tuple(out_i.shape),
                                 tuple(buf_out[:x.size(0), off:off + width].shape)
                                 if off + width <= buf_out.shape[1] else "OOB",
                                 layer._exl3_part_widths))
                svh_i = layer._exl3_bufs_svh[i]
                mid_i.zero_()
                _exl3_hadamard(x, xh_i, suh_i, None, 1.0)
                self._exl3_gemm_dispatch(xh_i, mid_i, trellis_i, bits, cb)
                _exl3_hadamard(mid_i, out_i, None, svh_i, 1.0)
                buf_out[:x.size(0), off:off + width] = out_i
            # Pad output to x.size(0) so downstream shape assertions
            # (e.g., GDN's assert z.shape == x_shape_og) pass. The buf_out
            # buffer is M_MAX rows; if x.size(0) > M_MAX, allocate a
            # zeros-padded copy. The first x.size(0) rows are kernel output;
            # the rest are zeros (padding that downstream doesn't use).
            # In dynamo this branch is unreachable (CG-PATH only entered
            # when x.size(0) <= M_MAX) but the code is kept for parity.
            if x.size(0) > layer._exl3_M_MAX:
                padded = torch.zeros(x.size(0), buf_out.shape[1],
                                     dtype=buf_out.dtype,
                                     device=buf_out.device)
                padded[:x.size(0)] = buf_out[:x.size(0)]
                return padded
            if (os.environ.get("VLLM_EXL3_INPUT_NAN_DBG") == "1"
                    and ".layers.17." in getattr(layer, "prefix", "")):
                print(f"[exl3_nan_dbg] CG-PATH OUTPUT prefix={getattr(layer, 'prefix', '?')} "
                      f"out_norm={buf_out[:x.size(0)].float().norm().item():.4f} "
                      f"out_max={buf_out[:x.size(0)].float().abs().max().item():.6f} "
                      f"has_nan={torch.isnan(buf_out[:x.size(0)].float()).any().item()}",
                      flush=True)
            return buf_out[:x.size(0)]

        # Fallback (eager / x.size(0) > M_MAX): dynamic allocation. Not
        # cudagraph-safe — re-introduces per-call allocations that produce
        # stale-pointer NaN under captured graphs. Use torch.zeros for all
        # dynamic buffers to commit GPU pages (RDNA2 doesn't auto-commit
        # from torch.empty).
        # Guard the debug log with is_compiling() — `%s % x.size(0)`
        # materializes the dynamic SymInt at trace time and triggers
        # dynamo ConstraintViolationError when input_ids.size()[0] is
        # also marked dynamic. The log is for human debugging only.
        if _exl3_dbg and _call_idx[0] < 8 \
                and not torch._dynamo.is_compiling():
            _call_idx[0] += 1
            _exl3_log("[exl3] FB-PATH %s M=%s (M_MAX=%s)"
                      % (str(getattr(layer, "prefix", "?")),
                         x.size(0),
                         getattr(layer, "_exl3_M_MAX", "?")))
        N = trellis.shape[1] * 16
        K = trellis.shape[0] * 16
        # Allocate out with x.size(0) rows for GDN's z.shape == x_shape_og.
        # torch.zeros commits all pages; the kernel writes only the first
        # M. x.size(0) is a dynamo-tracked SymInt, not a Python int — passing
        # _original_M here would specialize the dim to a compile-time constant.
        out = torch.zeros(x.size(0), N, dtype=torch.half, device=x.device)
        # Derive (suh, off, width) from _exl3_part_widths like CG-PATH does:
        # _exl3_suh_parts' own (off, width) entries are unreliable for fused
        # layers (loader quirk), which silently mis-slices the trellis/svh.
        suh_parts_raw = getattr(layer, "_exl3_suh_parts", [])
        suh_tensors = ([sp[0] for sp in suh_parts_raw]
                       if len(suh_parts_raw) == len(layer._exl3_part_widths)
                       else None)
        suh_parts = [
            ((suh_tensors[i] if suh_tensors else suh),
             sum(layer._exl3_part_widths[:i]), w)
            for i, w in enumerate(layer._exl3_part_widths)
        ]
        # The fused GEMM kernel caps M_PER at 8: at prefill M it re-decodes
        # every codebook tile M/8 times (256x redundant at M=2048, ~2.4
        # TFLOPS effective). The decode path instead decodes each tile once
        # into a shared fp16 scratch (trellis stays the weight store — no
        # fp16 residency) and runs rocBLAS (~18 TFLOPS measured on gfx1030).
        # VLLM_EXL3_PREFILL_DECODE=0 restores the fused-kernel path.
        prefill_decode = os.environ.get("VLLM_EXL3_PREFILL_DECODE", "1") == "1"
        for i, (suh_i, off, width) in enumerate(suh_parts):
            xh_i = torch.zeros_like(x)
            _exl3_hadamard(x, xh_i, suh_i, None, 1.0)
            trellis_i = trellis[:, off // 16:(off + width) // 16, :].contiguous()
            if prefill_decode:
                # Pool is populated at load; a dict store here would
                # decompile to dict.update and trip the cudagraph hook.
                w_raw = _prefill_scratch[(K, width, x.device)]
                mid_i = _exl3_mid(xh_i, trellis_i, w_raw, bits, cb)
            else:
                mid_i = torch.zeros(x.size(0), width, dtype=torch.half,
                                    device=x.device)
                self._exl3_gemm_dispatch(xh_i, mid_i, trellis_i, bits, cb)
            # The pre-allocated _exl3_bufs_svh is only M_MAX rows. When
            # x.size(0) > M_MAX (FB-PATH), the kernel processes x.size(0)
            # rows and would read past the pre-allocated buffer into
            # unmapped pages. Use the full-size svh slice in that case.
            svh_i = (layer._exl3_bufs_svh[i]
                     if (hasattr(layer, "_exl3_bufs_svh")
                         and x.size(0) <= layer._exl3_M_MAX)
                     else svh[off:off + width].to(x.device))
            out_i = torch.zeros(x.size(0), width, dtype=torch.half,
                                device=x.device)
            _exl3_hadamard(mid_i, out_i, None, svh_i, 1.0)
            out[:x.size(0), off:off + width] = out_i
        if (os.environ.get("VLLM_EXL3_INPUT_NAN_DBG") == "1"
                and ".layers.17." in getattr(layer, "prefix", "")):
            print(f"[exl3_nan_dbg] apply OUTPUT prefix={getattr(layer, 'prefix', '?')} "
                  f"out_norm={out[:_original_M].float().norm().item():.4f} "
                  f"out_has_nan={torch.isnan(out[:_original_M].float()).any().item()}",
                  flush=True)
        return out