# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen4ExpTextConfig

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_piecewise_capture
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_etp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    is_offload_process,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import is_fp8
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.platforms import current_platform
from vllm.platforms.rdna import on_rdna_family
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)

from ..common.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)
from ..common.ops.ple import ple_conv, ple_gate, ple_ngram_ids
from .ops.hc import grouped_gemma_rmsnorm

logger = init_logger(__name__)

# Output columns per conv1d launch in the flat prefill short-conv; bounds the
# per-launch temporaries independently of --max-num-batched-tokens.
_PREFILL_CONV_BLOCK = 2048


def short_conv_flat_prefill(
    x_p: torch.Tensor,
    q_starts: torch.Tensor,
    lengths: torch.Tensor,
    initial_state: torch.Tensor | None,
    weights: torch.Tensor,
    dilation: int,
    state_len: int,
    num_prefills: int,
    num_prefill_tokens: int,
    block: int = _PREFILL_CONV_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Causal depthwise dilated conv + SiLU over packed prefill requests.

    Requests sit back to back in one ``[hidden, tokens + P * state_len]``
    sequence, each preceded by its ``state_len`` columns of initial state, so
    an unpadded conv over it sees exactly each token's own causal window
    (``state_len == (kernel - 1) * dilation``). The conv runs in ``block``
    output columns at a time. Memory stays O(tokens * hidden) instead of the
    padded ``P * max_len * hidden`` copies.

    ``x_p``: [tokens, hidden]; ``initial_state``: [P, hidden, state_len] or
    None when ``state_len == 0``; ``weights``: [hidden, kernel]. Returns the
    [tokens, hidden] output and, when ``state_len > 0``, each request's last
    ``state_len`` columns of history as [P, hidden, state_len].
    """
    hidden = x_p.shape[1]
    device = x_p.device
    positions = torch.arange(num_prefill_tokens, device=device, dtype=torch.int64)
    req = torch.searchsorted(q_starts[1:], positions, right=True)
    total = num_prefill_tokens + num_prefills * state_len

    flat = x_p.new_empty((hidden, total))
    flat.index_copy_(1, positions + (req + 1) * state_len, x_p.t())
    seg_starts = None
    if state_len > 0:
        seg_starts = (
            q_starts[:-1]
            + torch.arange(num_prefills, device=device, dtype=torch.int64) * state_len
        )
        assert initial_state is not None
        state_cols = seg_starts.view(-1, 1) + torch.arange(
            state_len, device=device, dtype=torch.int64
        ).view(1, -1)
        flat.index_copy_(
            1,
            state_cols.reshape(-1),
            initial_state.to(x_p.dtype).permute(1, 0, 2).reshape(hidden, -1),
        )

    next_state = None
    if state_len > 0:
        assert seg_starts is not None
        cols = (
            seg_starts.view(-1, 1)
            + lengths.view(-1, 1)
            + torch.arange(state_len, device=device, dtype=torch.int64).view(1, -1)
        )
        next_state = (
            flat[:, cols.reshape(-1)]
            .view(hidden, num_prefills, state_len)
            .permute(1, 0, 2)
        )

    out_cols = total - state_len
    conv_out = x_p.new_empty((hidden, out_cols))
    w = weights.unsqueeze(1).contiguous()
    for o0 in range(0, out_cols, block):
        o1 = min(o0 + block, out_cols)
        y = F.conv1d(
            flat[:, o0 : o1 + state_len].unsqueeze(0),
            w,
            groups=hidden,
            dilation=dilation,
        )[0]
        conv_out[:, o0:o1] = F.silu(y)
    del flat
    output = conv_out.t().index_select(0, positions + req * state_len)
    return output, next_state


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        if (
            current_platform.is_rocm()
            and hidden_states.is_cuda
            and input_dtype in (torch.float16, torch.bfloat16)
            and hidden_states.stride(-1) == 1
        ):
            shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, shape[-1])
            num_groups = 1 if self.group_size is None else shape[-1] // self.group_size
            return grouped_gemma_rmsnorm(
                hidden_states, self.weight, self.eps, num_groups
            ).reshape(shape)

        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


# VLLM_PLE_CPU_OFFLOAD=1 (int4 table on disk): a CPU sidecar process owns the
# table and computes the embeddings (vllm/v1/ple_offload); the GPU worker only
# waits for them. Otherwise this is a plain module (upstream path).
_NGramEmbeddingBase: type[nn.Module] = (
    PleOffloadLayer if envs.VLLM_PLE_CPU_OFFLOAD else nn.Module
)


class Qwen4ExpNGramEmbedding(_NGramEmbeddingBase):  # type: ignore[valid-type,misc]
    positions_buffer: torch.Tensor
    padded_buffer: torch.Tensor
    ngram_heads_vocab_sizes: torch.Tensor
    ngram_heads_offsets: torch.Tensor

    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        prefix: str,
        layer_name: str,
        *,
        data_parallel_rank: int = 0,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.use_sequence_parallel = use_sequence_parallel
        self.embedding_dim = embedding_dim
        self.layer_name = layer_name
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        self._ngram_sizes = list(sizes)
        self._ngram_offsets = list(offsets)
        self._max_total_tokens = int(max_total_tokens)
        self._max_num_reqs = int(
            get_current_vllm_config().scheduler_config.max_num_seqs
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = round_up(total_vocab_size, divisor)
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        embedding_prefix = f"{prefix}.ngram_embedding"
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        if isinstance(embedding_quant_method, Qwen4ExpPLENvFp4EmbeddingMethod):
            raise NotImplementedError(
                "NVFP4 Qwen4Exp PLE tables are not supported on ROCm yet; use a "
                "checkpoint with a BF16 or FP8 PLE table."
            )
        engram_config = get_current_vllm_config().engram_config
        embedding_cls = (
            Qwen4ExpPLEPinnedHostEmbedding
            if engram_config is not None
            and engram_config.cpu_offload
            and not envs.VLLM_PLE_CPU_OFFLOAD
            else Qwen4ExpPLEDeviceEmbedding
        )
        # A TP reduce-scatter covers all vocabulary owners only when ETP=TP.
        self.use_reduce_scatter = (
            self.use_sequence_parallel
            and get_etp_group().world_size == get_tp_group().world_size
        )
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=not self.use_reduce_scatter,
        )
        logger.info(
            "Initialized AMD PLE embedding %s: "
            "quantization_config=%s, quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(quant_config).__name__,
            type(embedding_quant_method).__name__,
            self.ngram_embedding.weight.dtype,
            self.ngram_embedding.weight.device,
            self.ngram_embedding.weight.is_pinned(),
        )
        if envs.VLLM_PLE_CPU_OFFLOAD:
            # Workspace of the sidecar's torch n-gram-id path: the Triton op the
            # GPU path uses cannot run on CPU.
            self.register_buffer(
                "positions_buffer",
                torch.arange(max_total_tokens, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "padded_buffer",
                torch.full(
                    (self._max_num_reqs, max_total_tokens),
                    self.eos_token_id,
                    dtype=torch.int64,
                ),
                persistent=False,
            )

    def _materialize_workspace(self, device: torch.device) -> None:
        """Rebuild workspace buffers left on meta by the sidecar's meta build."""
        if (
            not self.positions_buffer.is_meta
            and not self.padded_buffer.is_meta
            and not self.ngram_heads_vocab_sizes.is_meta
            and int(self.ngram_heads_vocab_sizes.sum()) != 0
        ):
            return
        self.positions_buffer = torch.arange(
            self._max_total_tokens, dtype=torch.int64, device=device
        )
        self.padded_buffer = torch.full(
            (self._max_num_reqs, self._max_total_tokens),
            self.eos_token_id,
            dtype=torch.int64,
            device=device,
        )
        if (
            self.ngram_heads_vocab_sizes.is_meta
            or int(self.ngram_heads_vocab_sizes.sum()) == 0
        ):
            self.ngram_heads_vocab_sizes = torch.tensor(
                self._ngram_sizes, dtype=torch.long, device=device
            )
            self.ngram_heads_offsets = torch.tensor(
                self._ngram_offsets, dtype=torch.long, device=device
            )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    if not envs.VLLM_PLE_CPU_OFFLOAD:

        def forward(
            self,
            input_ids: torch.Tensor,
            query_start_loc: torch.Tensor,
            ngram_context: torch.Tensor,
        ) -> torch.Tensor:
            return self.forward_impl(None, input_ids, query_start_loc, ngram_context)

    def _sp_shard_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Shard embedding token rows, optionally reducing vocabulary contributions."""
        # Each embedding element has one vocabulary owner, including FP8 bytes.
        output_dtype = embeddings.dtype
        if is_fp8(embeddings):
            embeddings = embeddings.view(torch.int8)
        if self.use_reduce_scatter:
            embeddings = sp_reduce_scatter(embeddings)
        else:
            embeddings = sp_shard(embeddings)
        return embeddings.view(output_dtype)

    def _ngram_ids_torch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """N-gram ids in plain torch, for the CPU sidecar.

        The GPU path hashes in a Triton op (``qwen4_exp_amd_ple_ngram_ids``),
        which cannot run on CPU.
        """
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )

        # The sidecar is never graph-captured: narrow the pack width to the
        # batch, and drop the graph padding (the runner sends the padded
        # token count with an unpadded query_start_loc).
        self._materialize_workspace(input_ids.device)
        max_seq_len = max(
            1, int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
        )
        num_valid_tokens = min(int(query_start_loc[-1].item()), num_tokens)

        positions = self.positions_buffer[:num_tokens]
        packed = self.padded_buffer[:num_reqs, :max_seq_len]
        packed.fill_(self.eos_token_id)
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices[:num_valid_tokens], columns[:num_valid_tokens]] = (
            input_ids[:num_valid_tokens]
        )
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor | None,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return full embeddings or a token shard after combining vocabulary owners."""
        del hidden_states, output_buffer
        input_ids = input_ids.reshape(-1)
        offload = is_offload_process()
        if offload:
            ngram_ids = self._ngram_ids_torch(input_ids, query_start_loc, ngram_context)
        else:
            ngram_ids = input_ids.new_empty(
                (input_ids.shape[0], self.ngram_heads), dtype=torch.int64
            )
            torch.ops.vllm.qwen4_exp_amd_ple_ngram_ids(
                input_ids,
                query_start_loc,
                ngram_context,
                ngram_ids,
                self.layer_name,
            )
        embedding = self.ngram_embedding
        if offload:
            # In the sidecar the table is the mmapped INT4 store attached by
            # vllm/v1/ple_offload (or the plain CPU table without one).
            output = ngram_ids.new_empty(
                (ngram_ids.shape[0], self.embedding_dim),
                dtype=embedding.params_dtype,
            )
            quant = getattr(embedding, "_ple_quant", None)
            if quant is not None:
                quant.gather_into(
                    ngram_ids.reshape(-1), output.reshape(-1, self.head_dim)
                )
            else:
                output.copy_(F.embedding(ngram_ids, embedding.weight).flatten(-2))
            return output
        if embedding.supports_prefetch:
            output = ngram_ids.new_empty(
                (ngram_ids.shape[0], self.embedding_dim),
                dtype=embedding.lookup_dtype,
            )
            torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding_pinned(
                ngram_ids,
                output,
                self.layer_name,
            )
        else:
            output = ngram_ids.new_empty(
                (ngram_ids.shape[0], self.embedding_dim),
                dtype=embedding.params_dtype,
            )
            torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
                ngram_ids,
                output,
                self.layer_name,
            )
        if not self.use_sequence_parallel:
            return output
        return self._sp_shard_embeddings(output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""
        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            # The sidecar owns the table; this GPU-side placeholder has no
            # parameters (PleOffloadLayer skips its __init__).
            for _ in weights:
                pass
            return set()
        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                if buffer.is_meta:
                    # Sidecar meta build: copy_ into meta is a no-op; bind it.
                    setattr(self, name, loaded_weight.to(dtype=buffer.dtype))
                else:
                    buffer.copy_(
                        loaded_weight.to(device=buffer.device, dtype=buffer.dtype)
                    )
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                method = getattr(self.ngram_embedding, "embedding_method", None)
                if getattr(method, "lookup_dequantizes", False):
                    # The table comes from VLLM_PLE_QUANT_DIR after loading.
                    loaded.add("ngram_embedding.weight")
                    continue
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.use_sequence_parallel = use_sequence_parallel
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        # RDNA: compute the n-gram embedding outside the compiled graph. The
        # in-graph n-gram ids go wrong on padded piecewise HIP-graph replays:
        # every in-sequence position (row >= ngram_size - 1) of a prefill
        # whose token count is padded up to a capture size diverges.
        self._rdna_opaque_embedding = on_rdna_family() and not envs.VLLM_PLE_CPU_OFFLOAD
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            f"{prefix}.ple_embedding",
            prefix,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
            quant_config=quant_config,
            params_dtype=model_config.dtype,
            use_sequence_parallel=self.use_sequence_parallel,
        )
        # Projection weights stay replicated even when their token rows are sharded.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor | None,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        has_non_spec = has_prefill or has_decode
        inputs = inputs[: metadata.num_actual_tokens]
        residual = residual[: metadata.num_actual_tokens]
        if outer_residual is not None:
            outer_residual = outer_residual[: metadata.num_actual_tokens]

        spec_token_indices = None
        non_spec_token_indices = None
        if has_spec and has_non_spec:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            spec_token_indices = metadata.spec_token_indx
            non_spec_token_indices = metadata.non_spec_token_indx

        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            query_start_loc = metadata.spec_query_start_loc
            num_accepted_tokens = metadata.num_accepted_tokens
            assert query_start_loc is not None
            assert num_accepted_tokens is not None
            spec_state_indices = metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ]
            # Mixed batches stay in their original row order; the kernels map
            # logical spec/non-spec rows instead of materializing both groups.
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=spec_state_indices,
                outer_residual=outer_residual,
                mode="spec",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
                token_indices=spec_token_indices,
            )

        if not has_non_spec:
            return

        state_indices = metadata.state_indices_tensor
        assert state_indices is not None
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            if non_spec_token_indices is None:
                inputs_d, inputs_p = torch.split(
                    inputs, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                residual_d, residual_p = torch.split(
                    residual, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                outer_residual_d: torch.Tensor | None = None
                outer_residual_p: torch.Tensor | None = None
                if outer_residual is not None:
                    outer_residual_d, outer_residual_p = torch.split(
                        outer_residual,
                        [num_decode_tokens, num_prefill_tokens],
                        dim=0,
                    )
                token_indices_d = None
                token_indices_p = None
            else:
                inputs_d = inputs_p = inputs
                residual_d = residual_p = residual
                outer_residual_d = outer_residual_p = outer_residual
                token_indices_d, token_indices_p = torch.split(
                    non_spec_token_indices,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )

            if has_decode:
                ple_conv(
                    inputs=inputs_d,
                    residual=residual_d,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices=state_indices_d,
                    outer_residual=outer_residual_d,
                    mode="decode",
                    dilation=self.short_conv_dilation,
                    has_initial_states=metadata.has_initial_states_d,
                    token_indices=token_indices_d,
                )

            query_start_loc = metadata.query_start_loc_p
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
            has_initial_states = metadata.has_initial_states_p
            if has_initial_states is None:
                raise ValueError(
                    "has_initial_states_p is required for prefill short-conv"
                )
            ple_conv(
                inputs=inputs_p,
                residual=residual_p,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices_p,
                outer_residual=outer_residual_p,
                mode="prefill",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                has_initial_states=has_initial_states,
                token_indices=token_indices_p,
            )
        else:
            num_decode_rows = (
                non_spec_token_indices.numel()
                if non_spec_token_indices is not None
                else inputs.size(0)
            )
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices[:num_decode_rows],
                outer_residual=outer_residual,
                mode="decode",
                dilation=self.short_conv_dilation,
                has_initial_states=metadata.has_initial_states_d,
                token_indices=non_spec_token_indices,
            )

    def _short_conv(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor | None,
    ) -> None:
        """Update convolution state, optionally fusing both residual additions."""
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Profiling omits all metadata or this Mamba entry. Short convolution
        # is a no-op there, but preserve the outer residual addition.
        if attn_metadata is None:
            if outer_residual is not None:
                residual.add_(outer_residual)
            return

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            if outer_residual is not None:
                residual.add_(outer_residual)
            return
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, channel, window].
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        self._short_conv_dilated_dispatch(
            inputs=inputs,
            residual=residual,
            outer_residual=outer_residual,
            metadata=layer_attn_metadata,
            conv_state=conv_state,
            conv_weights=conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if (
            not self.use_sequence_parallel
            and input_ids.shape[0] != hidden_states.shape[0]
        ):
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        if envs.VLLM_PLE_CPU_OFFLOAD:
            # Sidecar: PleOffloadLayer.forward waits for the CPU result, already
            # in the activation dtype.
            embeddings = self.ple_embedding(
                hidden_states, input_ids, query_start_loc, ngram_context
            )
        elif self._rdna_opaque_embedding:
            ngram = self.ple_embedding.ngram_embedding
            dtype = (
                ngram.lookup_dtype if ngram.supports_prefetch else ngram.params_dtype
            )
            embeddings = hidden_states.new_empty(
                (input_ids.shape[0], self.ple_embedding.embedding_dim), dtype=dtype
            )
            torch.ops.vllm.qwen4_exp_amd_ple_embedding_rdna(
                input_ids, query_start_loc, ngram_context, embeddings, self.prefix
            )
        else:
            embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        if not envs.VLLM_PLE_CPU_OFFLOAD:
            embeddings = self.ple_embedding.ngram_embedding.dequantize(
                embeddings, hidden_states.dtype
            )
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = torch.ops.vllm.qwen4_exp_ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        # The short-conv op is a piecewise-graph splitting op: state routing
        # reads runtime request metadata. It accumulates the convolution and
        # the outer residual into gated_output.
        if self.use_sequence_parallel:
            # Only convolution needs full tokens. Add residuals on local shards.
            conv_input = sp_all_gather(conv_input)[: input_ids.shape[0]]
            # Metadata may omit padding tokens or all tokens during profiling.
            conv_output = torch.zeros_like(conv_input)
            torch.ops.vllm.qwen4_exp_ple_short_conv(
                conv_input, conv_output, None, self.prefix
            )
            gated_output.add_(sp_shard(conv_output))
            gated_output.add_(hidden_states)
        else:
            torch.ops.vllm.qwen4_exp_ple_short_conv(
                conv_input,
                gated_output,
                hidden_states,
                self.prefix,
            )
        return gated_output


@eager_break_during_piecewise_capture
def qwen4_exp_amd_ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Hash the current request layout into n-gram embedding indices.

    The launch sizes its request binary search from the request count, which
    is symbolic under Dynamo, so it runs outside Inductor's FX graph.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    embedding = layer.ple_embedding
    ple_ngram_ids(
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        ngram_context=ngram_context,
        layer_multipliers=embedding.layer_multipliers,
        ngram_heads_vocab_sizes=embedding.ngram_heads_vocab_sizes,
        ngram_heads_offsets=embedding.ngram_heads_offsets,
        eos_token_id=embedding.eos_token_id,
        heads_per_ngram=embedding.heads_per_ngram,
        output=output,
    )


@eager_break_during_piecewise_capture
def qwen4_exp_amd_ple_ngram_embedding(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run the large PLE embedding lookup outside Inductor's FX graph.

    Keeping the embedding weight in ``static_forward_context`` prevents AOT
    compile-time autotuning from materializing a synthetic copy of the weight.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    result = layer.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
    output.copy_(result)


@eager_break_during_piecewise_capture
def qwen4_exp_amd_ple_ngram_embedding_pinned(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run the pinned PLE UVA lookup outside Inductor's FX graph.

    Same rationale as the device-path escape: keeping the large embedding weight
    out of the graph prevents AOT compile-time autotuning from materializing a
    synthetic copy of the weight.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    result = layer.ple_embedding.ngram_embedding.sync_lookup(ngram_ids).flatten(-2)
    output.copy_(result)


@eager_break_during_piecewise_capture
def qwen4_exp_amd_ple_embedding_rdna(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """RDNA: n-gram ids and lookup as one graph-splitting op (runs eagerly)."""
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    output.copy_(layer.ple_embedding(input_ids, query_start_loc, ngram_context))


@eager_break_during_piecewise_capture
def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    outer_residual: torch.Tensor | None,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._short_conv(inputs, residual, outer_residual)


def _ple_gate_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gate output shapes for AMD graph tracing."""
    return torch.empty_like(hidden), torch.empty_like(hidden)


# Keep the AMD gate opaque to Dynamo while reusing the shared Triton kernel.
direct_register_custom_op(
    op_name="qwen4_exp_ple_gate",
    op_func=ple_gate,
    mutates_args=[],
    fake_impl=_ple_gate_fake,
)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_ids",
    op_func=qwen4_exp_amd_ple_ngram_ids,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_embedding",
    op_func=qwen4_exp_amd_ple_ngram_embedding,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_embedding_pinned",
    op_func=qwen4_exp_amd_ple_ngram_embedding_pinned,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_embedding_rdna",
    op_func=qwen4_exp_amd_ple_embedding_rdna,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["residual"],
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
