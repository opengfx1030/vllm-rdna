# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One real GDN decode layer (or a stack of them) on a paged cache.

``QwenGatedDeltaNetAttention`` needs a model config, TP groups and quantized
projections to construct, so this binds the layer's real methods
(``forward_cuda``, ``_forward_core``, ``_forward_core_decode_non_spec``,
``_output_projection`` and the state arena helpers) to a ``SimpleNamespace``
(the pattern of ``test_gdn_forward_core_split.py``) and feeds them metadata
from the real ``GDNAttentionMetadataBuilder``. Only the three projections are
stand-ins (plain fp16 matmuls). The state lives in a unified-page paged cache
laid out the way the engine lays it out: conv state in SD layout and the fp32
SSM state are two strided views of one uint8 page tensor, the page stride
shared (HI2 TP2 page: 2 MiB).

The dimensions are the HI2 (Qwen3.8-27B) per-rank dimensions at TP=2, so a
single-rank run here executes the same kernels at the same shapes as one rank
of the TP2 server: 8 key heads, 24 value heads, head dims 128, conv width 4.
Used by ``test_gdn_decode_inplace_rdna2.py`` and
``tools/rdna/rdna_hint_flex/gdn_glue_profile.py``.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

# HI2 Qwen3.8-27B per-rank dimensions at TP=2 (16 key / 48 value heads in total).
H, HV, K, V = 8, 24, 128, 128
CONV_WIDTH = 4
STATE_LEN = CONV_WIDTH - 1
KEY_DIM, VALUE_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 5120
QKVZ_DIM = CONV_DIM + VALUE_DIM  # 8192
PAGE_BYTES = 2 << 20  # unified page of the HI2 recipe (2 MiB)
BLOCK_SIZE = 16
MAX_BLOCKS = 8  # block-table columns; seq_len <= BLOCK_SIZE * MAX_BLOCKS
EPS = 1e-6
PREFIX = "model.layers.{}.linear_attn"


@functools.cache
def _model_dir() -> Path:
    """A config-only model dir (the builder reads dtype/limits from it)."""
    path = Path(tempfile.mkdtemp(prefix="gdn_glue_model_"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "vocab_size": 128,
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-5,
                "hidden_act": "silu",
                "model_type": "llama",
                "torch_dtype": "float16",
            }
        )
    )
    return path


def make_paged_state(
    num_blocks: int, device: str = "cuda", page_bytes: int = PAGE_BYTES
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(raw, conv, ssm): both caches are views of one page tensor.

    ``conv`` is [blocks, state_len, dim] (the engine's SD layout, transposed by
    the layer before the kernels see it); ``ssm`` is [blocks, HV, V, K] fp32.
    """
    conv_bytes = STATE_LEN * CONV_DIM * 2
    ssm_bytes = HV * V * K * 4
    ssm_off = -(-conv_bytes // 16) * 16
    assert ssm_off + ssm_bytes <= page_bytes
    raw = torch.zeros(num_blocks * page_bytes, dtype=torch.uint8, device=device)
    conv = torch.as_strided(
        raw.view(torch.float16),
        (num_blocks, STATE_LEN, CONV_DIM),
        (page_bytes // 2, CONV_DIM, 1),
        0,
    )
    ssm = torch.as_strided(
        raw.view(torch.float32),
        (num_blocks, HV, V, K),
        (page_bytes // 4, V * K, K, 1),
        ssm_off // 4,
    )
    return raw, conv, ssm


def randomize_state(
    conv: torch.Tensor,
    ssm: torch.Tensor,
    blocks: list[int] | None = None,
    seed: int = 0,
) -> None:
    """Fill ``blocks`` (default: all) with finite random state."""
    gen = torch.Generator(conv.device).manual_seed(seed)
    if blocks is None:
        conv.copy_(
            (torch.randn(conv.shape, device=conv.device, generator=gen) * 0.5).to(
                torch.float16
            )
        )
        ssm.normal_(0.0, 0.05, generator=gen)
        return
    for b in blocks:
        conv[b].copy_(
            (torch.randn(conv.shape[1:], device=conv.device, generator=gen) * 0.5).to(
                torch.float16
            )
        )
        ssm[b].normal_(0.0, 0.05, generator=gen)


def make_weights(hidden: int, device: str, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device).manual_seed(seed)

    def rand(*shape: int, scale: float) -> torch.Tensor:
        return (
            torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)
            * scale
        ).to(torch.float16)

    return {
        "w_qkvz": rand(QKVZ_DIM, hidden, scale=hidden**-0.5),
        "w_ba": rand(2 * HV, hidden, scale=hidden**-0.5),
        "w_out": rand(hidden, VALUE_DIM, scale=VALUE_DIM**-0.5),
        "conv_w": rand(CONV_DIM, CONV_WIDTH, scale=0.5),
        # vLLM keeps A_log fp32 and casts dt_bias to the model dtype on load.
        "A_log": torch.randn(HV, device=device, generator=gen, dtype=torch.float32),
        "dt_bias": rand(HV, scale=1.0),
        "norm_w": (1.0 + rand(V, scale=0.1)).to(torch.float16),
    }


_BOUND = {
    name: fn
    for name, fn in vars(QwenGatedDeltaNetAttention).items()
    if inspect.isfunction(fn)
}


def make_layer(
    idx: int,
    weights: dict[str, torch.Tensor],
    conv: torch.Tensor,
    ssm: torch.Tensor,
    *,
    inplace: bool,
    max_bs: int,
    device: str,
) -> types.SimpleNamespace:
    """The real layer methods bound to a stand-in with HI2's decode config."""
    layer = types.SimpleNamespace()
    for name, fn in _BOUND.items():
        setattr(layer, name, types.MethodType(fn, layer))
    layer.prefix = PREFIX.format(idx)
    layer.tp_size = 1
    layer.num_k_heads, layer.num_v_heads = H, HV
    layer.head_k_dim, layer.head_v_dim = K, V
    layer.key_dim, layer.value_dim = KEY_DIM, VALUE_DIM
    layer.conv_kernel_size = CONV_WIDTH
    layer.num_spec = 0
    layer.activation = "silu"
    layer.A_log = weights["A_log"]
    layer.dt_bias = weights["dt_bias"]
    layer.conv1d = types.SimpleNamespace(
        weight=weights["conv_w"].unsqueeze(1), bias=None
    )
    layer.kv_cache = [conv, ssm]
    layer.enable_packed_recurrent_decode = True
    layer.enable_fused_gdn_spec_decode = False
    layer.gqa_interleaved_layout = False
    layer.disable_tp_for_ba_proj = False
    layer._mono_gdn_op = None
    layer._lq_layout = None
    layer._rdna_opaque_forward = True
    layer._rdna_strided_decode_inputs = True  # HI2: rdna2 decode, strided env on
    layer._rdna_pack_z = False
    layer._gdn_arena_max_bs = max_bs
    layer._conv_state_arena = None
    layer._ssm_state_arena = None
    layer._rdna_decode_inplace = inplace
    layer.get_state_shape = (
        lambda: MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_WIDTH, 0
        )
    )
    layer.get_state_dtype = lambda: (torch.float16, torch.float32)
    w_qkvz, w_ba, w_out = weights["w_qkvz"], weights["w_ba"], weights["w_out"]
    layer.in_proj_qkvz = lambda h: (F.linear(h, w_qkvz), None)
    layer.in_proj_ba = lambda h: (F.linear(h, w_ba), None)
    layer.out_proj = lambda x: (F.linear(x, w_out), None)
    layer.norm = RMSNormGated(
        V,
        eps=EPS,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
        device=device,
        dtype=torch.float16,
    )
    layer.norm.weight.data.copy_(weights["norm_w"])
    return layer


@contextlib.contextmanager
def _env(**values: str):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class GdnDecodeRig:
    """``num_layers`` GDN layers sharing one metadata builder (one KV group).

    ``inplace`` flips ``VLLM_RDNA_GDN_DECODE_INPLACE`` for this rig: the flag
    is read when the builder and the layers are constructed (here) and, for the
    kernels, per call (``forward`` re-applies the rig's environment).
    ``conv_hip`` leaves ``VLLM_CAUSAL_CONV1D_RDNA2_UPDATE`` on (the HI2 recipe
    runs it at 0: Triton conv update).
    """

    def __init__(
        self,
        *,
        inplace: bool,
        num_blocks: int = 64,
        num_layers: int = 1,
        max_num_seqs: int = 8,
        mamba_cache_mode: str = "none",
        hidden: int = 256,
        seed: int = 0,
        conv_hip: bool = False,
        device: str = "cuda",
        page_bytes: int = PAGE_BYTES,
        init_state: bool = True,
    ):
        self.device = device
        self.hidden = hidden
        self.inplace_requested = inplace
        self.mamba_cache_mode = mamba_cache_mode
        self.num_blocks = num_blocks
        # Set to a big tensor to read it after every layer: the real model
        # streams ~60 MB of weights between two GDN layers, which evicts the
        # state pages from L2 and the Infinity Cache; a one-layer loop does not.
        self.thrash: torch.Tensor | None = None
        self.env = {
            "VLLM_RDNA_GDN_DECODE_INPLACE": "1" if inplace else "0",
            "VLLM_CAUSAL_CONV1D_RDNA2_UPDATE": "1" if conv_hip else "0",
        }
        self.vllm_config = create_vllm_config(
            model_name=str(_model_dir()),
            block_size=BLOCK_SIZE,
            max_num_seqs=max_num_seqs,
            dtype=torch.float16,
        )
        self.vllm_config.compilation_config.cudagraph_mode = (
            CUDAGraphMode.FULL_AND_PIECEWISE
        )
        self.vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
        weights = make_weights(hidden, device, seed)
        self.raws: list[torch.Tensor] = []
        self.convs: list[torch.Tensor] = []
        self.ssms: list[torch.Tensor] = []
        self.layers: list[types.SimpleNamespace] = []
        with self._env(), set_current_vllm_config(self.vllm_config):
            spec = MambaSpec(
                block_size=BLOCK_SIZE,
                shapes=((STATE_LEN, CONV_DIM), (HV, V, K)),
                dtypes=(torch.float16, torch.float32),
            )
            self.builder = GDNAttentionMetadataBuilder(
                kv_cache_spec=spec,
                layer_names=[PREFIX.format(i) for i in range(num_layers)],
                vllm_config=self.vllm_config,
                device=torch.device(device),
            )
            for i in range(num_layers):
                raw, conv, ssm = make_paged_state(num_blocks, device, page_bytes)
                if init_state:
                    randomize_state(conv, ssm, seed=seed + 1 + i)
                layer = make_layer(
                    i,
                    weights,
                    conv,
                    ssm,
                    inplace=self.builder_inplace,
                    max_bs=self.builder.decode_cudagraph_max_bs,
                    device=device,
                )
                self.raws.append(raw)
                self.convs.append(conv)
                self.ssms.append(ssm)
                self.layers.append(layer)
                self.vllm_config.compilation_config.static_forward_context[
                    layer.prefix
                ] = layer
            if not self.builder_inplace:
                # The arenas exist before any capture, as in the layer __init__.
                for layer in self.layers:
                    layer._init_gdn_state_arenas()

    @property
    def builder_inplace(self) -> bool:
        """Whether the builder actually took the in-place path."""
        return bool(getattr(self.builder, "_decode_inplace", False))

    def _env(self):
        return _env(**self.env)

    def build_md(
        self,
        slots: list[int],
        *,
        pad_to: int | None = None,
        seq_lens: list[int] | None = None,
    ) -> GDNAttentionMetadata:
        """Metadata for a FULL-graph decode batch: one token per real request,
        zero-length NULL requests up to ``pad_to`` (token-padded like V2)."""
        n = len(slots)
        batch = pad_to if pad_to is not None else n
        if seq_lens is None:
            seq_lens = [17 + 13 * i for i in range(n)]
        assert (
            len(seq_lens) == n and max(seq_lens, default=1) <= BLOCK_SIZE * MAX_BLOCKS
        )
        table = torch.zeros(batch, MAX_BLOCKS, dtype=torch.int32)
        for i, slot in enumerate(slots):
            col = (
                (seq_lens[i] - 1) // BLOCK_SIZE
                if self.mamba_cache_mode != "none"
                else 0
            )
            table[i, col] = slot
        spec = BatchSpec(
            seq_lens=seq_lens + [0] * (batch - n),
            query_lens=[1] * n + [0] * (batch - n),
        )
        common = create_common_attn_metadata(
            spec, BLOCK_SIZE, torch.device(self.device)
        )
        common = common.replace(
            block_table_tensor=table.to(self.device), num_actual_tokens=batch
        )
        with self._env():
            return self.builder.build(0, common)

    def forward(self, hidden: torch.Tensor, md: GDNAttentionMetadata) -> torch.Tensor:
        """Every layer on ``hidden`` (unchained); returns the last layer's output."""
        meta = {layer.prefix: md for layer in self.layers}
        out = None
        with (
            self._env(),
            set_forward_context(meta, self.vllm_config, num_tokens=hidden.size(0)),
        ):
            for layer in self.layers:
                out = layer.forward_cuda(hidden)
                if self.thrash is not None:
                    self.thrash.sum()
        assert out is not None
        return out

    def random_hidden(self, tokens: int, gen: torch.Generator) -> torch.Tensor:
        return torch.randn(
            tokens, self.hidden, device=self.device, dtype=torch.float32, generator=gen
        ).to(torch.float16)

    def snapshot(self) -> list[torch.Tensor]:
        return [raw.clone() for raw in self.raws]

    def restore(self, snap: list[torch.Tensor]) -> None:
        for raw, s in zip(self.raws, snap):
            raw.copy_(s)
