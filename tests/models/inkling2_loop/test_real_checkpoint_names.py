# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling-2 text model against the *real* Inkling-2-Mini checkpoint, on the meta
device.

The tiny checkpoints of ``make_tiny_checkpoint.py`` exercise the numerics; this file
checks that ``Inkling2LoopForConditionalGeneration.load_weights`` accepts every tensor
of the real checkpoint (names, shapes, dtypes) without reading any tensor data: the
model is built with the real config under ``torch.device("meta")``, the safetensors
*headers* of ``~/hfcache/inkling2-mini`` are turned into meta tensors of the same dtype
and shape, and the real ``load_weights`` (including ``process_weights_after_loading``)
runs on them.  Skipped when the checkpoint is not on the machine.

``test_w13_*`` need no checkpoint: they pin the gate/up row layout of the experts.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("vllm.models.inkling2_loop.amd.model")

from paged_harness import fake_vllm_config, patched  # noqa: E402

from vllm.models.inkling2_loop.amd import loader  # noqa: E402
from vllm.models.inkling2_loop.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import moe as moe_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402
from vllm.models.inkling2_loop.configs import Inkling2LoopConfig  # noqa: E402

CKPT = Path("~/hfcache/inkling2-mini").expanduser()
SHARD = CKPT / "model-00001-of-00001.safetensors"
MTP = CKPT / "mtp.safetensors"
TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"
GIB = 1 << 30

needs_real = pytest.mark.skipif(
    not (SHARD.exists() and (CKPT / "config.json").exists()),
    reason=f"real Inkling-2-Mini checkpoint not found at {CKPT}",
)

_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}


def read_header(path: Path) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """``{name: (dtype, shape)}`` from the safetensors header (no tensor data)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {
        k: (_DTYPES[v["dtype"]], tuple(v["shape"])) for k, v in sorted(header.items())
    }


def build_meta_model(ckpt: Path = CKPT):
    """The serving model of ``ckpt`` on the meta device (fp16, TP=1), plus the vLLM
    config slice it was built with."""
    hf_config = Inkling2LoopConfig.from_pretrained(str(ckpt))
    fake = fake_vllm_config(ckpt, hf_config)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with (
            patched(state_mod, "get_current_vllm_config", lambda: fake),
            torch.device("meta"),
        ):
            lm = model_mod.Inkling2LoopForConditionalGeneration(vllm_config=fake)
    finally:
        torch.set_default_dtype(old)
    return lm, fake


def load_header_weights(lm, header) -> set[str]:
    """Run the real ``load_weights`` over meta tensors shaped like ``header``."""
    weights = (
        (k, torch.empty(s, dtype=d, device="meta")) for k, (d, s) in header.items()
    )
    with torch.no_grad():
        return lm.load_weights(weights)


@pytest.fixture(scope="module")
def real():
    header = read_header(SHARD)
    lm, fake = build_meta_model()
    # parameters / persistent buffers before ``process_weights_after_loading``
    # replaces the experts by kernel buffers
    before = {k: (v.dtype, tuple(v.shape)) for k, v in lm.state_dict().items()}
    loaded = load_header_weights(lm, header)
    return types.SimpleNamespace(
        header=header, lm=lm, fake=fake, before=before, loaded=loaded
    )


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


@needs_real
def test_header_census(real):
    visual = [k for k in real.header if k.startswith("model.visual.")]
    llm = [k for k in real.header if k.startswith("model.llm.")]
    assert len(real.header) == 1244
    assert len(visual) == 422
    assert len(llm) == len(real.header) - len(visual)  # nothing else in the shard
    mtp = read_header(MTP)
    assert len(mtp) == 54
    assert not any(k.startswith("model.") for k in mtp)  # raw ``layers.N`` / ``norm``


@needs_real
def test_every_language_tensor_is_consumed(real):
    """``load_weights`` raises on an unmapped name or a missing parameter; on top of
    that every text parameter / persistent buffer must have been filled."""
    llm = {k for k in real.header if k.startswith("model.llm.")}
    cross = {
        "model.llm.middle_cycle_cross_pass_queries",
        "model.llm.middle_cycle_cross_pass_kv_queries",
    }
    assert cross <= llm
    lm = real.lm
    mapped = {lm._map_name(k) for k in llm - cross}
    assert None not in mapped, sorted(k for k in llm - cross if lm._map_name(k) is None)
    assert len(mapped) == len(llm) - len(cross)  # injective: no two tensors collide
    assert mapped | {"model.message_input.query"} == real.loaded
    unfilled = set(real.before) - real.loaded
    assert not unfilled, sorted(unfilled)
    assert real.loaded <= set(real.before)


# ---------------------------------------------------------------------------
# shapes and dtypes
# ---------------------------------------------------------------------------


@needs_real
def test_shapes_and_dtypes(real):
    lm = real.lm
    cross = {
        "model.llm.middle_cycle_cross_pass_queries",
        "model.llm.middle_cycle_cross_pass_kv_queries",
    }
    n_loop = lm.config.num_loop_layers
    for name, (dtype, shape) in real.header.items():
        if not name.startswith("model.llm."):
            continue
        if name in cross:
            assert shape == (2 * n_loop, lm.config.hidden_size), name
            continue
        dst = lm._map_name(name)
        want_dtype, want_shape = real.before[dst]
        if dst == "lm_head.weight":
            # padded on disk, the real ids only in the model
            assert shape == (lm.config.padded_vocab_size, lm.config.hidden_size)
            assert want_shape == (lm.config.unpadded_vocab_size, shape[1])
            continue
        assert shape == want_shape, (name, dst, shape, want_shape)
        if ".mlp.w" in dst:  # packed experts: bit-exact storage, never a cast
            assert dtype == want_dtype, (name, dtype, want_dtype)
        else:
            assert dtype in (torch.bfloat16, torch.float32), (name, dtype)


@needs_real
def test_vocab_padding(real):
    cfg = real.lm.config
    assert (cfg.padded_vocab_size, cfg.unpadded_vocab_size) == (201024, 200058)
    assert cfg.vocab_size == 200058
    assert real.before["model.embed_tokens.weight"][1] == (201024, cfg.hidden_size)
    assert real.before["lm_head.weight"][1] == (200058, cfg.hidden_size)
    assert real.header["model.llm.embed.weight"][1] == (201024, cfg.hidden_size)
    assert real.header["model.llm.unembed.weight"][1] == (201024, cfg.hidden_size)


# ---------------------------------------------------------------------------
# expert formats
# ---------------------------------------------------------------------------


@needs_real
def test_expert_formats(real):
    quant = json.loads((CKPT / "hf_quant_config.json").read_text())
    n = real.lm.config.num_hidden_layers
    fmts = loader.expert_formats(quant, n)
    assert fmts == ["nvfp4"] * 20 + ["mxfp8"] * 4
    for i, fmt in enumerate(fmts):
        p = f"model.llm.layers.{i}.mlp.experts."
        if fmt == "nvfp4":
            assert real.header[p + "w13_weight"][0] == torch.uint8
            assert real.header[p + "w13_weight_scale"][0] == torch.float8_e4m3fn
            assert real.header[p + "w13_weight_amax"] == (torch.float32, (128,))
            assert real.header[p + "w2_weight_outer_scale_denominator"] == (
                torch.float32,
                (1,),
            )
        else:
            assert real.header[p + "w13_weight"][0] == torch.float8_e4m3fn
            assert real.header[p + "w13_weight_scale"][0] == torch.uint8
            assert p + "w13_weight_amax" not in real.header
        assert real.lm.model.layers[i].mlp.quant_format == fmt


# ---------------------------------------------------------------------------
# device memory (TP=1, text only)
# ---------------------------------------------------------------------------


def _bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


@needs_real
def test_weights_fit_one_v620(real):
    """Parameters + kernel buffers after ``process_weights_after_loading``."""
    lm = real.lm
    params = _bytes(lm.parameters())
    buffers = _bytes(lm.buffers())
    print(
        f"[mem] parameters {params / GIB:.3f} GiB, "
        f"kernel/persistent buffers {buffers / GIB:.3f} GiB, "
        f"total {(params + buffers) / GIB:.3f} GiB"
    )
    # the checkpoint-layout expert tensors are gone once they are repacked
    assert not any(".w13_weight" in n for n, _ in lm.named_parameters())
    assert params + buffers < 24 * GIB


@needs_real
def test_kv_state_per_token(real):
    """Per-token cost of the cache owners, from their ``get_kv_cache_spec``."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        SlidingWindowSpec,
    )

    vllm_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=16)
    )
    full_per_token = 0
    swa_per_seq = 0
    conv_per_seq = 0
    n_full = n_swa = n_conv = 0
    for layer in real.lm.model.layers:
        for owner in layer.kv_owners:
            spec = owner.get_kv_cache_spec(vllm_config)
            if isinstance(spec, SlidingWindowSpec):
                n_swa += 1
                swa_per_seq += (
                    spec.page_size_bytes
                    // spec.block_size
                    * min(spec.sliding_window, 1 << 30)
                )
            else:
                assert isinstance(spec, FullAttentionSpec)
                n_full += 1
                full_per_token += spec.page_size_bytes // spec.block_size
        for owner in layer.conv_owners:
            spec = owner.get_kv_cache_spec(vllm_config)
            n_conv += 1
            conv_per_seq += (
                spec.page_size_bytes // spec.block_size * spec.sliding_window
            )
    print(
        f"[kv] full-attention owners {n_full}: {full_per_token} B/token; "
        f"sliding owners {n_swa}: {swa_per_seq / (1 << 20):.2f} MiB/sequence (window "
        f"full); conv owners {n_conv}: {conv_per_seq / (1 << 20):.3f} MiB/sequence"
    )
    assert (n_full, n_swa, n_conv) == (8, 40, 48)
    assert full_per_token == 8 * 2 * 8 * 128 * 2  # K|V, 8 kv heads, D=128, fp16


# ---------------------------------------------------------------------------
# gate / up row layout of w13 (no checkpoint needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("cols", [(), (5,)])
def test_w13_copy_deinterleaves_rows(dtype, cols):
    e, inter = 3, 7
    gen = torch.Generator().manual_seed(0)
    gate = torch.randint(0, 256, (e, inter, *cols), generator=gen, dtype=torch.uint8)
    up = torch.randint(0, 256, (e, inter, *cols), generator=gen, dtype=torch.uint8)
    src = torch.empty(e, 2 * inter, *cols, dtype=torch.uint8)
    src[:, 0::2], src[:, 1::2] = gate, up  # checkpoint: g0 u0 g1 u1 ...
    target = torch.zeros(e, 2 * inter, *cols, dtype=dtype)
    model_mod.copy_gate_up_deinterleaved(target, src.view(dtype))
    got = target.view(torch.uint8)
    assert torch.equal(got[:, :inter], gate)
    assert torch.equal(got[:, inter:], up)


def _tiny_tool(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0]
    + [-6.0]
)


def _dequant_kernel_buffers(mlp, stage: str) -> torch.Tensor:
    """fp32 ``[E, N, K]`` from the kernel-layout buffers of one stage (``w13`` /
    ``w2``), undoing the repack: words are K-major, scales are ``[E, K/block, N]``."""
    q = getattr(mlp, f"{stage}_qweight")  # int32 [E, K/words, N]
    s = getattr(mlp, f"{stage}_scales")  # uint8 [E, K/block, N]
    num_e, kw, n = q.shape
    word = q.to(torch.int64) & 0xFFFFFFFF
    if mlp.quant_format == "nvfp4":
        nib = torch.stack([(word >> (4 * i)) & 0xF for i in range(8)], dim=2)
        vals = _E2M1[nib.reshape(num_e, kw * 8, n)]
        scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        glob = getattr(mlp, f"{stage}_globals")[:, None, None]
        out = vals * scale * glob
    else:
        byte = torch.stack([(word >> (8 * i)) & 0xFF for i in range(4)], dim=2)
        vals = byte.reshape(num_e, kw * 4, n).to(torch.uint8).view(torch.float8_e4m3fn)
        scale = torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)
        out = vals.float() * scale
    return out.permute(0, 2, 1)


def test_load_weights_deinterleaves_expert_rows(tmp_path):
    """The tiny checkpoint stores ``w13`` interleaved like the real one.  After
    ``load_weights`` the kernel buffers (a) equal the repack of the same experts in
    ``[gate; up]`` order and (b) dequantize back to the reference's ``gate_up_proj``
    / ``down_proj`` (NVFP4 and MXFP8 layers)."""
    make = _tiny_tool("make_tiny_checkpoint")
    ref = _tiny_tool("tiny_reference")
    tensors = make.write_checkpoint(tmp_path, make.TinySpec(), seed=1)

    hf_config = Inkling2LoopConfig.from_pretrained(str(tmp_path))
    fake = fake_vllm_config(tmp_path, hf_config)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with patched(state_mod, "get_current_vllm_config", lambda: fake):
            lm = model_mod.Inkling2LoopForConditionalGeneration(vllm_config=fake)
    finally:
        torch.set_default_dtype(old)
    lm.load_weights(iter(tensors.items()))

    seen = set()
    for i, layer in enumerate(lm.model.layers):
        p = f"model.llm.layers.{i}.mlp.experts"
        w, s = tensors[p + ".w13_weight"], tensors[p + ".w13_weight_scale"]
        w_raw = w.view(torch.uint8)
        # the model order: [gate rows; up rows]
        w, s = (
            torch.cat([t.view(torch.uint8)[:, 0::2], t.view(torch.uint8)[:, 1::2]], 1)
            for t in (w, s)
        )
        assert not torch.equal(w, w_raw)
        if layer.mlp.quant_format == "nvfp4":
            q, sc, g = moe_mod.repack_nvfp4(
                w,
                s,
                tensors[p + ".w13_weight_amax"],
                tensors[p + ".w13_weight_outer_scale_denominator"],
            )
            assert torch.equal(layer.mlp.w13_globals, g)
        else:
            q, sc = moe_mod.repack_mxfp8(w, s)
        assert torch.equal(layer.mlp.w13_qweight, q.view(torch.int32))
        assert torch.equal(layer.mlp.w13_scales, sc)

        gate_up, down = ref.dequant_experts(tensors, p)  # reference [gate; up] rows
        torch.testing.assert_close(
            _dequant_kernel_buffers(layer.mlp, "w13"), gate_up, rtol=1e-5, atol=0
        )
        torch.testing.assert_close(
            _dequant_kernel_buffers(layer.mlp, "w2"), down, rtol=1e-5, atol=0
        )
        seen.add(layer.mlp.quant_format)
    assert seen == {"nvfp4", "mxfp8"}
