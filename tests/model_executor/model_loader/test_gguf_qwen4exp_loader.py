# SPDX-License-Identifier: Apache-2.0
"""CPU registry, TP1/TP2 load_model, and fail-closed tests for qwen4exp_gguf."""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from gguf import quants

HF_TEMPLATE_CONFIG = Path(__file__).parent / "fixtures/qwen4exp_hf_config.json"
GEOMETRY = {
    0: (1, 4),
    7: (32, 24),
    8: (32, 34),
    12: (256, 144),
    13: (256, 176),
    20: (32, 18),
    30: (1, 2),
}


def _gguf_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf_shard(
    path: Path,
    *,
    shard_no: int,
    shard_count: int,
    total_tensors: int,
    tensors: list[tuple[str, int, tuple[int, ...], bytes]],
    architecture: str | None,
) -> None:
    md: list[tuple[str, int, int | str]] = [
        ("split.count", 2, shard_count),
        ("split.no", 2, shard_no),
        ("split.tensors.count", 5, total_tensors),
    ]
    if architecture is not None:
        md.append(("general.architecture", 8, architecture))
    buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(md)))
    for key, typ, val in md:
        buf += _gguf_str(key) + struct.pack("<I", typ)
        if typ == 8:
            assert isinstance(val, str)
            buf += _gguf_str(val)
        else:
            fmt = {2: "<H", 5: "<i"}[typ]
            buf += struct.pack(fmt, int(val))
    offsets: list[int] = []
    pos = 0
    for name, typ, dims, raw in tensors:
        block, bpb = GEOMETRY[typ]
        expected_nbytes = math.prod(dims) // block * bpb
        assert len(raw) == expected_nbytes, (name, len(raw), expected_nbytes)
        off = (pos + 31) // 32 * 32
        offsets.append(off)
        pos = off + len(raw)
        buf += _gguf_str(name) + struct.pack("<I", len(dims))
        for d in dims:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", typ, off)
    if tensors:
        buf += bytes((-len(buf)) % 32)
        payload = bytearray(pos)
        for (_, _, _, raw), off in zip(tensors, offsets, strict=True):
            payload[off : off + len(raw)] = raw
        buf += payload
    path.write_bytes(buf)


def _forward_v_reorder(
    t: torch.Tensor, dim: int, num_k: int, num_v_per_k: int, head_dim: int
) -> torch.Tensor:
    ndim = t.ndim
    dim %= ndim
    shape = list(t.shape)
    reshaped = shape[:dim] + [num_k, num_v_per_k, head_dim] + shape[dim + 1 :]
    perm = list(range(len(reshaped)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.reshape(*reshaped).permute(*perm).contiguous().reshape(*shape)


def _encode_f32(t: torch.Tensor) -> bytes:
    return t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes()


def _encode_bf16(t: torch.Tensor) -> bytes:
    bits = (
        t.detach()
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2")
    )
    return bits.tobytes()


def _encode_q8_0(t: torch.Tensor) -> bytes:
    arr = t.detach().to(torch.float32).contiguous().numpy()
    return quants.quantize(arr, 8).tobytes()


def _make_toy_artifact(
    tmp_path: Path, *, mismatch_gate_up: bool = False, drop_last_dense: bool = False
) -> tuple[Path, Path, dict[str, torch.Tensor], dict[str, bytes]]:
    cfg = json.loads(HF_TEMPLATE_CONFIG.read_text())
    cfg.pop("quantization_config", None)
    cfg["architectures"] = ["Qwen4ExpForCausalLM"]
    cfg["language_model_only"] = True
    tc = cfg["text_config"]
    # Miniature 4-layer Qwen4Exp: GDN(L0), GDN+PLE(L1), Q5_K/Q8_0(L2), QSA(L3)
    tc.update(
        {
            "vocab_size": 128,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [3, 3, 2],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 128,
            "hc_count": 4,
            "hc_lowrank": 64,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 64,
            "indexer_compress_ratio": 4,
            "indexer_budget": 64,
            "ple_layer_ids": [2],  # 1-based -> layer 1
            "ple_embed_dim": 640,
            "ngram_vocab_size_base": 64,
            "heads_per_ngram": 2,
            "ngram_size": 3,
            "ple_conv_kernel_size": 4,
        }
    )
    hf_dir = tmp_path / "hf-config"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    from vllm.models.qwen4_exp.amd.model import Qwen4ExpForCausalLM  # noqa: F401
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpNGramEmbedding
    from vllm.utils.math_utils import round_up

    _, _, raw_ple_rows = Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=64,
        ngram_heads=2 * (3 - 1),
        ple_dense_layer_id=0,
    )
    ple_rows = round_up(raw_ple_rows, int(tc["make_ngram_vocab_size_divisible_by"]))

    g = torch.Generator().manual_seed(1234)

    def rand(shape: tuple[int, ...], scale: float = 0.25) -> torch.Tensor:
        return (
            (torch.randn(shape, generator=g, dtype=torch.float32) * scale)
            .half()
            .float()
        )

    truth: dict[str, torch.Tensor] = {}
    packed_truth: dict[str, bytes] = {}
    records: list[tuple[str, int, tuple[int, ...], bytes]] = []

    def add_q8(
        gguf_name: str,
        native_key: str,
        shape: tuple[int, ...],
        val: torch.Tensor | None = None,
    ):
        w = rand(shape) if val is None else val
        raw = _encode_q8_0(w)
        # Truth after Q8_0 round-trip so assertion checks exact loader math
        decoded = torch.from_numpy(
            quants.dequantize(np.frombuffer(raw, np.uint8), 8).copy()
        ).reshape(shape)
        truth[native_key] = decoded.half()
        records.append((gguf_name, 8, tuple(reversed(shape)), raw))

    def add_f32(
        gguf_name: str,
        native_key: str,
        shape: tuple[int, ...],
        *,
        plus_one: bool = False,
        target_fp32: bool = False,
        export_fn=None,
    ):
        w = rand(shape)
        truth[native_key] = w.float() if target_fp32 else w.half()
        exported = (w + 1.0) if plus_one else w
        if export_fn is not None:
            exported = export_fn(exported)
        records.append(
            (gguf_name, 0, tuple(reversed(exported.shape)), _encode_f32(exported))
        )

    add_q8("output.weight", "lm_head.weight", (128, 256))
    add_q8("token_embd.weight", "model.embed_tokens.weight", (128, 256))
    add_q8(
        "output_hc_down.weight",
        "model.hyper_connection_mixer.input_mix_weight_down.weight",
        (64, 1024),
    )
    add_q8(
        "output_hc_up.weight",
        "model.hyper_connection_mixer.input_mix_weight_up.weight",
        (1024, 64),
    )
    add_f32(
        "output_hc_norm.weight",
        "model.hyper_connection_mixer.hc_norm.weight",
        (1024,),
        plus_one=True,
    )

    for layer in range(4):
        pfx = f"model.layers.{layer}"
        gpfx = f"blk.{layer}"
        add_f32(f"{gpfx}.ffn_gate_inp.weight", f"{pfx}.mlp.gate.weight", (4, 256))
        shexp_gate = rand((256,))
        truth[f"{pfx}.mlp.shared_expert_gate.weight"] = shexp_gate.half().reshape(
            1, 256
        )
        records.append(
            (f"{gpfx}.ffn_gate_inp_shexp.weight", 0, (256,), _encode_f32(shexp_gate))
        )

        add_q8(
            f"{gpfx}.ffn_gate_shexp.weight",
            f"{pfx}.mlp.shared_expert.gate_proj.weight",
            (128, 256),
        )
        add_q8(
            f"{gpfx}.ffn_up_shexp.weight",
            f"{pfx}.mlp.shared_expert.up_proj.weight",
            (128, 256),
        )
        add_q8(
            f"{gpfx}.ffn_down_shexp.weight",
            f"{pfx}.mlp.shared_expert.down_proj.weight",
            (256, 128),
        )

        for hc_tag, hc_mod in (
            ("attn", "attn_hyper_connection"),
            ("ffn", "mlp_hyper_connection"),
        ):
            add_f32(
                f"{gpfx}.hc_{hc_tag}_norm.weight",
                f"{pfx}.{hc_mod}.hc_norm.weight",
                (1024,),
                plus_one=True,
            )
            add_q8(
                f"{gpfx}.hc_{hc_tag}_up.weight",
                f"{pfx}.{hc_mod}.input_mix_weight_up.weight",
                (1024, 64),
            )
            add_q8(
                f"{gpfx}.hc_{hc_tag}_down.weight",
                f"{pfx}.{hc_mod}.input_mix_weight_down.weight",
                (64, 1024),
            )
            add_f32(
                f"{gpfx}.hc_{hc_tag}_inject.weight",
                f"{pfx}.{hc_mod}.block_inject_weight.weight",
                (4, 1024),
            )

        # Packed routed experts: L2 uses Q5_K(13)+Q8_0(8); others use Q4_K(12)+Q5_1(7)
        gate_type = 13 if layer == 2 else 12
        up_type = 12 if mismatch_gate_up and layer == 0 else gate_type
        if mismatch_gate_up and layer == 0:
            up_type = 13
        down_type = 8 if layer == 2 else 7
        for ename, etype, dims in (
            ("ffn_gate_exps.weight", gate_type, (256, 256, 4)),
            ("ffn_up_exps.weight", up_type, (256, 256, 4)),
            ("ffn_down_exps.weight", down_type, (256, 256, 4)),
        ):
            blk, bpb = GEOMETRY[etype]
            nbytes = math.prod(dims) // blk * bpb
            raw = bytes(
                torch.randint(
                    0, 256, (nbytes,), generator=g, dtype=torch.uint8
                ).tolist()
            )
            packed_truth[f"{gpfx}.{ename}"] = raw
            records.append((f"{gpfx}.{ename}", etype, dims, raw))

        if layer == 1:
            add_f32(
                f"{gpfx}.ple_norm_key.weight",
                f"{pfx}.ple.norm_key.weight",
                (1024,),
                plus_one=True,
            )
            add_f32(
                f"{gpfx}.ple_norm_query.weight",
                f"{pfx}.ple.norm_query.weight",
                (1024,),
                plus_one=True,
            )
            add_f32(
                f"{gpfx}.ple_norm_conv.weight",
                f"{pfx}.ple.norm_conv.weight",
                (1024,),
                plus_one=True,
            )
            pconv = rand((1024, 1, 4))
            truth[f"{pfx}.ple.conv1d.weight"] = pconv.half()
            records.append(
                (
                    f"{gpfx}.ple_conv1d.weight",
                    0,
                    (4, 1024),
                    _encode_f32(pconv.squeeze(1)),
                )
            )
            add_q8(f"{gpfx}.ple_key.weight", f"{pfx}.ple.key_proj.weight", (1024, 640))
            add_q8(
                f"{gpfx}.ple_value.weight", f"{pfx}.ple.value_proj.weight", (256, 640)
            )

        if layer < 3:
            # GDN: num_k=2, num_v=6, head_dim=64 -> Q=128, K=128, V=384, Z=384
            qkv = rand((640, 256))
            qkv_exp = torch.cat(
                [qkv[:256], _forward_v_reorder(qkv[256:], 0, 2, 3, 64)], dim=0
            )
            raw_qkv = _encode_q8_0(qkv_exp)
            dec_qkv = torch.from_numpy(
                quants.dequantize(np.frombuffer(raw_qkv, np.uint8), 8).copy()
            ).reshape(640, 256)
            # Re-invert decoded Q8_0 so truth reflects exact Q8_0 roundtrip
            from vllm.model_executor.model_loader.gguf_gdn_transforms import (
                inverse_v_heads,
            )

            truth[f"{pfx}.linear_attn.in_proj_qkv.weight"] = torch.cat(
                [dec_qkv[:256], inverse_v_heads(dec_qkv[256:], 0, 2, 3, 64)], dim=0
            ).half()
            records.append((f"{gpfx}.attn_qkv.weight", 8, (256, 640), raw_qkv))

            z = rand((384, 256))
            raw_z = _encode_q8_0(_forward_v_reorder(z, 0, 2, 3, 64))
            dec_z = torch.from_numpy(
                quants.dequantize(np.frombuffer(raw_z, np.uint8), 8).copy()
            ).reshape(384, 256)
            truth[f"{pfx}.linear_attn.in_proj_z.weight"] = inverse_v_heads(
                dec_z, 0, 2, 3, 64
            ).half()
            records.append((f"{gpfx}.attn_gate.weight", 8, (256, 384), raw_z))

            for tag, nkey in (
                ("ssm_beta.weight", "in_proj_b.weight"),
                ("ssm_alpha.weight", "in_proj_a.weight"),
            ):
                ba = rand((6, 256))
                truth[f"{pfx}.linear_attn.{nkey}"] = ba.half()
                records.append(
                    (
                        f"{gpfx}.{tag}",
                        0,
                        (256, 6),
                        _encode_f32(_forward_v_reorder(ba, 0, 2, 3, 1)),
                    )
                )

            dt = rand((6,))
            truth[f"{pfx}.linear_attn.dt_bias"] = dt.half()
            records.append(
                (
                    f"{gpfx}.ssm_dt.bias",
                    0,
                    (6,),
                    _encode_f32(_forward_v_reorder(dt, 0, 2, 3, 1)),
                )
            )

            alog = -torch.linspace(0.25, 2.5, 6, dtype=torch.float32) - layer * 0.1
            truth[f"{pfx}.linear_attn.A_log"] = alog
            ssm_a_gguf = -torch.exp(_forward_v_reorder(alog, 0, 2, 3, 1))
            records.append((f"{gpfx}.ssm_a", 0, (6,), _encode_f32(ssm_a_gguf)))

            conv = rand((640, 1, 4))
            truth[f"{pfx}.linear_attn.conv1d.weight"] = conv.half()
            conv_s = conv.squeeze(1)
            conv_exp = torch.cat(
                [conv_s[:256], _forward_v_reorder(conv_s[256:], 0, 2, 3, 64)], dim=0
            )
            records.append(
                (f"{gpfx}.ssm_conv1d.weight", 0, (4, 640), _encode_f32(conv_exp))
            )

            # linear_attn.norm.weight is NOT zero-centered (+1 is skipped by converter!)
            add_f32(
                f"{gpfx}.ssm_norm.weight",
                f"{pfx}.linear_attn.norm.weight",
                (64,),
                plus_one=False,
            )

            out_w = rand((256, 384))
            raw_out = _encode_q8_0(_forward_v_reorder(out_w, 1, 2, 3, 64))
            dec_out = torch.from_numpy(
                quants.dequantize(np.frombuffer(raw_out, np.uint8), 8).copy()
            ).reshape(256, 384)
            truth[f"{pfx}.linear_attn.out_proj.weight"] = inverse_v_heads(
                dec_out, 1, 2, 3, 64
            ).half()
            records.append((f"{gpfx}.ssm_out.weight", 8, (384, 256), raw_out))
        else:
            # QSA full attention: Q gated 4*2*64=512 rows; K=128, V=128
            add_q8(
                f"{gpfx}.attn_q.weight", f"{pfx}.self_attn.q_proj.weight", (512, 256)
            )
            add_q8(
                f"{gpfx}.attn_k.weight", f"{pfx}.self_attn.k_proj.weight", (128, 256)
            )
            add_q8(
                f"{gpfx}.attn_v.weight", f"{pfx}.self_attn.v_proj.weight", (128, 256)
            )
            add_q8(
                f"{gpfx}.attn_output.weight",
                f"{pfx}.self_attn.o_proj.weight",
                (256, 256),
            )
            add_f32(
                f"{gpfx}.attn_q_norm.weight",
                f"{pfx}.self_attn.q_norm.weight",
                (64,),
                plus_one=True,
            )
            add_f32(
                f"{gpfx}.attn_k_norm.weight",
                f"{pfx}.self_attn.k_norm.weight",
                (64,),
                plus_one=True,
            )
            add_f32(
                f"{gpfx}.indexer.q_norm.weight",
                f"{pfx}.self_attn.indexer.q_layernorm.weight",
                (64,),
                plus_one=True,
            )
            add_f32(
                f"{gpfx}.indexer.k_norm.weight",
                f"{pfx}.self_attn.indexer.k_layernorm.weight",
                (64,),
                plus_one=True,
            )
            iq = rand((256, 256)).to(torch.bfloat16)
            ik = rand((64, 256)).to(torch.bfloat16)
            truth[f"{pfx}.self_attn.indexer.index_qk_proj.weight"] = torch.cat(
                [iq, ik], dim=0
            ).half()
            records.append(
                (f"{gpfx}.indexer.q_proj.weight", 30, (256, 256), _encode_bf16(iq))
            )
            records.append(
                (f"{gpfx}.indexer.k_proj.weight", 30, (256, 64), _encode_bf16(ik))
            )

    if drop_last_dense:
        records.pop()

    # PLE IQ4_NL table in shard 4
    ple_raw = np.zeros((ple_rows, 5, 18), dtype=np.uint8)
    scales = ((np.arange(ple_rows * 5, dtype=np.float32) % 31 + 1) / 512.0).astype(
        "<f2"
    )
    ple_raw[:, :, :2] = scales.view(np.uint8).reshape(ple_rows, 5, 2)
    codes = np.arange(16, dtype=np.uint8)[None, None, :] | (
        ((15 - np.arange(16, dtype=np.uint8))[None, None, :]) << 4
    )
    ple_raw[:, :, 2:] = (
        codes + (np.arange(ple_rows, dtype=np.uint8)[:, None, None] & 3)
    ) & 0xFF
    ple_bytes = ple_raw.tobytes()
    packed_truth["per_layer_token_embd.weight"] = ple_bytes
    ple_record = [("per_layer_token_embd.weight", 20, (160, ple_rows), ple_bytes)]

    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    mid = len(records) // 2
    total = len(records) + 1
    shards = [
        (0, [], "qwen4exp"),
        (1, records[:mid], None),
        (2, records[mid:], None),
        (3, ple_record, None),
    ]
    for idx, items, arch in shards:
        _write_gguf_shard(
            gguf_dir / f"toy-0000{idx + 1}-of-00004.gguf",
            shard_no=idx,
            shard_count=4,
            total_tensors=total,
            tensors=items,
            architecture=arch,
        )
    return hf_dir, gguf_dir, truth, packed_truth


def _patch_cpu_qsa_noop(monkeypatch):
    import importlib

    q4_pkg = importlib.import_module("vllm.models.qwen4_exp")
    amd_model = importlib.import_module("vllm.models.qwen4_exp.amd.model")
    amd_mtp = importlib.import_module("vllm.models.qwen4_exp.amd.mtp")
    qsa_mod = importlib.import_module("vllm.models.qwen4_exp.amd.qsa")

    monkeypatch.setattr(
        q4_pkg, "Qwen4ExpForCausalLM", amd_model.Qwen4ExpForCausalLM, raising=False
    )
    monkeypatch.setattr(q4_pkg, "Qwen4ExpMTP", amd_mtp.Qwen4ExpMTP, raising=False)
    monkeypatch.setattr(
        qsa_mod, "Qwen4ExpQSAFlashAttentionImpl", lambda *_a, **_k: None
    )


def test_registry_resolves_qwen4exp_gguf_loader(tmp_path):
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.qwen4exp_gguf_loader import (
        Qwen4ExpGGUFModelLoader,
    )

    loader = get_model_loader(
        LoadConfig(
            load_format="qwen4exp_gguf",
            model_loader_extra_config={"gguf_dir": str(tmp_path)},
        )
    )
    assert isinstance(loader, Qwen4ExpGGUFModelLoader)


def test_end_to_end_tp1_load_model_populates_every_parameter_accurately(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("VLLM_PLE_QUANT_DIR", raising=False)
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    _patch_cpu_qsa_noop(monkeypatch)
    hf_dir, gguf_dir, truth, packed_truth = _make_toy_artifact(tmp_path)

    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        EngramConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model_loader

    mc = ModelConfig(
        model=str(hf_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=512,
    )
    lc = LoadConfig(
        load_format="qwen4exp_gguf",
        model_loader_extra_config={"gguf_dir": str(gguf_dir)},
    )
    vc = VllmConfig(
        model_config=mc,
        load_config=lc,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=16,
            max_num_seqs=1,
        ),
        compilation_config=CompilationConfig(mode=0),
        engram_config=EngramConfig(cpu_offload=False, dp_shared_memory=False),
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp_path / 'gloo-tp1'}",
        local_rank=0,
        backend="gloo",
    )
    try:
        with set_current_vllm_config(vc):
            initialize_model_parallel(1, 1)
        model = get_model_loader(lc).load_model(vllm_config=vc, model_config=mc)
        params = dict(model.named_parameters())

        # Direct dense checks
        torch.testing.assert_close(
            params["lm_head.weight"], truth["lm_head.weight"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            params["model.embed_tokens.weight"],
            truth["model.embed_tokens.weight"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            params["model.hyper_connection_mixer.hc_norm.weight"],
            truth["model.hyper_connection_mixer.hc_norm.weight"],
            rtol=0,
            atol=1e-3,
        )
        # GDN merged QKVZ + BA + A_log + conv1d + norm (no -1!) + out_proj
        expected_qkvz0 = torch.cat(
            [
                truth["model.layers.0.linear_attn.in_proj_qkv.weight"],
                truth["model.layers.0.linear_attn.in_proj_z.weight"],
            ],
            dim=0,
        )
        torch.testing.assert_close(
            params["model.layers.0.linear_attn.in_proj_qkvz.weight"],
            expected_qkvz0,
            rtol=0,
            atol=0,
        )
        expected_ba0 = torch.cat(
            [
                truth["model.layers.0.linear_attn.in_proj_b.weight"],
                truth["model.layers.0.linear_attn.in_proj_a.weight"],
            ],
            dim=0,
        )
        torch.testing.assert_close(
            params["model.layers.0.linear_attn.in_proj_ba.weight"],
            expected_ba0,
            rtol=0,
            atol=0,
        )
        assert params["model.layers.0.linear_attn.A_log"].dtype == torch.float32
        torch.testing.assert_close(
            params["model.layers.0.linear_attn.A_log"],
            truth["model.layers.0.linear_attn.A_log"],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            params["model.layers.0.linear_attn.norm.weight"],
            truth["model.layers.0.linear_attn.norm.weight"],
            rtol=0,
            atol=0,
        )
        # HC padded rows (64 + 4 = 68 -> padded to 80; rows 68..80 must be exact zero!)
        hc0 = params[
            "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
        ]
        assert hc0.shape == (80, 1024)
        assert torch.count_nonzero(hc0[68:]) == 0
        torch.testing.assert_close(
            hc0[:64],
            truth["model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            hc0[64:68],
            truth["model.layers.0.attn_hyper_connection.block_inject_weight.weight"],
            rtol=0,
            atol=0,
        )
        # QSA + indexer merged QK
        torch.testing.assert_close(
            params["model.layers.3.self_attn.indexer.index_qk_proj.weight"],
            truth["model.layers.3.self_attn.indexer.index_qk_proj.weight"],
            rtol=0,
            atol=0,
        )
        # Packed experts: L0 (Q4_K=12, Q5_1=7) and L2 (Q5_K=13, Q8_0=8)
        r0 = model.model.layers[0].mlp.experts.routed_experts
        r2 = model.model.layers[2].mlp.experts.routed_experts
        assert (r0.w13_weight_type.weight_type, r0.w2_weight_type.weight_type) == (
            12,
            7,
        )
        assert (r2.w13_weight_type.weight_type, r2.w2_weight_type.weight_type) == (
            13,
            8,
        )
        g0 = torch.frombuffer(
            bytearray(packed_truth["blk.0.ffn_gate_exps.weight"]), dtype=torch.uint8
        ).reshape(4, 256, 144)
        u0 = torch.frombuffer(
            bytearray(packed_truth["blk.0.ffn_up_exps.weight"]), dtype=torch.uint8
        ).reshape(4, 256, 144)
        assert torch.equal(r0.w13_weight, torch.cat([g0, u0], dim=1))

        # Packed IQ4_NL PLE forward lookup
        ple_emb = model.model.layers[1].ple.ple_embedding.ngram_embedding
        got_ple = ple_emb(torch.tensor([0, 7]))
        raw_ple = np.frombuffer(
            packed_truth["per_layer_token_embd.weight"], dtype=np.uint8
        ).reshape(-1, 5, 18)
        exp_ple = (
            torch.from_numpy(
                quants.IQ4_NL.dequantize_blocks(raw_ple[[0, 7]].reshape(-1, 18)).copy()
            )
            .reshape(2, 160)
            .half()
        )
        torch.testing.assert_close(got_ple, exp_ple, rtol=0, atol=0)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _tp2_worker(rank: int, init_file: str, hf_dir: str, gguf_dir: str, out_dir: str):
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ.pop("VLLM_PLE_QUANT_DIR", None)
    os.environ.pop("VLLM_PLE_CPU_OFFLOAD", None)
    import importlib

    q4_pkg = importlib.import_module("vllm.models.qwen4_exp")
    amd_model = importlib.import_module("vllm.models.qwen4_exp.amd.model")
    qsa_mod = importlib.import_module("vllm.models.qwen4_exp.amd.qsa")

    q4_pkg.Qwen4ExpForCausalLM = amd_model.Qwen4ExpForCausalLM
    qsa_mod.Qwen4ExpQSAFlashAttentionImpl = lambda *_a, **_k: None
    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        EngramConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model_loader

    mc = ModelConfig(
        model=hf_dir, skip_tokenizer_init=True, dtype="float16", max_model_len=512
    )
    lc = LoadConfig(
        load_format="qwen4exp_gguf",
        model_loader_extra_config={"gguf_dir": gguf_dir},
    )
    vc = VllmConfig(
        model_config=mc,
        load_config=lc,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2, pipeline_parallel_size=1
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=16,
            max_num_seqs=1,
        ),
        compilation_config=CompilationConfig(mode=0),
        engram_config=EngramConfig(cpu_offload=False, dp_shared_memory=False),
    )
    init_distributed_environment(
        world_size=2,
        rank=rank,
        distributed_init_method=f"file://{init_file}",
        local_rank=rank,
        backend="gloo",
    )
    try:
        with set_current_vllm_config(vc):
            initialize_model_parallel(2, 1)
        model = get_model_loader(lc).load_model(vllm_config=vc, model_config=mc)
        torch.save(
            {
                "qkvz0": model.model.layers[0]
                .linear_attn.in_proj_qkvz.weight.detach()
                .cpu(),
                "w13_0": model.model.layers[0]
                .mlp.experts.routed_experts.w13_weight.detach()
                .cpu(),
                "w2_0": model.model.layers[0]
                .mlp.experts.routed_experts.w2_weight.detach()
                .cpu(),
            },
            Path(out_dir) / f"rank{rank}.pt",
        )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_tp2_sharding_partitions_merged_gdn_qkvz_and_packed_experts_correctly(tmp_path):
    hf_dir, gguf_dir, truth, packed_truth = _make_toy_artifact(tmp_path)
    out_dir = tmp_path / "tp2-out"
    out_dir.mkdir()
    mp.spawn(
        _tp2_worker,
        args=(str(tmp_path / "gloo-tp2"), str(hf_dir), str(gguf_dir), str(out_dir)),
        nprocs=2,
        join=True,
    )
    qkv = truth["model.layers.0.linear_attn.in_proj_qkv.weight"]
    z = truth["model.layers.0.linear_attn.in_proj_z.weight"]
    q, k, v = qkv[:128], qkv[128:256], qkv[256:]
    g0 = torch.frombuffer(
        bytearray(packed_truth["blk.0.ffn_gate_exps.weight"]), dtype=torch.uint8
    ).reshape(4, 256, 144)
    u0 = torch.frombuffer(
        bytearray(packed_truth["blk.0.ffn_up_exps.weight"]), dtype=torch.uint8
    ).reshape(4, 256, 144)
    d0 = torch.frombuffer(
        bytearray(packed_truth["blk.0.ffn_down_exps.weight"]), dtype=torch.uint8
    ).reshape(4, 256, 192)
    for r in range(2):
        saved = torch.load(out_dir / f"rank{r}.pt", weights_only=True)
        exp_qkvz = torch.cat(
            [
                q[r * 64 : (r + 1) * 64],
                k[r * 64 : (r + 1) * 64],
                v[r * 192 : (r + 1) * 192],
                z[r * 192 : (r + 1) * 192],
            ],
            dim=0,
        )
        torch.testing.assert_close(saved["qkvz0"], exp_qkvz, rtol=0, atol=0)
        exp_w13 = torch.cat(
            [g0[:, r * 128 : (r + 1) * 128], u0[:, r * 128 : (r + 1) * 128]], dim=1
        )
        assert torch.equal(saved["w13_0"], exp_w13)
        assert torch.equal(saved["w2_0"], d0[:, :, r * 96 : (r + 1) * 96])


@pytest.mark.parametrize(
    ("case", "expected_match"),
    [
        ("mismatch_gate_up", "gate/up quant mismatch"),
        ("missing_tensor", "missing|unloaded|incomplete"),
        ("mtp_without_sidecar", "no MTP tensors"),
    ],
)
def test_loader_fails_closed_on_invalid_contracts(
    tmp_path, monkeypatch, case, expected_match
):
    monkeypatch.delenv("VLLM_PLE_QUANT_DIR", raising=False)
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    _patch_cpu_qsa_noop(monkeypatch)
    hf_dir, gguf_dir, _, _ = _make_toy_artifact(
        tmp_path,
        mismatch_gate_up=(case == "mismatch_gate_up"),
        drop_last_dense=(case == "missing_tensor"),
    )
    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        EngramConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model_loader

    overrides = (
        {"architectures": ["Qwen4ExpMTP"]} if case == "mtp_without_sidecar" else {}
    )
    mc = ModelConfig(
        model=str(hf_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=512,
        hf_overrides=overrides,
    )
    lc = LoadConfig(
        load_format="qwen4exp_gguf",
        model_loader_extra_config={"gguf_dir": str(gguf_dir)},
    )
    vc = VllmConfig(
        model_config=mc,
        load_config=lc,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=16,
            max_num_seqs=1,
        ),
        compilation_config=CompilationConfig(mode=0),
        engram_config=None
        if case == "mtp_without_sidecar"
        else EngramConfig(cpu_offload=False, dp_shared_memory=False),
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp_path / f'gloo-{case}'}",
        local_rank=0,
        backend="gloo",
    )
    try:
        with set_current_vllm_config(vc):
            initialize_model_parallel(1, 1)
        with pytest.raises(ValueError, match=expected_match):
            get_model_loader(lc).load_model(vllm_config=vc, model_config=mc)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_explicit_mtp_sidecar_loads_all_31_unquantized_mtp_keys(tmp_path, monkeypatch):
    import safetensors.torch

    monkeypatch.delenv("VLLM_PLE_QUANT_DIR", raising=False)
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    _patch_cpu_qsa_noop(monkeypatch)
    # Dev VM has no compiled x86 _C_AVX2 prepack op; ROCm uses GPU MoE experts.
    import vllm.model_executor.layers.fused_moe.experts.cpu_moe as cpu_moe_mod

    monkeypatch.setattr(cpu_moe_mod, "cpu_prepack_moe_weight", lambda w, _isa: w)
    hf_dir, gguf_dir, _, _ = _make_toy_artifact(tmp_path)

    mtp_dir = tmp_path / "mtp-sidecar"
    mtp_dir.mkdir()
    mtp_tensors = {
        "mtp.fc_embedding.weight": torch.ones((256, 256), dtype=torch.bfloat16) * 0.1,
        "mtp.fc_hidden.weight": torch.ones((256, 256), dtype=torch.bfloat16) * 0.2,
        "mtp.pre_fc_norm_embedding.weight": torch.ones((256,), dtype=torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.ones((1024,), dtype=torch.bfloat16),
        "mtp.hyper_connection_mixer.hc_norm.weight": torch.ones(
            (1024,), dtype=torch.bfloat16
        ),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": torch.ones(
            (64, 1024), dtype=torch.bfloat16
        ),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": torch.ones(
            (1024, 64), dtype=torch.bfloat16
        ),
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": torch.ones(
            (4, 1024), dtype=torch.bfloat16
        ),
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight": torch.ones(
            (1024,), dtype=torch.bfloat16
        ),
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": torch.ones(
            (64, 1024), dtype=torch.bfloat16
        ),
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": torch.ones(
            (1024, 64), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp.experts.down_proj": torch.ones(
            (4, 256, 256), dtype=torch.bfloat16
        )
        * 0.5,
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.ones(
            (4, 512, 256), dtype=torch.bfloat16
        )
        * 0.25,
        "mtp.layers.0.mlp.gate.weight": torch.ones((4, 256), dtype=torch.bfloat16),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": torch.ones(
            (256, 128), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.ones(
            (128, 256), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.ones(
            (128, 256), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp.shared_expert_gate.weight": torch.ones(
            (1, 256), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": torch.ones(
            (4, 1024), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": torch.ones(
            (1024,), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": torch.ones(
            (64, 1024), dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": torch.ones(
            (1024, 64), dtype=torch.bfloat16
        ),
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": torch.ones(
            (320, 256), dtype=torch.bfloat16
        ),
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": torch.ones(
            (64,), dtype=torch.bfloat16
        ),
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": torch.ones(
            (64,), dtype=torch.bfloat16
        ),
        "mtp.layers.0.self_attn.k_norm.weight": torch.ones((64,), dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.k_proj.weight": torch.ones(
            (128, 256), dtype=torch.bfloat16
        ),
        "mtp.layers.0.self_attn.o_proj.weight": torch.ones(
            (256, 256), dtype=torch.bfloat16
        ),
        "mtp.layers.0.self_attn.q_norm.weight": torch.ones((64,), dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight": torch.ones(
            (512, 256), dtype=torch.bfloat16
        ),
        "mtp.layers.0.self_attn.v_proj.weight": torch.ones(
            (128, 256), dtype=torch.bfloat16
        ),
    }
    safetensors.torch.save_file(mtp_tensors, str(mtp_dir / "model-mtp.safetensors"))
    (mtp_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-mtp.safetensors" for k in mtp_tensors}})
    )

    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        EngramConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model_loader

    mc = ModelConfig(
        model=str(hf_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=512,
    )
    lc = LoadConfig(
        load_format="qwen4exp_gguf",
        model_loader_extra_config={
            "gguf_dir": str(gguf_dir),
            "mtp_safetensors_dir": str(mtp_dir),
        },
    )
    pc = ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1)
    sc = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=2,
        target_model_config=mc,
        target_parallel_config=pc,
    )
    vc = VllmConfig(
        model_config=mc,
        load_config=lc,
        speculative_config=sc,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=pc,
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=16,
            max_num_seqs=1,
        ),
        compilation_config=CompilationConfig(mode=0),
        engram_config=EngramConfig(cpu_offload=False, dp_shared_memory=False),
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp_path / 'gloo-mtp-ok'}",
        local_rank=0,
        backend="gloo",
    )
    try:
        with set_current_vllm_config(vc):
            initialize_model_parallel(1, 1)
        loader = get_model_loader(lc)
        # First load GGUF target (installs Qwen4ExpGGUFQuantConfig onto vc)
        _target = loader.load_model(vllm_config=vc, model_config=mc)
        # Now load MTP draft model exactly as EagleProposer._get_model does
        draft = loader.load_model(vllm_config=vc, model_config=sc.draft_model_config)
        routed = draft.model.layers[0].mlp.experts.routed_experts
        assert routed.w13_weight.dtype == torch.float16
        assert torch.all(routed.w13_weight == 0.25)
        assert torch.all(routed.w2_weight == 0.5)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
