# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Registered native writer/read gates for both serving cache layouts."""

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm._rocm_C  # noqa: F401
from benchmark_native import check_writer

from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import get_kv_cache_torch_dtype, is_quantized_kv_cache
from vllm.v1.attention.backends.rdna_attn import (
    RdnaAttentionBackend,
    _int4_fp16_cache_view,
    _int4_fp16_read_views,
)
from vllm.v1.attention.ops import fa_rdna2_backend as fa
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode, get_kv_quant_mode


def encode(cpu, codec):
    rows = cpu.flatten(0, -2).tolist()
    quantized = codec["quantize_int4_fp16_block_packed"](rows)
    payload = []
    for row, words in zip(rows, quantized.scales):
        codes = []
        for group, word in enumerate(words):
            scale, zp = codec["f16_from_bits"](word & ~15), word & 15
            codes.extend(
                max(0, min(15, codec["_round_half_away"](v / scale) + zp))
                for v in row[group * 32 : (group + 1) * 32]
            )
        payload.append(list(codec["pack_int4"](codes)))
    return (
        torch.tensor(payload, dtype=torch.uint8).reshape(*cpu.shape[:-1], -1),
        torch.tensor(quantized.scales, dtype=torch.uint16).reshape(*cpu.shape[:-1], -1),
        torch.tensor(quantized.reconstructed, dtype=torch.float16).reshape(cpu.shape),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "2,3,4,5"
    assert torch.accelerator.device_count() == 4
    assert all(
        "gfx1030" in torch.cuda.get_device_properties(i).gcnArchName for i in range(4)
    )
    names = ("reshape_and_cache_int4_fp16_rdna2", "fa_rdna2_paged_int4_fp16_block")
    schemas = [
        str(s)
        for s in torch._C._jit_get_all_schemas()
        if str(s).startswith("_rocm_C::")
    ]
    assert len(schemas) > 10 and any("all_reduce" in s for s in schemas)
    tags = {}
    for name in names:
        op = getattr(torch.ops._rocm_C, name).default
        assert torch.Tag.needs_exact_strides in op.tags, op.tags
        assert any(s.startswith("_rocm_C::" + name + "(") for s in schemas)
        tags[name] = [str(tag) for tag in op.tags]
    assert fa.int4_fp16_kernels_available()
    config = CacheConfig(cache_dtype="int4_fp16_block")
    assert get_kv_cache_torch_dtype(config.cache_dtype) == torch.uint8
    assert is_quantized_kv_cache(config.cache_dtype)
    assert get_kv_quant_mode(config.cache_dtype) == KVQuantMode.INT4_FP16_BLOCK
    spec = FullAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT4_FP16_BLOCK,
    )
    packed_spec = RdnaAttentionBackend.customize_spec(spec)
    assert packed_spec.num_head_slots == 2 and packed_spec.state_content_bytes == 144
    assert packed_spec.real_page_size_bytes == 294912
    assert RdnaAttentionBackend.customize_spec(packed_spec) == packed_spec
    dense_spec = RdnaAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=1024, num_kv_heads=1, head_size=256, dtype=torch.float16
        )
    )
    assert dense_spec.real_page_size_bytes == 1048576
    writer = torch.ops._rocm_C.reshape_and_cache_int4_fp16_rdna2
    legacy_writer_gate = check_writer(
        SimpleNamespace(write_packed=writer), page_size=1024
    )
    root = Path(__file__).resolve().parents[3]
    codec = runpy.run_path(
        str(root / "docs/explore/kv-int4-fp16-scales/reference_codec.py")
    )
    evidence = []
    torch.manual_seed(12345)
    for dim in (128, 256):
        for layout in ("LBHNC", "LHBNC"):
            # Frozen Flash-Next snapshot TP4: D256, Hq6, Hkv1 (replicated KV).
            # Keep D128/GQA4 as additional supported geometry coverage.
            pages, block, heads, tokens = 2, 1024, (1 if dim == 256 else 2), 64
            q_heads = 6 if dim == 256 else 8
            record = dim // 2 + 2 * (dim // 32)
            if layout == "LBHNC":
                logical = torch.full(
                    (pages, 2, block, heads * record),
                    0xA5,
                    dtype=torch.uint8,
                    device="cuda",
                )
            else:
                logical = torch.full(
                    (2, pages, block, heads * record),
                    0xA5,
                    dtype=torch.uint8,
                    device="cuda",
                ).transpose(0, 1)
            cache = _int4_fp16_cache_view(logical, heads, dim)
            assert cache.data_ptr() == logical.data_ptr()
            inputs = torch.randn(
                2, tokens, heads, dim + 8, dtype=torch.float16, device="cuda"
            )[..., :dim]
            slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
            writer(inputs[0], inputs[1], cache, slots)
            query = (
                torch.randn(3, q_heads, dim, dtype=torch.float16, device="cuda") * 0.1
            )
            table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
            cu = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
            lengths = torch.tensor([tokens], dtype=torch.int32, device="cuda")
            out_decode, out_prefill = torch.empty_like(query), torch.empty_like(query)

            def call(
                inputs=inputs,
                cache=cache,
                slots=slots,
                dim=dim,
                out_decode=out_decode,
                out_prefill=out_prefill,
                query=query,
                table=table,
                cu=cu,
                lengths=lengths,
                block=block,
            ):
                writer(inputs[0], inputs[1], cache, slots)
                # Match production: derive typed aliases inside the compiled
                # function, not as differently typed aliasing graph inputs.
                payload_k, payload_v, scale_k, scale_v = _int4_fp16_read_views(
                    cache, dim
                )
                for decode, out in ((True, out_decode), (False, out_prefill)):
                    fa.fa_rdna2_paged_int4_fp16_block(
                        query,
                        payload_k,
                        payload_v,
                        scale_k,
                        scale_v,
                        table,
                        cu,
                        lengths,
                        block,
                        kv_splits=16,
                        decode=decode,
                        out=out,
                    )

            def verify(
                inputs=inputs,
                cache=cache,
                tokens=tokens,
                dim=dim,
                q_heads=q_heads,
                heads=heads,
                query=query,
                out_decode=out_decode,
                out_prefill=out_prefill,
            ):
                data, words, dense = encode(inputs.cpu(), codec)
                for side in range(2):
                    torch.testing.assert_close(
                        cache[side, 0, :tokens, :, : dim // 2].cpu(),
                        data[side],
                        rtol=0,
                        atol=0,
                    )
                    actual = (
                        cache[side, 0, :tokens, :, dim // 2 :].view(torch.uint16).cpu()
                    )
                    torch.testing.assert_close(actual, words[side], rtol=0, atol=0)
                    assert torch.all(cache[side, 0, tokens:] == 0xA5)
                    assert torch.all(cache[side, 1] == 0xA5)
                key, value = [
                    x.cuda().repeat_interleave(q_heads // heads, dim=1).float()
                    for x in dense
                ]
                scores = torch.einsum("qhd,khd->hqk", query.float(), key) * dim**-0.5
                positions = torch.arange(tokens, device="cuda")
                mask = (
                    positions[None, :]
                    <= (tokens - 3 + torch.arange(3, device="cuda"))[:, None]
                )
                scores.masked_fill_(~mask[None, :, :], float("-inf"))
                expected = torch.einsum(
                    "hqk,khd->qhd", scores.softmax(-1), value
                ).half()
                errors = []
                for output in (out_decode, out_prefill):
                    assert torch.isfinite(output).all()
                    torch.testing.assert_close(output, expected, atol=0.003, rtol=0.003)
                    errors.append(float((output - expected).abs().max()))
                return errors

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call()
            torch.cuda.current_stream().wait_stream(stream)
            errors = [verify()]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for _ in range(2):
                inputs.mul_(0.9).add_(0.01)
                query.mul_(0.8).add_(0.01)
                graph.replay()
                errors.append(verify())
            print(
                f"D{dim} {layout} registered bytes/eager/changed CUDA graphs PASS",
                flush=True,
            )
            compiled = torch.compile(call, fullgraph=True)
            compiled()
            errors.append(verify())
            inputs.mul_(0.9).add_(0.01)
            query.mul_(0.8).add_(0.01)
            compiled()
            errors.append(verify())
            evidence.append(
                {
                    "dim": dim,
                    "layout": layout,
                    "query_heads": q_heads,
                    "kv_heads": heads,
                    "logical_strides": list(logical.stride()),
                    "writer_strides": list(cache.stride()),
                    "changed_inputs_decode_prefill_maxabs": errors,
                    "status": "PASS",
                }
            )
    result = {
        "status": "PASS",
        "registered_schema_count": len(schemas),
        "tags": tags,
        "actual_tp4_attention_page_bytes": {
            "fp16": dense_spec.real_page_size_bytes,
            "packed": packed_spec.real_page_size_bytes,
        },
        "writer_fullrange_gate": legacy_writer_gate,
        "layouts": evidence,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "vllm").glob("*.so")
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
