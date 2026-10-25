# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the gfx1030 Glm5Next path.

The HIP objects are not built in this environment. These tests lock the
source contract (current stream, no packed bf16 dot, no WMMA/MFMA, no
hardware fp8 conversion) and the SM100 family alias.
"""

import re
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _awq_uint8_qzeros_to_rdna2,
)
from vllm.platforms.interface import DeviceCapability

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

REPO = Path(__file__).resolve().parents[3]
KERNEL_DIR = REPO / "csrc" / "rocm" / "rdna" / "glm5next"
_COMMENT = re.compile(r"/\*.*?\*/|//.*?$", re.S | re.M)


def _code(path: Path) -> str:
    return _COMMENT.sub("", path.read_text())


def test_gfx1030_capability_aliases_sm100_family_bucket():
    cap = DeviceCapability(major=10, minor=3)
    assert cap.to_int() == 103
    assert cap.to_int() // 10 == 100 // 10


def test_rdna_family_check_is_not_sm100(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms import rocm

    monkeypatch.setattr(rocm, "on_gfx10x", lambda: True)
    monkeypatch.setattr(rocm, "on_rdna", lambda: False)
    monkeypatch.setattr(
        rocm.RocmPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    assert rocm.RocmPlatform.is_device_capability_family(100) is False
    assert rocm.RocmPlatform.is_device_capability_family(90) is False

    monkeypatch.setattr(rocm, "on_gfx10x", lambda: False)
    monkeypatch.setattr(rocm, "on_rdna", lambda: True)
    monkeypatch.setattr(
        rocm.RocmPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(11, 0)),
    )
    assert rocm.RocmPlatform.is_device_capability_family(110) is False
    assert rocm.RocmPlatform.is_device_capability_family(120) is False

    # gfx12 aliases SM120 the same way gfx1030 aliases SM100.
    monkeypatch.setattr(
        rocm.RocmPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(12, 0)),
    )
    assert rocm.RocmPlatform.is_device_capability_family(120) is False
    assert rocm.RocmPlatform.is_device_capability_family(100) is False

    # CDNA keeps the generic family test.
    monkeypatch.setattr(rocm, "on_gfx10x", lambda: False)
    monkeypatch.setattr(rocm, "on_rdna", lambda: False)
    monkeypatch.setattr(
        rocm.RocmPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    assert rocm.RocmPlatform.is_device_capability_family(90) is True
    assert rocm.RocmPlatform.is_device_capability_family(100) is False


def test_glm5next_kernels_match_gfx1030_contract():
    sources = sorted(KERNEL_DIR.glob("*.cu"))
    assert sources, f"no kernels in {KERNEL_DIR}"
    cmake = (REPO / "CMakeLists.txt").read_text()
    gfx1030 = cmake.split('if(VLLM_GPU_ARCHES MATCHES "gfx1030")', 1)[1]
    gfx1030 = gfx1030.split("endif()", 1)[0]
    gfx1100 = cmake.split('if(VLLM_GPU_ARCHES MATCHES "gfx1100")', 1)[1]
    gfx1100 = gfx1100.split("endif()", 1)[0]
    for path in sources:
        code = _code(path)
        assert "at::cuda::getCurrentCUDAStream()" in code
        assert "getCurrentHIPStream" not in code
        assert "hip/hip_fp8.h" not in code
        assert "v_wmma" not in code
        assert "v_mfma" not in code
        assert "v_dot2_f32_bf16" not in code
        assert "0xffffffffu" not in code
        if "__shfl_xor_sync" in code:
            assert "0xffffffffffffffffULL" in code
            assert "__shfl_xor(" not in code
        for launch in re.findall(r"<<<[^;]*?>>>", code):
            assert "stream" in launch, path.name
        rel = path.relative_to(REPO).as_posix()
        assert rel in gfx1030
        assert rel not in gfx1100


def test_awq_moe_zeros_apply_gptqv1_offset():
    # [E, N/2, G] = [1, 4, 1], columns 1..8 packed low-nibble-even.
    raw = torch.tensor([0x21, 0x43, 0x65, 0x87], dtype=torch.uint8)
    qz = raw.view(1, 4, 1)
    packed = _awq_uint8_qzeros_to_rdna2(qz)
    assert packed.shape == (1, 1, 1)
    assert packed.dtype == torch.int32
    expected = 0
    for nibble in range(8):
        expected |= nibble << (4 * nibble)
    assert int(packed.reshape(-1)[0].item()) == expected


def test_kda_slot_rule_matches_null_and_pad():
    """Slot 0 is NULL_BLOCK_ID and -1 is PAD_SLOT_ID. Both stay empty."""
    from vllm.models.glm5next.amd.rdna.ops import glm5_kda_slot_is_empty
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

    assert NULL_BLOCK_ID == 0
    assert PAD_SLOT_ID == -1
    num_slots = 4
    assert glm5_kda_slot_is_empty(0, num_slots) is True
    assert glm5_kda_slot_is_empty(PAD_SLOT_ID, num_slots) is True
    assert glm5_kda_slot_is_empty(1, num_slots) is False
    assert glm5_kda_slot_is_empty(num_slots, num_slots) is True

    recurrent = _code(KERNEL_DIR / "glm5_kda_recurrent_rdna2.cu")
    decode = _code(KERNEL_DIR / "glm5_kda_decode_rdna2.cu")
    for code in (recurrent, decode):
        assert "glm5_kda_slot_is_empty" in code
        assert "slot < 0" not in code
        assert "slot <= 0" not in code
    header = (KERNEL_DIR / "glm5_kda_slot.cuh").read_text()
    assert "slot <= 0" in header


def test_kda_hip_decode_is_opt_in(monkeypatch: pytest.MonkeyPatch):
    from vllm.models.glm5next.common import kda

    monkeypatch.delenv("VLLM_GLM5_KDA_HIP", raising=False)
    assert kda._glm5_kda_hip_enabled() is False
    monkeypatch.setenv("VLLM_GLM5_KDA_HIP", "0")
    assert kda._glm5_kda_hip_enabled() is False
    monkeypatch.setenv("VLLM_GLM5_KDA_HIP", "1")
    assert kda._glm5_kda_hip_enabled() is True


def test_rdna2_swiglu_limit_matches_shared_expert_clamp():
    from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
        _swiglu_split,
    )

    limit = 10.0
    x = torch.tensor([[12.0, -3.0, 11.0, -12.0, 1.0, 4.0]], dtype=torch.float32)
    got = _swiglu_split(x, limit)
    gate, up = x.chunk(2, dim=-1)
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(got, expected)
    unclamped = _swiglu_split(x, None)
    assert not torch.equal(got, unclamped)


_W4 = {
    "format": "pack-quantized",
    "weights": {"num_bits": 4, "group_size": 128},
}


class _CtConfig:
    def __init__(self, ignore, targets):
        self.ignore = ignore
        self.target_scheme_map = {target: dict(_W4) for target in targets}
        self.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}


class _AwqConfig:
    def __init__(self, skipped, lm_head=False):
        self.modules_to_not_convert = skipped
        self.lm_head_quantized = lm_head
        self.packed_modules_mapping = {}


def test_dense_leftovers_follow_the_w4a16_checkpoint():
    """wtdcode/GLM-5.3-Flash-AWQ-W4A16 ignores these and quantizes experts."""
    from vllm.models.glm5next.common.dense_leftovers import (
        glm5_leftover_is_quantized,
        require_glm5_dense_leftover,
    )

    ignore = [
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*visual\\..*",
        "re:.*self_attn\\..*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.gate$",
        "re:.*layers\\.(0|1|2)\\.mlp\\..*",
        "re:.*layers\\.45\\..*",
    ]
    cfg = _CtConfig(ignore, ["Linear"])
    dense = [
        (
            "model.layers.4.mlp.shared_experts.gate_up_proj",
            "MergedColumnParallelLinear",
        ),
        ("model.layers.4.mlp.shared_experts.down_proj", "RowParallelLinear"),
        ("model.layers.4.mlp.gate", "GateLinear"),
        ("lm_head", "ParallelLMHead"),
        ("language_model.lm_head", "ParallelLMHead"),
        ("model.embed_tokens", "VocabParallelEmbedding"),
        ("model.layers.0.self_attn.in_proj_qkvbfg_a", "MergedColumnParallelLinear"),
        ("model.layers.3.self_attn.indexer.wq_b", "ReplicatedLinear"),
        (
            "model.layers.3.self_attn.indexer.wk_weights_proj",
            "MergedColumnParallelLinear",
        ),
    ]
    for prefix, cls_name in dense:
        assert glm5_leftover_is_quantized(cfg, prefix, cls_name) is False
        require_glm5_dense_leftover(cfg, prefix, "leftover", cls_name)

    # Routed experts are the quantized target and are not leftovers.
    expert = "model.layers.4.mlp.experts.0.gate_proj"
    assert glm5_leftover_is_quantized(cfg, expert, "ReplicatedLinear") is True

    quantized = _CtConfig(["re:.*visual\\..*"], ["Linear"])
    with pytest.raises(ValueError, match="shared expert"):
        require_glm5_dense_leftover(
            quantized,
            "model.layers.4.mlp.shared_experts.down_proj",
            "shared expert",
            "RowParallelLinear",
        )

    awq_dense = _AwqConfig(
        ["shared_experts", "lm_head", "embed_tokens", "self_attn", "gate"]
    )
    assert (
        glm5_leftover_is_quantized(
            awq_dense,
            "model.layers.4.mlp.shared_experts.gate_up_proj",
            "MergedColumnParallelLinear",
        )
        is False
    )
    awq_q = _AwqConfig(["lm_head"], lm_head=False)
    assert (
        glm5_leftover_is_quantized(
            awq_q,
            "model.layers.4.mlp.shared_experts.down_proj",
            "RowParallelLinear",
        )
        is True
    )
    awq_head = _AwqConfig([], lm_head=True)
    assert glm5_leftover_is_quantized(awq_head, "lm_head", "ParallelLMHead") is True
    assert glm5_leftover_is_quantized(awq_dense, "lm_head", "ParallelLMHead") is False


def test_sparse_mla_metadata_skips_aiter_for_rope_free_non_fp8_models():
    """GLM-5.3-Flash (NoPE latent, non-fp8 KV) never reaches the AITER decode,
    so the builder must not import aiter for it; DeepSeek-V3.2 shapes still do.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        _use_rocm_sparse_triton,
        sparse_mla_needs_aiter_metadata,
    )

    assert sparse_mla_needs_aiter_metadata("auto", 512, 0) is False
    assert sparse_mla_needs_aiter_metadata("float16", 512, 0) is False
    assert sparse_mla_needs_aiter_metadata("fp8", 512, 0) is True
    assert sparse_mla_needs_aiter_metadata("auto", 512, 64) is True
    for kv_dtype, rope in (("auto", 0), ("fp8", 0), ("auto", 64), ("fp8_e4m3", 64)):
        triton = _use_rocm_sparse_triton(
            kv_cache_dtype=kv_dtype,
            head_size=512 + rope,
            kv_lora_rank=512,
            num_prefills=1,
            num_decodes=0,
            num_decode_tokens=0,
            max_query_len=1,
        )
        assert triton is (not sparse_mla_needs_aiter_metadata(kv_dtype, 512, rope))


def test_kpool_indexer_hip_path_opens_on_rdna_without_aiter(
    monkeypatch: pytest.MonkeyPatch,
):
    """The AMD indexer module itself registers a CustomOp and cannot be
    imported next to the NVIDIA one, so the gate lives in ``rdna.ops``.
    """
    import vllm.platforms.rdna as rdna
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.glm5next.amd.rdna.ops import hip_sparse_indexer_available

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", classmethod(lambda cls: False))
    monkeypatch.setattr(rdna, "_ON_RDNA_FAMILY", True)
    assert hip_sparse_indexer_available() is True
    monkeypatch.setattr(rdna, "_ON_RDNA_FAMILY", False)
    assert hip_sparse_indexer_available() is False
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", classmethod(lambda cls: True))
    assert hip_sparse_indexer_available() is True
