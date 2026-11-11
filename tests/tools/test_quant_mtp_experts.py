# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline MTP-expert W4A16 quantizer (CPU only, no vLLM import)."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "quant_mtp_experts", ROOT / "tools/rdna/quant_mtp_experts.py"
)
qm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qm)

GS = 32


def test_quant_matches_minmax_grid():
    torch.manual_seed(0)
    scale = torch.rand(4, 64 // GS) + 0.5
    q = torch.randint(-7, 8, (4, 64)).float()
    # Every group's absmax is 7.5 * scale, on the negative side: the RTN scale
    # (absmax / 7.5) is exact and that element rounds to q = -8.
    w = (q.reshape(4, -1, GS) * scale[..., None]).reshape(4, 64)
    w.view(4, -1, GS)[:, :, 0] = -7.5 * scale
    packed, s = qm.quant_rtn(w, GS)
    assert packed.dtype == torch.int32 and packed.shape == (4, 8)
    assert s.dtype == torch.bfloat16 and s.shape == (4, 2)
    expect = q.clone()
    expect.view(4, -1, GS)[:, :, 0] = -8
    deq = qm.unpack(packed, s, GS)
    exp_w = (expect.reshape(4, -1, GS) * s.float()[..., None]).reshape(4, 64)
    torch.testing.assert_close(deq, exp_w)


def test_nibble_order_low_bits_first():
    w = torch.zeros(1, GS)
    w[0, 0] = 7.0  # 7 * 7.5 / 7 = 7.5 -> clamped to q=7 -> nibble 15
    w[0, 1] = -7.0  # -7.5 rounds to q=-8 -> nibble 0
    w[0, 2] = 3.5  # 3.75 -> q=4 -> nibble 12
    packed, _ = qm.quant_rtn(w, GS)
    word = int(packed[0, 0]) & 0xFFFFFFFF
    assert word & 0xF == 15
    assert (word >> 4) & 0xF == 0
    assert (word >> 8) & 0xF == 12
    assert (word >> 12) & 0xF == 8  # zero -> nibble 8


def test_rejects_bad_k():
    with pytest.raises(ValueError):
        qm.quant_rtn(torch.zeros(2, GS + 8), GS)


def test_ignore_list_keeps_mtp_except_experts():
    ignore = ["lm_head", "re:.*mtp\\..*", "re:.*shared_expert_gate$"]
    out = qm.mtp_ignore_list(ignore)
    assert "re:.*mtp\\..*" not in out
    assert out[:2] == ["lm_head", "re:.*shared_expert_gate$"]
    pattern = out[-1].removeprefix("re:")
    import re

    assert re.match(pattern, "mtp.fc")
    assert re.match(pattern, "mtp.layers.0.self_attn.qkv_proj")
    assert re.match(pattern, "mtp.layers.0.mlp.shared_expert.gate_proj")
    assert not re.match(pattern, "mtp.layers.0.mlp.experts")
    assert not re.match(pattern, "mtp.layers.0.mlp.experts.3.down_proj")
    assert qm.mtp_ignore_list(out) == out  # idempotent


def test_quantize_mtp_experts_layout():
    torch.manual_seed(0)
    e, inter, hidden = 3, 64, 96
    tensors = {
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(e, 2 * inter, hidden),
        "mtp.layers.0.mlp.experts.down_proj": torch.randn(e, hidden, inter),
        "mtp.fc.weight": torch.randn(4, 4),
    }
    out, rel = qm.quantize_mtp_experts(tensors, GS)
    assert "mtp.fc.weight" in out
    assert not any(k.endswith(("gate_up_proj", "experts.down_proj")) for k in out)
    p = "mtp.layers.0.mlp.experts.2."
    assert out[p + "gate_proj.weight_packed"].shape == (inter, hidden // 8)
    assert out[p + "down_proj.weight_packed"].shape == (hidden, inter // 8)
    assert out[p + "up_proj.weight_scale"].shape == (inter, hidden // GS)
    assert out[p + "down_proj.weight_shape"].tolist() == [hidden, inter]
    # up_proj is the second half of gate_up_proj.
    w_up = tensors["mtp.layers.0.mlp.experts.gate_up_proj"][2, inter:]
    deq = qm.unpack(
        out[p + "up_proj.weight_packed"], out[p + "up_proj.weight_scale"], GS
    )
    assert ((deq - w_up).abs().mean() / w_up.abs().mean()) < 0.2
    assert 0.0 < rel < 0.2


def test_end_to_end_derived_checkpoint(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch.manual_seed(0)
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    main_w = torch.randn(32, 64)
    packed, scale = qm.quant_rtn(main_w, GS)
    base = "model.language_model.layers.0.mlp.experts.0.gate_proj."
    safetensors_torch.save_file(
        {base + "weight_packed": packed, base + "weight_scale": scale},
        str(src / "model-00001.safetensors"),
    )
    safetensors_torch.save_file(
        {
            "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 64, 64),
            "mtp.layers.0.mlp.experts.down_proj": torch.randn(2, 64, 32),
            "mtp.fc.weight": torch.randn(8, 8),
        },
        str(src / "model-00002.safetensors"),
    )
    index = {
        "metadata": {},
        "weight_map": {
            base + "weight_packed": "model-00001.safetensors",
            base + "weight_scale": "model-00001.safetensors",
            "mtp.layers.0.mlp.experts.gate_up_proj": "model-00002.safetensors",
            "mtp.layers.0.mlp.experts.down_proj": "model-00002.safetensors",
            "mtp.fc.weight": "model-00002.safetensors",
        },
    }
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    config = {
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "group_size": GS, "symmetric": True}
                }
            },
            "ignore": ["lm_head", "re:.*mtp\\..*"],
        }
    }
    (src / "config.json").write_text(json.dumps(config))
    (src / "tokenizer.json").write_text("{}")

    qm.main([str(src), str(dst)])

    new_index = json.loads((dst / "model.safetensors.index.json").read_text())
    wm = new_index["weight_map"]
    assert wm[base + "weight_packed"] == "model-00001.safetensors"
    assert wm["mtp.fc.weight"] == qm.NEW_SHARD
    assert wm["mtp.layers.0.mlp.experts.1.down_proj.weight_packed"] == qm.NEW_SHARD
    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in wm
    new_cfg = json.loads((dst / "config.json").read_text())
    assert new_cfg["quantization_config"]["ignore"] == [
        "lm_head",
        qm.MTP_IGNORE_EXCEPT_EXPERTS,
    ]
    assert (dst / "tokenizer.json").is_symlink()
    assert (dst / "model-00001.safetensors").is_symlink()
    assert not (dst / "model-00002.safetensors").exists()
