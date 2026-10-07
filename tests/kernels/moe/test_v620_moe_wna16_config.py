# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V620 Triton WNA16 fused-MoE tile table from leapdragon/vllm-rdna2-qwen.

E=128 is local experts at EP=4 (512 global). N=640 is unpacked w2 N.
int4_w4a16 is the Flash-Next AWQ expert path. Missing this file makes
get_moe_configs() log the default-tile warning on AMD_Radeon_Pro_V620.
"""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "configs"
    / "E=128,N=640,device_name=AMD_Radeon_Pro_V620,dtype=int4_w4a16.json"
)


def test_v620_int4_w4a16_config_exists_and_decode_tiles_are_skinny():
    assert _CONFIG.is_file()
    cfg = json.loads(_CONFIG.read_text())
    assert cfg["1"]["BLOCK_SIZE_M"] == 1
    assert cfg["8"]["BLOCK_SIZE_M"] == 8
    assert cfg["16"]["BLOCK_SIZE_M"] == 16
    assert cfg["32"]["BLOCK_SIZE_M"] == 16


def test_v620_int4_w4a16_prefill_tiles_use_one_stage():
    cfg = json.loads(_CONFIG.read_text())
    for key in ("512", "1024", "2048", "4096"):
        tile = cfg[key]
        assert tile["BLOCK_SIZE_M"] == 64
        assert tile["BLOCK_SIZE_N"] == 64
        assert tile["BLOCK_SIZE_K"] == 32
        assert tile["num_warps"] == 4
        assert tile["num_stages"] == 1
