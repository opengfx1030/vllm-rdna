# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCR_VISIBLE_DEVICES → amdsmi handle index.

Kept import-light so CPU tests can load it without initializing HIP.
Ported from leapdragon/vllm-rdna2-qwen (Aron Hsiao).
"""


def amdsmi_index_from_rocr(base: int, rocr: str | None) -> int:
    """Map a torch-visible ordinal through ROCR_VISIBLE_DEVICES.

    amdsmi enumerates every physical GPU and ignores ROCR. Leap measured
    ROCR=1,2,3,4 with a display card at physical 0 looking up the fused-MoE
    JSON as AMD_Radeon_RX_6700_XT instead of AMD_Radeon_Pro_V620.
    """
    if not rocr:
        return base
    ids = [int(x) for x in rocr.split(",") if x.strip()]
    return ids[base] if 0 <= base < len(ids) else base
