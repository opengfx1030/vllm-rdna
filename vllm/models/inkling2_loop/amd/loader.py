# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert storage format of an Inkling-2-Mini checkpoint.

The name mapping and the tensor copies live in ``model.py``
(``Inkling2LoopForConditionalGeneration.load_weights``); the only thing that has to
be known before the modules are built is which layers carry NVFP4 and which MXFP8
routed experts, because the two formats have different parameter shapes.
"""

from __future__ import annotations

from typing import Literal

import regex as re

_EXPERT_W13_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.w13_weight$")

ExpertFormat = Literal["nvfp4", "mxfp8"]


def expert_formats(quant_config: dict, num_layers: int) -> list[ExpertFormat]:
    """Per-layer expert storage (``"nvfp4"`` / ``"mxfp8"``) from the
    ``model.parameter_groups`` of the checkpoint's ``hf_quant_config.json``.

    A group's ``compute_dtype`` names the format of every ``layers.N.mlp.experts
    .w13_weight`` it lists (the real checkpoint is NVFP4 for layers 0-19 and MXFP8
    for 20-23, the tiny checkpoints put MXFP8 on the last layer only).
    """
    formats: dict[int, ExpertFormat] = {}
    for group in quant_config["model"]["parameter_groups"]:
        fmt = group["compute_dtype"]
        if fmt not in ("nvfp4", "mxfp8"):
            continue
        for name in group["tensors"]:
            m = _EXPERT_W13_RE.match(name)
            if m:
                formats[int(m.group(1))] = fmt
    missing = [i for i in range(num_layers) if i not in formats]
    if missing:
        raise ValueError(
            f"hf_quant_config.json lists no nvfp4/mxfp8 experts for layers {missing}"
        )
    return [formats[i] for i in range(num_layers)]


def read_expert_formats(
    model: str, revision: str | None, num_layers: int
) -> list[ExpertFormat]:
    """:func:`expert_formats` of the ``hf_quant_config.json`` next to ``model``."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    cfg = get_hf_file_to_dict("hf_quant_config.json", model, revision)
    if cfg is None:
        raise FileNotFoundError(
            f"{model}: hf_quant_config.json is required to tell the NVFP4 and "
            "MXFP8 expert layers apart"
        )
    return expert_formats(cfg, num_layers)
