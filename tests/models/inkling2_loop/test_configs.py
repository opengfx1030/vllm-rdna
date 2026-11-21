# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocabulary handling of ``vllm.models.inkling2_loop.configs``.

The checkpoint pads ``embed`` / ``unembed`` to ``vocab_size`` rows (201024 real,
512 tiny) while only ``unpadded_vocab_size`` ids exist (200058 / 500).  vLLM sizes
logits, sampler state and token-id validation from ``vocab_size``, so the serving
config exposes the real vocabulary there and keeps the padded table height in
``padded_vocab_size``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "_i2l_configs_vocab_under_test", _REPO / "vllm/models/inkling2_loop/configs.py"
)
assert _spec is not None and _spec.loader is not None
cfgs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cfgs
_spec.loader.exec_module(cfgs)


def _triple(text) -> tuple[int, int, int]:
    return text.vocab_size, text.unpadded_vocab_size, text.padded_vocab_size


def test_real_checkpoint_defaults():
    assert _triple(cfgs.Inkling2LoopTextConfig()) == (200058, 200058, 201024)


def test_tiny_checkpoint_values():
    text = cfgs.Inkling2LoopTextConfig(vocab_size=512, unpadded_vocab_size=500)
    assert _triple(text) == (500, 500, 512)


def test_text_config_round_trip_is_idempotent():
    text = cfgs.Inkling2LoopTextConfig(vocab_size=512, unpadded_vocab_size=500)
    again = cfgs.Inkling2LoopTextConfig.from_dict(text.to_dict())
    assert _triple(again) == (500, 500, 512)
    assert _triple(cfgs.Inkling2LoopTextConfig.from_dict(again.to_dict())) == (
        500,
        500,
        512,
    )


def test_composite_config_exposes_the_real_vocabulary():
    raw = {
        "model_type": "inkling2_loop",
        "text_config": {
            "vocab_size": 512,
            "unpadded_vocab_size": 500,
            "hidden_size": 256,
        },
    }
    cfg = cfgs.Inkling2LoopConfig.from_dict(raw)
    assert cfg.vocab_size == 500
    assert _triple(cfg.get_text_config()) == (500, 500, 512)
    again = cfgs.Inkling2LoopConfig.from_dict(cfg.to_dict())
    assert again.vocab_size == 500
    assert again.get_text_config().padded_vocab_size == 512
