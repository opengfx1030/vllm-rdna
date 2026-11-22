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


_MTP_CONFIG = {
    "hidden_size": 2304,
    "num_hidden_layers": 3,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9216,
    "max_position_embeddings": 1048576,
    "torch_dtype": "bfloat16",
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "local_layer_ids": [0, 2],
    "sliding_window_size": 1024,
    "block_size": 8,
    "markov_rank": 256,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [5, 23, 41, 48],
}


def test_draft_config_asks_the_target_for_the_entry_taps_only():
    """The draft reads four taps, but the runner taps only the three layer entries: the
    target appends its final output itself, and the last id (the number of layer
    visits) is not a layer the target could tap."""
    target = cfgs.Inkling2LoopConfig.from_dict(
        {"model_type": "inkling2_loop", "mtp_config": _MTP_CONFIG}
    )
    draft = cfgs.Inkling2DSparkConfig.from_target(target)
    assert target.text_config.total_num_hidden_layers == 48
    assert draft.target_layer_ids == [5, 23, 41, 48]
    assert draft.eagle_aux_hidden_state_layer_ids == [5, 23, 41]
