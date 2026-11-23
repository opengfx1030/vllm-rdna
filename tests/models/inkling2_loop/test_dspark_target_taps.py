# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tap hook of the Inkling-2 target model: the inputs of the DSpark draft.

The draft reads the AttnRes partial block at the *entry* of global layer visits
``target_layer_ids[:-1]`` (visit index counted over the prelude, every pass of the
loop and the coda) and the final normalized output.  The test runs the serving
model on the tiny checkpoints (CPU, the engine emulation of ``test_paged_model``),
records every layer's input with a forward pre-hook and requires the returned taps to
be exactly those tensors, in visit order, plus the normalized output.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs a transformers build that ships inkling2_loop",
)

from test_paged_model import _model, _prompts, variant  # noqa: E402, F401


def _visits(cfg) -> int:
    """Layer visits of one forward: every layer once, the loop's extra passes."""
    loop_layers = cfg.loop_layer_index_end - cfg.loop_layer_index_start
    return cfg.num_hidden_layers + (cfg.loop_num_passes - 1) * loop_layers


def _run_with_taps(pm, ids: tuple[int, ...]):
    lm = pm.lm
    entries: list[torch.Tensor] = []
    handles = [
        layer.register_forward_pre_hook(
            lambda _m, args: entries.append(args[0].clone())
        )
        for layer in lm.model.layers
    ]
    seen: dict[str, object] = {}
    plain = lm.forward

    def tapped(*args, **kwargs):
        out = plain(*args, **kwargs)
        assert isinstance(out, tuple), "taps were requested"
        seen["hidden"], seen["taps"] = out
        return out[0]

    try:
        lm.set_aux_hidden_state_layers(ids)
        lm.forward = tapped
        pm.step([(0, _prompts()[1])])
    finally:
        del lm.forward
        lm.set_aux_hidden_state_layers(())
        for h in handles:
            h.remove()
    return entries, seen


def test_taps_are_the_entry_block_states_of_the_selected_visits(variant):  # noqa: F811
    pm = _model(variant)
    n_visits = _visits(pm.cfg)
    assert n_visits >= 6
    ids = (0, 3, n_visits - 1)
    entries, seen = _run_with_taps(pm, ids)
    assert len(entries) == n_visits
    taps = seen["taps"]
    assert len(taps) == len(ids) + 1
    for tap, visit in zip(taps, ids):
        assert tap.dtype == torch.float32 and tap.shape == entries[visit].shape
        assert torch.equal(tap, entries[visit]), f"visit {visit}"
    # the last tap is the normalized output the head sees
    assert torch.equal(taps[-1].to(seen["hidden"].dtype), seen["hidden"])
    # different visits carry different states (the loop does not just repeat a block)
    assert not torch.equal(taps[0], taps[1])


def test_plain_forward_is_unchanged_without_taps(variant):  # noqa: F811
    pm = _model(variant)
    lm = pm.lm
    assert lm.supports_eagle3
    assert lm.get_eagle3_default_aux_hidden_state_layers() == (5, 23, 41)
    prompt = _prompts()[0]
    plain = pm.step([(0, prompt)])[0]
    pm.reset()
    entries, seen = _run_with_taps(pm, (1,))
    pm.reset()
    again = pm.step([(0, prompt)])[0]  # the hook was removed: tensors out again
    assert torch.equal(plain, again)
    assert len(seen["taps"]) == 2 and entries


def test_the_served_classes_satisfy_the_runners_eagle3_interface():
    """The runner asks ``supports_eagle3`` (a runtime-checkable protocol that needs
    every member, the ``SupportsEagleBase`` flags included) before it taps the
    target."""
    from vllm.model_executor.models.interfaces import supports_eagle3
    from vllm.models.inkling2_loop.amd import model as model_mod

    assert supports_eagle3(model_mod.Inkling2LoopTextForCausalLM)
    assert supports_eagle3(model_mod.Inkling2LoopForConditionalGeneration)


def test_the_unquantized_draft_does_not_inherit_the_targets_quantization():
    """``mtp.safetensors`` is bf16 next to an NVFP4 target: a draft config that took
    the target's ``modelopt_fp4`` would try to parse a ModelOpt config for it."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

    def spec(model_type: str, text_model_type: str):
        target = SimpleNamespace(
            model="ckpt",
            quantization="modelopt_fp4",
            hf_config=SimpleNamespace(model_type=model_type),
            hf_text_config=SimpleNamespace(model_type=text_model_type),
        )
        return SimpleNamespace(quantization=None, target_model_config=target)

    inkling = spec("inkling2_loop", "inkling2_loop_text")
    assert DSparkSpeculatorConfig.default_model(inkling) == "ckpt"
    assert inkling.quantization is None
    # a DSpark draft that ships inside the quantized target checkpoint still inherits
    other = spec("deepseek_v4", "deepseek_v4")
    assert DSparkSpeculatorConfig.default_model(other) == "ckpt"
    assert other.quantization == "modelopt_fp4"
