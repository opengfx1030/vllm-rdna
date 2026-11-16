# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import check_logprobs_close
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.platforms import current_platform

from ...utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    get_spec_decode_metric_value,
    get_test_prompts,
)
from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
    ],
    ids=["qwen3_5-hybrid"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graphs"])
@single_gpu_only
def test_mtp_multimodal_chunked_prefill_coordinates(
    monkeypatch: pytest.MonkeyPatch, vllm_runner, image_assets, enforce_eager: bool
):
    """Real MTP loading and mixed image/text batches preserve target outputs."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    image = image_assets[0].pil_image
    image_marker = "<|vision_start|><|image_pad|><|vision_end|>"
    prompts = [
        f"{image_marker} Describe the image.",
        "A note before the image. " * 80 + f"{image_marker} Describe the image.",
        "Repeat the word hello ten times:",
    ]
    results = []
    for speculate in (False, True):
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B-Base",
            dtype="half",
            block_size=None,
            max_model_len=2048,
            max_num_seqs=4,
            max_num_batched_tokens=128,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs={"min_pixels": 3136, "max_pixels": 50176},
            speculative_config=(
                {"method": "mtp", "num_speculative_tokens": 3}
                if speculate else None
            ),
            disable_log_stats=False,
            seed=0,
        ) as runner:
            mixed = runner.generate_greedy_logprobs(
                prompts, max_tokens=16, num_logprobs=5, images=[image, image, None]
            )
            # Reuse freed request slots and a smaller captured batch.
            recycled = runner.generate_greedy_logprobs(
                ["Count from one to ten:"], max_tokens=16, num_logprobs=5
            )
            assert all(output[0] for output in mixed + recycled)
            results.append(mixed + recycled)
            if speculate:
                metrics = runner.llm.get_metrics()
                assert get_spec_decode_metric_value(
                    metrics, "vllm:spec_decode_num_draft_tokens"
                ) > 0
                assert get_spec_decode_metric_value(
                    metrics, "vllm:spec_decode_num_accepted_tokens"
                ) > 0
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="target",
        name_1="multimodal MTP",
    )


@single_gpu_only
def test_mtp_draft_lm_head_quantization(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """A quantized draft copy of the shared lm_head only changes drafting:
    greedy outputs match stock MTP and acceptance stays within 10%."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    results = {}
    # NVFP4 needs a CUDA W4A16 kernel.
    quants = (None, "fp8", "nvfp4") if current_platform.is_cuda() else (None, "fp8")
    for quant in quants:
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B-Base",
            block_size=None,
            enable_chunked_prefill=None,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "video": 0},
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 3,
                "draft_lm_head_quantization": quant,
            },
            disable_log_stats=False,
        ) as runner:
            outputs = runner.llm.chat(prompts, sampling_config)
            results[quant] = (outputs, compute_acceptance_len(runner.llm.get_metrics()))

    ref_outputs, ref_al = results.pop(None)
    for quant, (outputs, al) in results.items():
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            # Not bit-exact: a different acceptance length changes the target's
            # verify batch shapes, which can flip a near-tied greedy token.
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context=f"draft_lm_head_quantization={quant}",
        )
        assert al >= 0.9 * ref_al, (
            f"{quant}: acceptance length {al:.3f} vs {ref_al:.3f}"
        )


@single_gpu_only
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="nvfp4 needs a CUDA W4A16 kernel"
)
def test_mtp_draft_lm_head_quantization_survives_sleep_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """The quantized draft head is derived from the target head: it must come
    back intact after a level 1 sleep and be re-derived after a target weight
    reload, so outputs and acceptance match the run before."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    with vllm_runner(
        "Qwen/Qwen3.5-0.8B-Base",
        block_size=None,
        enable_chunked_prefill=None,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "draft_lm_head_quantization": "nvfp4",
        },
        enable_sleep_mode=True,
        disable_log_stats=False,
    ) as runner:
        llm = runner.llm

        def run():
            before = llm.get_metrics()
            outputs = llm.chat(prompts, sampling_config)
            return outputs, compute_acceptance_len(llm.get_metrics(), before)

        ref_outputs, ref_al = run()

        llm.sleep(level=1)
        llm.wake_up()
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after level 1 sleep",
        )
        assert al == pytest.approx(ref_al, rel=0.03)

        llm.collective_rpc("reload_weights")
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after weight reload",
        )
        assert al == pytest.approx(ref_al, rel=0.03)
