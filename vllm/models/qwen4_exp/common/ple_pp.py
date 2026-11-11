# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel placement check for the Qwen4Exp n-gram PLE layers.

PLE consumes the raw ``input_ids`` and the n-gram context, which only the
first pipeline stage receives (the V1 runner injects them there; the PLE CPU
offload worker is spawned by global rank 0 and expects registrations from one
stage). PP > 1 is therefore safe exactly when every PLE layer lives on the
first stage. Upstream rejects PP > 1 outright; on RDNA we allow it when the
layer partition keeps the PLE layers on stage 0 (3x V620 PP=3, 8x V620
TP=4 x PP=2 across two PCIe switches).
"""

from collections.abc import Iterable

PLE_PP_UNSUPPORTED = (
    "Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 "
    "because non-first pipeline ranks do not receive the raw input_ids "
    "it needs. Please run with PP=1."
)


def ple_layers_outside_first_stage(
    ple_layer_ids: Iterable[int], first_stage_end: int
) -> list[int]:
    """PLE layer ids whose decoder layer is not on the first PP stage.

    ``ple_layer_ids`` are 1-based (the PLE module lives in decoder layer
    ``id - 1``, see ``Qwen4ExpDecoderLayer``); ``first_stage_end`` is the
    exclusive end layer of stage 0.
    """
    return sorted({int(i) for i in ple_layer_ids if int(i) - 1 >= first_stage_end})


def ple_pp_unsupported_reason(
    text_config, pipeline_parallel_size: int, *, allow_pp: bool
) -> str | None:
    """Return why PLE cannot run at this PP size, or None if it can."""
    ple_layer_ids = getattr(text_config, "ple_layer_ids", None)
    if not ple_layer_ids or pipeline_parallel_size <= 1:
        return None
    if not allow_pp:
        return PLE_PP_UNSUPPORTED

    from vllm.distributed.utils import get_pp_indices

    _, first_stage_end = get_pp_indices(
        text_config.num_hidden_layers, 0, pipeline_parallel_size
    )
    outside = ple_layers_outside_first_stage(ple_layer_ids, first_stage_end)
    if not outside:
        return None
    return (
        "Qwen4Exp N-gram PLE layers must sit on the first pipeline stage "
        f"(layers 0..{first_stage_end - 1} with pipeline_parallel_size="
        f"{pipeline_parallel_size}), but PLE layer ids {outside} fall on later "
        "stages. Set VLLM_PP_LAYER_PARTITION so the first stage holds decoder "
        f"layer {max(outside) - 1}."
    )


def rdna_ple_pp_unsupported_reason(
    text_config, pipeline_parallel_size: int
) -> str | None:
    """`ple_pp_unsupported_reason` with PP allowed only on RDNA."""
    if not getattr(text_config, "ple_layer_ids", None) or pipeline_parallel_size <= 1:
        return None
    from vllm.platforms.rdna import on_rdna_family

    return ple_pp_unsupported_reason(
        text_config, pipeline_parallel_size, allow_pp=on_rdna_family()
    )
