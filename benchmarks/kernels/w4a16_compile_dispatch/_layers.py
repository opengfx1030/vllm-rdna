# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 layer introspection, importable in worker processes.

``LLM.apply_model`` pickles its callable and sends it to newly spawned workers
(the probe runs as ``python -m ...probe``, so a helper defined in ``__main__``
cannot be resolved there). A helper that lives in a real module pickles by
reference on both sides.
"""

from __future__ import annotations


def _collect_w4a16_layers(model) -> list[tuple[str, int, int, bool, int, bool]]:
    from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (
        RDNA2W4A16LinearKernel,
    )
    from vllm.scalar_type import scalar_types

    layers = []
    for name, module in model.named_modules():
        # compressed-tensors keeps the kernel on the scheme (layer.scheme.kernel),
        # the plain AWQ/GPTQ methods on quant_method; check both owners.
        for owner in (
            getattr(module, "quant_method", None),
            getattr(module, "scheme", None),
        ):
            for kernel in vars(owner).values() if owner is not None else ():
                if isinstance(kernel, RDNA2W4A16LinearKernel):
                    k, n = kernel.config.partition_weight_shape
                    is_awq = kernel.config.weight_type == scalar_types.uint4
                    layers.append(
                        (name, k, n, is_awq, kernel.config.group_size,
                         bool(getattr(kernel, "_w4a8", False)))
                    )
    return layers
