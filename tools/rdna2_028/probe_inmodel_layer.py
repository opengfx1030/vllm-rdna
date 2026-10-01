#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-model layer test: real checkpoint tensors, production compressed-tensors
layout, processed by the actual RDNA2W4A16LinearKernel; W4A8 on vs off.

This is the faithful path: it exercises the real AWQ qweight/qzeros/scales
layout the model produces, RDNA2W4A16LinearKernel.process_weights_after_loading
(the AWQ qzeros transpose included), and the W4A8 dispatcher + kernel.
Single GPU.
"""
import os
import sys

import torch

torch.manual_seed(0)

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (  # noqa: E402
    RDNA2W4A16LinearKernel,
)
from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types  # noqa: E402
from safetensors import safe_open  # noqa: E402

MODEL = (
    "/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4"
    "/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea"
)
GROUP = 32


def _find(name):
    for i in range(1, 20):
        fn = f"{MODEL}/model-{i:05d}-of-00005.safetensors"
        if os.path.exists(fn):
            with safe_open(fn, framework="pt") as f:
                if name in f.keys():
                    return f.get_tensor(name)
    raise KeyError(name)


def _build_layer(layer_name):
    wp = _find(layer_name + ".weight_packed").to(torch.int32).cuda()
    ws = _find(layer_name + ".weight_scale").to(torch.float16).cuda()
    wz = _find(layer_name + ".weight_zero_point").to(torch.int32).cuda()
    wsh = _find(layer_name + ".weight_shape").cpu().numpy()
    n, k = int(wsh[0]), int(wsh[1])

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=wp, weight_loader=lambda *a, **kk: None,
            input_dim=1, output_dim=0, packed_dim=1, packed_factor=8,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=ws, weight_loader=lambda *a, **kk: None,
            input_dim=1, output_dim=0,
        ),
    )
    layer.register_parameter(
        "weight_zero_point",
        PackedvLLMParameter(
            data=wz, weight_loader=lambda *a, **kk: None,
            input_dim=1, output_dim=0, packed_dim=0, packed_factor=8,
        ),
    )
    return layer, k, n


def _ensure_tp_group(vcfg):
    from vllm.distributed.parallel_state import get_tp_group

    try:
        get_tp_group()
        return
    except AssertionError:
        pass
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    init_distributed_environment(
        world_size=1, rank=0,
        distributed_init_method="tcp://127.0.0.1:0", local_rank=0,
    )
    ensure_model_parallel_initialized(1, 1)


def run_layer(layer_name, m):
    _, k, n = _build_layer(layer_name)
    x = (0.25 * torch.randn((m, k), device="cuda", dtype=torch.float32)).to(
        torch.float16
    )
    config = MPLinearLayerConfig(
        full_weight_shape=(k, n), partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4, act_type=torch.float16,
        group_size=GROUP, zero_points=True, has_g_idx=False,
    )

    def _apply(w4a8: bool):
        os.environ.pop("VLLM_RDNA2_W4A8_SDOT4", None)
        if w4a8:
            os.environ["VLLM_RDNA2_W4A8_SDOT4"] = "1"
        layer, _, _ = _build_layer(layer_name)  # fresh params per arm
        kernel = RDNA2W4A16LinearKernel(
            config, w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
            w_gidx_param_name=None,
        )
        kernel.process_weights_after_loading(layer)
        return kernel.apply_weights(layer, x)

    out_off = _apply(False)
    out_on = _apply(True)
    den = out_off.float().norm()
    rel = float("nan") if den == 0 else ((out_on.float() - out_off.float()).norm() / den).item()
    print(
        f"[{layer_name.split('.')[-1]}] k={k} n={n} m={m}: "
        f"w4a8_vs_w4a16 rel-L2 = {rel:.5f}  (off norm={den.item():.1f})"
    )


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.3.self_attn.o_proj"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2001
    vcfg = VllmConfig()
    with set_current_vllm_config(vcfg):
        _ensure_tp_group(vcfg)
        run_layer(name, m)


if __name__ == "__main__":
    main()
