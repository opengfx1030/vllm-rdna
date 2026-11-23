# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp RDNA2 MonoKernel switch (``--monokernel``): CPU checks.

The flag and its env alias set ``kernel_config.monokernel``, off by default;
every deployment the kernels do not cover is refused at startup with
``--monokernel <reason>``; the switch adds no parameters and leaves the loaded
values alone.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import KernelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.models.qwen4_exp.amd import mono_decode_rdna2 as md
from vllm.models.qwen4_exp.amd.mono.rdna2 import OPTS, Options
from vllm.utils.argparse_utils import FlexibleArgumentParser

FLASH_NEXT = (
    "/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/"
    "snapshots/0939125b929543a783ce700c90e36dd1a575c00c"
)


def _config(
    monokernel=True,
    arch="Qwen4ExpForConditionalGeneration",
    pp=1,
    knobs=None,
    cache_dtype=None,
):
    extra = {} if knobs is None else {"rocm_mono_decode": knobs}
    cfg = SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        model_config=SimpleNamespace(architectures=[arch]),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        additional_config=extra,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    if cache_dtype is not None:
        cfg.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    return cfg


@pytest.fixture
def deployment(monkeypatch):
    """A deployment the kernels take: gfx10x, every op built."""
    from vllm import envs

    monkeypatch.setattr(md, "_on_gfx10x", lambda: True)
    monkeypatch.setattr(md, "ops_refusal", lambda: None)
    monkeypatch.setattr(envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    saved = Options()
    saved.assign(OPTS)
    yield
    OPTS.assign(saved)


# ---- the switch -----------------------------------------------------------


def test_off_by_default():
    assert KernelConfig().monokernel is False
    assert md.options(_config(monokernel=False)) is None


def test_flag_parses_into_engine_args():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    assert parser.parse_args([]).monokernel is False
    assert parser.parse_args(["--monokernel"]).monokernel is True
    assert EngineArgs.monokernel is False  # the dataclass default


def test_flag_turns_on_every_part_with_the_validated_defaults():
    o = md.options(_config())
    assert o is not None
    assert (o.k1, o.k2, o.gdn, o.qsa, o.timing) == (True, True, True, True, False)
    assert (o.k1_max_rows, o.k2_max_rows, o.gdn_max_rows) == (24, 8, 64)


def test_knobs_dict_and_unknown_keys():
    o = md.options(_config(knobs={"k2_max_rows": 4, "gdn": False, "k1_max_rows": 99}))
    assert o is not None
    assert (o.k2_max_rows, o.gdn, o.k1_max_rows) == (4, False, 24)
    with pytest.raises(ValueError, match="unknown rocm_mono_decode keys"):
        md.options(_config(knobs={"rdna_mono_ffn": True}))
    # the knobs are only read with the switch on
    assert md.options(_config(monokernel=False, knobs={"bogus": 1})) is None


def test_mtp_drafter_is_opt_in_and_has_independent_row_caps():
    defaults = md.options(_config())
    assert defaults is not None
    assert not defaults.mtp_drafter
    assert defaults.mtp_k2_max_rows == 24
    opts = md.options(
        _config(
            knobs={
                "mtp_drafter": True,
                "mtp_k1_max_rows": 32,
                "mtp_k2_max_rows": 3,
            }
        )
    )
    assert opts is not None
    assert opts.mtp_drafter
    assert (opts.mtp_k1_max_rows, opts.mtp_k2_max_rows) == (24, 3)


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"k1": True, "k2": False},
        {"k1": False, "k2": True},
        {"k1": False, "k2": False, "mtp_drafter": True},
    ],
    ids=["default-target", "target-k1-only", "target-k2-only", "drafter-only"],
)
def test_timeout_nan_failstop_is_required_for_mono_kernels(
    monkeypatch, deployment, knobs
):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    cfg = _config(knobs=knobs)
    with pytest.raises(
        ValueError,
        match=(
            "requires VLLM_RAISE_ON_LOGIT_NANS=1.*"
            "fail-stop before generated tokens are published"
        ),
    ):
        md.install(cfg)


@pytest.mark.parametrize(
    "knobs",
    [{}, {"k1": False, "k2": False, "mtp_drafter": True}],
    ids=["default-target", "drafter-only"],
)
def test_timeout_nan_failstop_accepts_mono_kernels_when_enabled(
    monkeypatch, deployment, knobs
):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    # Stop after the guard so this CPU test does not import/register GPU ops.
    monkeypatch.setattr(md, "ops_refusal", lambda: "guard passed")
    with pytest.raises(ValueError, match="^--monokernel guard passed\\.$"):
        md.install(_config(knobs=knobs))


def test_timeout_nan_failstop_is_not_required_when_target_kernels_are_disabled(
    monkeypatch, deployment
):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    # The persistent target kernels are disabled, so install reaches the later
    # mocked ops check without requiring the global NaN fail-stop setting.
    monkeypatch.setattr(md, "ops_refusal", lambda: "guard passed")
    with pytest.raises(ValueError, match="^--monokernel guard passed\\.$"):
        md.install(_config(knobs={"k1": False, "k2": False}))


def test_timeout_nan_failstop_is_not_required_without_monokernel(
    monkeypatch, deployment
):
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    assert md.install(_config(monokernel=False)) is None


def _engine_config(monkeypatch, *args, env=None):
    from vllm.platforms import current_platform

    if not current_platform.is_rocm() or not os.path.isdir(FLASH_NEXT):
        pytest.skip("needs ROCm and the Flash-Next checkpoint")
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("needs a gfx10x GPU")
    if env is not None:
        monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", env)
    else:
        monkeypatch.delenv("VLLM_ROCM_MONO_DECODE", raising=False)
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = parser.parse_args(
        [
            "--model",
            FLASH_NEXT,
            "--max-model-len",
            "4096",
            "--dtype",
            "float16",
            "--language-model-only",
            *args,
        ]
    )
    return EngineArgs.from_cli_args(ns).create_engine_config()


@pytest.mark.parametrize(
    "args, env, expected",
    [((), None, False), (("--monokernel",), None, True), ((), "1", True)],
    ids=["off", "flag", "env-alias"],
)
def test_flag_and_env_alias_reach_kernel_config(monkeypatch, args, env, expected):
    """End to end through create_engine_config (API-server side): the flag and
    VLLM_ROCM_MONO_DECODE=1 both set kernel_config.monokernel."""
    assert _engine_config(monkeypatch, *args, env=env).kernel_config.monokernel is (
        expected
    )


# ---- refusals -------------------------------------------------------------


@pytest.mark.parametrize(
    "patch, cfg, reason",
    [
        ({"_on_gfx10x": lambda: False}, {}, "needs a gfx10x"),
        ({}, {"arch": "Qwen3NextForCausalLM"}, "supports Qwen4Exp only"),
        ({}, {"pp": 2}, "pipeline parallelism"),
        ({"ops_refusal": lambda: "needs the _rocm_C ops ['x']"}, {}, "_rocm_C ops"),
    ],
    ids=["not-gfx10x", "not-qwen4exp", "pp", "ops"],
)
def test_unsupported_deployment_is_refused(monkeypatch, deployment, patch, cfg, reason):
    for name, value in patch.items():
        monkeypatch.setattr(md, name, value)
    with pytest.raises(ValueError, match=rf"^--monokernel .*{reason}.*\.$"):
        md.install(_config(**cfg))


def test_mtp_drafter_config_is_accepted(deployment):
    """The MTP drafter's own VllmConfig (arch Qwen4ExpMTP) carries the flag."""
    md.check_config(_config(arch="Qwen4ExpMTP"))
    assert md.config_refusal(_config(arch="Qwen4ExpMTP")) is None


@pytest.mark.parametrize("raw", [None, "1"], ids=["unset", "explicitly-enabled"])
def test_api_config_enables_nan_failstop_for_supported_monokernel(
    monkeypatch, deployment, raw
):
    if raw is None:
        monkeypatch.delenv("VLLM_RAISE_ON_LOGIT_NANS", raising=False)
    else:
        monkeypatch.setenv("VLLM_RAISE_ON_LOGIT_NANS", raw)

    md.check_config(_config())

    assert os.environ["VLLM_RAISE_ON_LOGIT_NANS"] == "1"


def test_api_config_respects_explicit_nan_failstop_opt_out(monkeypatch, deployment):
    monkeypatch.setenv("VLLM_RAISE_ON_LOGIT_NANS", "0")

    with pytest.raises(
        ValueError,
        match="explicitly setting it to 0 disables fail-stop",
    ):
        md.check_config(_config())
    assert os.environ["VLLM_RAISE_ON_LOGIT_NANS"] == "0"


def test_api_config_does_not_set_nan_failstop_without_protected_mono_parts(
    monkeypatch, deployment
):
    monkeypatch.delenv("VLLM_RAISE_ON_LOGIT_NANS", raising=False)

    md.check_config(_config(knobs={"k1": False, "k2": False}))

    assert "VLLM_RAISE_ON_LOGIT_NANS" not in os.environ


def test_api_config_does_not_set_nan_failstop_without_monokernel(
    monkeypatch, deployment
):
    monkeypatch.delenv("VLLM_RAISE_ON_LOGIT_NANS", raising=False)

    md.check_config(_config(monokernel=False))

    assert "VLLM_RAISE_ON_LOGIT_NANS" not in os.environ


def test_unsupported_gpu_does_not_set_nan_failstop(monkeypatch, deployment):
    monkeypatch.setattr(md, "_on_gfx10x", lambda: False)
    monkeypatch.delenv("VLLM_RAISE_ON_LOGIT_NANS", raising=False)

    with pytest.raises(ValueError, match="^--monokernel needs a gfx10x"):
        md.check_config(_config())
    assert "VLLM_RAISE_ON_LOGIT_NANS" not in os.environ


@pytest.mark.parametrize("cache_dtype", ["auto", "float16", "bfloat16"])
def test_fp16_kv_cache_is_accepted(deployment, cache_dtype):
    """The QSA kernels read the main KV cache as FP16/BF16 words."""
    assert md.config_refusal(_config(cache_dtype=cache_dtype)) is None


@pytest.mark.parametrize(
    "cache_dtype", ["int4_fp16_block", "fp8", "fp8_e4m3", "int8_per_token_head"]
)
def test_quantized_kv_cache_is_refused(deployment, cache_dtype):
    """A packed or quantized KV cache has a record layout the mono QSA kernels
    do not read: refuse it at the switch instead of reading garbage."""
    reason = md.config_refusal(_config(cache_dtype=cache_dtype))
    assert reason is not None
    assert "FP16 or BF16 KV cache" in reason
    assert cache_dtype in reason
    with pytest.raises(ValueError, match="^--monokernel .*KV cache.*\\.$"):
        md.install(_config(cache_dtype=cache_dtype))


def test_missing_ops_are_named(monkeypatch):
    monkeypatch.setattr(md, "REQUIRED_OPS", ("mono_does_not_exist_rdna2",))
    reason = md.ops_refusal()
    assert reason is not None
    assert "mono_does_not_exist_rdna2" in reason


def test_api_server_check_leaves_other_gpus_and_models_alone(monkeypatch):
    monkeypatch.setattr(md, "_on_gfx10x", lambda: False)
    md.check_config(_config(arch="DeepseekV41ForCausalLM"))  # another MonoKernel
    with pytest.raises(ValueError, match="^--monokernel needs a gfx10x"):
        md.check_config(_config())


class CompressedTensorsWNA16RDNA2MoEMethod:  # name-matched stand-in
    def __init__(self, symmetric=True):
        self.symmetric = symmetric


class _Experts(nn.Module):
    def __init__(self, qm):
        super().__init__()
        self.quant_method = qm
        self.w13_weight_packed = nn.Parameter(
            torch.randint(0, 99, (4, 8, 16), dtype=torch.int32), requires_grad=False
        )


class _Layer(nn.Module):
    """The parts of a Qwen4Exp decoder layer the switch looks at."""

    def __init__(self, qm):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = _Experts(qm)
        self.mlp.gate = nn.Linear(16, 4, bias=False)
        self.attn_hyper_connection = nn.Linear(8, 8)
        self.mlp_hyper_connection = nn.Linear(8, 8)


@pytest.mark.parametrize(
    "qm, reason",
    [
        (SimpleNamespace(), "got SimpleNamespace"),
        (CompressedTensorsWNA16RDNA2MoEMethod(symmetric=False), "asymmetric"),
    ],
    ids=["other-method", "asymmetric"],
)
@pytest.mark.parametrize("w4a8", [False, True], ids=["w4a16", "w4a8"])
def test_moe_not_symmetric_w4a16_rdna2_is_refused(
    monkeypatch, deployment, qm, reason, w4a8
):
    """W4A8 is not a refusal by itself (test_amd_mono_rdna2_w4a8.py) and does
    not hide the ones for another method or asymmetric zeros."""
    import vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe as moe

    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: w4a8)
    with pytest.raises(ValueError, match=rf"^--monokernel needs the MoE .*{reason}"):
        md.layer_hook(_Layer(qm), _config(), "model.layers.3", True)


# ---- parameters -----------------------------------------------------------


def _state(m: nn.Module):
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_switch_adds_no_parameters_and_keeps_values(monkeypatch, deployment):
    """named_parameters() and the loaded values are the same with the switch
    on or off: the per-layer MonoKernel state lives in the forward context,
    not in the module tree."""
    import vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe as moe

    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: False)
    torch.manual_seed(0)
    off_layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    on_layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    on_layer.load_state_dict(off_layer.state_dict())

    off_cfg, on_cfg = _config(monokernel=False), _config()
    md.install(off_cfg)  # no-op without the switch
    assert md.layer_hook(off_layer, off_cfg, "model.layers.3", True) == (None, None)
    OPTS.assign(md.options(on_cfg))
    names = md.layer_hook(on_layer, on_cfg, "model.layers.3", True)
    assert names == ("model.layers.3.mono_rdna2",) * 2
    assert not off_cfg.compilation_config.static_forward_context
    assert list(on_cfg.compilation_config.static_forward_context) == [names[0]]

    assert [(n, p.shape) for n, p in off_layer.named_parameters()] == [
        (n, p.shape) for n, p in on_layer.named_parameters()
    ]
    on_state, off_state = _state(on_layer), _state(off_layer)
    assert on_state.keys() == off_state.keys()
    assert all(torch.equal(on_state[k], off_state[k]) for k in on_state)


def test_mtp_drafter_stays_on_modules_by_default(deployment):
    cfg = _config()
    OPTS.assign(md.options(cfg))
    layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    assert md.layer_hook(layer, cfg, "mtp.layers.0", True) == (None, None)
    assert md.layer_hook(layer, cfg, "model.layers.0", False) == (None, None)


def test_mtp_drafter_opt_in_registers_isolated_mono_layer(monkeypatch, deployment):
    import vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe as moe

    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: False)
    cfg = _config(
        knobs={"mtp_drafter": True, "mtp_k1_max_rows": 3, "mtp_k2_max_rows": 4}
    )
    OPTS.assign(md.options(cfg))
    layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    names = md.layer_hook(layer, cfg, "mtp.layers.0", True)
    assert names == ("mtp.layers.0.mono_rdna2",) * 2
    mono = cfg.compilation_config.static_forward_context[names[0]]
    assert mono.is_mtp
    assert (mono.k1_max_rows, mono.k2_max_rows) == (3, 4)

    # Opting in for the drafter must not change the target-model route.
    target_layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    target_names = md.layer_hook(target_layer, cfg, "model.layers.0", True)
    assert target_names == ("model.layers.0.mono_rdna2",) * 2
    assert not cfg.compilation_config.static_forward_context[target_names[0]].is_mtp


def test_unsupported_mtp_drafter_keeps_module_fallback(monkeypatch, deployment):
    import vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe as moe

    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: False)
    cfg = _config(knobs={"mtp_drafter": True})
    OPTS.assign(md.options(cfg))
    layer = _Layer(SimpleNamespace())
    names = md.layer_hook(layer, cfg, "mtp.layers.0", True)
    assert names == ("mtp.layers.0.mono_rdna2",) * 2
    mono = cfg.compilation_config.static_forward_context[names[0]]
    assert not mono.setup()
    assert not mono.ready


def test_mtp_k1_setup_is_independent_of_unsupported_k2(monkeypatch):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import layout

    def hc():
        down = SimpleNamespace(
            weight=torch.empty((4, 8), dtype=torch.float16), weight_i8=None
        )
        up = SimpleNamespace(
            weight=torch.empty((8, 4), dtype=torch.float16), weight_i8=None
        )
        return SimpleNamespace(
            hc_count=4,
            use_combine=True,
            input_mix_weight_down_block_inject=down,
            input_mix_weight_up=up,
            hc_norm=SimpleNamespace(weight=torch.ones(8, dtype=torch.float16)),
            config=SimpleNamespace(rms_norm_eps=1e-6),
        )

    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            gate=object(), experts=SimpleNamespace(quant_method=SimpleNamespace())
        ),
        mlp_hyper_connection=hc(),
        attn_hyper_connection=hc(),
    )
    monkeypatch.setattr(layout, "workspace", lambda device, scope: object())
    monkeypatch.setattr(layout.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(layout, "ZEROS", {})

    mono = layout.MonoLayer(layer, is_mtp=True)
    assert mono.setup_k1()
    assert mono.k1_ready and mono.hc_ok
    assert not mono.setup()
    assert not mono.ready
    assert mono.k1_ready


class UnquantizedFusedMoEMethod:  # name-matched stand-in
    pass


def _dense_mtp_layer(*, w13_shape=(128, 1280, 2560)):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import layout

    def meta(shape):
        return torch.empty(shape, dtype=torch.float16, device="meta")

    hc = SimpleNamespace(
        hc_count=4,
        use_combine=True,
        hc_norm=SimpleNamespace(weight=meta((10240,))),
        input_mix_weight_down_block_inject=SimpleNamespace(
            weight=meta((336, 10240)), weight_i8=None
        ),
        input_mix_weight_up=SimpleNamespace(weight=meta((10240, 320)), weight_i8=None),
        config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    experts = SimpleNamespace(
        quant_method=UnquantizedFusedMoEMethod(),
        w13_weight=meta(w13_shape),
        w2_weight=meta((128, 2560, 640)),
        activation=layout.MoEActivation.SILU,
        w13_bias=None,
        w2_bias=None,
        expert_map=torch.arange(128, dtype=torch.int32),
        top_k=10,
        renormalize=True,
        apply_router_weight_on_input=False,
    )
    shared = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=meta((320, 2560))),
        down_proj=SimpleNamespace(weight=meta((2560, 160))),
        expert_gate=SimpleNamespace(weight=meta((1, 2560))),
    )
    mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=meta((512, 2560)), bias=None),
        experts=experts,
        shared_expert=shared,
    )
    return layout, SimpleNamespace(
        mlp=mlp,
        mlp_hyper_connection=hc,
        config=SimpleNamespace(shared_expert_intermediate_size=640),
    )


def test_dense_unquantized_mtp_ffn_is_supported_only_for_drafter(monkeypatch):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import layout

    _, layer = _dense_mtp_layer()
    monkeypatch.setattr(layout, "workspace", lambda device, scope: object())

    mono = layout.MonoLayer(layer, is_mtp=True)
    assert mono.setup()
    assert mono.dense_mtp
    assert mono.t["w13"].shape == (128, 1280, 2560)
    assert mono.t["w2"].shape == (128, 2560, 640)
    assert mono.t["se_w1"].shape == (320, 2560)
    assert mono.t["se_w2"].shape == (2560, 160)

    target = layout.MonoLayer(layer, is_mtp=False)
    assert not target.setup()
    assert not target.dense_mtp


def test_dense_unquantized_mtp_wrong_geometry_falls_back(monkeypatch):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import layout

    _, layer = _dense_mtp_layer(w13_shape=(128, 1278, 2560))
    monkeypatch.setattr(layout, "workspace", lambda device, scope: object())

    mono = layout.MonoLayer(layer, is_mtp=True)
    assert not mono.setup()
    assert not mono.dense_mtp


def test_dense_unquantized_mtp_wrong_shared_shard_falls_back(monkeypatch):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import layout

    _, layer = _dense_mtp_layer()
    layer.mlp.shared_expert.gate_up_proj.weight = torch.empty(
        (640, 2560), dtype=torch.float16, device="meta"
    )
    monkeypatch.setattr(layout, "workspace", lambda device, scope: object())

    mono = layout.MonoLayer(layer, is_mtp=True)
    assert not mono.setup()
    assert not mono.dense_mtp
    # The shared expert was given the global width where the local TP4 width
    # was expected, so the reason has to say the layout is unsupported and
    # name the widths it saw.
    assert "shared expert layout is unsupported" in mono.reason
    assert "640" in mono.reason and "160" in mono.reason


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("scoring_func", "sigmoid", "routing"),
        ("use_grouped_topk", True, "routing"),
        ("apply_router_weight_on_input", True, "router weights on input"),
    ],
)
def test_dense_unquantized_mtp_unsupported_routing_falls_back(
    monkeypatch, field, value, reason
):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import layout

    _, layer = _dense_mtp_layer()
    setattr(layer.mlp.experts, field, value)
    monkeypatch.setattr(layout, "workspace", lambda device, scope: object())

    mono = layout.MonoLayer(layer, is_mtp=True)
    assert not mono.setup()
    assert not mono.dense_mtp
    assert reason in mono.reason


def test_gdn_and_qsa_parts_follow_the_switch(monkeypatch):
    assert md.qsa_part(_config(monokernel=False)) is None
    assert md.qsa_part(_config(knobs={"qsa": False})) is None
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa

    assert md.qsa_part(_config()) is qsa


def test_hw_queue_budget(monkeypatch):
    monkeypatch.setattr(md, "_on_gfx10x", lambda: True)
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "2")
    md.set_worker_hw_queue_budget(_config(monokernel=False))
    assert os.environ["GPU_MAX_HW_QUEUES"] == "2"
    md.set_worker_hw_queue_budget(_config())
    assert os.environ["GPU_MAX_HW_QUEUES"] == "4"
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "2")
    md.set_worker_hw_queue_budget(_config(knobs={"hw_queues": 0}))
    assert os.environ["GPU_MAX_HW_QUEUES"] == "2"


# ---- map mode: K2 reads the layer's live expert_map (EPLB load counters) ------


def _meta(shape, dtype=torch.float16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _block_map(base):
    """int32 [512]: the contiguous block [base, base + 128) in rows 0..127."""
    emap = torch.full((512,), -1, dtype=torch.int32)
    emap[base : base + 128] = torch.arange(128, dtype=torch.int32)
    return emap


def _scattered_map(seed):
    """int32 [512]: 128 random global experts in random rows (no block)."""
    gen = torch.Generator().manual_seed(seed)
    emap = torch.full((512,), -1, dtype=torch.int32)
    owned = torch.randperm(512, generator=gen)[:128]
    emap[owned] = torch.randperm(128, generator=gen).to(torch.int32)
    return emap


def _quantized_layer(emap, **experts_extra):
    """A decoder layer with one TP4 / EP4 rank's quantized resident experts.
    The weights are on the meta device (only their shapes are read); the router
    and the expert map are real."""

    def hc(count=4):
        return SimpleNamespace(
            hc_count=count,
            use_combine=True,
            hc_norm=SimpleNamespace(weight=_meta((10240,))),
            input_mix_weight_down_block_inject=SimpleNamespace(
                weight=_meta((336, 10240)), weight_i8=None
            ),
            input_mix_weight_up=SimpleNamespace(
                weight=_meta((10240, 320)), weight_i8=None
            ),
            config=SimpleNamespace(rms_norm_eps=1e-6),
        )

    experts = SimpleNamespace(
        quant_method=CompressedTensorsWNA16RDNA2MoEMethod(),
        w13_weight_packed=_meta((128, 320, 1280), torch.int32),
        w13_weight_scale=_meta((128, 20, 1280)),
        w13_qzeros=_meta((128, 20, 160), torch.int32),
        w2_weight_packed=_meta((128, 80, 2560), torch.int32),
        w2_weight_scale=_meta((128, 5, 2560)),
        w2_qzeros=_meta((128, 5, 320), torch.int32),
        expert_map=emap,
        top_k=10,
        renormalize=True,
        **experts_extra,
    )
    shared = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=_meta((320, 2560))),
        down_proj=SimpleNamespace(weight=_meta((2560, 160))),
        expert_gate=SimpleNamespace(weight=_meta((1, 2560))),
    )
    mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=torch.zeros((512, 2560), dtype=torch.float16)),
        experts=experts,
        shared_expert=shared,
    )
    # the attention side is K1's business: hc_count 3 leaves K1 unready
    return SimpleNamespace(
        mlp=mlp, mlp_hyper_connection=hc(), attn_hyper_connection=hc(3)
    )


def _layout(monkeypatch, *, map_on):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import layout

    monkeypatch.setattr(layout, "workspace", lambda device, scope: object())
    monkeypatch.setattr(layout, "map_mode", lambda: map_on)
    monkeypatch.setattr(layout, "op_takes_map", lambda: True)
    return layout


def test_setup_without_map_mode_still_wants_one_contiguous_block(monkeypatch):
    layout = _layout(monkeypatch, map_on=False)
    mono = layout.MonoLayer(_quantized_layer(_block_map(256)))
    assert mono.setup(), mono.reason
    assert mono.base == 256
    assert not mono.map_mode and mono.experts is None and mono.ids_out is None
    scattered = layout.MonoLayer(_quantized_layer(_scattered_map(1)))
    assert not scattered.setup()
    assert "one contiguous block of 128" in scattered.reason


def test_setup_in_map_mode_takes_any_placement(monkeypatch):
    layout = _layout(monkeypatch, map_on=True)
    for emap in (_block_map(256), _scattered_map(2)):
        layer = _quantized_layer(emap)
        mono = layout.MonoLayer(layer)
        assert mono.setup(), mono.reason
        assert mono.map_mode
        assert mono.experts is layer.mlp.experts
        assert mono.experts.expert_map is emap  # the live tensor, never a copy
        assert mono.ids_out.shape == (layout.MAX_ROWS, layout.TOP_K)
        assert mono.ids_out.dtype == torch.int32


@pytest.mark.parametrize(
    "case, reason",
    [
        ("duplicate row", "128 local slots"),
        ("127 experts", "128 local slots"),
        ("int64 map", "int32"),
        ("short map", "one entry per global expert"),
        ("expert cache", "expert cache"),
        ("old extension", "built without"),
    ],
)
def test_setup_in_map_mode_refuses_what_k2_cannot_run(monkeypatch, case, reason):
    layout = _layout(monkeypatch, map_on=True)
    emap = _scattered_map(3)
    owned = (emap >= 0).nonzero().flatten()
    extra = {}
    if case == "duplicate row":
        emap[owned[0]] = emap[owned[1]]
    elif case == "127 experts":
        emap[owned[0]] = -1
    elif case == "int64 map":
        emap = emap.long()
    elif case == "short map":
        emap = emap[:511]
    elif case == "expert cache":
        extra["_rdna_expert_cache"] = SimpleNamespace(
            logical_num_experts=128, slot_count=128
        )
    else:
        monkeypatch.setattr(layout, "op_takes_map", lambda: False)
    mono = layout.MonoLayer(_quantized_layer(emap, **extra))
    assert not mono.setup()  # the layer stays on the modules
    assert not mono.map_mode and mono.ids_out is None
    assert reason in mono.reason


def _k2_inputs(rows=3):
    """(mono, (hidden, attn_out, injection)) enough for ``k2.run`` to build the
    op call; the op itself is replaced by a recorder."""
    names = [
        "norm_w",
        "wd",
        "wu",
        "wr",
        "se_w1",
        "se_w2",
        "se_wg",
        "w13",
        "s13",
        "z13",
        "w2",
        "s2",
        "z2",
    ]
    mono = SimpleNamespace(
        t={name: torch.empty(0) for name in names},
        ws=torch.empty(0),
        base=0,
        eps=1e-6,
        is_mtp=False,
        dense_mtp=False,
    )
    inputs = (
        torch.zeros(rows, 10240, dtype=torch.float16),
        torch.zeros(rows, 2560, dtype=torch.float16),
        torch.zeros(rows, 4, dtype=torch.float16),
    )
    return mono, inputs


@pytest.fixture
def op_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops._rocm_C,
        "mono_k2_rdna2",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    return calls


def test_k2_run_without_map_mode_is_todays_call(op_calls):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    mono, inputs = _k2_inputs()
    k2.run(mono, *inputs)
    ((args, kwargs),) = op_calls
    # 28 positional arguments, none of the map-mode ones
    assert kwargs == {} and len(args) == 28
    assert args[19] is mono.ws
    assert args[20:25] == (0, 1, k2.OPTS.k2_num_wg, 1e-6, 2000.0)
    assert args[25:] == (False, False, False)


def test_k2_run_in_map_mode_passes_the_live_map_and_counts_the_ids(
    monkeypatch, op_calls
):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    registered: list = []
    counted: list = []
    stats = SimpleNamespace(count=counted.append)

    def layer_stats(layer, ids):
        registered.append((layer, ids))
        return stats

    monkeypatch.setattr(k2, "EPLB_STATS", SimpleNamespace(layer_stats=layer_stats))
    mono, inputs = _k2_inputs(rows=3)
    emap = _scattered_map(4)
    experts = SimpleNamespace(expert_map=emap)
    mono.map_mode, mono.experts = True, experts
    mono.ids_out = torch.zeros((24, 10), dtype=torch.int32)
    k2.run(mono, *inputs)
    ((args, kwargs),) = op_calls
    assert kwargs == {} and len(args) == 30
    assert args[28] is emap and args[29] is mono.ids_out
    ((layer, ids),) = registered
    assert layer is experts and ids.shape == (1, 10)
    assert len(counted) == 1
    assert counted[0].shape == (3, 10)
    assert counted[0].data_ptr() == mono.ids_out.data_ptr()
    # read at every call, never cached: the tensor the layer holds now is used
    experts.expert_map = other = emap.clone()
    k2.run(mono, *inputs)
    assert op_calls[1][0][28] is other


@pytest.mark.parametrize("collector", ["off", "layer unregistered"])
def test_k2_run_in_map_mode_without_counters_asks_for_no_ids(
    monkeypatch, op_calls, collector
):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    # counters off, or a layer that first ran inside a graph capture
    monkeypatch.setattr(
        k2,
        "EPLB_STATS",
        None if collector == "off" else SimpleNamespace(layer_stats=lambda *a: None),
    )
    mono, inputs = _k2_inputs()
    emap = _block_map(0)
    mono.map_mode, mono.experts = True, SimpleNamespace(expert_map=emap)
    mono.ids_out = torch.zeros((24, 10), dtype=torch.int32)
    k2.run(mono, *inputs)
    ((args, _),) = op_calls
    assert len(args) == 30 and args[28] is emap and args[29] is None
