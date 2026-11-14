# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU proof that Qwen3.8-27B-HI2-3.0bpw routes its weights under MTP and TP2.

No GPU and no real weights. The inputs are the real ``config.json`` and the
real safetensors headers (tensor name, dtype, shape) of
``BlivionIaG/Qwen3.8-27B-HI2-3.0bpw`` (``fixtures/qwen38_27b_hi2``; the vision
tower is left out). The tests build the target and drafter classes on the
``meta`` device with a faked tensor-parallel group (world size 1 or 2) and push
every checkpoint tensor name through ``load_weights`` as a meta tensor:

* every ``mtp.*`` tensor reaches a parameter of ``Qwen3_5MTP`` (the loader raises
  on an unmapped name) and every drafter parameter is loaded;
* the drafter's layers are dense (``UnquantizedLinearMethod``), not hi2;
* every ``.hi2_*`` tensor of the target reaches the loader of the layer that
  owns it, with the shard ids its fused layer expects.

A hi2 ``rdna_hint_flex`` checkpoint carries no ``ignore`` list: which modules are
quantized comes from the tensor names (``RdnaHintFlexConfig``). Regenerate the
fixture from a snapshot with the header dump in
``docs/rdna2/hi2-27b-mtp-2026-11-14.md``.
"""

import gzip
import json
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="rdna_hint_flex is a ROCm (gfx10x) quantization method",
)

FIXTURE = Path(__file__).parent / "fixtures" / "qwen38_27b_hi2"
SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I32": torch.int32,
}
MTP_TENSORS = 15  # mtp.fc + mtp.layers.0.* + three norms (checkpoint index)


def _tensors() -> dict[str, tuple[str, list[int]]]:
    with gzip.open(FIXTURE / "tensors.json.gz", "rt") as f:
        return {k: (v[0], v[1]) for k, v in json.load(f).items()}


def _meta_weights(names, tensors):
    for n in names:
        dtype, shape = tensors[n]
        yield n, torch.empty(shape, dtype=SAFETENSORS_DTYPES[dtype], device="meta")


def _write_checkpoint(path: Path, names) -> str:
    """config.json plus one real safetensors file holding tiny tensors under the
    given names (the readers under test only look at names)."""
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text((FIXTURE / "config.json").read_text())
    save_file(
        {n: torch.zeros(1, dtype=torch.float16) for n in names},
        path / "model.safetensors",
    )
    return str(path)


class _FakeGroup:
    def __init__(self, world: int, rank: int):
        self.world_size = world
        self.rank_in_group = rank
        self.rank = rank
        self.local_rank = rank
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world - 1
        self.ranks = list(range(world))
        self.device_communicator = None


@pytest.fixture(scope="module")
def _patched_platform():
    """Device capability without a GPU (gfx1030 reports 10.3)."""
    from vllm.platforms.interface import DeviceCapability

    mp = pytest.MonkeyPatch()
    mp.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    yield
    mp.undo()


def _engine_config(model_dir: str, tp: int, num_spec: int = 2):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model=model_dir,
        dtype="float16",
        tensor_parallel_size=tp,
        skip_tokenizer_init=True,
        max_model_len=4096,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_spec,
            "use_local_argmax_reduction": True,
        },
    )
    return args.create_engine_config()


@pytest.fixture
def tp_group(request):
    """Fake a TP group of ``request.param`` ranks (rank 1 of 2 for TP2, so the
    slicing is not the trivial rank-0 slice)."""
    import vllm.distributed.parallel_state as ps

    tp = getattr(request, "param", 1)
    mp = pytest.MonkeyPatch()
    mp.setattr(ps, "_TP", _FakeGroup(tp, tp - 1))
    mp.setattr(ps, "_PP", _FakeGroup(1, 0))
    for name in ("_DP", "_EP", "_PCP", "_DCP", "_EPLB", "_WORLD"):
        if hasattr(ps, name):
            mp.setattr(ps, name, _FakeGroup(1, 0))
    yield tp
    mp.undo()


@pytest.fixture
def hi2_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path / "hi2", _tensors())


def _build(cls, cfg, prefix: str = "", *, draft: bool = False):
    """Construct ``cls`` on the meta device the way the loader does
    (``initialize_model``: packed-module mapping handed to the quant config,
    the draft model config for the drafter)."""
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    model_config = cfg.speculative_config.draft_model_config if draft else None
    with set_default_torch_dtype(torch.float16), torch.device("meta"):
        return initialize_model(
            vllm_config=cfg,
            model_class=cls,
            prefix=prefix,
            model_config=model_config,
        )


# --------------------------------------------------------------------------
# (a) detection and dense construction of the MTP layer
# --------------------------------------------------------------------------
def test_checkpoint_has_dense_mtp_and_hi2_body():
    names = list(_tensors())
    mtp = [n for n in names if n.startswith("mtp.")]
    assert len(mtp) == MTP_TENSORS
    assert all(n.endswith(".weight") for n in mtp)
    assert not any(".hi2_" in n for n in mtp)
    # lm_head and embeddings are dense too; the body is hi2.
    assert "lm_head.weight" in names
    assert "model.language_model.embed_tokens.weight" in names
    assert any(".hi2_codes" in n for n in names)


def test_mtp_detected_unquantized(hi2_checkpoint, _patched_platform):
    from vllm.model_executor.models.qwen3_5_mtp import _mtp_weights_unquantized

    cfg = _engine_config(hi2_checkpoint, 1)
    assert _mtp_weights_unquantized(cfg.model_config)


def test_hi2_mtp_is_not_detected_as_unquantized(tmp_path, _patched_platform):
    """A future hi2 checkpoint that also quantizes the drafter stores
    ``mtp.*.hi2_*`` tensors; the detection must not strip the quant config."""
    from vllm.model_executor.models.qwen3_5_mtp import _mtp_weights_unquantized

    tensors = _tensors()
    names = [n for n in tensors if not n.startswith("mtp.")]
    names += [
        f"mtp.layers.0.mlp.{p}.hi2_codes" for p in ("gate_proj", "up_proj", "down_proj")
    ]
    names += ["mtp.fc.weight", "mtp.norm.weight"]
    model_dir = _write_checkpoint(tmp_path / "hi2mtp", names)
    cfg = _engine_config(model_dir, 1)
    assert not _mtp_weights_unquantized(cfg.model_config)


@pytest.mark.parametrize("tp_group", [1, 2], indirect=True)
def test_mtp_builds_dense_layers(hi2_checkpoint, _patched_platform, tp_group):
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    tp = tp_group
    cfg = _engine_config(hi2_checkpoint, tp)
    assert cfg.quant_config is not None
    assert cfg.quant_config.get_name() == "rdna_hint_flex"
    model = _build(Qwen3_5MTP, cfg, draft=True)
    # The drafter's own quant_config switch is restored after the layers exist.
    assert cfg.quant_config is not None

    linears = {n: m for n, m in model.named_modules() if isinstance(m, LinearBase)}
    assert set(linears) == {
        "model.fc",
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
    }
    for name, m in linears.items():
        assert isinstance(m.quant_method, UnquantizedLinearMethod), name
    assert not any("hi2_" in n for n, _ in model.named_parameters())
    assert isinstance(model.lm_head.quant_method, UnquantizedEmbeddingMethod)

    # Per-rank shapes (checkpoint: fc 5120x10240, q 12288x5120 incl. the output
    # gate, k/v 1024x5120, o 5120x6144, gate/up 17408x5120, down 5120x17408).
    p = dict(model.named_parameters())
    assert p["model.fc.weight"].shape == (5120 // tp, 10240)
    assert p["model.layers.0.self_attn.qkv_proj.weight"].shape == (
        (12288 + 2 * 1024) // tp,
        5120,
    )
    assert p["model.layers.0.self_attn.o_proj.weight"].shape == (5120, 6144 // tp)
    assert p["model.layers.0.mlp.gate_up_proj.weight"].shape == (2 * 17408 // tp, 5120)
    assert p["model.layers.0.mlp.down_proj.weight"].shape == (5120, 17408 // tp)
    assert p["lm_head.weight"].shape == (248320 // tp, 5120)
    # Row-parallel K per rank is a multiple of 64 (hi2 spec) at TP2.
    assert (6144 // tp) % 64 == 0 and (17408 // tp) % 64 == 0


@pytest.mark.parametrize("tp_group", [2], indirect=True)
def test_hi2_config_alone_keeps_dense_mtp_dense(
    hi2_checkpoint, _patched_platform, tp_group, monkeypatch
):
    """Second safety net: even if the unquantized-MTP detection said no, the
    drafter's linears are dense, because RdnaHintFlexConfig decides per module
    from the tensor names (``mtp.*`` has no ``.hi2_*`` tensors and ``_norm_name``
    keeps the ``mtp.`` prefix, so it cannot alias ``layers.N`` of the body)."""
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models import qwen3_5_mtp
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    monkeypatch.setattr(qwen3_5_mtp, "_mtp_weights_unquantized", lambda _cfg: False)
    cfg = _engine_config(hi2_checkpoint, tp_group)
    model = _build(Qwen3_5MTP, cfg, draft=True)
    linears = [m for m in model.modules() if isinstance(m, LinearBase)]
    assert len(linears) == 5
    assert all(isinstance(m.quant_method, UnquantizedLinearMethod) for m in linears)
    prefixes = {m.prefix for m in linears}
    assert "mtp.fc" in prefixes and "mtp.layers.0.mlp.gate_up_proj" in prefixes


@pytest.mark.parametrize("tp_group", [1, 2], indirect=True)
def test_mtp_load_weights_maps_every_tensor(
    hi2_checkpoint, _patched_platform, tp_group
):
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    cfg = _engine_config(hi2_checkpoint, tp_group)
    model = _build(Qwen3_5MTP, cfg, draft=True)
    tensors = _tensors()

    kept = [n for n in tensors if not model.is_unused_checkpoint_weight(n)]
    mtp = sorted(n for n in kept if n.startswith("mtp."))
    assert len(mtp) == MTP_TENSORS
    # The drafter also reads embed_tokens and lm_head (shared with the target
    # after loading); nothing else of the checkpoint (hi2 body included).
    assert sorted(set(kept) - set(mtp)) == [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
    ]

    # AutoWeightsLoader raises on a name without a module/parameter, and the
    # default loaders assert the (TP-sliced) shape of every dense tensor.
    loaded = model.load_weights(_meta_weights(tensors, tensors))
    params = {n for n, _ in model.named_parameters()}
    assert loaded == params
    assert {
        "model.fc.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.norm.weight",
        "model.pre_fc_norm_hidden.weight",
        "model.pre_fc_norm_embedding.weight",
    } <= loaded


def _recording_loader(monkeypatch):
    """Replace the hi2 loader by one that records ``(field, shard_id)`` per layer
    prefix (the real one copies the tensor to the CPU, which a meta tensor
    cannot do; the data path is covered by test_rdna_hint_flex_rdna2.py)."""
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf

    seen: dict[str, set[tuple[str, str]]] = {}

    def make_loader(layer, field):
        def loader(param, loaded_weight, shard_id=None):
            if shard_id is None:
                shard_id = getattr(loaded_weight, "shard_id", None)
            seen.setdefault(layer.prefix, set()).add((field, str(shard_id)))

        return loader

    monkeypatch.setattr(
        rhf.RdnaHintFlexLinearMethod, "_make_loader", staticmethod(make_loader)
    )
    return seen


@pytest.mark.parametrize("tp_group", [2], indirect=True)
def test_drafter_shares_embedding_and_head_under_tp2(
    hi2_checkpoint, _patched_platform, tp_group
):
    """The drafter reads embed_tokens/lm_head from the checkpoint but the
    speculator replaces both by the target's (no second 1.27 GB/rank copy
    stays resident); the vocab shards are identical, so sharing is legal, and
    the local-argmax path needs no vocab padding handling at TP2."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        maybe_share_target_embed,
    )

    cfg = _engine_config(hi2_checkpoint, tp_group)
    draft = _build(Qwen3_5MTP, cfg, draft=True)
    target = _build(Qwen3_5ForCausalLM, cfg, prefix="language_model")
    assert hasattr(draft, "get_top_tokens")  # use_local_argmax_reduction
    assert not getattr(draft, "has_own_embed_tokens", False)
    assert not getattr(draft, "has_own_lm_head", False)
    assert _should_share(draft, "has_own_lm_head", draft.lm_head, target.lm_head)
    maybe_share_target_embed(draft, draft.model, target.model)
    assert draft.model.embed_tokens is target.model.embed_tokens

    # rank 1 of 2 owns vocab rows [124160, 248320); no padding rows, and the
    # global index fits a float32 exactly (the pair all-gather path).
    shard = target.lm_head.shard_indices
    assert shard.org_vocab_start_index == 248320 // 2
    assert shard.num_org_vocab_padding == 0
    assert target.lm_head.weight.shape == (248320 // 2, 5120)
    assert 248320 < 2**24


@pytest.mark.parametrize("tp_group", [2], indirect=True)
def test_hi2_mtp_checkpoint_would_build_hi2_layers(
    tmp_path, _patched_platform, tp_group, monkeypatch
):
    """Synthetic (no such checkpoint exists yet): if the quantizer also encodes
    the drafter, its layers must stay hi2 and every hi2 tensor must reach its
    layer. The ``mtp.layers.0.*_proj`` tensors are replaced by layer 3's
    (a full-attention layer of the same shapes), renamed."""
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    tensors = _tensors()
    synthetic = {
        n: v
        for n, v in tensors.items()
        if not (n.startswith("mtp.layers.0.") and "_proj." in n)
    }
    for n, v in tensors.items():
        if n.startswith("model.language_model.layers.3.") and ".hi2_" in n:
            synthetic["mtp.layers.0." + n.split("layers.3.", 1)[1]] = v
    model_dir = _write_checkpoint(tmp_path / "hi2mtp", synthetic)
    cfg = _engine_config(model_dir, tp_group)
    seen = _recording_loader(monkeypatch)

    model = _build(Qwen3_5MTP, cfg, draft=True)
    layer = model.model.layers[0]
    hi2 = {
        "mtp.layers.0.self_attn.qkv_proj": layer.self_attn.qkv_proj,
        "mtp.layers.0.self_attn.o_proj": layer.self_attn.o_proj,
        "mtp.layers.0.mlp.gate_up_proj": layer.mlp.gate_up_proj,
        "mtp.layers.0.mlp.down_proj": layer.mlp.down_proj,
    }
    for prefix, m in hi2.items():
        assert isinstance(m.quant_method, rhf.RdnaHintFlexLinearMethod), prefix
    # fc stays dense: it is stored as mtp.fc.weight.
    assert not isinstance(model.model.fc.quant_method, rhf.RdnaHintFlexLinearMethod)

    model.load_weights(_meta_weights(synthetic, synthetic))
    assert set(seen) == set(hi2)
    for prefix, items in seen.items():
        assert {f for f, _ in items} >= set(rhf.hu.FIELDS), prefix


# --------------------------------------------------------------------------
# target side: every hi2 tensor reaches the layer that owns it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tp_group", [2], indirect=True)
def test_target_hi2_tensors_reach_their_layers(
    hi2_checkpoint, _patched_platform, tp_group, monkeypatch
):
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM

    cfg = _engine_config(hi2_checkpoint, tp_group)
    text = cfg.model_config.hf_text_config
    seen = _recording_loader(monkeypatch)
    # ``--language-model-only`` / the inner language model of the VL class.
    model = _build(Qwen3_5ForCausalLM, cfg, prefix="language_model")

    hi2_layers = {
        n: m
        for n, m in model.named_modules()
        if isinstance(getattr(m, "quant_method", None), rhf.RdnaHintFlexLinearMethod)
    }
    n_full = text.num_hidden_layers // text.full_attention_interval
    n_gdn = text.num_hidden_layers - n_full
    # gate_up + down on every layer, qkv + o on full attention, in_proj_qkvz +
    # in_proj_ba + out_proj on GDN (496 checkpoint tensors before fusion).
    assert len(hi2_layers) == 2 * text.num_hidden_layers + 2 * n_full + 3 * n_gdn
    # TP2: every row-parallel K per rank is a multiple of 64 (spec); the layer
    # constructor checks it, this pins the actual sizes.
    ks = {m.rdna_hint_flex_k for m in hi2_layers.values()}
    assert ks == {5120, 8704, 3072}, ks
    assert all(k % rhf.hu.K_MULTIPLE == 0 for k in ks)

    tensors = _tensors()
    body = [n for n in tensors if not n.startswith("mtp.")]
    n_hi2 = sum(1 for n in body if n.endswith(".hi2_codes"))
    assert n_hi2 == 496
    # A name without a module/parameter, or a dense tensor of the wrong
    # (TP-sliced) shape, raises here.
    loaded = model.load_weights(_meta_weights(body, tensors))
    assert loaded == {n for n, _ in model.named_parameters()}

    # Loader keys are the layer prefixes as the quant config saw them.
    assert set(seen) == {f"language_model.{n}" for n in hi2_layers}
    for prefix, items in seen.items():
        fields = {f for f, _ in items}
        assert fields >= set(rhf.hu.FIELDS), prefix
        shards = {s for _, s in items}
        last = prefix.rsplit(".", 1)[-1]
        if last == "qkv_proj":
            assert shards == {"q", "k", "v"}, shards
        elif last in ("gate_up_proj", "in_proj_ba"):
            assert shards == {"0", "1"}, (prefix, shards)
        elif last == "in_proj_qkvz":
            assert len(shards) == 2, (prefix, shards)
        else:
            assert shards == {"None"}, (prefix, shards)
