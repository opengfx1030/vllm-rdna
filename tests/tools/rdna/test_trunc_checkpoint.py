# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/rdna/rdna3/trunc_checkpoint.py on a synthetic mini checkpoint and the
compare logic of tools/rdna/rdna3/trunc_e2e.py on synthetic dumps.

The tools are imported by path (no vllm import).
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "rdna3"
LAYERS = 8
PREFIX = "model.language_model"
BYTES_PER = {"F16": 2, "BF16": 2, "F32": 4, "I32": 4, "I64": 8}


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tc() -> Any:
    pytest.importorskip("regex")
    return _load("trunc_checkpoint")


# --- an independent safetensors writer / reader (the tool has its own) -------

Tensors = dict[str, tuple[str, list[int], bytes]]


def write_st(path: Path, tensors: Tensors) -> None:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for _, _, raw in tensors.values():
            f.write(raw)


def read_st(path: Path) -> Tensors:
    data = path.read_bytes()
    (hlen,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + hlen])
    header.pop("__metadata__", None)
    base = 8 + hlen
    return {
        name: (
            e["dtype"],
            e["shape"],
            data[base + e["data_offsets"][0] : base + e["data_offsets"][1]],
        )
        for name, e in header.items()
    }


def _tensor(rng: np.random.Generator, dtype: str, shape: list[int]) -> Any:
    n = int(np.prod(shape)) * BYTES_PER[dtype]
    return dtype, shape, rng.integers(0, 256, n, dtype=np.uint8).tobytes()


def _layer_tensors(rng: np.random.Generator, i: int) -> Tensors:
    p = f"{PREFIX}.layers.{i}"
    out: Tensors = {
        f"{p}.attn_hyper_connection.hc_norm.weight": _tensor(rng, "BF16", [16]),
        f"{p}.mlp.gate.weight": _tensor(rng, "BF16", [4, 16]),
        f"{p}.mlp.experts.0.down_proj.weight_packed": _tensor(rng, "I32", [4, 2]),
        f"{p}.mlp.experts.0.down_proj.weight_scale": _tensor(rng, "BF16", [4, 1]),
        f"{p}.mlp.experts.0.down_proj.weight_shape": _tensor(rng, "I64", [2]),
    }
    if i % 4 == 3:
        out[f"{p}.self_attn.q_proj.weight"] = _tensor(rng, "BF16", [8, 16])
    else:
        out[f"{p}.linear_attn.A_log"] = _tensor(rng, "F32", [4])
        out[f"{p}.linear_attn.conv1d.weight"] = _tensor(rng, "BF16", [8, 1, 4])
    if i == 1:
        out[f"{p}.ple.key_proj.weight"] = _tensor(rng, "BF16", [8, 16])
    return out


def make_checkpoint(root: Path) -> dict[str, Tensors]:
    """A mini Flash-Next layout: 8 layers, PLE in layer 1, MTP in its own file."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    shards: dict[str, Tensors] = {
        "model-00001-of-00003.safetensors": {
            f"{PREFIX}.layers.1.ple.ple_embedding.ngram_embedding.shard_{k}.weight": (
                _tensor(rng, "F16", [16, 8])
            )
            for k in range(2)
        },
        "model-00002-of-00003.safetensors": {
            f"{PREFIX}.embed_tokens.weight": _tensor(rng, "BF16", [32, 16]),
            "model.visual.blocks.0.attn.qkv.weight": _tensor(rng, "F16", [8, 8]),
        },
        "model-00003-of-00003.safetensors": {
            "lm_head.weight": _tensor(rng, "BF16", [32, 16]),
            f"{PREFIX}.hyper_connection_mixer.hc_norm.weight": _tensor(
                rng, "BF16", [16]
            ),
        },
        "model_mtp.safetensors": {
            "mtp.fc_hidden.weight": _tensor(rng, "BF16", [16, 16]),
            "mtp.layers.0.mlp.experts.down_proj": _tensor(rng, "F16", [2, 4, 4]),
        },
    }
    for i in range(LAYERS):
        target = "model-00002-of-00003.safetensors"
        if i >= 6:
            target = "model-00003-of-00003.safetensors"
        shards[target].update(_layer_tensors(rng, i))
    weight_map = {}
    for name, tensors in shards.items():
        write_st(root / name, tensors)
        weight_map.update({t: name for t in tensors})
    total = sum(len(r) for t in shards.values() for _, _, r in t.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    pattern = ["linear_attention"] * 3 + ["full_attention"]
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "quantization_config": {
            "format": "pack-quantized",
            "ignore": ["lm_head", "re:.*mtp\\..*"],
            "config_groups": {"group_0": {"weights": {"group_size": 128}}},
        },
        "text_config": {
            "num_hidden_layers": LAYERS,
            "layer_types": pattern * (LAYERS // 4),
            "full_attention_interval": 4,
            "hidden_size": 16,
            "ple_layer_ids": [2],
            "mtp_num_hidden_layers": 1,
            "mtp": {"layer_types": ["full_attention"], "num_hidden_layers": 1},
        },
        "vision_config": {"depth": 1},
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "tokenizer.json").write_text('{"tok": 1}')
    (root / "chat_template.jinja").write_text("{{ x }}")
    (root / "generation_config.json").write_text('{"temperature": 1.0}')
    (root / "README.md").write_text("readme")
    return shards


def all_tensors(shards: dict[str, Tensors]) -> Tensors:
    return {n: t for tensors in shards.values() for n, t in tensors.items()}


def layer_of(name: str) -> int | None:
    if name.startswith("mtp."):
        return None
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1]) if "layers" in parts else None


@pytest.fixture()
def ckpt(tmp_path: Path) -> tuple[Path, dict[str, Tensors]]:
    src = tmp_path / "src"
    return src, make_checkpoint(src)


def dst_tensors(dst: Path) -> Tensors:
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    out: Tensors = {}
    for fname in sorted(set(index["weight_map"].values())):
        out.update(read_st(dst / fname))
    return out


def run_build(tc: Any, src: Path, dst: Path, layers: int, **kwargs: Any) -> Any:
    return tc.build(src, dst, layers, log=lambda *_: None, **kwargs)


# --- trunc_checkpoint ---------------------------------------------------------


def test_truncate_to_four_layers_config(tc, ckpt, tmp_path):
    src, _ = ckpt
    dst = tmp_path / "dst"
    run_build(tc, src, dst, 4)
    old = json.loads((src / "config.json").read_text())
    new = json.loads((dst / "config.json").read_text())
    text = new["text_config"]
    assert text["num_hidden_layers"] == 4
    assert text["layer_types"] == old["text_config"]["layer_types"][:4]
    assert text["layer_types"][-1] == "full_attention"
    assert text["ple_layer_ids"] == [2]  # layer 1 is kept
    assert text["mtp_num_hidden_layers"] == 0
    # Nothing else changes: quantization config, vision config, other text keys.
    for key in old:
        if key != "text_config":
            assert new[key] == old[key]
    changed = {k for k in text if text[k] != old["text_config"].get(k)}
    assert changed == {"num_hidden_layers", "layer_types", "mtp_num_hidden_layers"}
    assert old["text_config"]["num_hidden_layers"] == LAYERS  # source untouched


def test_kept_tensors_and_bytes(tc, ckpt, tmp_path):
    src, shards = ckpt
    dst = tmp_path / "dst"
    report = run_build(tc, src, dst, 4)
    source = all_tensors(shards)
    expected = {
        n
        for n in source
        if not n.startswith("mtp.") and layer_of(n) in (None, 0, 1, 2, 3)
    }
    got = dst_tensors(dst)
    assert set(got) == expected
    assert report["tensors"] == len(expected)
    for name in expected:
        assert got[name] == source[name], name  # dtype, shape and bytes
    dropped = set(source) - expected
    assert any(layer_of(n) == 7 for n in dropped)
    assert "mtp.fc_hidden.weight" in dropped
    assert not (dst / "model_mtp.safetensors").exists()


def test_index_matches_shards(tc, ckpt, tmp_path):
    src, shards = ckpt
    dst = tmp_path / "dst"
    run_build(tc, src, dst, 4)
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    got = dst_tensors(dst)
    assert set(index["weight_map"]) == set(got)
    for fname in set(index["weight_map"].values()):
        names = {n for n, f in index["weight_map"].items() if f == fname}
        assert names == set(read_st(dst / fname))
    assert index["metadata"]["total_size"] == sum(len(t[2]) for t in got.values())
    # Shard 3 only had layers 6-7 and the head: the head and mixer survive.
    assert "lm_head.weight" in got
    assert f"{PREFIX}.hyper_connection_mixer.hc_norm.weight" in got


def test_fully_kept_shard_is_symlinked(tc, ckpt, tmp_path):
    src, shards = ckpt
    dst = tmp_path / "dst"
    report = run_build(tc, src, dst, 4)
    ple = dst / "model-00001-of-00003.safetensors"
    assert ple.is_symlink()
    assert ple.resolve() == (src / ple.name).resolve()
    assert not (dst / "model-00002-of-00003.safetensors").is_symlink()
    assert report["linked_bytes"] > 0
    copied = tmp_path / "copied"
    run_build(tc, src, copied, 4, copy_shards=True)
    assert not (copied / ple.name).is_symlink()
    assert read_st(copied / ple.name) == shards[ple.name]


def test_non_weight_files_copied(tc, ckpt, tmp_path):
    src, _ = ckpt
    dst = tmp_path / "dst"
    run_build(tc, src, dst, 4)
    for name in ("tokenizer.json", "chat_template.jinja", "generation_config.json"):
        assert (dst / name).read_bytes() == (src / name).read_bytes()
    assert (dst / "README.md").exists()
    info = json.loads((dst / "trunc_info.json").read_text())
    assert info["layers"] == 4 and info["keep_mtp"] is False


def test_keep_mtp(tc, ckpt, tmp_path):
    src, shards = ckpt
    dst = tmp_path / "dst"
    run_build(tc, src, dst, 4, keep_mtp=True)
    got = dst_tensors(dst)
    mtp = shards["model_mtp.safetensors"]
    assert {n: got[n] for n in mtp} == mtp
    text = json.loads((dst / "config.json").read_text())["text_config"]
    assert text["mtp_num_hidden_layers"] == 1
    assert text["mtp"] == {"layer_types": ["full_attention"], "num_hidden_layers": 1}


def test_ragged_layer_count_rejected(tc, ckpt, tmp_path, capsys):
    src, _ = ckpt
    with pytest.raises(ValueError, match="multiple of the layer-pattern period 4"):
        run_build(tc, src, tmp_path / "d3", 3)
    assert not (tmp_path / "d3").exists()
    assert (
        tc.main(["--src", str(src), "--dst", str(tmp_path / "d6"), "--layers", "6"])
        == 2
    )
    assert "multiple" in capsys.readouterr().err
    report = run_build(tc, src, tmp_path / "ok", 3, allow_ragged=True)
    text = json.loads((tmp_path / "ok" / "config.json").read_text())["text_config"]
    assert text["num_hidden_layers"] == 3
    assert len(text["layer_types"]) == 3
    assert max(layer_of(n) or 0 for n in report["weight_map"]) == 2


def test_cut_before_the_ple_layer(tc, ckpt, tmp_path):
    src, _ = ckpt
    dst = tmp_path / "dst"
    run_build(tc, src, dst, 1, allow_ragged=True)
    text = json.loads((dst / "config.json").read_text())["text_config"]
    assert text["ple_layer_ids"] == []
    assert not (dst / "model-00001-of-00003.safetensors").exists()
    assert all("ple" not in n for n in dst_tensors(dst))


def test_full_depth_keeps_every_decoder_tensor(tc, ckpt, tmp_path):
    src, shards = ckpt
    dst = tmp_path / "dst"
    run_build(tc, src, dst, LAYERS)
    source = all_tensors(shards)
    got = dst_tensors(dst)
    assert set(got) == {n for n in source if not n.startswith("mtp.")}
    assert (
        json.loads((dst / "config.json").read_text())["text_config"]["layer_types"]
        == json.loads((src / "config.json").read_text())["text_config"]["layer_types"]
    )


@pytest.mark.parametrize("layers", [0, LAYERS + 4])
def test_layer_count_out_of_range(tc, ckpt, tmp_path, layers):
    with pytest.raises(ValueError, match="--layers must be in"):
        run_build(tc, ckpt[0], tmp_path / "d", layers, allow_ragged=True)


def test_dst_guards(tc, ckpt, tmp_path):
    src, _ = ckpt
    dst = tmp_path / "dst"
    run_build(tc, src, dst, 4)
    with pytest.raises(FileExistsError):
        run_build(tc, src, dst, 4)
    run_build(tc, src, dst, 8, force=True)
    assert (
        json.loads((dst / "config.json").read_text())["text_config"][
            "num_hidden_layers"
        ]
        == 8
    )
    with pytest.raises(ValueError, match="inside --src"):
        run_build(tc, src, src / "out", 4)


def test_missing_layer_in_source_is_an_error(tc, ckpt, tmp_path):
    src, shards = ckpt
    victim = "model-00002-of-00003.safetensors"
    tensors = {n: t for n, t in shards[victim].items() if layer_of(n) != 2}
    write_st(src / victim, tensors)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    index["weight_map"] = {
        n: f for n, f in index["weight_map"].items() if layer_of(n) != 2
    }
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="layer 2 has no tensors"):
        run_build(tc, src, tmp_path / "d", 4)


def test_index_disagreeing_with_headers_is_an_error(tc, ckpt, tmp_path):
    src, _ = ckpt
    index = json.loads((src / "model.safetensors.index.json").read_text())
    index["weight_map"].pop("lm_head.weight")
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="disagrees"):
        run_build(tc, src, tmp_path / "d", 4)


# --- trunc_e2e compare ----------------------------------------------------------


@pytest.fixture(scope="module")
def e2e() -> Any:
    pytest.importorskip("regex")
    pytest.importorskip("torch")
    return _load("trunc_e2e")


NL, NR, DIM, HID = 6, 10, 64, 16


def make_dump(torch: Any, seed: int = 0, names: tuple[str, ...] = ("a", "b")) -> Any:
    g = torch.Generator().manual_seed(seed)
    prompts = []
    for k, name in enumerate(names):
        ids = list(range(100 + 20 * k, 140 + 20 * k))
        prompts.append(
            {
                "name": name,
                "num_tokens": len(ids),
                "token_ids": ids,
                "rows": list(range(NR)),
                "layer_ids": list(range(NL)),
                "out": torch.randn(NL, NR, DIM, generator=g),
                "attn": torch.randn(NL, NR, HID, generator=g),
                "mlp": torch.randn(NL, NR, HID, generator=g),
                "final": torch.randn(NR, HID, generator=g),
                "greedy_token": 7 + k,
                "top_ids": list(range(20)),
                "top_logprobs": [-0.1 * i for i in range(20)],
                "prompt_logprobs": None,
                "multi_call": [],
            }
        )
    return {"version": 1, "meta": {}, "prompts": prompts}


def perturb(torch: Any, dump: Any, rel: float, from_layer: int = 0, keys=("out",)):
    """Copy of ``dump`` with relative noise ``rel`` on the chosen keys/layers."""
    g = torch.Generator().manual_seed(1)
    new: dict[str, Any] = {"version": 1, "meta": {}, "prompts": []}
    for p in dump["prompts"]:
        q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}
        for key in keys:
            t = p[key].clone()
            noise = torch.randn(t.shape, generator=g)
            for layer in range(from_layer, t.shape[0]):
                scale = rel * t[layer].norm() / noise[layer].norm()
                t[layer] += noise[layer] * scale
            q[key] = t
        new["prompts"].append(q)
    return new


def test_identical_dumps(e2e):
    import torch

    d = make_dump(torch)
    report = e2e.compare_dumps(d, d, 2e-3)
    assert report.first_divergence() is None
    assert all(r.worst == 0.0 and r.maxabs == 0.0 for r in report.rows)
    assert report.exit_code() == 0
    text = e2e.format_report(report, "A", "B")
    assert "first divergence: none" in text and "RESULT: PASS" in text
    # every layer has attn, mlp and out rows in forward order
    labels = [r.label for r in report.rows[:3]]
    assert labels == ["L00.attn", "L00.mlp", "L00"]


def test_fp16_noise_has_no_divergence_and_growth_is_reported(e2e):
    import torch

    d = make_dump(torch)
    b = perturb(torch, d, 1e-4, keys=("out", "attn", "mlp"))
    report = e2e.compare_dumps(d, b, 2e-3)
    assert report.first_divergence() is None
    assert all(abs(r.worst - 1e-4) < 2e-5 for r in report.rows)
    assert report.exit_code() == 0
    text = e2e.format_report(report, "A", "B")
    assert "relative L2 of the layer output" in text
    assert "per layer" in text


def test_one_diverging_layer_is_named(e2e):
    import torch

    d = make_dump(torch)
    b = perturb(torch, d, 5e-2, from_layer=3, keys=("out",))
    report = e2e.compare_dumps(d, b, 2e-3)
    first = report.first_divergence()
    assert first.label == "L03" and first.layer == 3
    assert report.exit_code() == 0  # layer >= 1 diverges, greedy token agrees
    assert "first divergence: L03" in e2e.format_report(report, "A", "B")
    # The MoE half of the layer is the culprit when only its output moves.
    c = perturb(torch, d, 5e-2, from_layer=3, keys=("mlp", "out"))
    assert e2e.compare_dumps(d, c, 2e-3).first_divergence().label == "L03.mlp"
    # The rtol threshold decides.
    assert e2e.compare_dumps(d, b, 0.5).first_divergence() is None


def test_divergence_in_layer_zero_fails(e2e):
    import torch

    d = make_dump(torch)
    b = perturb(torch, d, 5e-2, from_layer=0, keys=("out",))
    report = e2e.compare_dumps(d, b, 2e-3)
    assert report.first_divergence().layer == 0
    assert report.exit_code() == 1
    assert "RESULT: FAIL" in e2e.format_report(report, "A", "B")


def test_greedy_token_mismatch_fails(e2e):
    import torch

    d = make_dump(torch)
    b = perturb(torch, d, 1e-4)
    b["prompts"][1]["greedy_token"] = 99
    report = e2e.compare_dumps(d, b, 2e-3)
    assert report.first_divergence() is None
    assert report.token_mismatches() == ["b"]
    assert report.exit_code() == 1


def test_non_finite_counts_as_divergence(e2e):
    import torch

    d = make_dump(torch)
    b = perturb(torch, d, 1e-4)
    b["prompts"][0]["out"][2, 0, 0] = float("nan")
    report = e2e.compare_dumps(d, b, 2e-3)
    assert report.first_divergence().label == "L02"
    assert report.first_divergence().worst == float("inf")


def test_logprob_and_ppl_agreement(e2e):
    import torch

    d = make_dump(torch)
    b = perturb(torch, d, 1e-4)
    d["prompts"][0]["prompt_logprobs"] = [-2.0, -1.0, -3.0]
    b["prompts"][0]["prompt_logprobs"] = [-2.01, -1.0, -3.0]
    b["prompts"][0]["top_logprobs"][3] -= 0.02
    report = e2e.compare_dumps(d, b, 2e-3)
    assert report.top_overlap == [20, 20]
    assert report.top_maxdiff[0] == pytest.approx(0.02, abs=1e-6)
    ((name, ppl_a, ppl_b, worst, _),) = report.ppl
    assert name == "a" and ppl_a != ppl_b and worst == pytest.approx(0.01)
    assert "ppl probe a" in e2e.format_report(report, "A", "B")


def test_incomparable_dumps_are_rejected(e2e):
    import torch

    d = make_dump(torch)
    b = make_dump(torch)
    b["prompts"][0]["token_ids"] = [1, 2, 3]
    with pytest.raises(ValueError, match="token ids"):
        e2e.compare_dumps(d, b, 2e-3)
    with pytest.raises(ValueError, match="no prompt in common"):
        e2e.compare_dumps(d, make_dump(torch, names=("x",)), 2e-3)
    bad = dict(d, version=99)
    with pytest.raises(ValueError, match="version"):
        e2e.compare_dumps(bad, d, 2e-3)


def test_compare_cli_roundtrip(e2e, tmp_path, capsys):
    import torch

    d = make_dump(torch)
    torch.save(d, tmp_path / "a.pt")
    torch.save(perturb(torch, d, 5e-2, from_layer=2), tmp_path / "b.pt")
    code = e2e.main(["compare", str(tmp_path / "a.pt"), str(tmp_path / "a.pt")])
    assert code == 0
    code = e2e.main(["compare", str(tmp_path / "a.pt"), str(tmp_path / "b.pt")])
    out = capsys.readouterr().out
    assert code == 0 and "<-- FIRST" in out and "L02" in out
    code = e2e.main(
        ["compare", str(tmp_path / "a.pt"), str(tmp_path / "b.pt"), "--strict"]
    )
    assert code == 1
    assert e2e.main(["compare", str(tmp_path / "a.pt"), str(tmp_path / "no.pt")]) == 2


def test_row_sampling_and_layer_spec(e2e):
    assert e2e.sample_rows(600) == [0, 67, 133, 200, 266, 333, 399, 466, 532, 599]
    assert e2e.sample_rows(20)[0] == 0 and e2e.sample_rows(20)[-1] == 19
    assert len(e2e.sample_rows(20)) == 10
    assert e2e.sample_rows(3) == [0, 1, 2]
    assert e2e.parse_layer_spec("all", 4) == [0, 1, 2, 3]
    assert e2e.parse_layer_spec("0-1,3", 4) == [0, 1, 3]
    with pytest.raises(ValueError, match="outside"):
        e2e.parse_layer_spec("5", 4)


def test_run_refuses_without_a_device_pin(e2e, monkeypatch, capsys):
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    code = e2e.main(["run", "--model", "/nonexistent", "--out", "/nonexistent/x.pt"])
    assert code == 2
    assert "HIP_VISIBLE_DEVICES" in capsys.readouterr().err


def _stub_model(torch: Any, num_layers: int = 3) -> Any:
    """A tiny model with the module names and call shape the capture relies on."""
    from torch import nn

    hc_count, hidden = 4, 8

    class HC(nn.Module):
        def combine(self, hidden_states, block_output, injection):
            return hidden_states + block_output.repeat(1, hc_count) * injection

    class Qwen4ExpDecoderLayer(nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            self.layer_idx = idx
            self.linear_attn = nn.Linear(hidden, hidden)
            self.mlp = nn.Linear(hidden, hidden)
            self.mlp_hyper_connection = HC()

        def forward(self, hidden_states):
            attn = self.linear_attn(hidden_states[:, :hidden])
            return hidden_states, self.mlp(attn), torch.full_like(hidden_states, 0.5)

    class Mixer(nn.Module):
        def combine_and_mix(self, hidden_states, block_output, injection):
            multi = self.combine(hidden_states, block_output, injection)
            return multi, multi[:, :hidden] * 2, None

        def combine(self, hidden_states, block_output, injection):
            return hidden_states + block_output.repeat(1, hc_count) * injection

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(i) for i in range(num_layers)
            )
            self.hyper_connection_mixer = Mixer()

        def forward(self, hidden_states):
            block, inj = None, None
            for layer in self.layers:
                if block is not None:
                    hidden_states = HC().combine(hidden_states, block, inj)
                hidden_states, block, inj = layer(hidden_states)
            return self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block, inj
            )[1]

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.model = Inner()

    return Outer()


def test_capture_hooks_on_a_stub_model(e2e):
    import torch

    model = _stub_model(torch)
    inner = model.language_model.model
    assert e2e._install_capture(model, [0, 2], True) == [0, 2]
    x = torch.randn(12, 32)
    rows = e2e.sample_rows(12)
    inner(x)  # not armed: nothing recorded
    assert model._trunc_e2e["out"] == {}
    e2e._arm_capture(model, 12, rows)
    final = inner(x)
    got = e2e._collect_capture(model)
    assert sorted(got["out"]) == [0, 2] and sorted(got["attn"]) == [0, 2]
    assert sorted(got["mlp"]) == [0, 2]
    # layer 0 output = materialized combine of that layer's own tuple
    layer0 = inner.layers[0]
    hs, blk, inj = layer0(x)
    want = layer0.mlp_hyper_connection.combine(hs, blk, inj)[rows]
    assert torch.allclose(got["out"][0], want)
    assert (
        got["out"][0].shape == (len(rows), 32) and got["out"][0].dtype == torch.float32
    )
    assert torch.allclose(got["attn"][0], layer0.linear_attn(x[:, :8])[rows])
    assert torch.allclose(got["final"], final[rows])
    assert got["multi_call"] == []
    # collect disarms: a later (decode-sized) forward records nothing new
    inner(x[:1])
    assert model._trunc_e2e["armed"] is False
    # a forward with fewer rows than the prompt (decode / other batch) is skipped
    e2e._arm_capture(model, 12, rows)
    inner(x[:1])
    assert e2e._collect_capture(model)["out"] == {}
    with pytest.raises(RuntimeError, match="not in the model"):
        e2e._install_capture(model, [7], True)
