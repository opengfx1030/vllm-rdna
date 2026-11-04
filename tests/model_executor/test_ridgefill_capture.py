# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the RidgeFill statistics capture and its fitter.

The modules load by file path, so no vllm package import (and no GPU) is
needed: ``capture_probe.py`` (worker extension), ``capture_activations.py``
(driver) and ``fit_stats.py`` (fitter).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPLORE = ROOT / "docs" / "explore" / "ridgefill"
PKG = ROOT / "vllm" / "model_executor" / "layers" / "ridgefill"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe: Any = _load(PKG / "capture_probe.py", "ridgefill_test_capture_probe")
capture: Any = _load(EXPLORE / "capture_activations.py", "ridgefill_test_capture")
fit: Any = _load(EXPLORE / "fit_stats.py", "ridgefill_test_fit_stats")
rf_math: Any = _load(PKG / "math.py", "ridgefill_test_math")
loader: Any = _load(PKG / "loader.py", "ridgefill_test_loader")


def _worker(model: Any, rank: int = 0) -> Any:
    """VLLM adds the extension as a base of the worker; this is that worker."""
    worker = probe.RidgeFillProbe()
    worker.model_runner = SimpleNamespace(model=model)
    worker.rank = rank
    return worker


def _synthetic_model(din: int = 8, dout: int = 4, layers: int = 4, vocab: int = 32):
    torch.manual_seed(3)
    return capture._synthetic_classes(torch)(din, dout, layers, vocab=vocab)


def _record(model, splits: list[int], end: int) -> dict:
    """Every forward's split states and block inputs, in fp64."""
    rec: dict = {"x": {s: [] for s in splits}, "y": {}}
    for split in splits:

        def split_hook(module, args, kwargs, split=split):
            state = probe.split_state(module, args, kwargs)
            rec["x"][split].append(state.double())

        model.layers[split].register_forward_pre_hook(split_hook, with_kwargs=True)
    for layer in range(min(splits), end):
        rec["y"][layer] = []

        def attn_hook(module, args, kwargs, layer=layer):
            rec["y"][layer].append(kwargs["hidden_states"].double())

        model.layers[layer].self_attn.register_forward_pre_hook(
            attn_hook, with_kwargs=True
        )
    return rec


def _close(got, want, rel: float = 1e-5) -> None:
    scale = float(want.abs().max()) or 1.0
    assert float((got.double() - want.double()).abs().max()) <= rel * scale


def test_real_capture_needs_a_corpus(tmp_path: Path) -> None:
    """Only --synthetic runs without a corpus; nothing is written first."""
    out = tmp_path / "out"
    model = str(tmp_path / "ckpt")
    with pytest.raises(SystemExit, match="--prompts-file or --corpus-json"):
        capture.main(["--model", model, "--output", str(out)])
    with pytest.raises(SystemExit, match="--held-json"):
        capture.main(
            [
                "--model",
                model,
                "--stats",
                "--corpus-json",
                "fit.json",
                "--output",
                str(out),
            ]
        )
    with pytest.raises(SystemExit, match="--synthetic"):
        capture.main(["--output", str(out)])
    # The placeholder corpus is gone: no prompts file, no prompts.
    with pytest.raises(SystemExit, match="--prompts-file"):
        capture._read_prompts(SimpleNamespace(prompts_file=""))
    assert not out.exists()
    capture._require_corpus(
        SimpleNamespace(synthetic=True, stats=True, corpus_json="", held_json="")
    )


def test_corpus_windows_are_checked(tmp_path: Path) -> None:
    """A window must hold meta.length ids whose ``ppl_long.ids_sha`` matches."""
    path = tmp_path / "windows.json"
    ids = [5, 6, 7, 8]
    sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    good = {"meta": {"length": 4}, "windows": [{"ids": ids, "sha": sha}]}
    path.write_text(json.dumps(good))
    meta, windows = capture._read_windows(str(path))
    assert meta["length"] == 4 and windows[0]["ids"] == ids
    path.write_text(json.dumps({"meta": {"length": 8}, "windows": [{"ids": ids}]}))
    with pytest.raises(SystemExit, match="want 8"):
        capture._read_windows(str(path))
    bad = {"meta": {"length": 4}, "windows": [{"ids": ids, "sha": "0" * 16}]}
    path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="sha"):
        capture._read_windows(str(path))


@pytest.mark.parametrize("device_sums", [True, False])
def test_stats_accumulator_matches_direct_sums(
    tmp_path: Path, device_sums: bool
) -> None:
    """Rank 0's fp64 sums equal the direct sums of the prefill rows.

    Two chunked prefill forwards (6 + 3 rows) count; the one-row decode
    forward does not. The sums are of shifted rows; adding the shift back
    gives sum x, X^T X, sum y, sum y^2 and X^T Y. With ``device_sums`` the
    forwards add into fp32 accumulators that the mode switch folds once.
    """
    model = _synthetic_model()
    worker = _worker(model)
    splits, end = [1, 2], 4
    begun = worker.ridgefill_stats_begin(splits, end, 2, 2, 3, 0, device_sums)
    assert begun == {"rank": 0, "layers": 4, "splits": splits}
    rec = _record(model, splits, end)
    ids = torch.randint(32, (10,))
    worker.ridgefill_stats_mode("stats", 0)
    with torch.no_grad():
        model(input_ids=ids[:6], positions=torch.arange(6))
        model(input_ids=ids[6:9], positions=torch.arange(6, 9))
        model(input_ids=ids[9:], positions=torch.tensor([9]))
    done = worker.ridgefill_stats_mode("off", -1)
    assert done["mode"] == "stats" and done["chunks"] == [(0, 6), (6, 3)]
    snap = worker.ridgefill_stats_snapshot(str(tmp_path / "stats.pt"))
    stats = torch.load(tmp_path / "stats.pt", weights_only=True)
    n = 9
    assert stats["rows"] == n and stats["forwards"] == 2 and snap["rows"] == n
    assert snap["bytes"] == (tmp_path / "stats.pt").stat().st_size
    assert snap["device_sums"] is device_sums
    assert snap["folds"] == (1 if device_sums else 0)
    for split in splits:
        x = torch.cat(rec["x"][split][:2])
        sx = stats["x"][split]
        shift, total = sx["shift"], sx["sum"]
        _close(total + n * shift, x.sum(dim=0))
        gram = (
            sx["gram"]
            + torch.outer(shift, total)
            + torch.outer(total, shift)
            + n * torch.outer(shift, shift)
        )
        _close(gram, x.T @ x)
        _close(torch.tensor(snap["check"][str(split)]), x.sum(dim=0))
        for layer in range(split, end):
            y = torch.cat(rec["y"][layer][:2])
            sy = stats["y"][layer]
            cross = (
                stats["cross"][split][layer]
                + torch.outer(shift, sy["sum"])
                + torch.outer(total, sy["shift"])
                + n * torch.outer(shift, sy["shift"])
            )
            _close(cross, x.T @ y)
            _close(sy["sum"] + n * sy["shift"], y.sum(dim=0))
            sumsq = sy["sumsq"] + 2 * sy["shift"] * sy["sum"] + n * sy["shift"] ** 2
            _close(sumsq, (y * y).sum(dim=0))
    assert sorted(stats["cross"][2]) == [2, 3]
    # Stats mode keeps the fit prompts' first positions as fp16 rows.
    worker.ridgefill_stats_save_rows("head", str(tmp_path / "head.pt"))
    head = torch.load(tmp_path / "head.pt", weights_only=True)
    assert head["positions"].tolist() == [0, 1, 2]
    assert torch.equal(head["x"][1], rec["x"][1][0][:3].half())
    assert torch.equal(head["y"][3], rec["y"][3][0][:3].half())
    # Raw mode keeps positions below 2 and every 2nd one, and adds no sums.
    worker.ridgefill_stats_mode("raw", 5)
    with torch.no_grad():
        model(input_ids=ids[:6], positions=torch.arange(6))
    worker.ridgefill_stats_mode("off", -1)
    worker.ridgefill_stats_save_rows("held", str(tmp_path / "held.pt"))
    held = torch.load(tmp_path / "held.pt", weights_only=True)
    keep = [0, 1, 2, 4]
    assert held["positions"].tolist() == keep
    assert held["prompts"].tolist() == [5] * len(keep)
    assert torch.equal(held["x"][2], rec["x"][2][3][keep].half())
    assert torch.equal(held["y"][1], rec["y"][1][3][keep].half())
    assert worker.ridgefill_stats_snapshot(str(tmp_path / "again.pt"))["rows"] == n


def test_stats_accumulator_folds_every_forward_when_device_sums_do_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device accumulator that does not fit ends device sums for the run.

    The first forward's cross term runs out of memory: the sums already
    added on the device fold at once and every later forward folds directly,
    so the totals still equal the direct sums.
    """
    zeros = torch.zeros

    def no_room(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("dtype") is torch.float32 and args[0] == torch.Size([4, 2]):
            raise torch.OutOfMemoryError("no room for the cross term")
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", no_room)
    acc = probe.StatsAccumulator([0], 2, device_sums=True)
    xs = [torch.randn(5, 4) for _ in range(2)]
    ys = [torch.randn(5, 2) for _ in range(2)]
    for x, y in zip(xs, ys):
        acc.add_split(0, x)
        acc.add_target(0, y)
        acc.add_target(1, 2 * y)
    stats = acc.payload()
    assert stats["device_sums"] is False and acc.folds == 1
    assert stats["rows"] == 10 and stats["forwards"] == 2
    x, y = torch.cat(xs).double(), torch.cat(ys).double()
    sx = stats["x"][0]
    _close(sx["sum"] + 10 * sx["shift"], x.sum(dim=0))
    u = x - sx["shift"]
    _close(sx["gram"], u.T @ u)
    for layer, target in ((0, y), (1, 2 * y)):
        sy = stats["y"][layer]
        v = target - sy["shift"]
        _close(sy["sum"], v.sum(dim=0))
        _close(sy["sumsq"], (v * v).sum(dim=0))
        _close(stats["cross"][0][layer], u.T @ v)


def _stats_from_rows(x, ys: dict, split: int) -> dict:
    """An exact fp64 snapshot of ``x`` and ``ys`` with nonzero shifts."""
    shift = x[0].clone()
    u = x - shift
    stats: dict = {
        "format": fit.STATS_FORMAT,
        "rows": int(x.shape[0]),
        "x": {split: {"shift": shift, "sum": u.sum(dim=0), "gram": u.T @ u}},
        "y": {},
        "cross": {split: {}},
    }
    for layer, y in ys.items():
        shift_y = y[1].clone()
        v = y - shift_y
        stats["y"][layer] = {
            "shift": shift_y,
            "sum": v.sum(dim=0),
            "sumsq": (v * v).sum(dim=0),
        }
        stats["cross"][split][layer] = u.T @ v
    return stats


def test_stats_fit_matches_fit_ridge() -> None:
    """The eigendecomposition fit from sums is ``math.fit_ridge`` on the rows."""
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(40, 6, generator=gen, dtype=torch.float64) * 3 + 7
    w = torch.randn(6, 3, generator=gen, dtype=torch.float64)
    y = x @ w + 0.1 * torch.randn(40, 3, generator=gen, dtype=torch.float64) + 2
    moments = fit.Moments(_stats_from_rows(x, {0: y}, 0), 0)
    solver = fit.RidgeSolver(moments)
    mu_y, cross, centered, raw = moments.layer(0)
    proj = solver.project(cross)
    assert raw == pytest.approx(float((y * y).sum()), rel=1e-12)
    for lam in (0.003, 0.03, 0.3):
        a, b = solver.maps(proj, moments.mu_x, mu_y, lam, torch.float64)
        want_a, want_b = rf_math.fit_ridge(x.tolist(), y.tolist(), lam)
        want_a = torch.tensor(want_a, dtype=torch.float64)
        want_b = torch.tensor(want_b, dtype=torch.float64)
        assert torch.allclose(a, want_a, rtol=1e-9, atol=1e-11)
        assert torch.allclose(b, want_b, rtol=1e-9, atol=1e-10)
        sse = float(((x @ a.T + b - y) ** 2).sum())
        assert solver.train_sse(proj, centered, lam) == pytest.approx(sse, rel=1e-8)
    # Subtracting head rows from the sums fits the remaining rows.
    head = {"x": x[:5], "y": {0: y[:5]}}
    moments = fit.Moments(_stats_from_rows(x, {0: y}, 0), 0, head)
    solver = fit.RidgeSolver(moments)
    mu_y, cross, _, _ = moments.layer(0)
    a, b = solver.maps(solver.project(cross), moments.mu_x, mu_y, 0.03, torch.float64)
    want_a, want_b = rf_math.fit_ridge(x[5:].tolist(), y[5:].tolist(), 0.03)
    assert torch.allclose(a, torch.tensor(want_a, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(b, torch.tensor(want_b, dtype=torch.float64), atol=1e-8)


class _FakeHC(torch.nn.Module):
    def combine(self, hidden_states, block_output, injection):
        return hidden_states + injection * block_output


class _FakeAttn(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _FakeLayer(torch.nn.Module):
    """Delayed combine as in Qwen4ExpDecoderLayer: block output stays pending."""

    def __init__(self) -> None:
        super().__init__()
        self.attn_hyper_connection = _FakeHC()
        self.self_attn = _FakeAttn()

    def forward(
        self, hidden_states, prev_block_output=None, prev_injection=None, positions=None
    ):
        if prev_block_output is not None and prev_injection is not None:
            hidden_states = self.attn_hyper_connection.combine(
                hidden_states, prev_block_output, prev_injection
            )
        out = self.self_attn(hidden_states=2 * hidden_states)
        injection = torch.full((hidden_states.shape[0], 1), 0.5)
        return hidden_states, out + 1, injection


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(3)])

    def forward(self, hidden_states, positions=None):
        block_output = injection = None
        for layer in self.layers:
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
            )
        return hidden_states


def test_probe_records_the_combined_split_state(tmp_path: Path) -> None:
    """Every capture path records the split layer's materialized HC state.

    Layer 0 leaves ``(2 x + 1, 0.5)`` pending, so the state entering layer 1
    is ``x + 0.5 (2 x + 1) = 2 x + 0.5``, not the ``x`` its forward receives.
    Layer 0 has nothing pending and records ``x`` itself.
    """
    # Multiples of 1/8: every sum below is exact in fp32 and fp16.
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 8
    combined = 2 * x + 0.5
    positions = torch.arange(5)

    # Legacy raw capture (rank file).
    model = _FakeModel()
    worker = _worker(model)
    worker.ridgefill_capture(1, 3)
    with torch.no_grad():
        model(x, positions=positions)
    worker.ridgefill_save(str(tmp_path / "raw"))
    saved = torch.load(tmp_path / "raw" / "rank0.pt", weights_only=True)
    assert torch.equal(saved["split"][0], combined)
    assert torch.equal(saved["inputs"][1][0], 2 * combined)

    # The driver's in-process hooks.
    model = _FakeModel()
    hooks = capture._CaptureHooks(split=1, end=3)
    hooks.register(model)
    with torch.no_grad():
        model(x, positions=positions)
    assert torch.equal(hooks.split_state[0], combined)

    # Stats and raw modes, with a split that has nothing pending.
    model = _FakeModel()
    worker = _worker(model)
    worker.ridgefill_stats_begin([0, 1], 3, 1, 0, 0, 0)
    worker.ridgefill_stats_mode("stats", 0)
    with torch.no_grad():
        model(x, positions=positions)
    worker.ridgefill_stats_mode("raw", 0)
    with torch.no_grad():
        model(x, positions=positions)
    worker.ridgefill_stats_mode("off", -1)
    worker.ridgefill_stats_snapshot(str(tmp_path / "stats.pt"))
    stats = torch.load(tmp_path / "stats.pt", weights_only=True)
    for split, state in ((0, x), (1, combined)):
        sx = stats["x"][split]
        _close(sx["sum"] + 5 * sx["shift"], state.double().sum(dim=0))
    worker.ridgefill_stats_save_rows("held", str(tmp_path / "held.pt"))
    held = torch.load(tmp_path / "held.pt", weights_only=True)
    assert torch.equal(held["x"][1], combined.half())
    assert torch.equal(held["x"][0], x.half())


def test_synthetic_stats_capture_and_fit_end_to_end(tmp_path: Path) -> None:
    """Capture (two fake TP ranks, chunked prefill) -> fit -> checked bundles."""
    cap = tmp_path / "capture"
    rc = capture.main(
        [
            "--synthetic",
            "--stats",
            "--output",
            str(cap),
            "--prompts",
            "4",
            "--tokens",
            "40",
            "--held-prompts",
            "2",
            "--splits",
            "1,2",
            "--snapshots",
            "2,4",
            "--max-batched-tokens",
            "16",
            "--raw-stride",
            "2",
            "--raw-head",
            "4",
            "--stats-head",
            "4",
            "--synthetic-din",
            "8",
            "--synthetic-dout",
            "4",
            "--synthetic-layers",
            "4",
            "--synthetic-vocab",
            "64",
            "--tp",
            "2",
        ]
    )
    assert rc == 0
    meta = json.loads((cap / "capture_meta.json").read_text())
    assert meta["complete"] and meta["splits"] == [1, 2] and meta["end"] == 4
    assert [s["prompts"] for s in meta["snapshots"]] == [2, 4]
    assert [s["rows"] for s in meta["snapshots"]] == [80, 160]
    assert all(s["rank_sum_x_max_rel"] == 0.0 for s in meta["snapshots"])
    # Device sums by default: one fold per fit prompt, none per forward.
    assert [s["folds"] for s in meta["snapshots"]] == [2, 4]
    assert all(s["device_sums"] for s in meta["snapshots"])
    assert all(p["forward_rows"] == [16, 16, 8] for p in meta["prompts"])
    # Held rows: positions 0-3 and the even ones, 22 per prompt.
    assert meta["rows"]["held"]["rows"] == 44
    assert meta["rows"]["head"]["rows"] == 16
    with pytest.raises(SystemExit, match="already holds a capture"):
        capture.main(["--synthetic", "--stats", "--output", str(cap)])

    out = tmp_path / "fit"
    common = ["--capture", str(cap), "--lams", "0.01,0.1", "--fill-limit", "24"]
    assert fit.main([*common, "--out", str(out), "--threads", "1"]) == 0
    assert (out / "DONE").exists()
    for split, layers in ((1, [1, 2, 3]), (2, [2, 3])):
        bundle = out / f"bundle_s{split}"
        names = sorted(p.name for p in bundle.iterdir())
        assert names == ["maps.pt", "pins.json", "ridgefill.json"]
        manifest = json.loads((bundle / "ridgefill.json").read_text())
        assert manifest["split"] == split
        assert sorted(int(k) for k in manifest["layers"]) == layers
        assert manifest["lambda_mode"] == "per-layer"
        assert all(v["lambda"] in (0.01, 0.1) for v in manifest["layers"].values())
        loader.require_bundle(str(bundle), str(bundle / "pins.json"))
        maps = torch.load(bundle / "maps.pt", weights_only=True)
        assert sorted(maps) == layers and maps[layers[0]][0].dtype == torch.float16
    report = json.loads((out / "report.json").read_text())
    assert set(report["splits"]) == {"1", "2"}
    assert set(report["splits"]["1"]["snapshots"]) == {"2", "4"}
    assert set(report["splits"]["2"]["curve"]) == {"2", "4"}
    score = report["splits"]["1"]["snapshots"]["4"]["layers"]["3"]["0.1"]
    assert set(score) == {"held", "fill", "buckets", "train"}
    assert set(score["buckets"]) == {"0", "1-3", "4-15", "16-63", "64-511", "512+"}
    assert score["buckets"]["512+"] is None and score["held"] > 0
    with pytest.raises(SystemExit, match="DONE"):
        fit.main([*common, "--out", str(out)])

    # Scored as old maps on the same rows, the split-1 bundle gives back its
    # own held-out errors.
    old = out / "bundle_s1" / "maps.pt"
    out2 = tmp_path / "fit2"
    args = ["--out", str(out2), "--splits", "1", "--old-maps", str(old)]
    assert fit.main([*common, *args, "--old-split", "1", "--threads", "1"]) == 0
    scored = json.loads((out2 / "report.json").read_text())["splits"]["1"]["old_maps"]
    manifest = json.loads((out / "bundle_s1" / "ridgefill.json").read_text())
    for layer, info in manifest["layers"].items():
        want = info["fit"]["held_rel_rms_fp16"]
        assert scored["layers"][layer]["held"] == pytest.approx(want, rel=1e-12)
