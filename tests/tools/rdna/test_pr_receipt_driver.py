# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/rdna/pr_receipt.sh against stub trees and a stub serve_validate.sh.

The driver is executed for real; the servers are not: SERVE_VALIDATE points at
a stub that writes the files serve_validate.sh would (summary.txt, bench JSON,
concurrency probe output, quality collects) and records how it was called.
No GPU, server or network is touched.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "tools/rdna/pr_receipt.sh"

PLAN_STUB = """#!/usr/bin/env bash
tree=$(cd "$(dirname "$0")/../.." && pwd)
echo "export PYTHONPATH='$tree'"
echo "# args: $*" | sed "s#$tree#<tree>#g" > /dev/null
extra=$(cat "$tree/plan_extra" 2> /dev/null)
echo "'$tree/bin/python' '--compilation-config' '{\\"mode\\":3}' $extra"
"""

VALIDATE_WRAPPER = """#!/usr/bin/env bash
exec "$STUB_PYTHON" "$(dirname "$0")/stub_validate.py" "$@"
"""

VALIDATE_STUB = r"""
import json
import os
import shutil
import sys
from pathlib import Path

out = Path(os.environ["OUT"])
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads(Path(os.environ["STUB_CONFIG"]).read_text())
boot = os.environ["TAG"].rsplit("-", 1)[1]
role = "head" if boot in ("XB", "B1", "B2") else "base"
args = sys.argv[1:]
keys = ("CELLS", "BENCH_EXTRA", "RESET_PREFIX_CACHE", "CONC_PROBE", "QUIET_HOLD",
        "LOAD_HOOK", "SERVE_TREE", "VENV", "QUALITY_WINDOWS", "QUALITY_LIMIT",
        "HIP_VISIBLE_DEVICES", "QUIET_OWN", "QUIET_WIN_MAX", "RECEIPT_PY")
with open(cfg["calls"], "a") as f:
    f.write(json.dumps({"boot": boot, "args": args,
                        "env": {k: os.environ.get(k) for k in keys}}) + "\n")
if boot in cfg.get("fail_boots", []):
    (out / "summary.txt").write_text("[10:00:00] BOOT FAILED after 1800s\n")
    sys.exit(1)

lines = ["[10:00:00] boot stub", "[10:02:00] READY in 100s served=m",
         "[10:02:10] PROBE PASS", "[10:02:20] PPL PROBE PASS", "[10:02:30] PREFIX PASS"]
exact = "VLLM_RDNA2_MOE_DETERMINISTIC=1" in args
if os.environ.get("CONC_PROBE") == "1":
    tokens = ["token_id:1", "token_id:2"]
    head_tokens = ["token_id:1", "token_id:9"] if cfg.get("exact_differs") else tokens
    toks = head_tokens if role == "head" else tokens
    rec = lambda t: {"tokens": t, "text": "x"}
    saved = {"solo": {"0": rec(toks)}, "solo2": {"0": rec(toks)},
             "long_solo": {"0": rec(tokens)}, "together": {}}
    (out / "concurrency.json").write_text(json.dumps(saved))
    failing = cfg.get("probe_failing", {}).get(role, 0) if exact else 0
    (out / "concurrency.txt").write_text(
        f"RESULT {'PASS' if not failing else 'FAIL'}: {failing} failing, 1 near-tie, "
        "0 soft, 0 solo-noise, 5 identical of 6 checks (1.0s)\n")
    lines += ["[10:03:30] CONCURRENCY PASS", "    RESULT PASS: 0 failing"]
cells = os.environ["CELLS"].split()
for cell in ([] if cells == ["none"] else cells):
    parts = cell.split(":")
    in_len, out_len, conc = (int(p) for p in parts[:3])
    n = int(parts[3]) if len(parts) > 3 else (4 if conc == 1 else max(conc * 4, 8))
    name = f"bench-{in_len}x{out_len}-c{conc}"
    speed = dict(cfg["speed"][role])
    ttft = speed["ttft"] * (1 + 0.05 * conc)
    data = {"completed": n, "failed": 0, "duration": 20.0,
            "total_input_tokens": n * in_len, "max_concurrency": conc,
            "mean_ttft_ms": ttft, "median_ttft_ms": ttft, "p99_ttft_ms": ttft * 1.5,
            "mean_tpot_ms": 22.0, "median_itl_ms": speed["itl"],
            "mean_itl_ms": speed["itl"], "p99_itl_ms": speed["itl"] * 2,
            "output_throughput": speed["out"],
            "spec_decode_acceptance_length": 2.9}
    if "--save-detailed" in os.environ.get("BENCH_EXTRA", ""):
        data["generated_texts"] = [f"text {i}" for i in range(n)]
    (out / f"{name}.json").write_text(json.dumps(data))
    evict = cfg.get("invalid", {}).get(f"{boot}:{name}")
    lines += [f"[10:04:00] PREFIX RESET PASS before {name}",
              f"[10:04:00] bench {name} (n={n})", f"    {name} prefill 1.0",
              "    evicted_ms Worker_TP0=+0 -> " + ("INVALID (x)" if evict else "ok")]
if os.environ.get("LOAD_HOOK"):
    shutil.copy(cfg["npz"][boot], out / "topk.npz")
    shutil.copy(cfg["quality"][boot], out / "quality.json")
    lines.append("[10:08:00] LOAD HOOK rc=0 PASS pr_receipt quality collects")
lines += ["[10:09:00] PROBE-AFTER-LOAD PASS",
          "[10:09:01] evicted_ms since READY: Worker_TP0=+0 -> ok",
          "[10:09:02] fault markers in serve.log: 0",
          "[10:09:03] new amdgpu dmesg faults: 0", "[10:09:04] done"]
(out / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
"""


def make_tree(root: Path, name: str, sha: str, extra: str = "") -> Path:
    tree = root / name
    (tree / "tools/rdna/recipes").mkdir(parents=True)
    (tree / "tools/rdna/recipes/unit.env").write_text("")
    serve = tree / "tools/rdna/serve_rdna.sh"
    serve.write_text(PLAN_STUB)
    serve.chmod(0o755)
    (tree / "vllm").mkdir()
    (tree / "vllm/_rocm_C.abi3.so").write_bytes(b"so " + name.encode())
    (tree / ".sha").write_text(sha + "\n")
    if extra:
        (tree / "plan_extra").write_text(extra)
    return tree


class Run:
    def __init__(self, tmp_path: Path, **config):
        self.tmp = tmp_path
        self.base = make_tree(tmp_path, "base", "b" * 40)
        self.head = make_tree(tmp_path, "head", "c" * 40, config.pop("head_plan", ""))
        self.calls = tmp_path / "calls.jsonl"
        self.config = {
            "calls": str(self.calls),
            "speed": {
                "base": {"ttft": 700.0, "itl": 20.0, "out": 160.0},
                "head": {"ttft": 700.0, "itl": 20.0, "out": 160.0},
            },
            **config,
        }
        (tmp_path / "stub_validate.py").write_text(VALIDATE_STUB)
        wrapper = tmp_path / "stub_validate.sh"
        wrapper.write_text(VALIDATE_WRAPPER)
        wrapper.chmod(0o755)
        (tmp_path / "home").mkdir()
        self.out = tmp_path / "out"

    def run(self, extra_env=None, args=()):
        cfg = self.tmp / "stub_config.json"
        cfg.write_text(json.dumps(self.config))
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.tmp / "home"),
            "USER": "tester",
            "BASE_TREE": str(self.base),
            "HEAD_TREE": str(self.head),
            "VENV": str(self.tmp / "no-venv"),
            "PYTHON": sys.executable,
            "STUB_PYTHON": sys.executable,
            "STUB_CONFIG": str(cfg),
            "SERVE_VALIDATE": str(self.tmp / "stub_validate.sh"),
            "RECIPE": "unit",
            "MODEL": "/models/snapshots/" + "d" * 40,
            "GPUS": "2,3,4,5",
            "OUT": str(self.out),
            "CELLS": "1024:512:1 1024:512:4",
            "EXACT_CELLS": "1024:128:1:2",
            "TEMPWATCH": "0",
            **(extra_env or {}),
        }
        return subprocess.run(
            ["bash", str(DRIVER), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

    def boots(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def receipt(self):
        return json.loads((self.out / "receipt.json").read_text())


@pytest.fixture()
def harness(tmp_path):
    made: list[Path] = []

    def make(**config):
        root = tmp_path / f"run{len(made)}"
        root.mkdir()
        made.append(root)
        return Run(root, **config)

    return make


def test_dry_run_lists_the_boots(harness):
    h = harness()
    expect = {
        ("ab", "0", "1"): "XA:base:exact XB:head:exact A1:base:speed B1:head:speed",
        ("ab", "1", "1"): (
            "XA:base:exact XB:head:exact A1:base:speed B1:head:speed A2:base:floor"
        ),
        ("abba", "1", "1"): (
            "XA:base:exact XB:head:exact A1:base:speed B1:head:speed "
            "B2:head:speed A2:base:speed"
        ),
        ("ab", "0", "0"): "A1:base:speed B1:head:speed",
    }
    for i, ((mode, quality, exact), boots) in enumerate(expect.items()):
        h.out = h.tmp / f"dry{i}"
        res = h.run(
            {
                "MODE": mode,
                "QUALITY": quality,
                "EXACT": exact,
                "QUALITY_WINDOWS": str(h.tmp / "w.json"),
                "DRY_RUN": "1",
            }
        )
        assert res.returncode == 0, res.stderr
        assert f"boots: {boots}\n" in res.stdout
        assert not h.boots()  # nothing was booted


def test_dry_run_reports_launch_plan_differences(harness):
    same = harness()
    res = same.run({"DRY_RUN": "1"})
    assert "launch plans: identical;" in res.stdout
    assert '{"mode":3}' in res.stdout
    other = harness(head_plan="'--new-flag'")
    other.out = other.tmp / "dry2"
    res = other.run({"DRY_RUN": "1"})
    assert "launch plans: 2 lines differ;" in res.stdout
    diff = (other.out / "plans/diff.txt").read_text()
    assert "--new-flag" in diff and "<TREE>" in diff  # tree paths are masked
    assert str(other.head) not in diff and str(other.base) not in diff


def test_inputs_are_checked(harness):
    h = harness()
    assert h.run({"EAGER": "1"}).returncode == 2
    res = h.run(args=["EAGER=1"])
    assert res.returncode == 2 and "production mode" in res.stderr
    assert h.run({"MODE": "abab"}).returncode == 2
    assert h.run({"RECIPE": "nope"}).returncode == 2
    assert h.run({"CELLS": "1024:512"}).returncode == 2
    assert h.run({"EXACT": "0", "CELLS": "none"}).returncode == 2
    assert not h.boots()


def test_ab_run_writes_the_receipt(harness):
    h = harness(
        speed={
            "base": {"ttft": 700.0, "itl": 20.0, "out": 160.0},
            "head": {"ttft": 800.0, "itl": 20.0, "out": 160.0},
        }
    )
    res = h.run(args=["SEQS=4"])
    assert res.returncode == 0, res.stdout + res.stderr
    boots = h.boots()
    assert [b["boot"] for b in boots] == ["XA", "XB", "A1", "B1"]
    xa, xb, a1, b1 = boots
    for b in (xa, xb):  # deterministic boots: cells for hashing, no speed window
        assert "VLLM_RDNA2_MOE_DETERMINISTIC=1" in b["args"]
        assert b["env"]["CELLS"] == "1024:128:1:2"
        assert b["env"]["BENCH_EXTRA"] == "--temperature 0 --save-detailed"
        assert b["env"]["QUIET_HOLD"] == "0" and b["env"]["CONC_PROBE"] == "1"
    for b in (a1, b1):  # production boots: no deterministic flag, measured window
        assert "VLLM_RDNA2_MOE_DETERMINISTIC=1" not in b["args"]
        assert b["env"]["CELLS"] == "1024:512:1 1024:512:4"
        assert b["env"]["BENCH_EXTRA"] == "--temperature 0"
        assert b["env"]["QUIET_HOLD"] == "1" and not b["env"]["LOAD_HOOK"]
    for b in boots:
        assert b["args"][:3] == ["VLLM_SERVER_DEV_MODE=1", "SEQS=4"] + b["args"][2:3]
        assert b["env"]["RESET_PREFIX_CACHE"] == "1"
        assert b["env"]["HIP_VISIBLE_DEVICES"] == "2,3,4,5"
        assert b["env"]["QUIET_OWN"] == "tester" and b["env"]["QUIET_WIN_MAX"] == "240"
    assert xa["env"]["SERVE_TREE"] == str(h.base) == a1["env"]["SERVE_TREE"]
    assert xb["env"]["SERVE_TREE"] == str(h.head) == b1["env"]["SERVE_TREE"]
    receipt = h.receipt()
    assert receipt["gates"] == {"exact": "PASS", "quality": "NOT RUN"}
    assert receipt["ok"] is True
    assert {c["verdict"] for c in receipt["speed"]["cells"]} == {"worse"}
    text = (h.out / "receipt.md").read_text()
    assert f"# RDNA receipt: {'c' * 10} against {'b' * 10}" in text
    assert "ABBA needed" in text and "MODE=abba" in text
    assert '`{"mode":3}`' in text  # compilation config of the head plan
    assert "bash tools/rdna/pr_receipt.sh SEQS=4" in text
    assert "GPUS=2,3,4,5" in text
    manifest = json.loads((h.out / "manifest.json").read_text())
    assert [b["id"] for b in manifest["boots"]] == ["XA", "XB", "A1", "B1"]
    assert manifest["settings"]["model_rev"] == "d" * 40


def test_abba_run_marks_regression_and_invalid_cells(harness):
    h = harness(
        speed={
            "base": {"ttft": 700.0, "itl": 20.0, "out": 160.0},
            "head": {"ttft": 800.0, "itl": 20.0, "out": 160.0},
        },
        invalid={"A2:bench-1024x512-c4": True},
    )
    res = h.run({"MODE": "abba", "EXACT": "0"})
    assert res.returncode == 0, res.stdout + res.stderr
    assert [b["boot"] for b in h.boots()] == ["A1", "B1", "B2", "A2"]
    speed = h.receipt()["speed"]
    verdicts = {c["name"]: c["verdict"] for c in speed["cells"]}
    assert verdicts == {
        "bench-1024x512-c1": "regression",  # both head runs slower than both base
        "bench-1024x512-c4": "invalid",  # KFD eviction in A2
    }
    text = (h.out / "receipt.md").read_text()
    assert "REGRESSION" in text and "INVALID" in text
    assert '`CELLS="1024:512:4"`' in text  # how to rerun the invalid cell
    assert "Speculative acceptance length" in text


def test_exact_failure_stops_the_run_unless_told_otherwise(harness):
    h = harness(exact_differs=True)
    res = h.run()
    assert res.returncode == 1
    assert [b["boot"] for b in h.boots()] == ["XA", "XB"]  # fail fast
    assert h.receipt()["gates"]["exact"] == "FAIL"
    assert "long_solo" not in (h.out / "receipt.md").read_text()
    h2 = harness(exact_differs=True)
    res = h2.run({"FAIL_FAST": "0"})
    assert res.returncode == 1
    assert [b["boot"] for b in h2.boots()] == ["XA", "XB", "A1", "B1"]


def test_a_failing_concurrency_probe_fails_the_gate(harness):
    h = harness(probe_failing={"head": 2})
    assert h.run().returncode == 1
    assert h.receipt()["exact"]["probe"]["head"]["failing"] == 2


def test_a_boot_that_fails_ends_the_run(harness):
    h = harness(fail_boots=["A1"])
    res = h.run()
    assert res.returncode == 1
    assert [b["boot"] for b in h.boots()] == ["XA", "XB", "A1"]
    receipt = h.receipt()
    assert receipt["boot_failures"] == ["A1"] and receipt["ok"] is False


def test_a_second_run_in_the_same_directory_is_refused(harness):
    h = harness()
    assert h.run({"EXACT": "0"}).returncode == 0
    res = h.run({"EXACT": "0"})
    assert res.returncode == 2 and "already holds" in res.stderr


# ------------------------------------------------------------------ quality

np = pytest.importorskip("numpy")


def write_topk(path: Path, seed: int, noise: float) -> None:
    """A tiny topk_kl.py collect file: 6 windows of 32 positions, vocab 10."""
    windows, length, vocab, top = 6, 32, 10, 4
    shared = np.random.default_rng(0)
    logits = shared.normal(0, 3.0, size=(windows, length, vocab))
    act_id = shared.integers(0, vocab, size=(windows, length)).astype(np.int32)
    rng = np.random.default_rng(seed)
    logits = logits + rng.normal(0, noise, size=logits.shape)
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    order = np.argsort(-lp, axis=-1)[..., :top]
    act_lp = np.take_along_axis(lp, act_id[..., None], -1)[..., 0]
    meta = {
        "windows_sha256": "w",
        "top": top,
        "seq_len": length + 1,
        "domains": {"docs": {"windows": windows}},
    }
    np.savez_compressed(
        path,
        **{
            "docs.act_lp": act_lp,
            "docs.act_rank": (lp > act_lp[..., None]).sum(-1) + 1,
            "docs.top_ids": order.astype(np.int32),
            "docs.top_lp": np.take_along_axis(lp, order, -1),
            "docs.act_id": act_id,
            "meta": np.asarray(json.dumps(meta)),
        },
    )


def write_quality(path: Path, ppl: float) -> None:
    tokens = ["a", "b", "c", "d"]
    passage = {
        "prompt_tokens": 4,
        "nll": 1.6,
        "ppl": ppl,
        "token_logprobs": [None, -1.5, -1.6, -1.7],
        "tokens": tokens,
        "greedy_text": "abc",
        "greedy_tokens": ["x", "y"],
        "greedy_logprobs": [-0.1, -0.2],
    }
    path.write_text(json.dumps({"model": "m", "passages": [passage], "ppl": ppl}))


def quality_config(tmp: Path, head_noise: float, head_ppl: float = 5.0) -> dict:
    npz, quality = {}, {}
    for i, (boot, noise) in enumerate(
        [("A1", 0.05), ("A2", 0.05), ("B1", head_noise), ("B2", head_noise)]
    ):
        npz[boot] = str(tmp / f"{boot}.npz")
        write_topk(Path(npz[boot]), seed=10 + i, noise=noise)
        quality[boot] = str(tmp / f"{boot}-q.json")
        write_quality(Path(quality[boot]), 5.0 if boot[0] == "A" else head_ppl)
    return {"npz": npz, "quality": quality}


def windows_file(tmp: Path) -> str:
    path = tmp / "windows.json"
    path.write_text(json.dumps({"sha256": "f" * 64}))
    return str(path)


def test_quality_run_compares_the_collects(harness, tmp_path):
    h = harness(**quality_config(tmp_path, head_noise=0.05))
    res = h.run({"QUALITY": "1", "QUALITY_WINDOWS": windows_file(tmp_path)})
    assert res.returncode == 0, res.stdout + res.stderr
    boots = h.boots()
    assert [b["boot"] for b in boots] == ["XA", "XB", "A1", "B1", "A2"]
    speed = [b for b in boots if b["boot"] in ("A1", "B1")]
    floor = boots[-1]
    for b in speed + [floor]:
        assert b["env"]["LOAD_HOOK"] == str(ROOT / "tools/rdna/pr_receipt_hook.sh")
        assert b["env"]["QUALITY_WINDOWS"] == str(tmp_path / "windows.json")
        assert b["env"]["QUALITY_LIMIT"] == "64"
    assert floor["env"]["CELLS"] == "none" and floor["env"]["QUIET_HOLD"] == "0"
    assert floor["env"]["CONC_PROBE"] == "0"
    quality = h.receipt()["quality"]
    assert quality["verdict"] == "PASS", quality["checks"]
    assert set(quality["ppl"]) == {"floor", "head"}
    assert (h.out / "quality/kl.json").exists() and (h.out / "quality/kl.md").exists()
    text = (h.out / "receipt.md").read_text()
    assert "floor (base vs base)" in text and "head (base vs head)" in text
    assert "ids sha256 " + "f" * 16 in text


def test_quality_run_fails_a_head_that_leaves_the_floor(harness, tmp_path):
    h = harness(**quality_config(tmp_path, head_noise=1.5))
    res = h.run({"QUALITY": "1", "QUALITY_WINDOWS": windows_file(tmp_path)})
    assert res.returncode == 1, res.stdout + res.stderr
    quality = h.receipt()["quality"]
    assert quality["verdict"] == "FAIL"
    failed = [c["name"] for c in quality["checks"] if not c["ok"]]
    assert "docs: head KL within floor" in failed


def test_quality_run_fails_a_ppl_rise(harness, tmp_path):
    h = harness(**quality_config(tmp_path, head_noise=0.05, head_ppl=5.5))
    res = h.run({"QUALITY": "1", "QUALITY_WINDOWS": windows_file(tmp_path)})
    assert res.returncode == 1, res.stdout + res.stderr
    failed = [c["name"] for c in h.receipt()["quality"]["checks"] if not c["ok"]]
    assert failed == ["PPL rise, head"]


def test_abba_quality_run_compares_both_head_boots(harness, tmp_path):
    h = harness(**quality_config(tmp_path, head_noise=0.05))
    env = {"QUALITY": "1", "MODE": "abba", "EXACT": "0"}
    res = h.run({**env, "QUALITY_WINDOWS": windows_file(tmp_path)})
    assert res.returncode == 0, res.stdout + res.stderr
    assert [b["boot"] for b in h.boots()] == ["A1", "B1", "B2", "A2"]
    quality = h.receipt()["quality"]
    assert set(quality["ppl"]) == {"floor", "head1", "head2"}
    heads = quality["domains"]["docs"]["heads"]
    assert set(heads) == {"head1", "head2"}
