#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One arm of the mono + W4A8 comparison, measured against a running server.

Run by ``hook.sh`` (the ``serve_validate.sh`` LOAD_HOOK) after the standard
probes. Phases, in this order:

``perf``    the cells of ``CELLS`` through ``vllm bench serve`` (random prompts,
            greedy, a seed of its own per cell so no prompt is in the prefix
            cache), each bracketed by KFD eviction snapshots of every worker.
            A cell whose window shows an eviction jump is INVALID and runs again
            with another seed (``--cell-tries``); a cell that stays invalid is
            recorded as such and fails the arm.
``parity``  greedy top-5 logprobs of fixed prompts (``tools/rdna/mono/
            logprob_parity.py``): the decode path, solo and at c8.
``topk``    teacher-forced 2048-token windows with top-20 prompt logprobs
            (``tools/rdna/port_v031/topk_kl.py collect``): every scored position
            runs the prefill kernels (the W4A8 MoE GEMM among them), then a
            within-boot repeat of the first windows for the noise floor.

Writes ``<out>/arm/{cells.jsonl,summary.json,evict.jsonl}`` plus the raw files of
the phases; ``analyze.py`` turns the arm directories into the tables.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import regex as re

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "port_v031"))
import kfd_evictions as kfd  # noqa: E402


@dataclass(frozen=True)
class Cell:
    name: str
    input_len: int
    output_len: int
    concurrency: int
    requests: int
    seed: int
    what: str


# Solo TTFT, decode per request / ITL at the MTP2 widths 3, 18 and 24 rows (c1,
# c6, c8), and the 16k c8 aggregate prefill. ``requests`` is a whole number of
# waves of ``concurrency``.
CELLS = (
    Cell("ttft4k", 4096, 64, 1, 8, 4101, "solo TTFT 4k"),
    Cell("ttft16k", 16384, 64, 1, 8, 4102, "solo TTFT 16k"),
    Cell("c1", 1024, 512, 1, 4, 4201, "decode c1"),
    Cell("c6", 1024, 512, 6, 24, 4206, "decode c6"),
    Cell("c8", 1024, 512, 8, 32, 4208, "decode c8"),
    Cell("agg16k", 16384, 1024, 8, 16, 4301, "16k/1k c8 aggregate prefill"),
)
# Not measured: one pass over each prompt shape before the cells (4k and 16k
# prefill chunks, the c8 decode batch), with prompts no cell reuses.
WARMUPS = (
    Cell("warm4k", 4096, 16, 1, 2, 4901, "warm-up"),
    Cell("warm16k", 16384, 16, 1, 2, 4902, "warm-up"),
    Cell("warm-c8", 1024, 64, 8, 8, 4903, "warm-up"),
)
COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scrape(base: str) -> dict[str, float]:
    """Cumulative server counters (prefix cache, preemptions) from /metrics."""
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=30) as resp:
            text = resp.read().decode()
    except OSError:
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in COUNTERS:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def summarize(path: Path, cell: Cell) -> dict:
    d = json.loads(path.read_text())
    done = max(int(d.get("completed", 0)), 1)
    in_len = d.get("total_input_tokens", 0) / done
    mean_ttft = d.get("mean_ttft_ms", 0.0) / 1000.0
    p99_ttft = d.get("p99_ttft_ms", 0.0) / 1000.0
    wave = min(int(d.get("max_concurrency") or 1), done)
    tpot = d.get("mean_tpot_ms", 0.0)
    return {
        "completed": int(d.get("completed", 0)),
        "failed": int(d.get("failed", 0)),
        "in_len": in_len,
        "ttft_mean_s": mean_ttft,
        "ttft_median_s": d.get("median_ttft_ms", 0.0) / 1000.0,
        "ttft_p99_s": p99_ttft,
        "prefill_tok_s_req": in_len / mean_ttft if mean_ttft > 0 else 0.0,
        "prefill_tok_s_agg": wave * in_len / p99_ttft if p99_ttft > 0 else 0.0,
        "tpot_mean_ms": tpot,
        "decode_tok_s_req": 1000.0 / tpot if tpot > 0 else 0.0,
        "itl_median_ms": d.get("median_itl_ms", 0.0),
        "itl_mean_ms": d.get("mean_itl_ms", 0.0),
        "itl_p99_ms": d.get("p99_itl_ms", 0.0),
        "out_tok_s": d.get("output_throughput", 0.0),
        "duration_s": d.get("duration", 0.0),
        "accept_len": d.get("spec_decode_acceptance_length"),
        "accept_rate_pct": d.get("spec_decode_acceptance_rate"),
        "accept_per_pos": d.get("spec_decode_per_position_acceptance_rates"),
    }


def run_bench(args, cell: Cell, seed: int, name: str) -> int:
    cmd = [
        args.py,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--base-url",
        args.url,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(cell.input_len),
        "--random-output-len",
        str(cell.output_len),
        "--num-prompts",
        str(cell.requests),
        "--max-concurrency",
        str(cell.concurrency),
        "--ignore-eos",
        "--request-rate",
        "inf",
        "--seed",
        str(seed),
        "--temperature",
        "0",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--result-dir",
        str(args.cells_dir),
        "--result-filename",
        f"{name}.json",
        "--disable-tqdm",
    ]
    with open(args.cells_dir / f"{name}.log", "w") as log:
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


def take(root: int) -> dict:
    """The evicted ms of every worker (KFD) now, with the time of the reading."""
    return {"t": time.time(), "ms": kfd.snapshot(root)}


def sample(root: int, trail: Path) -> dict[str, int]:
    """One reading of the background sampler, appended to ``trail`` (jsonl): which
    worker was evicted, and when."""
    ms = kfd.snapshot(root)
    with open(trail, "a") as f:
        f.write(json.dumps({"t": round(time.time(), 1), "ms": ms}) + "\n")
    return ms


def window(before: dict, after: dict, workers: int) -> dict:
    """What KFD did to the workers between two ``take`` readings. ``ok`` needs
    all ``workers`` visible and none evicted, none gone: a window that saw no
    worker proves nothing."""
    delta = kfd.jumps(before["ms"], after["ms"])
    gone = sorted(set(before["ms"]) - set(after["ms"]))
    jumped = {w: ms for w, ms in delta.items() if ms > 0}
    reasons = []
    if jumped:
        reasons.append(
            "evicted: " + " ".join(f"{w}=+{ms}ms" for w, ms in jumped.items())
        )
    if gone:
        reasons.append(f"worker(s) gone: {gone}")
    if len(delta) < workers:
        reasons.append(f"saw {len(delta)} KFD worker(s), need {workers}")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "delta_ms": delta,
        "gone": gone,
        "workers": len(delta),
        "window_s": round(after["t"] - before["t"], 3),
        "total_delta_ms": sum(jumped.values()),
    }


def window_line(v: dict) -> str:
    if v["ok"]:
        return f"EVICT OK ({v['workers']} workers, {v['window_s']:.0f} s)"
    return f"EVICT INVALID: {'; '.join(v['reasons'])}"


def measure(args, cell: Cell, root: int) -> dict:
    """One cell, repeated with fresh prompts until a window shows no eviction."""
    rec: dict = {}
    for attempt in range(1, args.cell_tries + 1):
        seed = cell.seed + 1000 * (attempt - 1)
        name = cell.name if attempt == 1 else f"{cell.name}-try{attempt}"
        before, m0 = take(root), scrape(args.url)
        t0 = time.time()
        rc = run_bench(args, cell, seed, name)
        m1, after = scrape(args.url), take(root)
        verdict = window(before, after, args.tp)
        rec = {
            "cell": cell.name,
            "what": cell.what,
            "attempt": attempt,
            "seed": seed,
            "file": f"cells/{name}.json",
            "wall_s": round(time.time() - t0, 1),
            "evict": verdict,
            "counters": {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in COUNTERS},
            "cell_spec": asdict(cell),
        }
        result = args.cells_dir / f"{name}.json"
        if rc != 0 or not result.exists():
            rec["status"] = "bench-failed"
        else:
            rec["metrics"] = summarize(result, cell)
            if rec["metrics"]["failed"] or rec["metrics"]["completed"] != cell.requests:
                rec["status"] = "bench-failed"
            else:
                rec["status"] = "valid" if verdict["ok"] else "invalid"
        with open(args.arm_dir / "cells.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        m = rec.get("metrics", {})
        say(
            f"cell {cell.name} try {attempt} seed {seed}: {rec['status']} | "
            f"{window_line(verdict)} | TTFT {m.get('ttft_mean_s', 0):.2f} s "
            f"prefill {m.get('prefill_tok_s_req', 0):.0f}/"
            f"{m.get('prefill_tok_s_agg', 0):.0f} tok/s "
            f"ITL {m.get('itl_median_ms', 0):.1f} ms "
            f"decode {m.get('decode_tok_s_req', 0):.1f} tok/s/req "
            f"accept {m.get('accept_len')}"
        )
        if rec["status"] == "valid" or not alive(root):
            break
    return rec


def worker_mem(root: int, proc: str = "/proc") -> dict:
    """Locked memory (VmLck, kB) of each worker and the RLIMIT_MEMLOCK it runs
    under: the proof that the pinned PLE table is mlocked (a worker that could
    not lock it logs a warning and serves without the lock)."""
    out = {}
    for label, pid in kfd.worker_pids(root, proc).items():
        try:
            status = Path(f"{proc}/{pid}/status").read_text()
            limits = Path(f"{proc}/{pid}/limits").read_text()
        except OSError:
            continue
        locked = re.search(r"^VmLck:\s+(\d+) kB", status, re.MULTILINE)
        limit = re.search(r"^Max locked memory\s+(\S+)\s+(\S+)", limits, re.MULTILINE)
        out[str(pid)] = {
            "comm": label,
            "vmlck_kb": int(locked.group(1)) if locked else None,
            "memlock_soft": limit.group(1) if limit else None,
            "memlock_hard": limit.group(2) if limit else None,
        }
    return out


def env_snapshot() -> dict:
    load = Path("/proc/loadavg").read_text().split()[:3]
    ps = subprocess.run(
        ["ps", "-eo", "pcpu,comm", "--sort=-pcpu"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()[1:]
    # the ps that takes the snapshot always ranks first (CPU time over milliseconds)
    top = [t.strip() for t in ps if t.split()[-1:] != ["ps"]][:2]
    dmesg = subprocess.run(
        ["sudo", "-n", "dmesg"], capture_output=True, text=True, check=False
    ).stdout
    aer = len(re.findall(r"pcieport.*(Timeout|BadTLP|NonFatalErr)", dmesg))
    return {
        "t": time.time(),
        "loadavg": load,
        "top_cpu": top,
        "aer": aer if dmesg else None,
    }


def phase_perf(args, root: int) -> list[dict]:
    wanted = args.cells.split(",") if args.cells else [c.name for c in CELLS]
    for cell in WARMUPS:  # first-use compiles and tuning stay out of the cells
        t0 = time.time()
        rc = run_bench(args, cell, cell.seed, cell.name)
        say(f"warm-up {cell.name}: rc={rc} in {time.time() - t0:.0f} s")
    recs = []
    for cell in CELLS:
        if cell.name not in wanted:
            continue
        if not alive(root):
            say("server is gone: perf stops")
            break
        recs.append(measure(args, cell, root))
    return recs


def phase_parity(args) -> int:
    cmd = [
        args.py,
        str(HERE.parent / "logprob_parity.py"),
        "dump",
        "--url",
        args.url,
        "--model",
        args.model,
        "--out",
        str(args.out / "parity.json"),
    ]
    with open(args.out / "parity.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    say(f"parity dump rc={rc}")
    return rc


def phase_topk(args) -> int:
    topk = HERE.parents[1] / "port_v031" / "topk_kl.py"
    rc = 0
    for label, limit in (("topk", args.topk_limit), ("topk_rep", args.topk_repeat)):
        if limit <= 0:
            continue
        t0 = time.time()
        cmd = [
            args.py,
            str(topk),
            "collect",
            "--url",
            args.url,
            "--model",
            args.model,
            "--windows",
            args.windows,
            "--top",
            "20",
            "--limit",
            str(limit),
            "--out",
            str(args.out / f"{label}.npz"),
        ]
        with open(args.out / f"{label}.log", "w") as log:
            prc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        say(f"{label} (limit {limit}/domain) rc={prc} in {time.time() - t0:.0f} s")
        rc = rc or prc
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="server base URL")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer", required=True, help="model dir for the client")
    ap.add_argument("--out", type=Path, required=True, help="the arm directory")
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--tp", type=int, default=4, help="KFD workers expected")
    ap.add_argument("--phases", default="perf,parity,topk")
    ap.add_argument("--cells", default="", help="comma list of CELLS names")
    ap.add_argument("--cell-tries", type=int, default=3)
    ap.add_argument("--windows", default="")
    ap.add_argument("--topk-limit", type=int, default=32)
    ap.add_argument("--topk-repeat", type=int, default=16)
    args = ap.parse_args()

    args.url = args.url.rstrip("/")
    args.arm_dir = args.out / "arm"
    args.cells_dir = args.arm_dir / "cells"
    args.cells_dir.mkdir(parents=True, exist_ok=True)
    root = int((args.out / "serve.pid").read_text().split()[0])
    phases = args.phases.split(",")
    if "topk" in phases and not Path(args.windows).is_file():
        say(f"topk needs --windows (got {args.windows!r})")
        return 2

    evict_trail = args.arm_dir / "evict.jsonl"
    sampler = kfd.Sampler(root, time.time, 2.0, read=lambda: sample(root, evict_trail))
    sampler.start()
    summary: dict = {
        "phases": phases,
        "env_start": env_snapshot(),
        "worker_mem": worker_mem(root),
        "rc": {},
    }
    start = take(root)
    rc = 0
    try:
        if "perf" in phases:
            recs = phase_perf(args, root)
            summary["perf"] = {r["cell"]: r["status"] for r in recs}
            bad = [r["cell"] for r in recs if r["status"] != "valid"]
            summary["rc"]["perf"] = 1 if bad or len(recs) < 1 else 0
            if bad:
                say(f"PERF CELLS NOT VALID: {bad}")
        if "parity" in phases and alive(root):
            summary["rc"]["parity"] = phase_parity(args)
        if "topk" in phases and alive(root):
            summary["rc"]["topk"] = phase_topk(args)
        rc = 1 if any(summary["rc"].values()) else 0
    finally:
        sampler.stop()
        end = take(root)
        summary["env_end"] = env_snapshot()
        summary["worker_mem_end"] = worker_mem(root)
        summary["evict_whole_hook"] = window(start, end, args.tp)
        summary["server_alive"] = alive(root)
        (args.arm_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    say(f"whole-hook {window_line(summary['evict_whole_hook'])}")
    aer0, aer1 = summary["env_start"]["aer"], summary["env_end"]["aer"]
    if aer0 is not None and aer1 is not None and aer1 > aer0:
        say(f"AER DIRTY: {aer1 - aer0} new PCIe AER line(s) during the hook")
        rc = rc or 4
    say("PASS" if rc == 0 else f"FAIL rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
