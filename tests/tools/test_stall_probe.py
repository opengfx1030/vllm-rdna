# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mixed-batch stall probe must see decoder stalls and how knobs move them.

Runs the stdlib probe against a tiny fake streaming server whose "engine"
mixes prefill chunks into decode steps (step time grows with the chunk), so a
long prompt freezes decoders exactly like a real chunked-prefill scheduler.
No GPU, no vLLM import.
"""

import importlib.util
import json
import math
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "stall_probe", ROOT / "tools/rdna/port_v031/stall_probe.py"
)
assert _spec is not None and _spec.loader is not None
sp = importlib.util.module_from_spec(_spec)
sys.modules["stall_probe"] = sp
_spec.loader.exec_module(sp)


class FakeEngine:
    """Chunked-prefill engine: every step decodes all decoders and takes up to
    `chunk` prompt tokens; the step lasts decode_ms + prompt_tokens / rate."""

    def __init__(self, chunk: int, decode_ms: float = 5.0, rate: float = 20000.0):
        self.chunk = chunk
        self.decode_s = decode_ms / 1000.0
        self.rate = rate
        self.lock = threading.Lock()
        self.reqs: dict[int, dict] = {}
        self.next_id = 0
        self.alive = True
        threading.Thread(target=self.loop, daemon=True).start()

    def add(self, n_prompt: int, max_tokens: int) -> dict:
        with self.lock:
            r = {
                "left": n_prompt,
                "max": max_tokens,
                "out": 0,
                "q": [],
                "cv": threading.Condition(),
                "dead": False,
            }
            self.reqs[self.next_id] = r
            self.next_id += 1
            return r

    def running(self) -> int:
        with self.lock:
            return sum(1 for r in self.reqs.values() if not r["dead"])

    def loop(self) -> None:
        while self.alive:
            with self.lock:
                live = [r for r in self.reqs.values() if not r["dead"]]
            if not live:
                time.sleep(0.002)
                continue
            budget = self.chunk
            for r in live:
                if r["left"] > 0 and budget > 0:
                    take = min(r["left"], budget)
                    r["left"] -= take
                    budget -= take
            time.sleep(self.decode_s + (self.chunk - budget) / self.rate)
            for r in live:
                if r["left"] == 0:
                    with r["cv"]:
                        r["out"] += 1
                        done = r["out"] >= r["max"]
                        r["q"].append(done)
                        if done:
                            r["dead"] = True
                        r["cv"].notify()


def make_server(engine: FakeEngine) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/metrics":
                text = (
                    f'vllm:num_requests_running{{model_name="m"}} {engine.running()}\n'
                )
                body = text.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                words = req["prompt"].split()
                self._json({"tokens": [100 + (hash(w) % 5000) for w in words]})
                return
            prompt = req["prompt"]
            r = engine.add(len(prompt), req["max_tokens"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                while True:
                    with r["cv"]:
                        while not r["q"]:
                            r["cv"].wait()
                        done = r["q"].pop(0)
                        n = r["out"] - len(r["q"])
                    chunk = {
                        "choices": [
                            {"text": "x", "finish_reason": "length" if done else None}
                        ],
                        "usage": {"prompt_tokens": len(prompt), "completion_tokens": n},
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    if done:
                        break
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                r["dead"] = True

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def chunks(times, n=1):
    return [(t, n) for t in times]


class WindowMathTests(unittest.TestCase):
    def test_stall_inside_window_is_measured(self):
        # Two decoders at 10 ms ITL; one freezes for 500 ms inside [1.0, 1.6].
        base = [i * 0.01 for i in range(101)]  # 0.00 .. 1.00
        a = sp.Stream("dec0", "decoder", 8, 100, submit=-0.1)
        a.chunks = chunks(base + [1.5] + [1.5 + i * 0.01 for i in range(1, 30)])
        b = sp.Stream("dec1", "decoder", 8, 100, submit=-0.1)
        b.chunks = chunks([i * 0.01 for i in range(200)])
        bl = sp.baseline_stats([a, b], 0.0, 1.0)
        self.assertAlmostEqual(bl["gaps_ms"]["p50"], 10.0, places=3)
        w = sp.window_stats([a, b], 1.0, 1.6, bl, stall_factor=5, stall_min_ms=0)
        self.assertAlmostEqual(w["max_gap_ms"], 500.0, places=3)
        self.assertAlmostEqual(w["stall_threshold_ms"], 50.0, places=3)
        per = {p["sid"]: p for p in w["per_decoder"]}
        self.assertAlmostEqual(per["dec0"]["stalled_s"], 0.5, places=6)
        self.assertEqual(per["dec1"]["n_stalls"], 0)
        self.assertEqual(w["decode_tokens_min"], per["dec0"]["tokens"])
        self.assertLess(w["jain"], 1.0)
        self.assertEqual(w["starved"], 0)

    def test_frozen_decoder_counts_as_starved(self):
        a = sp.Stream("dec0", "decoder", 8, 100, submit=0.0)
        a.chunks = chunks([0.1, 0.2, 0.3, 3.0, 3.1])
        bl = sp.baseline_stats([a], 0.0, 0.3)
        w = sp.window_stats([a], 0.5, 2.5, bl, 5, 0)
        self.assertEqual(w["decode_tokens_total"], 0)
        self.assertEqual(w["starved"], 1)
        self.assertAlmostEqual(w["frozen_frac_max"], 1.0, places=6)

    def test_reverse_counts_wait_for_first_token(self):
        a = sp.Stream("dec0", "decoder", 8, 100, submit=1.0)
        a.chunks = chunks([4.0, 4.01, 4.02])
        bl = {"gaps_ms": {"p50": 10.0}, "decode_tps_total": 100.0}
        without = sp.window_stats([a], 1.0, 3.5, bl, 5, 0)
        with_ttft = sp.window_stats([a], 1.0, 3.5, bl, 5, 0, include_ttft=True)
        self.assertTrue(math.isnan(without["max_gap_ms"]))
        self.assertAlmostEqual(with_ttft["max_gap_ms"], 3000.0, places=3)

    def test_prompts_are_exact_unique_and_reproducible(self):
        pool = list(range(1000, 1300))
        p1 = sp.Prompts(pool, seed=1, salt="s")
        p2 = sp.Prompts(pool, seed=1, salt="s")
        a, b = p1.make(16384, "inject|0|inj0"), p2.make(16384, "inject|0|inj0")
        self.assertEqual(len(a), 16384)
        self.assertEqual(a, b)
        self.assertNotEqual(a[:16], p1.make(16384, "inject|1|inj0")[:16])
        self.assertNotEqual(a[:16], sp.Prompts(pool, 1, "t").make(16384, "x")[:16])

    def test_server_iteration_log_is_attributed_to_windows(self):
        head = "(APIServer pid=1) INFO 10-09 12:00:0{} [loggers.py:190] Iteration({}): "
        body = (
            "{} context requests, {} context tokens, {} generation requests, "
            "{} generation tokens, iteration elapsed time: {} ms, cuda KV cache"
        )
        rows = [(0, 1, 0, 0, 8, 8, "40.00"), (1, 2, 1, 2048, 0, 0, "900.50"),
                (2, 3, 1, 2048, 8, 8, "950.00")]  # fmt: skip
        lines = [head.format(*r[:2]) + body.format(*r[2:]) for r in rows]
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write("\n".join(lines) + "\n")
        iters = sp.parse_iteration_log(f.name, 2026)
        Path(f.name).unlink()
        self.assertEqual([i.index for i in iters], [1, 2, 3])
        w0 = iters[1].wall
        s = sp.server_window(iters, w0 + 0.2, w0 + 1.5, n_decoders=8)
        self.assertEqual(s["steps"], 2)
        self.assertEqual(s["steps_prefill_only"], 1)
        self.assertEqual(s["steps_mixed"], 1)
        self.assertEqual(s["steps_missing_decoders"], 1)
        self.assertEqual(s["prefill_tokens"], 4096)
        self.assertAlmostEqual(s["max_step_ms"], 950.0)


class EndToEndTests(unittest.TestCase):
    ARGS = [
        "--model", "m", "--decoders", "3", "--decoder-prompt-tokens", "32",
        "--warmup-tokens", "5", "--baseline-s", "0.4", "--prefill-lens", "4096",
        "--recover-s", "0.25", "--periodic-len", "2048", "--period-s", "0.3",
        "--period-count", "2", "--reverse-delay-s", "0.05", "--settle-s", "0.05",
        "--repeats", "2", "--salt", "fixed", "--timeline",
    ]  # fmt: skip

    def run_probe(self, chunk: int, *extra: str) -> tuple[int, dict, Path]:
        engine = FakeEngine(chunk=chunk)
        srv = make_server(engine)
        out = Path(tempfile.mkdtemp())
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            rc = sp.main(["--base-url", url, "--out", str(out), *self.ARGS, *extra])
        finally:
            engine.alive = False
            srv.shutdown()
            srv.server_close()
        return rc, json.loads((out / "stall_probe.json").read_text()), out

    def test_detects_stall_and_reacts_to_chunk_cap(self):
        rc, big, out = self.run_probe(1024)
        self.assertEqual(rc, 0, (out / "summary.txt").read_text())
        self.assertEqual(big["verdict"], "NO-GATES")
        self.assertEqual(big["prompt_source"], "tokenize")
        self.assertTrue((out / "timeline.csv").exists())
        w = big["aggregate"]["windows"]
        self.assertEqual(
            set(w), {"inject:4096", "periodic:2048#0", "periodic:2048#1",
                     "periodic:busy", "reverse:4096"},
        )  # fmt: skip
        base_p50 = big["aggregate"]["baseline_gap_p50_ms"]["inject"]["median"]
        inj = w["inject:4096"]
        # 1024-token chunk at 20k tok/s: ~56 ms steps vs ~5 ms decode steps.
        self.assertGreater(inj["max_gap_ms"]["median"], 6 * base_p50)
        self.assertGreater(inj["stalled_s_max"]["median"], 0.1)
        self.assertGreater(inj["slowdown"]["median"], 0.5)
        rev = w["reverse:4096"]
        self.assertGreater(rev["max_gap_ms"]["median"], 100.0)

        _, small, _ = self.run_probe(256)
        small_inj = small["aggregate"]["windows"]["inject:4096"]
        self.assertLess(
            small_inj["max_gap_ms"]["median"], 0.6 * inj["max_gap_ms"]["median"]
        )
        self.assertGreater(
            small_inj["decode_tokens_total"]["median"],
            inj["decode_tokens_total"]["median"],
        )

    def test_gates_fail_on_stall(self):
        rc, res, _ = self.run_probe(
            1024, "--scenarios", "solo,inject", "--repeats", "1",
            "--max-stall-ms", "20", "--min-decode-tokens-during-prefill", "1",
        )  # fmt: skip
        self.assertEqual(rc, 1)
        self.assertEqual(res["verdict"], "FAIL")
        failed = {g["gate"] for g in res["gates"] if not g["ok"]}
        self.assertIn("max_stall_ms", failed)
        rc, res, _ = self.run_probe(
            1024, "--scenarios", "solo,inject", "--repeats", "1",
            "--max-stall-ms", "5000",
        )  # fmt: skip
        self.assertEqual(rc, 0)
        self.assertEqual(res["verdict"], "PASS")


class KfdEvictionTests(unittest.TestCase):
    """Per-worker KFD evicted_ms and INVALID windows (kfd_evictions.py)."""

    ev = sys.modules["kfd_evictions"]

    def fake_host(self, root: Path) -> tuple[str, str]:
        proc, kfd = root / "proc", root / "kfd"
        # 100 serve script -> 101 api server -> 102 engine -> 103/104 workers;
        # 200 is another server's worker.
        tree = {
            100: (1, "bash"),
            101: (100, "python3"),
            102: (101, "VLLM::EngineCore"),
            103: (102, "VLLM::Worker_TP0_EP0"),
            104: (102, "VLLM::Worker_TP1_EP1"),
            200: (1, "VLLM::Worker_TP0"),
        }
        for pid, (ppid, title) in tree.items():
            (proc / str(pid)).mkdir(parents=True)
            (proc / str(pid) / "stat").write_text(f"{pid} (a b) S {ppid} 0 0\n")
            (proc / str(pid) / "cmdline").write_bytes(title.encode() + b"\0")
        for pid, ms in ((103, 5), (104, 0), (200, 9)):
            for gpu in ("8425", "11353"):
                d = kfd / str(pid) / f"stats_{gpu}"
                d.mkdir(parents=True)
                (d / "evicted_ms").write_text(f"{ms}\n")
        return str(proc), str(kfd)

    def test_snapshot_reads_only_this_servers_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, kfd = self.fake_host(Path(tmp))
            self.assertEqual(
                self.ev.worker_pids(100, proc),
                {"Worker_TP0_EP0": 103, "Worker_TP1_EP1": 104},
            )
            snap = self.ev.snapshot(100, proc, kfd)
        self.assertEqual(snap, {"Worker_TP0_EP0": 5, "Worker_TP1_EP1": 0})

    def test_verdict_line(self):
        ok = self.ev.verdict_line(self.ev.jumps({"W0": 5, "W1": 0}, {"W0": 5, "W1": 0}))
        self.assertEqual(ok, "evicted_ms W0=+0 W1=+0 -> ok")
        bad = self.ev.verdict_line(self.ev.jumps({"W0": 5}, {"W0": 8205}))
        self.assertTrue(bad.startswith("evicted_ms W0=+8200 -> INVALID"), bad)

    def sampler(self, samples):
        s = self.ev.Sampler(0, time.monotonic, read=dict)
        s.samples = samples
        return s

    def test_overlap_attributes_an_eviction_to_the_windows_it_covers(self):
        # 8.2 s eviction of W1 that ended between t=20 and t=20.5.
        s = self.sampler(
            [(0.0, {"W0": 0, "W1": 0}), (20.0, {"W0": 0, "W1": 0}),
             (20.5, {"W0": 0, "W1": 8200}), (30.0, {"W0": 0, "W1": 8200})]
        )  # fmt: skip
        self.assertEqual(s.overlap_ms(13.0, 14.0), {"W1": 8200})
        self.assertEqual(s.overlap_ms(0.0, 11.0), {})
        self.assertEqual(s.overlap_ms(21.0, 30.0), {})
        self.assertEqual(s.totals(), {"W0": 0, "W1": 8200})

    def test_mark_evictions_invalidates_windows_baseline_and_solo(self):
        solo = sp.Run("solo", 0)
        s = sp.Stream("i0", "inject", 1024, 1, submit=1.0)
        s.chunks.append((2.0, 1))
        solo.streams.append(s)
        inject = sp.Run("inject", 0)
        analyses: list[dict[str, Any]] = [
            {"scenario": "solo", "repeat": 0, "windows": []},
            {"scenario": "inject", "repeat": 0,
             "baseline": {"t0": 3.0, "t1": 5.0},
             "windows": [{"len": 4096, "t0": 6.0, "t1": 8.0},
                         {"len": 8192, "t0": 9.0, "t1": 12.0}]},
        ]  # fmt: skip

        def overlap(t0, t1):
            return {"W0": 700} if t0 <= 10.0 <= t1 or t0 <= 1.5 <= t1 else {"W0": 0}

        invalid = sp.mark_evictions([solo, inject], analyses, overlap)
        self.assertEqual(
            invalid, ["solo r0 solo 1024: W0=+700", "inject r0 8192: W0=+700"]
        )
        self.assertEqual(analyses[1]["windows"][1]["invalid"], "kfd eviction")
        self.assertNotIn("invalid", analyses[1]["windows"][0])
        self.assertEqual(analyses[1]["baseline"]["evicted_ms"], {"W0": 0})


if __name__ == "__main__":
    unittest.main()
