#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record prefill, mixed and decode steps of a live rdna server under c=8 load.

    python3 tools/rdna/rdna_hint_flex/c8_profile.py --label base --gpus 4,5 \
        --pci-buses df,e2 --port 18215 --model /path/to/checkpoint \
        --workload c8 -- RECIPE=27b-hi2-tp2 EXTRA_ARGS=... KEY=VAL

Boots the recipe under ``rocprofv3 --kernel-trace --selected-regions`` (as
decode_profile.py: SIGUSR1/SIGUSR2 -> roctxProfilerResume/Pause through the
roctx shim, CUDA-graph replays traced kernel by kernel) with
``--enable-logging-iteration-details`` (log only) so every engine step's
composition and wall time is in serve.log, drives one of two workloads and
opens a recording window on the server state read from that log:

* ``c8``: the scoreboard cell (16 random 16384/1024 requests, 8 concurrent):
  window ``prefill`` while the first request prefills alone, window ``mixed``
  while seven requests decode and one prefills, window ``decode`` once all
  eight decode (about 4 minutes in);
* ``fast``: one 16384-token prompt alone (window ``prefill``), then seven
  1024/4000 decoders and a 16384-token prompt stream on top of them (window
  ``mixed``), about 2 minutes in; the decoders have a short context.

Steps are typed from their kernels by c8_breakdown.py (run at the end).
Output ``OUT_ROOT/LABEL``: serve.log, iterations.json (every engine step seen),
windows.json, ranks.json, thermal.csv, trace/ (kept), c8.{md,json} and the CSVs.

Rules of the box: the caller holds ``flock ~/w4a8_runs/measure.lock``; this
script waits for the quiet files, takes boot.lock for the boot, and touches its
own quiet file while it measures. Never HIP 0-3 (``--refused-gpus``).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import regex as re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_profile import DEFAULT_VENV, TREE, kill_group, log, metric  # noqa: E402

ITER = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([0-9.]+) ms"
)


class IterLog(threading.Thread):
    """Tails serve.log and keeps every ``Iteration(n)`` line with its read time."""

    def __init__(self, path: str) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.rows: list[dict] = []
        self.stop_flag = False

    def run(self) -> None:
        pos, buf = 0, b""
        while not self.stop_flag:
            try:
                with open(self.path, "rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
            except OSError:
                chunk = b""
            if chunk:
                buf += chunk
                *lines, buf = buf.split(b"\n")
                now = time.time()
                for ln in lines:
                    m = ITER.search(ln.decode("utf-8", "replace"))
                    if m:
                        n, cr, ct, gr, gt, ms = m.groups()
                        self.rows.append(
                            {
                                "iter": int(n),
                                "ctx_req": int(cr),
                                "ctx_tok": int(ct),
                                "gen_req": int(gr),
                                "gen_tok": int(gt),
                                "ms": float(ms),
                                "seen": now,
                            }
                        )
            time.sleep(0.2)

    def last(self, n: int) -> list[dict]:
        return self.rows[-n:]


class Sensors(threading.Thread):
    """Junction temperature, power and sclk of the buses at 2 s; trips an abort."""

    def __init__(self, buses: list[str], path: str, on_trip) -> None:
        super().__init__(daemon=True)
        self.dev = {}
        self.card = {}
        for card in glob.glob("/sys/class/drm/card*/device"):
            try:
                with open(os.path.join(card, "uevent")) as uf:
                    uev = uf.read()
            except OSError:
                continue
            for b in buses:
                if f"PCI_SLOT_NAME=0000:{b}:" in uev:
                    hw = glob.glob(os.path.join(card, "hwmon", "hwmon*"))
                    if hw:
                        self.dev[b] = hw[0]
                        self.card[b] = card
        self.path = path
        self.on_trip = on_trip
        self.stop_flag = False

    def vram_used_mib(self, b: str) -> int:
        return self._read(os.path.join(self.card[b], "mem_info_vram_used")) >> 20

    @staticmethod
    def _read(p: str) -> int:
        try:
            return int(open(p).read())
        except (OSError, ValueError):
            return 0

    def run(self) -> None:
        hist: dict[str, list[int]] = {b: [] for b in self.dev}
        with open(self.path, "w") as fh:
            fh.write("epoch bus edge_mC junction_mC mem_mC power_uW sclk_Hz\n")
            while not self.stop_flag:
                for b, hw in self.dev.items():
                    t = [self._read(f"{hw}/temp{i}_input") for i in (1, 2, 3)]
                    pw = self._read(f"{hw}/power1_average")
                    fq = self._read(f"{hw}/freq1_input")
                    fh.write(f"{time.time():.1f} {b} {t[0]} {t[1]} {t[2]} {pw} {fq}\n")
                    hist[b] = (hist[b] + [t[1]])[-3:]
                    if t[1] >= 95000 or (
                        len(hist[b]) == 3 and sum(hist[b]) / 3 >= 90000
                    ):
                        self.on_trip(f"junction {b} {t[1] / 1000:.0f} C")
                fh.flush()
                time.sleep(2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--venv", default=os.environ.get("VENV", DEFAULT_VENV))
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--out-root", default=os.path.expanduser("~/w4a8_runs/hi2-c8/prof"))
    ap.add_argument("--workload", choices=("c8", "fast"), default="c8")
    ap.add_argument("--windows", default="prefill,mixed,decode")
    ap.add_argument("--prefill-window-s", type=float, default=25.0)
    ap.add_argument("--mixed-window-s", type=float, default=28.0)
    ap.add_argument("--decode-window-s", type=float, default=3.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--state-timeout-s", type=float, default=420)
    ap.add_argument("--ready-timeout-s", type=float, default=1500)
    ap.add_argument("--pci-buses", default="", help="expected PCI buses of --gpus")
    ap.add_argument("--refused-gpus", default="2,3,4,5")
    ap.add_argument("--boot-lock", default=os.path.expanduser("~/w4a8_runs/boot.lock"))
    ap.add_argument(
        "--quiet-dir", default=os.path.expanduser("~/w4a8_runs/lanes/quiet")
    )
    ap.add_argument("--quiet-own", default="HiC8Opt")
    ap.add_argument("--no-analyse", action="store_true")
    ap.add_argument(
        "--serve-script", default="", help="default: tools/rdna/serve_rdna.sh"
    )
    ap.add_argument(
        "--bench-prefix",
        default="",
        help="command that replaces '<venv python> -m vllm.entrypoints.cli.main "
        "bench serve'",
    )
    ap.add_argument("serve_args", nargs="*", help="KEY=VAL for serve_rdna.sh")
    return ap.parse_args()


def fresh_quiet(qdir: str, own: str, minutes: float = 6.0) -> str:
    now = time.time()
    for f in glob.glob(os.path.join(qdir, "*")):
        if os.path.basename(f).startswith(own):
            continue
        with contextlib.suppress(OSError):
            if now - os.path.getmtime(f) < minutes * 60:
                return f
    return ""


def main() -> int:  # noqa: C901
    a = parse_args()
    for g in a.gpus.split(","):
        if g in a.refused_gpus.split(","):
            log(a.label, f"refusing HIP GPU {g} (--refused-gpus {a.refused_gpus})")
            return 2
    py = os.path.join(a.venv, "bin", "python")
    site = glob.glob(os.path.join(a.venv, "lib", "python3.*", "site-packages"))[0]
    core = os.path.join(site, "_rocm_sdk_core")
    rocprofv3 = os.environ.get("ROCPROFV3", os.path.join(core, "bin", "rocprofv3"))
    roctx = os.path.join(core, "lib", "librocprofiler-sdk-roctx.so.1")
    for p in (py, rocprofv3, roctx):
        if not os.path.exists(p):
            log(a.label, f"missing {p}")
            return 2

    out = os.path.join(a.out_root, a.label)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    cache = os.path.join(out, "cache")
    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    ldp = [
        os.path.join(site, p)
        for p in (
            "_rocm_sdk_libraries/lib",
            "_rocm_sdk_core/lib/host-math/lib",
            "_rocm_sdk_core/lib/rocm_sysdeps/lib",
            "_rocm_sdk_core/lib",
            "torch/lib",
        )
    ]
    if env.get("LD_LIBRARY_PATH"):
        ldp.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ldp)
    serve_args = list(a.serve_args)
    extra = "--enable-logging-iteration-details"
    for i, kv in enumerate(serve_args):
        if kv.startswith("EXTRA_ARGS="):
            serve_args[i] = f"{kv} {extra}"
            break
    else:
        serve_args.append(f"EXTRA_ARGS={extra}")
    env.update(
        HIP_VISIBLE_DEVICES=a.gpus,
        ROCPROFILER_REGISTER_ENABLED="1",
        RDNA_ROCTX_SHIM="1",
        ROCTX_LIB=roctx,
        PYTHONPATH=f"{os.path.join(HERE, 'roctx_shim')}:{a.tree}",
        VLLM_CACHE_ROOT=os.path.join(cache, "vllm"),
        TRITON_CACHE_DIR=os.path.join(cache, "triton"),
        TORCHINDUCTOR_CACHE_DIR=os.path.join(cache, "inductor"),
        LAUNCH_WRAPPER=f"{rocprofv3} --kernel-trace --selected-regions "
        f"--output-format csv -d {out}/trace -o %pid% --",
    )

    buses = a.pci_buses.split(",") if a.pci_buses else []
    if buses:
        chk = (
            "import torch;print(' '.join('%02x' % "
            "torch.cuda.get_device_properties(i).pci_bus_id "
            "for i in range(torch.accelerator.device_count())))"
        )
        got = subprocess.run(
            [py, "-c", chk],
            env={**env, "ROCPROFILER_REGISTER_ENABLED": "0"},
            capture_output=True,
            text=True,
        ).stdout.split()
        if got != buses:
            log(a.label, f"refusing: HIP {a.gpus} -> buses {got}, want {buses}")
            return 2
        log(a.label, f"bus check ok: HIP {a.gpus} -> {got}")

    tripped: list[str] = []
    sensors = Sensors(
        buses, os.path.join(out, "thermal.csv"), lambda m: tripped.append(m)
    )
    busy = [
        b for b in buses if b not in sensors.card or sensors.vram_used_mib(b) > 2048
    ]
    if busy:
        log(a.label, f"refusing: bus {busy} not found or its VRAM is in use")
        return 2
    sensors.start()
    quiet_file = os.path.join(a.quiet_dir, a.quiet_own)
    qstop = threading.Event()

    def quiet_beat() -> None:
        t0 = time.time()
        while not qstop.wait(60) and time.time() - t0 < 900:
            with contextlib.suppress(OSError):
                os.utime(quiet_file)

    def quiet_open() -> None:
        with open(quiet_file, "a"):
            pass
        threading.Thread(target=quiet_beat, daemon=True).start()
        # measurer protocol: a boot that began before the file appeared still runs
        while not tripped:
            try:
                with open(a.boot_lock, "a") as bl:
                    fcntl.flock(bl, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                log(a.label, f"{a.boot_lock} is held (a boot is in flight): waiting")
                time.sleep(15)

    def quiet_close() -> None:
        qstop.set()
        with contextlib.suppress(OSError):
            os.remove(quiet_file)

    # booter protocol: no fresh quiet file, then boot.lock, look again under it
    with contextlib.ExitStack() as stack:
        lock = stack.enter_context(open(a.boot_lock, "a"))
        while True:
            while (q := fresh_quiet(a.quiet_dir, a.quiet_own)) and not tripped:
                log(a.label, f"quiet window ({os.path.basename(q)}): waiting")
                time.sleep(30)
            log(a.label, f"waiting for boot lock {a.boot_lock}")
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not fresh_quiet(a.quiet_dir, a.quiet_own):
                break
            fcntl.flock(lock, fcntl.LOCK_UN)
        srv_log_path = os.path.join(out, "serve.log")
        srv_log = stack.enter_context(open(srv_log_path, "wb"))
        cmd = [
            "bash",
            a.serve_script or os.path.join(a.tree, "tools/rdna/serve_rdna.sh"),
            f"VENV={a.venv}",
            f"VLLM_TREE={a.tree}",
            f"PORT={a.port}",
            f"MODEL={a.model}",
            *serve_args,
        ]
        log(a.label, "boot: " + " ".join(cmd[2:]))
        t_boot = time.time()
        srv = subprocess.Popen(
            cmd,
            cwd=a.tree,
            env=env,
            stdout=srv_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(),
        )
        served = ""
        while time.time() - t_boot < a.ready_timeout_s and srv.poll() is None:
            if tripped:
                break
            try:
                url = f"http://127.0.0.1:{a.port}/v1/models"
                served = json.loads(urllib.request.urlopen(url, timeout=3).read())[
                    "data"
                ][0]["id"]
                break
            except (OSError, ValueError, KeyError):
                time.sleep(5)
    sid = srv.pid
    if not served:
        log(a.label, f"BOOTFAIL (see serve.log) tripped={tripped}")
        kill_group(sid)
        sensors.stop_flag = True
        return 1
    log(a.label, f"up after {time.time() - t_boot:.0f}s served={served}")

    def workers() -> dict[int, int]:
        r = subprocess.run(
            ["pgrep", "-s", str(sid), "-f", "[W]orker_TP"],
            capture_output=True,
            text=True,
        )
        res = {}
        for p in (int(x) for x in r.stdout.split()):
            with contextlib.suppress(OSError):
                with open(f"/proc/{p}/cmdline", "rb") as cf:
                    m = re.search(rb"Worker_TP(\d+)", cf.read())
                if m:
                    res[int(m.group(1))] = p
        return res

    ranks = workers()
    with open(os.path.join(out, "ranks.json"), "w") as fh:
        json.dump({str(k): v for k, v in ranks.items()}, fh)
    log(a.label, f"workers {ranks}")

    itl = IterLog(srv_log_path)
    itl.start()

    def bench(name, inl, outl, num, conc, seed):
        argv = [
            *(
                shlex.split(a.bench_prefix)
                or [py, "-m", "vllm.entrypoints.cli.main", "bench", "serve"]
            ),
            "--backend", "openai", "--endpoint", "/v1/completions",
            "--base-url", f"http://127.0.0.1:{a.port}", "--model", served,
            "--tokenizer", a.model, "--dataset-name", "random",
            "--random-input-len", str(inl), "--random-output-len", str(outl),
            "--num-prompts", str(num), "--max-concurrency", str(conc),
            "--ignore-eos", "--request-rate", "inf", "--seed", str(seed),
            "--temperature", "0",
        ]  # fmt: skip
        benv = {**env, "ROCPROFILER_REGISTER_ENABLED": "0", "RDNA_ROCTX_SHIM": "0"}
        benv["PYTHONPATH"] = a.tree
        benv.pop("LAUNCH_WRAPPER", None)
        with open(os.path.join(out, f"{name}.log"), "wb") as logf:
            return subprocess.Popen(
                argv,
                env=benv,
                cwd=a.tree,
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

    def wait_for(pred, what: str, timeout: float | None = None) -> bool:
        t0 = time.time()
        while time.time() - t0 < (timeout or a.state_timeout_s) and not tripped:
            if pred():
                log(a.label, f"state reached: {what} (+{time.time() - t0:.0f}s)")
                return True
            time.sleep(0.25)
        log(a.label, f"state NOT reached: {what} tripped={tripped}")
        return False

    running = lambda n: metric(a.port, "vllm:num_requests_running") >= n  # noqa: E731

    def last_ok(n: int, ok) -> bool:
        rows = itl.last(n)
        return len(rows) == n and all(ok(r) for r in rows)

    windows: list[dict] = []

    def record(kind: str, seconds: float) -> None:
        wp = workers()
        i0 = len(itl.rows)
        for pid in wp.values():
            os.kill(pid, signal.SIGUSR1)
        t0 = time.time()
        time.sleep(seconds)
        for pid in wp.values():
            os.kill(pid, signal.SIGUSR2)
        t1 = time.time()
        windows.append(
            {
                "kind": kind,
                "t_start": t0,
                "t_stop": t1,
                "iter_first": itl.rows[i0]["iter"] if len(itl.rows) > i0 else None,
                "iter_last": itl.rows[-1]["iter"] if itl.rows else None,
                "workers": wp,
            }
        )
        log(a.label, f"window {kind}: {t1 - t0:.1f} s, iterations {len(itl.rows) - i0}")

    want = set(a.windows.split(","))
    procs: list[subprocess.Popen] = []
    quiet_open()
    try:
        # the first window after boot records nothing: spend one on a warm bench
        w = bench("warm", 1024, 64, 4, 4, 11)
        time.sleep(4)
        for pid in workers().values():
            os.kill(pid, signal.SIGUSR1)
        time.sleep(1.0)
        for pid in workers().values():
            os.kill(pid, signal.SIGUSR2)
        w.wait()
        wait_for(lambda: not running(1), "idle after warm", 60)
        log(a.label, "priming window done")

        if a.workload == "c8":
            p = bench("c8", 16384, 1024, 16, 8, 12345)
            procs.append(p)
            if "prefill" in want and wait_for(
                lambda: running(1), "first request running", 120
            ):
                time.sleep(a.settle_s)
                record("prefill", a.prefill_window_s)
            if "mixed" in want and wait_for(
                lambda: last_ok(3, lambda r: r["gen_req"] >= 7 and r["ctx_tok"] > 0),
                "mixed: 7 decoders + prefill",
            ):
                record("mixed", a.mixed_window_s)
            if "decode" in want and wait_for(
                lambda: last_ok(6, lambda r: r["gen_req"] == 8 and r["ctx_tok"] == 0),
                "decode: 8 decoders, no prefill",
            ):
                time.sleep(a.settle_s)
                record("decode", a.decode_window_s)
        else:
            if "prefill" in want:
                p = bench("p1", 16384, 2, 1, 1, 21)
                procs.append(p)
                if wait_for(lambda: running(1), "prefill request running", 120):
                    time.sleep(a.settle_s)
                    record("prefill", a.prefill_window_s)
                p.wait()
            if "mixed" in want:
                d = bench("decoders", 1024, 4000, 7, 7, 22)
                procs.append(d)
                wait_for(
                    lambda: last_ok(
                        5, lambda r: r["gen_req"] >= 7 and r["ctx_tok"] == 0
                    ),
                    "7 decoders, no prefill",
                )
                p2 = bench("p2", 16384, 2, 3, 1, 23)
                procs.append(p2)
                if wait_for(
                    lambda: last_ok(
                        3, lambda r: r["gen_req"] >= 7 and r["ctx_tok"] > 0
                    ),
                    "mixed: 7 decoders + prefill",
                ):
                    record("mixed", a.mixed_window_s)
    except Exception:  # keep what was recorded: the flush below writes the traces
        import traceback

        log(
            a.label,
            "driver error, flushing what was recorded:\n" + traceback.format_exc(),
        )
    except BaseException:
        kill_group(sid)
        raise
    finally:
        for p in procs:
            kill_group(p.pid)
        quiet_close()
    itl.stop_flag = True
    sensors.stop_flag = True
    with open(os.path.join(out, "iterations.json"), "w") as fh:
        json.dump(itl.rows, fh)
    with open(os.path.join(out, "windows.json"), "w") as fh:
        json.dump({"windows": windows, "tripped": tripped}, fh, indent=1)

    # rocprofv3 owns SIGTERM in every traced process: flush the workers first
    wp = list(workers().values())
    for pid in wp:
        os.kill(pid, signal.SIGTERM)

    def alive(pid: int) -> bool:
        try:
            with open(f"/proc/{pid}/stat", "rb") as sf:
                return sf.read().rsplit(b")", 1)[1].split()[0] != b"Z"
        except OSError:
            return False

    t0 = time.time()
    while time.time() - t0 < 900 and any(alive(p) for p in wp):
        time.sleep(1)
    log(a.label, f"workers flushed in {time.time() - t0:.0f}s ({wp})")
    kill_group(sid)
    csvs = glob.glob(
        os.path.join(out, "trace", "**", "*kernel_trace.csv"), recursive=True
    )
    log(
        a.label,
        f"trace csv={len(csvs)} {sum(os.path.getsize(c) for c in csvs) / 1e6:.0f} MB",
    )
    rc = 0
    if not a.no_analyse:
        rc = subprocess.run(
            [
                py,
                os.path.join(HERE, "c8_breakdown.py"),
                os.path.join(out, "trace"),
                "--ranks-json",
                os.path.join(out, "ranks.json"),
                "--out-prefix",
                os.path.join(out, "c8"),
                "--title",
                a.label,
            ],
            stdout=subprocess.DEVNULL,
        ).returncode
    log(a.label, f"FINISHED rc={rc} tripped={tripped}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
