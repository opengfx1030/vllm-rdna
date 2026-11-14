#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile steady decode of a live rdna server with rocprofv3, per kernel family.

    python3 tools/rdna/rdna_hint_flex/decode_profile.py --label hi2-tp2-f16 \
        --gpus 2,3 --port 18212 --pci-buses d9,dc --model /path/to/checkpoint \
        --cells 1024:1024:1,1024:1024:4 -- RECIPE=27b-hi2-tp2

The server is started with tools/rdna/serve_rdna.sh under ``rocprofv3
--kernel-trace --selected-regions`` (ROCPROFILER_REGISTER_ENABLED=1; CUDA-graph
replays are traced kernel by kernel). tools/rdna/rdna_hint_flex/roctx_shim maps
SIGUSR1/SIGUSR2 to roctxProfilerResume/Pause in every process, so recording
starts only after the engine reached steady decode (all ``conc`` requests
running, prompt tokens flat). A cell is ``in:out:conc[:temperature]`` (random
prompts, ignore-eos; ``out`` must outlast the window). Per cell the driver
records the generated tokens over ``--paused-s`` seconds with the tool loaded
but paused and over the ``--window-s`` recording window (engine step time with
and without the profiler). When the cells are done the workers get SIGTERM (the
tool writes its CSV and chains to vLLM's handler), the server session is killed
and decode_breakdown.py writes breakdown.md/json.

Everything after ``--`` is passed to serve_rdna.sh as KEY=VAL. Output:
OUT_ROOT/LABEL/{serve.log, windows.json, breakdown.md, breakdown.json, trace/
(deleted unless --keep-trace)}.

Technique credit: rocprofv3 kernel tracing and roctx selected regions (ROCm
rocprofiler-sdk); triggering the window on server state follows
tools/rdna/prof/serve_profile.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_VENV = os.path.expanduser("~/Apps/vllm/venv-7.14.0_0.32.0")


def log(label: str, msg: str) -> None:
    print(f"[{label} {time.strftime('%T')}] {msg}", flush=True)


def metric(port: int, name: str) -> float:
    """Summed value of one /metrics series family (0 when the server is silent)."""
    try:
        url = f"http://127.0.0.1:{port}/metrics"
        txt = urllib.request.urlopen(url, timeout=5).read().decode()
    except OSError:
        return 0.0
    tot = 0.0
    for line in txt.splitlines():
        if line.startswith(name) and (
            len(line) == len(name) or line[len(name)] in "{ "
        ):
            tot += float(line.rsplit(" ", 1)[1])
    return tot


def kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpus", required=True, help="HIP_VISIBLE_DEVICES of the server")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--venv", default=os.environ.get("VENV", DEFAULT_VENV))
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--out-root", default=os.path.expanduser("~/w4a8_runs/prof"))
    ap.add_argument("--cache", default="", help="VLLM/Triton cache dir (OUT/cache)")
    ap.add_argument("--cells", default="1024:1024:1,1024:1024:4")
    ap.add_argument("--window-s", type=float, default=2.5)
    ap.add_argument("--paused-s", type=float, default=3.0)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--state-timeout-s", type=float, default=600)
    ap.add_argument(
        "--warm", default="1024:64:4:4", help="in:out:num:conc bench first ('' skips)"
    )
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--boot-lock", default=os.environ.get("BOOT_LOCK", ""))
    ap.add_argument("--refused-gpus", default=os.environ.get("REFUSED_GPUS", "0,1"))
    ap.add_argument("--pci-buses", default="", help="expected PCI buses of --gpus")
    ap.add_argument("--ready-timeout-s", type=float, default=1500)
    ap.add_argument("--keep-trace", action="store_true")
    ap.add_argument("--hbm-gbs", type=float, default=512.0)
    ap.add_argument("serve_args", nargs="*", help="KEY=VAL for serve_rdna.sh")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    for g in a.gpus.split(","):
        if g in a.refused_gpus.split(","):
            log(a.label, f"refusing HIP GPU {g} (REFUSED_GPUS={a.refused_gpus})")
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
    cache = a.cache or os.path.join(out, "cache")
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

    if a.pci_buses:
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
        if got != a.pci_buses.split(","):
            log(a.label, f"refusing: HIP {a.gpus} -> buses {got}, want {a.pci_buses}")
            return 2
        log(a.label, f"bus check ok: HIP {a.gpus} -> {got}")

    with contextlib.ExitStack() as stack:
        if a.boot_lock:
            lock = stack.enter_context(open(a.boot_lock, "w"))
            log(a.label, f"waiting for boot lock {a.boot_lock}")
            fcntl.flock(lock, fcntl.LOCK_EX)
        srv_log = stack.enter_context(open(os.path.join(out, "serve.log"), "wb"))
        cmd = [
            "bash",
            os.path.join(a.tree, "tools/rdna/serve_rdna.sh"),
            f"VENV={a.venv}",
            f"VLLM_TREE={a.tree}",
            f"PORT={a.port}",
            f"MODEL={a.model}",
            *a.serve_args,
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
        )
        served = ""
        while time.time() - t_boot < a.ready_timeout_s and srv.poll() is None:
            try:
                url = f"http://127.0.0.1:{a.port}/v1/models"
                body = urllib.request.urlopen(url, timeout=3).read()
                served = json.loads(body)["data"][0]["id"]
                break
            except (OSError, ValueError, KeyError):
                time.sleep(5)
    sid = srv.pid  # session leader: the bash running serve_rdna.sh
    if not served:
        log(a.label, "BOOTFAIL (see serve.log)")
        kill_group(sid)
        return 1
    log(a.label, f"up after {time.time() - t_boot:.0f}s served={served}")

    def workers() -> list[int]:
        r = subprocess.run(
            ["pgrep", "-s", str(sid), "-f", "[W]orker_TP"],
            capture_output=True,
            text=True,
        )
        return [int(x) for x in r.stdout.split()]

    def bench(name, inl, outl, num, conc, temp):
        argv = [
            py, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "openai", "--endpoint", "/v1/completions",
            "--base-url", f"http://127.0.0.1:{a.port}", "--model", served,
            "--tokenizer", a.model, "--dataset-name", "random",
            "--random-input-len", str(inl), "--random-output-len", str(outl),
            "--num-prompts", str(num), "--max-concurrency", str(conc),
            "--ignore-eos", "--request-rate", "inf", "--seed", str(a.seed),
            "--temperature", str(temp),
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

    def wait_decode(conc: int) -> bool:
        t0 = time.time()
        while time.time() - t0 < a.state_timeout_s:
            run = metric(a.port, "vllm:num_requests_running")
            p0 = metric(a.port, "vllm:prompt_tokens_total")
            time.sleep(2)
            p1 = metric(a.port, "vllm:prompt_tokens_total")
            if run >= conc and p1 == p0:
                return True
        return False

    def wait_idle() -> None:
        t0 = time.time()
        while (
            time.time() - t0 < 120 and metric(a.port, "vllm:num_requests_running") > 0
        ):
            time.sleep(1)

    gen = "vllm:generation_tokens_total"
    if a.warm:
        wi, wo, wn, wc = (int(x) for x in a.warm.split(":"))
        w = bench("warm", wi, wo, wn, wc, 0.0)
        w.wait()
        log(a.label, f"warm bench rc={w.returncode}")
        wait_idle()

    windows = []
    for k, cell in enumerate(c for c in a.cells.replace(" ", ",").split(",") if c):
        f = cell.split(":")
        inl, outl, conc = int(f[0]), int(f[1]), int(f[2])
        temp = float(f[3]) if len(f) > 3 else 0.0
        p = bench(f"cell{k}", inl, outl, conc, conc, temp)
        ok = wait_decode(conc)
        log(a.label, f"cell {cell}: decode state {'reached' if ok else 'NOT reached'}")
        time.sleep(a.settle_s)
        # tool loaded but paused: engine step time without recording
        g0, t0 = metric(a.port, gen), time.time()
        time.sleep(a.paused_s)
        g1, t1 = metric(a.port, gen), time.time()
        wp = workers()
        for pid in wp:
            os.kill(pid, signal.SIGUSR1)
        g2, t2 = metric(a.port, gen), time.time()
        time.sleep(a.window_s)
        g3, t3 = metric(a.port, gen), time.time()
        for pid in wp:
            os.kill(pid, signal.SIGUSR2)
        nan = float("nan")
        paused_itl = (t1 - t0) / ((g1 - g0) / conc) * 1e3 if g1 > g0 else nan
        win_itl = (t3 - t2) / ((g3 - g2) / conc) * 1e3 if g3 > g2 else nan
        windows.append(
            {
                "cell": cell,
                "conc": conc,
                "input": inl,
                "output": outl,
                "temperature": temp,
                "decode_state": ok,
                "workers": wp,
                "paused_s": t1 - t0,
                "paused_tokens": g1 - g0,
                "paused_itl_ms": paused_itl,
                "window_s": t3 - t2,
                "window_tokens": g3 - g2,
                "window_itl_ms": win_itl,
            }
        )
        log(
            a.label,
            f"cell {cell}: engine ITL paused {paused_itl:.2f} ms, recording "
            f"{win_itl:.2f} ms ({g3 - g2:.0f} tokens in {t3 - t2:.2f} s)",
        )
        kill_group(p.pid)
        p.wait()
        wait_idle()
        time.sleep(1)
    with open(os.path.join(out, "windows.json"), "w") as fh:
        json.dump(windows, fh, indent=1)

    # rocprofv3 owns SIGTERM in every traced process, and the API server's
    # handler then waits for children that are never told to stop. Flush the
    # workers first: each worker's tool handler writes its trace and chains to
    # vLLM's.
    wp = workers()
    for pid in wp:
        os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < 900 and any(os.path.exists(f"/proc/{p}") for p in wp):
        time.sleep(2)
    log(a.label, f"workers flushed in {time.time() - t0:.0f}s ({wp})")
    kill_group(sid)
    pattern = os.path.join(out, "trace", "**", "*kernel_trace.csv")
    csvs = glob.glob(pattern, recursive=True)
    mb = sum(os.path.getsize(c) for c in csvs) / 1e6
    log(a.label, f"trace csv={len(csvs)} size={mb:.0f} MB")

    labels = ",".join(w["cell"] for w in windows)
    rc = subprocess.run(
        [
            py,
            os.path.join(HERE, "decode_breakdown.py"),
            os.path.join(out, "trace"),
            "--labels",
            labels,
            "--hbm-gbs",
            str(a.hbm_gbs),
            "--md",
            os.path.join(out, "breakdown.md"),
            "--json",
            os.path.join(out, "breakdown.json"),
        ]
    ).returncode
    if not a.keep_trace:
        shutil.rmtree(os.path.join(out, "trace"), ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)
    log(a.label, f"FINISHED rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
