# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand-in for the rdna server and for `bench serve` in test_c8_profile_dry_run.py.

serve:  python3 c8_fake_server.py serve --port P
    /v1/models and /metrics (vllm:num_requests_running) like the real server, two
    fake VLLM::Worker_TP0/1 processes that log SIGUSR1/SIGUSR2 to
    $C8_FAKE_DIR/signals.log, and an `Iteration(n): ...` line on stdout for every
    scripted engine step. A bench request (GET /bench?inl=&outl=&num=) plays the
    c=8 16k/1k timeline in miniature: one request prefilling alone, then seven
    decoders next to a prefill chunk, then eight decoders.
bench:  python3 c8_fake_server.py bench --port P [bench serve flags...]
    Asks the server to play the timeline for the flags it was given and returns
    when it is over (the driver kills it earlier for the long cell).
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

WORKER = r"""
import os, signal, sys, time
log = sys.argv[1]
def note(name):
    def handler(*_):
        with open(log, "a") as fh:
            fh.write(f"{name} {os.getpid()}\n")
    return handler
signal.signal(signal.SIGUSR1, note("resume"))
signal.signal(signal.SIGUSR2, note("pause"))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.05)
"""

# (steps, seconds per step, context requests, context tokens, generation requests)
C8 = (
    (40, 0.05, 1, 2048, 0),  # request 0 prefills alone
    (60, 0.05, 1, 1024, 7),  # seven decoders next to a prefill chunk
    (200, 0.02, 0, 0, 8),  # eight decoders
)
WARM = ((30, 0.02, 0, 0, 4),)


class State:
    def __init__(self) -> None:
        self.running = 0
        self.iter = 0
        self.lock = threading.Lock()

    def step(self, cr: int, ct: int, gr: int, dt: float) -> None:
        with self.lock:
            self.iter += 1
            n = self.iter
        print(
            f"(EngineCore pid=1) INFO 11-15 18:00:00 [core.py:1] Iteration({n}): "
            f"{cr} context requests, {ct} context tokens, {gr} generation "
            f"requests, {gr} generation tokens, iteration elapsed time: "
            f"{dt * 1e3:.2f} ms",
            flush=True,
        )

    def play(self, script) -> None:
        for steps, dt, cr, ct, gr in script:
            self.running = cr + gr
            for _ in range(steps):
                self.step(cr, ct, gr, dt)
                time.sleep(dt)
        self.running = 0


def serve(port: int) -> None:
    state = State()
    sig_log = os.path.join(os.environ.get("C8_FAKE_DIR", os.getcwd()), "signals.log")
    workers = [
        subprocess.Popen(
            [
                "bash",
                "-c",
                'exec -a "$0" python3 -c "$1" "$2"',
                f"VLLM::Worker_TP{r}",
                WORKER,
                sig_log,
            ]
        )
        for r in range(2)
    ]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            if url.path == "/v1/models":
                self._send(json.dumps({"data": [{"id": "fake"}]}).encode())
            elif url.path == "/metrics":
                txt = (
                    "# TYPE vllm:num_requests_running gauge\n"
                    f'vllm:num_requests_running{{engine="0"}} {float(state.running)}\n'
                )
                self._send(txt.encode(), "text/plain")
            elif url.path == "/bench":
                q = urllib.parse.parse_qs(url.query)
                state.play(C8 if int(q["inl"][0]) >= 16384 else WARM)
                self._send(b"{}")
            else:
                self.send_error(404)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    try:
        srv.serve_forever()
    finally:
        for w in workers:
            w.terminate()


def bench(port: int, argv: list[str]) -> None:
    def flag(name: str) -> str:
        return argv[argv.index(name) + 1]

    q = urllib.parse.urlencode(
        {
            "inl": flag("--random-input-len"),
            "outl": flag("--random-output-len"),
            "num": flag("--num-prompts"),
        }
    )
    urllib.request.urlopen(f"http://127.0.0.1:{port}/bench?{q}", timeout=600).read()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("serve", "bench"))
    ap.add_argument("--port", type=int, required=True)
    args, rest = ap.parse_known_args()
    if args.mode == "serve":
        serve(args.port)
    else:
        bench(args.port, rest)
    sys.exit(0)
