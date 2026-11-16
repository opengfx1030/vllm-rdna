#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a finished NIXL producer's side-channel handshake (G0a spike).

``capture`` asks the live producer's side-channel listener for the handshake
payload of every TP rank (the same ZMQ request the decode worker sends) and saves
them. ``serve`` answers the decode workers' requests with the saved payloads on
another port, so a decode instance booted after the producer was stopped
completes the real ``_nixl_handshake`` (compatibility hash, metadata decode,
region geometry, descriptor planning).

  g0a_handshake_server.py capture --store DIR --port 5600 --tp 4
  g0a_handshake_server.py serve   --store DIR --port 5601
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import msgspec
import regex as re
import zmq

GET_META_MSG = b"get_meta_msg"


def capture(args: argparse.Namespace) -> int:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 10000)
    sock.connect(f"tcp://{args.host}:{args.port}")
    os.makedirs(args.store, exist_ok=True)
    for rank in range(args.tp):
        sock.send(msgspec.msgpack.encode((GET_META_MSG, 0, rank)))
        parts = sock.recv_multipart()
        assert len(parts) == 2, f"expected 2 reply parts, got {len(parts)}"
        path = os.path.join(args.store, f"handshake_rank{rank}.bin")
        with open(path, "wb") as f:
            f.write(parts[0])
        print(f"rank {rank}: saved {len(parts[0])} bytes -> {path}")
    return 0


def serve(args: argparse.Namespace) -> int:
    payloads: dict[tuple[int, int], bytes] = {}
    for path in sorted(glob.glob(os.path.join(args.store, "handshake_rank*.bin"))):
        rank = int(re.search(r"rank(\d+)", os.path.basename(path)).group(1))
        with open(path, "rb") as f:
            payloads[(0, rank)] = f.read()
    assert payloads, f"no handshake_rank*.bin in {args.store}"
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(f"tcp://{args.host}:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    print(f"serving {len(payloads)} payloads on {args.host}:{args.port}", flush=True)
    deadline = time.time() + args.max_seconds
    served = 0
    while time.time() < deadline:
        try:
            identity, _, msg = sock.recv_multipart()
        except zmq.Again:
            continue
        kind, pp, tp = msgspec.msgpack.decode(msg)
        if kind != GET_META_MSG:
            print(f"unexpected message {kind!r}", flush=True)
        ts = msgspec.msgpack.encode(time.perf_counter())
        sock.send_multipart((identity, b"", payloads[(pp, tp)], ts))
        served += 1
        print(f"{time.strftime('%T')} served pp={pp} tp={tp} (#{served})", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, required=True)
        if name == "capture":
            p.add_argument("--tp", type=int, default=4)
        else:
            p.add_argument("--max-seconds", type=float, default=6 * 3600)
    args = ap.parse_args()
    return capture(args) if args.cmd == "capture" else serve(args)


if __name__ == "__main__":
    sys.exit(main())
