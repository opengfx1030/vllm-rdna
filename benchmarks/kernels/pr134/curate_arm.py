# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Curate a pulled serving arm down to the evidence worth keeping in Git.

A raw arm carries the rebuildable inductor/triton cache, three 1 MB kernel
ring-buffer snapshots, the 2.7 MB normal-EOS probe dumps and the full serve
log. None of that is evidence; what is evidence is the metrics, the gate
results, the kernel delta the classifier read, the manifest that pinned the
binary and the handful of serve-log lines that say what was served.

This tool copies the evidence into a new directory and distils the serve log
into ``serve-evidence.log``: the TunableOp profile, the attention backend
selection, the KV cache size, the engine configuration and the speculative
decoding metrics. The raw arm is left untouched.

Usage: python curate_arm.py <raw-arm-dir> <curated-dir> [--force]
"""

import argparse
import shutil
from pathlib import Path

# Files copied verbatim. The kernel ring-buffer snapshots are deliberately
# absent: only the delta between them carries the fault attribution, and
# that delta is kept.
KEEP = (
    "bench-1024x512-c1.json",
    "bench-1024x512-c8.json",
    "bench-16384x1024-c1.json",
    "bench-16384x1024-c8.json",
    "bench-1024x512-c1.log",
    "bench-1024x512-c8.log",
    "bench-16384x1024-c1.log",
    "bench-16384x1024-c8.log",
    "concurrency.log",
    "greedy.log",
    "greedy-after.log",
    "kernel.delta.log",
    "kernel.events.json",
    "manifest.before.json",
    "manifest.after.json",
    "metrics.json",
    "models.json",
    "ppl.log",
    "ppl-paired.json",
    "prefix.log",
    "qualification.json",
    "reset-bench-1024x512-c1.json",
    "reset-bench-1024x512-c8.json",
    "reset-bench-16384x1024-c1.json",
    "reset-bench-16384x1024-c8.json",
    "run.json",
    "status.json",
    "warmup.log",
)

# The serve-log lines that say what was served and how it behaved. The
# order is the order they appear in the log.
SERVE_EVIDENCE_PATTERNS = (
    "TunableOp",
    "[api_utils.py:286]",  # the non-default engine arguments
    "[core.py:130]",  # the engine configuration
    "[rocm.py:766]",  # the attention backend selection
    "[kv_cache_utils.py:2469]",  # the KV cache size and max concurrency
    "[metrics.py:120]",  # the speculative decoding metrics
    # The scheduling levers the arm ran with: the mamba state block (which
    # sets the attention block and the prefill budget), the budget the
    # server actually used, the decode-stall cap, and the W4A8 kernel
    # announcing that its fast path is live.
    "Mamba prefix checkpoints",
    "[vllm.py:2380]",  # max_num_scheduled_tokens as the server set it
    "Decode stall bound",
    "w4a8_sdot4_rdna2.hip",
)


def size_of(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def distill_serve_log(raw: Path, target: Path) -> int:
    serve_log = raw / "serve.log"
    if not serve_log.exists():
        return 0
    kept = [
        line
        for line in serve_log.read_text(errors="replace").splitlines()
        if any(pattern in line for pattern in SERVE_EVIDENCE_PATTERNS)
    ]
    target.write_text("\n".join(kept) + "\n")
    return len(kept)


def curate(raw: Path, curated: Path, force: bool = False) -> None:
    if curated.exists():
        if not force:
            raise SystemExit(f"refusing to overwrite {curated}")
        # The raw arm is the source of truth, so replacing the curated copy
        # is what "refresh" means: an earlier curation may predate a file
        # that is now pulled (the manifest, the serve log).
        shutil.rmtree(curated)
    curated.mkdir(parents=True)
    copied = 0
    for name in KEEP:
        source = raw / name
        if source.exists():
            shutil.copy2(source, curated / name)
            copied += 1
    lines = distill_serve_log(raw, curated / "serve-evidence.log")
    print(
        f"curated {raw.name}: {copied} files + serve-evidence.log ({lines} lines)\n"
        f"  {size_of(raw) / 1e6:.1f} MB -> {size_of(curated) / 1e6:.1f} MB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("curated", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing curated directory (the raw arm wins)",
    )
    args = parser.parse_args()
    curate(args.raw, args.curated, force=args.force)


if __name__ == "__main__":
    main()
