#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Boot an HI2 Inkling-2-Mini directory on two V620s (TP=2), language model only.
#
#   MODEL=/path/to/bpw3.5 HIP_VISIBLE_DEVICES=0,1 VENV=/path/to/venv \
#       bash tools/inkling_hi2/serve.sh
#
# Required: MODEL (a directory written by the quantizer; it carries the tokenizer),
#           HIP_VISIBLE_DEVICES (the lease's V620s, see tools/inkling_hi2/README.md),
#           VENV (the box venv with the built vllm).
# Optional: TREE (default: this worktree), PORT=18230, MAXLEN=8192, MAXBAT=256,
#           SEQS=1, EAGER=0, GMEM=0.92, WAIT_QUIET=1 (wait out other agents' quiet
#           windows), BOOT_LOCK, QUIET_DIR, REGRESSION=1 (the HI2 kernel path; 0
#           boots with the torch glue reference for an A/B).
#
# The recipe (inkling2-mini-tp2) owns FULL_AND_PIECEWISE HIP graphs, prefix
# caching, the HIP all-reduce and the TP plan; this wrapper only pins the
# language-only + HI2 settings and the small first-run context.
set -euo pipefail

: "${MODEL:?MODEL must point at the HI2 output directory}"
: "${HIP_VISIBLE_DEVICES:?set HIP_VISIBLE_DEVICES to the leased V620 pair}"
: "${VENV:?set VENV to the box venv with the built vllm}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../.." && pwd)}
PORT=${PORT:-18230}
MAXLEN=${MAXLEN:-8192}
MAXBAT=${MAXBAT:-256}
SEQS=${SEQS:-1}
EAGER=${EAGER:-0}
GMEM=${GMEM:-0.92}
WAIT_QUIET=${WAIT_QUIET:-1}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet}
REGRESSION=${REGRESSION:-1}

if [[ ! -f $MODEL/model.safetensors.index.json || ! -f $MODEL/tokenizer.json ]]; then
    echo "serve.sh: $MODEL is not a servable HI2 directory (index or tokenizer missing)" >&2
    exit 2
fi
if [[ ! -x $VENV/bin/python ]]; then
    echo "serve.sh: $VENV/bin/python not found" >&2
    exit 2
fi

# --- lease / quiet / boot protocol (AGENTS.md, "GPU selection on par1-cs25") ---
# Booters wait until no measured window is younger than 6 minutes, then take the
# boot lock without holding it for the whole serve.
if [[ $WAIT_QUIET == 1 ]]; then
    while find "$QUIET_DIR" -type f -mmin -6 2>/dev/null | grep -q .; do
        echo "serve.sh: another agent's quiet window is active; waiting (60 s)"
        sleep 60
    done
fi
mkdir -p "$(dirname "$BOOT_LOCK")"
flock -w 900 "$BOOT_LOCK" true

# The HIP indexes must be V620s of the leased slot, not the W7800s (HIP 0/5).
HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES "$VENV/bin/python" - <<'PY'
import os, torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    name = p.name
    bus = "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)
    print(f"serve.sh: HIP {i} {bus} {name}")
    if "V620" not in name:
        raise SystemExit(f"serve.sh: HIP {i} is {name}, not a V620")
PY

EXTRA="--limit-mm-per-prompt {\"image\":0} --max-logprobs 128"
if [[ $REGRESSION == 0 ]]; then
    export VLLM_INKLING2_PATH=reference
fi

echo "serve.sh: MODEL=$MODEL TP=2 PORT=$PORT MAXLEN=$MAXLEN MAXBAT=$MAXBAT SEQS=$SEQS EAGER=$EAGER"
exec bash "$TREE/tools/rdna/serve_rdna.sh" \
    RECIPE=inkling2-mini-tp2 \
    MODEL="$MODEL" \
    VENV="$VENV" \
    HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
    PORT="$PORT" \
    DSPARK=0 MTP=0 EAGER="$EAGER" CG_MODE=FULL_AND_PIECEWISE \
    MAXLEN="$MAXLEN" MAXBAT="$MAXBAT" SEQS="$SEQS" GMEM="$GMEM" \
    EXTRA_ARGS="$EXTRA"
