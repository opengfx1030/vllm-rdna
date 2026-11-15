#!/usr/bin/env bash
# Scoreboard driver: boot ONE arm once, run the fixed cells, probes and quality
# collection, and leave scoreboard.json / scoreboard.md next to the raw results.
#
#   [ENV=...] bash tools/rdna/rdna_hint_flex/scoreboard.sh ARM [KEY=value ...]
#
# ARM names the result directory $OUT_ROOT/ARM. The arm itself is the set of
# KEY=value arguments (recipe overrides and env switches, exactly as
# serve_rdna.sh takes them), e.g.
#
#   scoreboard.sh base                                    # recipe defaults, flags off
#   scoreboard.sh attn VLLM_FA_RDNA2_UNFUSED_PREFILL=1    # one switch
#   scoreboard.sh mtp2 MTP=2
#   BASELINE=base FLOOR=base2 scoreboard.sh attn VLLM_FA_RDNA2_UNFUSED_PREFILL=1
#
# BASELINE / FLOOR name earlier arms of the SAME checkpoint (directories under
# OUT_ROOT): quality is compared to BASELINE, and FLOOR (a second boot of the
# baseline, run it as `scoreboard.sh base2`) gives the boot-to-boot noise a delta
# must exceed. The baseline and the floor are plain arms of this script.
#
# Everything is delegated to existing tools: boot, probes (greedy, PPL, prefix
# cache, concurrency), bench cells, KFD eviction validity and the quiet protocol
# to tools/rdna/port_v031/serve_validate.sh; the quality passes to quality_ab.py,
# topk_kl.py and decode_kl.py (scoreboard_hook.sh); the report to
# scoreboard_report.py. This script adds a 1 s thermal / VRAM sampler and meta.json.
#
# Variables (defaults in brackets):
#   OUT_ROOT [$HOME/w4a8_runs/scoreboard]   RECIPE [27b-hi2-tp2]   MODEL [HI2 27B snapshot]
#   GPUS [2,3]  BUSES [d9 dc] (PCI buses of GPUS; refuses to boot if HIP != BUSES)
#   PORT [18211]  SERVE_TREE [this tree]  VENV [$HOME/Apps/vllm/venv-7.14.0_0.32.0]
#   CELLS [1024:512:1:16 1024:512:8:16 16384:1024:1:8 16384:1024:8:16 32768:2048:1:4]
#       (input:output:concurrency:requests; 32k/2k needs a KV pool of 32k+ tokens)
#   QUALITY [1]  0 skips the quality collection (arm cannot then be judged on quality)
#   QUIET_HOLD [1]  the quiet protocol of serve_validate.sh (waits for any docker buildx
#       build and other lanes' windows); 0 labels the result "not for comparison"
#   KEEP_CACHE [0]  keep the per-arm torch / TunableOp cache directory
#
# The prefix cache is cleared before every cell (RESET_PREFIX_CACHE=1 needs
# VLLM_SERVER_DEV_MODE=1): the cells share one random-prompt seed, so without the
# reset a later cell would be served from the cache of an earlier one.
set -uo pipefail

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ARM=${1:?usage: scoreboard.sh ARM [KEY=value ...]}
shift
OUT_ROOT=${OUT_ROOT:-$HOME/w4a8_runs/scoreboard}
OUT=$OUT_ROOT/$ARM
RECIPE=${RECIPE:-27b-hi2-tp2}
SNAP=("$HOME"/.cache/huggingface/hub/models--BlivionIaG--Qwen3.8-27B-HI2-3.0bpw/snapshots/*/)
MODEL=${MODEL:-${SNAP[0]%/}}
GPUS=${GPUS:-2,3}
BUSES=${BUSES:-"d9 dc"}
PORT=${PORT:-18211}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
SERVE_TREE=${SERVE_TREE:-$TREE}
CELLS=${CELLS:-"1024:512:1:16 1024:512:8:16 16384:1024:1:8 16384:1024:8:16 32768:2048:1:4"}
QUALITY=${QUALITY:-1}
QUIET_HOLD=${QUIET_HOLD:-1}
KEEP_CACHE=${KEEP_CACHE:-0}
[[ -d $MODEL ]] || { echo "scoreboard: MODEL '$MODEL' is not a directory"; exit 2; }
[[ -e $OUT/summary.txt ]] && { echo "scoreboard: $OUT already holds a run (pick a new ARM or remove it)"; exit 2; }
mkdir -p "$OUT"

build_running() {
    pgrep -f '^(/usr/sbin/|/usr/bin/)?docker buildx build' > /dev/null && echo 1 || echo 0
}

# HIP index -> PCI bus must match BUSES before anything touches a GPU.
if [[ -n $BUSES ]]; then
    SITE=$VENV/lib/python3.12/site-packages
    export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    got=$(HIP_VISIBLE_DEVICES=$GPUS "$VENV/bin/python" -c "import torch; print(' '.join('%02x' % torch.cuda.get_device_properties(i).pci_bus_id for i in range(torch.cuda.device_count())))" 2> /dev/null | tail -1)
    if [[ $got != "$BUSES" ]]; then
        echo "scoreboard: HIP $GPUS -> PCI '$got', expected '$BUSES': REFUSING"
        exit 2
    fi
fi

ARM_MTP=0
for kv in "$@"; do [[ $kv == MTP=* ]] && ARM_MTP=${kv#MTP=}; done
TREE_SHA=$(git -C "$SERVE_TREE" rev-parse --short=10 HEAD 2> /dev/null || echo "${TREE_SHA:-unknown}")
ARM=$ARM RECIPE=$RECIPE MODEL=$MODEL GPUS=$GPUS BUSES=$BUSES CELLS=$CELLS QUIET_HOLD=$QUIET_HOLD \
    OVERRIDES="$*" TREE_SHA=$TREE_SHA BUILD_START=$(build_running) OUTDIR=$OUT python3 - << 'EOF'
import json, os, time
e = os.environ
meta = {
    "arm": e["ARM"], "recipe": e["RECIPE"], "model": e["MODEL"], "gpus": e["GPUS"],
    "buses": e["BUSES"], "cells": e["CELLS"].split(), "quiet_hold": e["QUIET_HOLD"],
    "overrides": e["OVERRIDES"], "tree_sha": e["TREE_SHA"],
    "build_running_start": e["BUILD_START"], "start": time.strftime("%F %T"),
    "env": {k: v for k, v in e.items() if k.startswith("VLLM_")},
}
json.dump(meta, open(os.path.join(e["OUTDIR"], "meta.json"), "w"), indent=1)
EOF

# 1 s sampler: junction / edge / memory temperature and VRAM of the arm's cards.
sample_sensors() {
    local out=$1 c s b hw
    declare -A DEV
    for c in /sys/class/drm/card*/device; do
        s=$(grep PCI_SLOT_NAME "$c/uevent" 2> /dev/null | cut -d= -f2)
        for b in $BUSES; do [[ $s == 0000:$b:* ]] && DEV[$b]=$c; done
    done
    echo "epoch bus edge_mC junction_mC mem_mC vram_used_MiB" > "$out"
    while true; do
        for b in "${!DEV[@]}"; do
            c=${DEV[$b]}
            hws=("$c"/hwmon/hwmon*)
            hw=${hws[0]}
            echo "$(date +%s) $b $(cat "$hw/temp1_input" 2> /dev/null || echo 0)" \
                "$(cat "$hw/temp2_input" 2> /dev/null || echo 0)" \
                "$(cat "$hw/temp3_input" 2> /dev/null || echo 0)" \
                "$(($(cat "$c/mem_info_vram_used" 2> /dev/null || echo 0) / 1048576))" >> "$out"
        done
        sleep 1
    done
}
sample_sensors "$OUT/thermal.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2> /dev/null' EXIT

export RECIPE MODEL GPUS PORT TAG=$ARM OUT CELLS QUIET_HOLD VENV SERVE_TREE
export CACHE=$OUT_ROOT/cache-$ARM OUT_ROOT ARM_MTP
export BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock} QUIET_NAME=${QUIET_NAME:-Scoreboard-$ARM}
export VLLM_SERVER_DEV_MODE=1 RESET_PREFIX_CACHE=1
export PREFIX_WORDS=${PREFIX_WORDS:-3000}
export BENCH_EXTRA="${BENCH_EXTRA:---temperature 0 --save-detailed}"
[[ $QUALITY == 1 ]] && export LOAD_HOOK=$TREE/tools/rdna/rdna_hint_flex/scoreboard_hook.sh

bash "$TREE/tools/rdna/port_v031/serve_validate.sh" "$@"
rc=$?
kill $SAMPLER 2> /dev/null
python3 - << EOF
import json, os
p = os.path.join("$OUT", "meta.json")
m = json.load(open(p))
m["build_running_end"] = "$(build_running)"
m["serve_validate_rc"] = $rc
json.dump(m, open(p, "w"), indent=1)
EOF
[[ $KEEP_CACHE == 1 ]] || rm -rf "$CACHE"

REPORT=("$VENV/bin/python" "$TREE/tools/rdna/rdna_hint_flex/scoreboard_report.py" arm "$OUT" --python "$VENV/bin/python")
[[ -n ${BASELINE:-} ]] && REPORT+=(--baseline "$OUT_ROOT/$BASELINE")
[[ -n ${FLOOR:-} ]] && REPORT+=(--floor "$OUT_ROOT/$FLOOR")
"${REPORT[@]}" > "$OUT/scoreboard.stdout" 2>&1 || tail -20 "$OUT/scoreboard.stdout"
echo "scoreboard: $OUT/scoreboard.md and scoreboard.json (serve_validate rc=$rc)"
exit $rc
