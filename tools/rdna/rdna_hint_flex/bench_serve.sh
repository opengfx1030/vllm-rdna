#!/usr/bin/env bash
# Serving benchmark of rdna_hint_flex vs its dense fp16 export (V620s,
# FULL_AND_PIECEWISE cudagraphs via RECIPE=full), the four standard cells,
# through tools/rdna/port_v031/serve_validate.sh.
#   GPU=5 [TP=1] [CENSUS_DIR=dir] bash tools/rdna/rdna_hint_flex/bench_serve.sh \
#       tag=/model/path ...
# CENSUS_DIR records the GEMM shapes without a TunableOp row
# (PYTORCH_TUNABLEOP_RECORD_UNTUNED) into CENSUS_DIR/<tag>/untuned<rank>.csv.
# TUNABLEOP=0 runs on rocBLAS heuristics (no rows at all). SERVE_OVERRIDES adds
# recipe KEY=value overrides (e.g. VLLM_USE_RDNA2_FA=0 for head size 64).
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${GPU:?GPU (HIP indices among ours) required}
TP=${TP:-1}
ROOT=${ROOT:-$HOME/w4a8_runs/hi2-support}
mkdir -p "$ROOT"
CELLS=${CELLS:-"1024:512:1 1024:512:8:32 16384:1024:1:4 16384:1024:8:16"}
PORT=${PORT:-18731}
SMI_OFFSET=${SMI_OFFSET:-2}  # rocm-smi GPU[i+2] = HIP i on par1-cs25
# Wait until the previous arm's server has released the GPUs: a boot right
# after a teardown sees the old process's VRAM and fails the
# gpu-memory-utilization check.
wait_vram_free() {
  local g u used
  for _ in $(seq 1 36); do
    used=0
    for g in ${GPU//,/ }; do
      u=$(rocm-smi -d $((g + SMI_OFFSET)) --showmeminfo vram 2>/dev/null |
        awk -F': ' '/Total Used Memory/ {print $NF}')
      ((${u:-0} > 1073741824)) && used=1
    done
    ((used == 0)) && return 0
    sleep 5
  done
  echo "[$(date +%H:%M:%S)] WARN: VRAM on HIP $GPU still in use after 180 s"
}
# Record every process on our GPUs during the run (another job sharing the
# card skews timings): ROOT/gpu-pids.log, one line per 30 s sample.
smi_ids=""
for g in ${GPU//,/ }; do smi_ids+=" $((g + SMI_OFFSET))"; done
(
  while sleep 30; do
    echo "$(date +%H:%M:%S) $(rocm-smi --showpidgpus 2>/dev/null |
      awk -v ids="$smi_ids" 'BEGIN{n=split(ids,a," ");for(i=1;i<=n;i++)w[a[i]]=1}
        /^PID/{p=$2} /^[0-9 ]+$/{for(i=1;i<=NF;i++) if($i in w) printf "%s@%s ", p, $i}')"
  done
) >>"$ROOT/gpu-pids.log" 2>&1 &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
for arm in "$@"; do
  wait_vram_free
  tag=${arm%%=*}
  model=${arm#*=}
  echo "[$(date +%H:%M:%S)] arm $tag $model TP=$TP GPU=$GPU host $(cut -d' ' -f1-3 /proc/loadavg)"
  if [[ -n ${CENSUS_DIR:-} ]]; then
    mkdir -p "$CENSUS_DIR/$tag"
    export PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
    export PYTORCH_TUNABLEOP_UNTUNED_FILENAME=$CENSUS_DIR/$tag/untuned.csv
  fi
  # shellcheck disable=SC2086
  VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final} RECIPE=full MODEL=$model \
    GPUS=$GPU PORT=$PORT TAG=$tag OUT=$ROOT/serve-$tag CACHE=$ROOT/cache-$tag \
    CELLS="$CELLS" CONC_PROBE=${CONC_PROBE:-1} PREFIX_PROBE=0 \
    bash "$TREE/tools/rdna/port_v031/serve_validate.sh" TP="$TP" MAXLEN=32768 \
    SEQS=8 KV="${KV:-8000000000}" TUNABLEOP="${TUNABLEOP:-1}" ${SERVE_OVERRIDES:-}
  echo "[$(date +%H:%M:%S)] arm $tag rc=$? host $(cut -d' ' -f1-3 /proc/loadavg)"
done
