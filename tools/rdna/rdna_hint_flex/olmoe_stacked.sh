#!/usr/bin/env bash
# OLMoE-1B-7B per-expert vs stacked rdna_hint_flex layouts on one free V620
# (holds a lock file while it runs): model load time with a cold page cache
# (the checkpoint's pages evicted with posix_fadvise) and a warm one, the
# serve probes, the 1k/512 cells at c=1/c=8, then held-out perplexity of every
# checkpoint through the fork.
#   GPU=2 bash tools/rdna/rdna_hint_flex/olmoe_stacked.sh
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${GPU:?GPU}
R=$HOME/w4a8_runs/hi2-support
S=$R/stacked
A=$HOME/w4a8_runs/hadamard-int2
M=$A/runsmoe
LOCK=$A/hi2support.gpu$GPU.lock
echo "rdna_hint_flex OLMoE stacked-layout check (fork side), started $(date)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
log() { echo "[$(date +%H:%M:%S)] $* (host $(cut -d' ' -f1-3 /proc/loadavg))"; }
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
evict() {
  "$VENV/bin/python" - "$1" <<'PY'
import os, sys
n = 0
for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        fd = os.open(os.path.join(root, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        n += 1
print(f"evicted {n} files under {sys.argv[1]}")
PY
}
B=$TREE/tools/rdna/rdna_hint_flex/bench_serve.sh
export SERVE_OVERRIDES="MAXLEN=4096 MIN_MAXLEN=0"
CKPTS="r4=$M/olmoe-r4 r4s=$M/olmoe-r4-stacked b30=$M/olmoe-b3.0 b30s=$M/olmoe-b3.0-stacked"
for arm in $CKPTS; do
  tag=${arm%%=*}
  model=${arm#*=}
  for state in cold warm; do
    [[ $state == cold ]] && evict "$model"
    log "load $tag $state"
    ROOT=$S/$state CELLS=none CONC_PROBE=0 WARMUP=0 GPU=$GPU PORT=18751 \
      bash "$B" "olmoe-$tag=$model"
    grep -h "Model loading took" "$S/$state/serve-olmoe-$tag/serve.log" | tail -1
  done
done
log "serving cells"
ROOT=$S/serve CELLS="1024:512:1 1024:512:8:32" GPU=$GPU PORT=18751 \
  bash "$B" "olmoe-r4=$M/olmoe-r4" "olmoe-r4s=$M/olmoe-r4-stacked"
MIX=$A/eval_mix.txt
for arm in $CKPTS; do
  tag=${arm%%=*}
  model=${arm#*=}
  GPU=$GPU TP=1 TAG=olmoe_${tag}_study_layout MODEL=$model bash "$TREE/tools/rdna/rdna_hint_flex/acc_tp.sh"
  GPU=$GPU TP=1 TAG=olmoe_${tag}_mix_layout MODEL=$model bash "$TREE/tools/rdna/rdna_hint_flex/acc_tp.sh" --text "$MIX" --skip-tokens 0
done
log "finished"
