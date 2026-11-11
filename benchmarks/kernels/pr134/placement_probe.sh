#!/usr/bin/env bash
# Live GPU placement probe.
#
# The placement assertion only means something while a server is up, so this
# launches the same server the campaign launches, waits for it to be ready,
# captures the placement, and then stops it. The probe output goes to a
# scratch directory, not into the evidence tree: it is a placement record, not
# a measurement.
#
# Usage: placement_probe.sh <output-json> [gpus]
set -uo pipefail

out=${1:?usage: $0 <output-json> [gpus]}
gpus=${2:-6,7,8,9}
readonly tree=/home/chenco_adm/vllm-pr-int4fp16
readonly probe=/home/chenco_adm/pr134-results/placement-probe
readonly model=/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea

cd "$tree" || exit 10
export HIP_VISIBLE_DEVICES=$gpus
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

# A stale probe dir would make qualify_serving refuse to start.
if [[ -e $probe ]]; then
  mv "$probe" "${probe}.stale.$(date +%s)"
fi

bash benchmarks/kernels/pr134/run_remote_micro.sh \
  benchmarks/kernels/pr134/qualify_serving.py \
  --model "$model" --recipe 27b-awq --layout LBHNC --kv-bytes 8000000000 \
  --hw-queues 2 --port 18134 --gpus "$gpus" --run --cache-dtype auto \
  --output "$probe" >/home/chenco_adm/pr134-results/placement-probe.log 2>&1 &
probe_pid=$!

echo "waiting for the server to become ready (probe pid $probe_pid)..."
for _ in $(seq 1 240); do
  if [[ -s $probe/models.json ]]; then
    echo "server is ready; capturing placement"
    sleep 10   # let the ranks finish allocating their KV cache
    python3 benchmarks/kernels/pr134/gpu_placement.py --expect "$gpus" -o "$out"
    rc=$?
    echo "placement capture exit=$rc"
    break
  fi
  if ! kill -0 "$probe_pid" 2>/dev/null; then
    echo "probe exited before the server was ready" >&2
    tail -20 /home/chenco_adm/pr134-results/placement-probe.log >&2
    exit 3
  fi
  sleep 5
done

# Stop the server and the probe. The server runs in its own session, so the
# recorded pgid is what has to go.
if [[ -s $probe/server-owner.json ]]; then
  pgid=$(python3 -c "import json;print(json.load(open('$probe/server-owner.json')).get('pgid',''))")
  if [[ -n $pgid ]]; then
    echo "stopping server pgid $pgid"
    kill -TERM -- "-$pgid" 2>/dev/null
  fi
fi
kill -TERM "$probe_pid" 2>/dev/null
sleep 5
kill -KILL "$probe_pid" 2>/dev/null
wait "$probe_pid" 2>/dev/null

# The scratch probe tree is not evidence; keep only the placement record.
if [[ -d $probe ]]; then
  mv "$probe" "/tmp/placement-probe-tree.$(date +%s)"
fi
echo "probe done"
