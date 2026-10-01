#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# PR #30 B1: run the compile-dispatch probe on three arms and compare.
#   python  = default inline dispatch (today)
#   op      = VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=1 (candidate fix)
#   eager   = --enforce-eager (no compile; selector follows M)
#
# Defaults are the serving-matrix config (27B AWQ, TP=4, F&P) so every arm
# answers the H1 question on the same model/CG mode the matrix runs at. The
# *graph-ops* evidence is identical at TP=1 and TP=4 because torch.compile
# traces apply_weights once with the trace-time M, so a TP=1 mechanism arm
# does not change the verdict. If you must run TP=1 (e.g. to fit a model on
# 1x GPU), set TP=1 *and* MECHANISM_ONLY=1 — the driver logs the label
# and the record's "mechanism_only" flag is set so H1 cannot be drawn from
# TP=1 perf numbers.
#
# All knobs are env args (no hardcoding beyond the serving-matrix defaults):
#   V            venv path
#   T            tree path
#   MODEL        model snapshot dir
#   TP           tensor-parallel size (default 4)
#   CG_MODE      cudagraph mode (default FULL_AND_PIECEWISE)
#   MAXLEN       max_model_len (default 4096)
#   GMEM         gpu-memory-utilization (default 0.85)
#   TAG          results dir suffix (default 2026-09-30_pr30-b1)
#   OUT          runs root (default /home/chenco_adm/w4a8_runs)
#   MECHANISM_ONLY=1 marks this run as mechanism-only (TP=1 is allowed)
#   GPUIDS       HIP_VISIBLE_DEVICES (default 0,1,2,3)
#
# One engine at a time. Fresh VLLM_CACHE_ROOT per arm.
#   bash tools/rdna2_028/pr30_b1_probe.sh
set -uo pipefail
V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea}
TP=${TP:-4}
CG_MODE=${CG_MODE:-FULL_AND_PIECEWISE}
MAXLEN=${MAXLEN:-4096}
GMEM=${GMEM:-0.85}
TAG=${TAG:-2026-09-30_pr30-b1}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
MECHANISM_ONLY=${MECHANISM_ONLY:-0}
D=$OUT/$TAG
mkdir -p "$D"

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn GPU_MAX_HW_QUEUES=2
export VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RDNA2_FA=1
export VLLM_USE_AOT_COMPILE=0
# The hybrid 27B graph is not saveable on the JIT path (>1 aot_autograd
# artifact -> compiler_interface.is_saveable_2_10 raises). Production 27B
# serves with the compile cache off; computation_graph.py is still written
# to local_cache_dir (backends.py:1359), so H1's graph-op signal survives.
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_RDNA2_W4A8_SDOT4=${W4A8:-0}
export HIP_VISIBLE_DEVICES=${GPUIDS:-0,1,2,3}
source "$T/tools/rdna2_028/tunableop_env.sh"
configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$T/tunableop" 2>&1 | tee -a "$D/driver.log"

cd "$T"
P="benchmarks.kernels.w4a16_compile_dispatch.probe"
LLMK='{"language_model_only": true, "dtype": "float16"}'
COMPILE="{\"cudagraph_mode\":\"$CG_MODE\",\"compile_ranges_endpoints\":[],\"max_cudagraph_capture_size\":16,\"cudagraph_capture_sizes\":[1,2,4,8,16],\"inductor_compile_config\":{\"combo_kernels\":false}}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }

# Scoped to this venv's exe so a co-tenant (venv-7.14.0) is never touched.
# A killed probe leaves its spawned EngineCore/Workers holding ~27 GiB/GPU;
# without this the next arm boots into "free memory 2.3/29.98 GiB".
our_pids() {
  local p exe
  for p in $(pgrep -f "entrypoints.cli.main serve|VLLM::Worker|VLLM::EngineCore" 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null || true)
    case "$exe" in */venv-7.14.0_0.28.0/bin/python*) echo "$p" ;; esac
  done
}
hygiene() {
  local p n=0
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n + 1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
  log "hygiene: terminated=$n remaining=[$(our_pids | tr '\n' ' ')]"
}

# Hard guard: refuse TP=1 unless explicitly labelled mechanism-only.
if [ "$TP" = "1" ] && [ "$MECHANISM_ONLY" != "1" ]; then
  log "FATAL: TP=1 requested without MECHANISM_ONLY=1. Performance numbers at TP=1 do not generalise to the serving-matrix config."
  log "       Re-run with MECHANISM_ONLY=1 if you only need the graph-ops evidence."
  exit 2
fi

hygiene
log "model=$MODEL tp=$TP cg=$CG_MODE maxlen=$MAXLEN w4a8=${W4A8:-0} mechanism_only=$MECHANISM_ONLY"
md5sum "$T/vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py" 2>/dev/null | tee -a "$D/driver.log"

for arm in python op eager; do
  extra=()
  [ "$arm" = op ] && extra=(env VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=1)
  [ "$arm" = eager ] && extra=(env VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=0)
  eager_flag=()
  [ "$arm" = eager ] && eager_flag=(--enforce-eager)
  log "arm $arm start"
  "${extra[@]}" "$V/bin/python" -m $P run \
    --model "$MODEL" --tp "$TP" --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$GMEM" --llm-kwargs "$LLMK" \
    --compilation-config "$COMPILE" \
    "${eager_flag[@]}" \
    --cache-root "$D/cache-$arm" --json "$D/$arm.json" \
    > "$D/$arm.log" 2>&1
  rc=$?
  log "arm $arm done rc=$rc"
  if [ "$rc" != "0" ]; then
    log "arm $arm FAILED; tail of log:"
    tail -30 "$D/$arm.log" | tee -a "$D/driver.log"
  fi
  hygiene
done

log "compare"
"$V/bin/python" -m $P compare "$D/python.json" "$D/op.json" "$D/eager.json" 2>&1 | tee -a "$D/driver.log"
log "done"
