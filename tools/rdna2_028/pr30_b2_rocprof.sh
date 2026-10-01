#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# PR #30 B2: rocprofv3 kernel trace per arm, then summarize W4A16 op families.
# Reuses the B1 compile caches so startup is a cache hit (the trace then covers
# the workload, not the compile). The python/op arms replay the same compiled
# graph; only the eager arm differs. One engine at a time.
#
# Defaults are the serving-matrix config (27B AWQ, TP=4, F&P). All knobs are
# env args; no hardcoding beyond serving-matrix defaults:
#   V/T/MODEL/TP/CG_MODE/MAXLEN/GMEM/TAG/OUT/B1/GPUIDS — see pr30_b1_probe.sh
#   CG_MODE defaults to FULL_AND_PIECEWISE here too. MECHANISM_ONLY=1 lets
#   a TP=1 arm run; the resulting kernel counts are still useful for H2.
#
#   bash tools/rdna2_028/pr30_b2_rocprof.sh
set -uo pipefail
V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea}
TP=${TP:-4}
CG_MODE=${CG_MODE:-FULL_AND_PIECEWISE}
MAXLEN=${MAXLEN:-4096}
GMEM=${GMEM:-0.85}
B1=${B1:-/home/chenco_adm/w4a8_runs/2026-09-30_pr30-b1}
TAG=${TAG:-2026-09-30_pr30-b2}
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
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RDNA2_FA=1 VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_RDNA2_W4A8_SDOT4=${W4A8:-0}
export HIP_VISIBLE_DEVICES=${GPUIDS:-0,1,2,3}
source "$T/tools/rdna2_028/tunableop_env.sh"
configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$T/tunableop" 2>&1 | tee -a "$D/driver.log"

cd "$T"
P="benchmarks.kernels.w4a16_compile_dispatch.probe"
LLMK='{"language_model_only": true, "dtype": "float16"}'
COMPILE="{\"cudagraph_mode\":\"$CG_MODE\",\"compile_ranges_endpoints\":[],\"max_cudagraph_capture_size\":16,\"cudagraph_capture_sizes\":[1,2,4,8,16],\"inductor_compile_config\":{\"combo_kernels\":false}}"
ROC=$V/bin/rocprofv3
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }

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
}

if [ "$TP" = "1" ] && [ "$MECHANISM_ONLY" != "1" ]; then
  log "FATAL: TP=1 requested without MECHANISM_ONLY=1."
  exit 2
fi
log "model=$MODEL tp=$TP cg=$CG_MODE maxlen=$MAXLEN mechanism_only=$MECHANISM_ONLY"
hygiene

run_arm() { # $1=name rest=extra env+args after the python cmd
  local name=$1; shift
  log "arm $name start"
  "$ROC" --kernel-trace --output-format csv --output-file "$D/prof_$name" -- \
    "$@" > "$D/${name}_prof.log" 2>&1
  log "arm $name done rc=$?"
  hygiene
}

run_arm python env \
  "$V/bin/python" -m $P run --model "$MODEL" --tp "$TP" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GMEM" --llm-kwargs "$LLMK" --compilation-config "$COMPILE" \
  --cache-root "$B1/cache-python" --repeat 1 --json "$D/python.json"

run_arm op env VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=1 \
  "$V/bin/python" -m $P run --model "$MODEL" --tp "$TP" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GMEM" --llm-kwargs "$LLMK" --compilation-config "$COMPILE" \
  --cache-root "$B1/cache-op" --repeat 1 --json "$D/op.json"

run_arm eager env \
  "$V/bin/python" -m $P run --model "$MODEL" --tp "$TP" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GMEM" --llm-kwargs "$LLMK" --enforce-eager \
  --cache-root "$B1/cache-eager" --repeat 1 --json "$D/eager.json"

log "kernels"
"$V/bin/python" -m $P kernels "$D"/prof_python_kernel_trace.csv \
  "$D"/prof_eager_kernel_trace.csv "$D"/prof_op_kernel_trace.csv 2>&1 | tee -a "$D/driver.log"
log "done"
