#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# STEP-0 in-model kernel profile for the gfx1030 fork.
#
# rocprofv3 in --run (spawn) mode around `vllm bench throughput`, which drives
# the real vLLM engine with the serving config and then exits cleanly so the
# rocprofiler tool finalizes and merges its output (the long-lived `vllm serve`
# path cannot be flushed: SIGINT does not exit vLLM and SIGKILL aborts the
# finalizer). One cell per invocation, one engine at a time.
#
#   MTP=0 TAG=step0_mtp0 bash tools/rdna2_028/step0_profile.sh
#   MTP=2 TAG=step0_mtp2 bash tools/rdna2_028/step0_profile.sh
#
# Cells: c=1 1k prefills + 128 decode steps, c=1 16k prefills + 128 decode
# steps. No /tmp (rocprof TMPDIR + all artifacts under $OUT).
set -uo pipefail

V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PLE=${PLE:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
MTP=${MTP:-0}
OUTLEN=${OUTLEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-1}
TAG=${TAG:-step0_mtp$MTP}
OUT=${OUT:-/home/chenco_adm/w4a8_runs/$TAG}
CELLS=${CELLS:-"1024 16384"}
READY_SO=${READY_SO:-}

mkdir -p "$OUT"
DRIVER_LOG=$OUT/driver.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

# --- ROCm/venv library resolution (mirror of rdna_launcher_common) ----------
RSL=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
RS=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$RSL:$RS/host-math/lib:$RS/rocm_sysdeps/lib:$RS/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_HOME=${ROCM_HOME:-/opt/rocm/core-7.14}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-7.14}
export HIP_PATH=${HIP_PATH:-/opt/rocm/core-7.14}
export HIP_ROOT_DIR=${HIP_ROOT_DIR:-/opt/rocm/core-7.14}
export PYTHONPATH="$T${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export GPU_MAX_HW_QUEUES=2
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MOE=0
export VLLM_RDNA_FORCE_FP16=1 TORCH_BLAS_PREFER_HIPBLASLT=0 VLLM_BATCH_INVARIANT=0
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$T/cache/step0-$TAG}
export TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor
export TORCH_EXTENSIONS_DIR=$VLLM_CACHE_ROOT/extensions
export VLLM_USE_RDNA2_FA=1 VLLM_FA_RDNA2_GQA_DECODE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_MOE_PREFILL=0 VLLM_GDN_HIP_PREFILL=0
export VLLM_RDNA_FUSED_SE=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_ONESHOT_KB=64
export VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=${CONV1D:-0}
export VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=${CONV1D:-0}
export VLLM_ENABLE_STARTUP_PLAN=0
export VLLM_TUNED_CONFIG_FOLDER=$T/tuned-moe
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_QUANT_DIR=$PLE
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1

# TunableOp: shared build-keyed rows (keeps GEMM selection deterministic).
# shellcheck source=tools/rdna2_028/tunableop_env.sh
source "$T/tools/rdna2_028/tunableop_env.sh"
configure_tunableop "$RSL/librocblas.so.5" "$T/tunableop" 2>&1 | tee -a "$DRIVER_LOG"

# --- optional .so swap (for A/B; default = whatever is installed) -----------
if [[ -n $READY_SO ]]; then
  cp "$READY_SO" "$T/vllm/_rocm_C.abi3.so"
fi
so=$(sha256sum "$T/vllm/_rocm_C.abi3.so" | cut -c1-16)

# --- PCI-SERR guard ---------------------------------------------------------
sel_count(){ timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
ss0=$(sel_count)
log "=== STEP0 MTP=$MTP tag=$TAG so=$so cells='$CELLS' ==="
log "PCI-SERR before=$ss0 uptime=$(uptime -p)"

# --- model compilation config ----------------------------------------------
COMPILE='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"compile_ranges_endpoints":[]}'
SPEC=()
if [[ $MTP != 0 ]]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"use_local_argmax_reduction\":true}")
fi

common_args=(--model "$MODEL" --tensor-parallel-size 4 --dtype float16
  --block-size 1024 --max-model-len 262144 --max-num-seqs 8
  --max-num-batched-tokens 2048 --long-prefill-token-threshold 0
  --prefill-schedule-interval 1 --kv-cache-memory-bytes 4026531840
  --compilation-config "$COMPILE" --enable-expert-parallel
  --enable-prefix-caching --mamba-cache-mode align
  --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":1605632}'
  --attention-backend RDNA_ATTN --gpu-memory-utilization 0.90
  --dataset-name random --num-prompts "$NUM_PROMPTS" --output-len "$OUTLEN"
  --seed 12345)
[[ ${#SPEC[@]} -gt 0 ]] && common_args+=("${SPEC[@]}")

for IN in $CELLS; do
  CD=$OUT/in$IN
  mkdir -p "$CD" "$CD/tmp"
  log "cell in=$IN out=$OUTLEN starting"
  t0=$(date +%s)
  TMPDIR="$CD/tmp" "$V/bin/rocprofv3" --kernel-trace -f csv -o "$CD/prof" -- \
    "$V/bin/python" -m vllm.entrypoints.cli.main bench throughput \
    "${common_args[@]}" --input-len "$IN" \
    >"$CD/bench.log" 2>&1
  rc=$?
  t1=$(date +%s)
  log "cell in=$IN rc=$rc elapsed=$((t1-t0))s"
  grep -m1 -iE "Throughput|Output token throughput|Mean TTFT|Mean TPOT" "$CD/bench.log" | tr -s ' ' | tee -a "$DRIVER_LOG"
  ls -la "$CD"/prof*.csv 2>/dev/null | tee -a "$DRIVER_LOG" || log "NO CSV for in=$IN"
  ss=$(sel_count)
  if [[ "$ss" -gt "$ss0" ]]; then
    log "STOP NEW PCI SERR ($ss0 -> $ss)"; echo "STOP NEW PCI SERR" >"$OUT/IM_STOP"; break
  fi
done
log "=== STEP0 done MTP=$MTP ==="
