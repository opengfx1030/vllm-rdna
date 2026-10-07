#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# STEP-0 in-model kernel profile via vLLM's built-in torch profiler.
#
# Why not rocprofv3: its spawn-mode launcher mishandles vLLM's
# multiprocessing-spawn process tree (the wrapped engine is reported as a
# 0.000 s child, the per-worker rocprofiler buffers are never merged, and no
# CSV is produced). vLLM's own profiler runs inside each worker and writes a
# complete chrome trace on stop_profile(), so it survives the spawn tree.
#
# Kernels are profiled with --enforce-eager so graph replay does not collapse
# them into a hipGraphLaunch. The kernel set (MTP sampling trio, QSA, conv1d,
# GEMMs) is the same as the captured path; shares are a ranking, not a
# per-phase budget.
#
#   MTP=0 TAG=step0_tp_mtp0 bash tools/rdna2_028/step0_torchprof.sh
#
# No /tmp. Artifacts: /home/chenco_adm/w4a8_runs/<TAG>/.
set -uo pipefail

V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PLE=${PLE:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
MTP=${MTP:-0}
OUTLEN=${OUTLEN:-128}
TAG=${TAG:-step0_tp_mtp$MTP}
OUT=${OUT:-/home/chenco_adm/w4a8_runs/$TAG}
CELLS=${CELLS:-"16384 1024"}
READY_SO=${READY_SO:-}

mkdir -p "$OUT"
DRIVER_LOG=$OUT/driver.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

RSL=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
RS=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$RSL:$RS/host-math/lib:$RS/rocm_sysdeps/lib:$RS/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_HOME=${ROCM_HOME:-/opt/rocm/core-7.14} ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-7.14}
export PYTHONPATH="$T${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn GPU_MAX_HW_QUEUES=2
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MOE=0
export VLLM_RDNA_FORCE_FP16=1 TORCH_BLAS_PREFER_HIPBLASLT=0 VLLM_BATCH_INVARIANT=0
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$T/cache/step0tp-$TAG}
export TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TORCH_EXTENSIONS_DIR=$VLLM_CACHE_ROOT/extensions
export VLLM_USE_RDNA2_FA=1 VLLM_FA_RDNA2_GQA_DECODE=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_MOE_PREFILL=0 VLLM_GDN_HIP_PREFILL=0 VLLM_RDNA_FUSED_SE=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_ONESHOT_KB=64 VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=${CONV1D:-0} VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=${CONV1D:-0}
export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_TUNED_CONFIG_FOLDER=$T/tuned-moe
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_QUANT_DIR=$PLE
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
source "$T/tools/rdna2_028/tunableop_env.sh"
configure_tunableop "$RSL/librocblas.so.5" "$T/tunableop" 2>&1 | tee -a "$DRIVER_LOG"

if [[ -n $READY_SO ]]; then cp "$READY_SO" "$T/vllm/_rocm_C.abi3.so"; fi
so=$(sha256sum "$T/vllm/_rocm_C.abi3.so" | cut -c1-16)
sel_count(){ timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
ss0=$(sel_count)
log "=== STEP0-TP MTP=$MTP tag=$TAG so=$so conv1d=${CONV1D:-0} cells='$CELLS' (eager) ==="
log "PCI-SERR before=$ss0 uptime=$(uptime -p)"

SPEC=()
[[ $MTP != 0 ]] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"use_local_argmax_reduction\":true}")

for IN in $CELLS; do
  CD=$OUT/in$IN
  rm -rf "$CD"; mkdir -p "$CD/prof"
  PROFCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$CD/prof\",\"torch_profiler_use_gzip\":false,\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_dump_cuda_time_total\":true}"
  log "cell in=$IN out=$OUTLEN starting"
  t0=$(date +%s)
  "$V/bin/python" -m vllm.entrypoints.cli.main bench throughput \
    --model "$MODEL" --tensor-parallel-size 4 --dtype float16 \
    --block-size 1024 --max-model-len 262144 --max-num-seqs 8 \
    --max-num-batched-tokens 2048 --long-prefill-token-threshold 0 \
    --prefill-schedule-interval 1 --kv-cache-memory-bytes 4026531840 \
    --enable-expert-parallel --enable-prefix-caching --mamba-cache-mode align \
    --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":1605632}' \
    --attention-backend RDNA_ATTN --gpu-memory-utilization 0.90 \
    --dataset-name random --num-prompts 1 --output-len "$OUTLEN" --seed 12345 \
    --input-len "$IN" --enforce-eager --profile --profiler-config "$PROFCFG" \
    "${SPEC[@]}" >"$CD/bench.log" 2>&1
  rc=$?
  log "cell in=$IN rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
  ls -la "$CD/prof"/ 2>/dev/null | tail -6 | tee -a "$DRIVER_LOG"
  ss=$(sel_count); if [[ "$ss" -gt "$ss0" ]]; then log "STOP NEW PCI SERR"; echo STOP >"$OUT/IM_STOP"; break; fi
done
log "=== STEP0-TP done MTP=$MTP ==="
