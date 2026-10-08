#!/usr/bin/env bash
# Run the fork's RDNA test files one pytest process per file.
#
#   bash tools/rdna/port_v031/unit.sh                       # this tree, HIP GPU 9
#   TREE=~/vllm-rdna-0.28.0 VENV=~/Apps/vllm/venv-7.14.0_0.28.0 \
#     TAG=baseline-028 GPU=8 bash tools/rdna/port_v031/unit.sh
#   SKIP='gdn_prefill_kkt' bash tools/rdna/port_v031/unit.sh   # skip a file
#   ONLY='gdn_prefill_kkt' TAG=kkt bash tools/rdna/port_v031/unit.sh
#
# Summary: ~/w4a8_runs/port-v031/unit-$TAG/summary.txt
set -uo pipefail

SELF_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TREE=${TREE:-$SELF_TREE}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0}
TAG=${TAG:-v031}
GPU=${GPU:-9}  # HIP index; on par1-cs25 0-1 are W7800s, 2-9 are V620s
PER_FILE_TIMEOUT=${PER_FILE_TIMEOUT:-1800}
SKIP=${SKIP:-}    # extended regex of test paths to skip, e.g. SKIP=gdn_prefill_kkt
ONLY=${ONLY:-}    # extended regex: run only matching test paths
OUT=$HOME/w4a8_runs/port-v031/unit-$TAG
mkdir -p "$OUT"
case ",$GPU," in
    *,0,* | *,1,*) echo "refusing HIP GPUs 0/1 (W7800s on par1-cs25)"; exit 2 ;;
esac

FILES=(
    tests/kernels/quantization/test_rdna2_w4a16.py
    tests/kernels/quantization/test_rdna2_w4a16_awq.py
    tests/kernels/quantization/test_rdna2_w4a16_selection.py
    tests/kernels/quantization/test_w4a16_kernel_selection.py
    tests/kernels/quantization/test_rdna2_w4a8.py
    tests/kernels/quantization/test_rdna_hybrid_w4a16.py
    tests/kernels/quantization/test_rdna2_moe_w4a16.py
    tests/kernels/quantization/test_rdna2_moe_w4a8.py
    tests/kernels/quantization/test_rdna2_w8a16_fp8_moe.py
    tests/kernels/quantization/test_mxfp4_rdna2.py
    tests/kernels/quantization/test_exl3_rdna2.py
    tests/kernels/quantization/test_rocm_skinny_gemms.py
    tests/kernels/quantization/test_rocm_moe_skinny.py
    tests/kernels/quantization/test_reshape_and_cache_flash_rdna2.py
    tests/kernels/quantization/test_gdn_decode_rdna2.py
    tests/kernels/quantization/test_gdn_prefill_prep_rdna2.py
    tests/kernels/quantization/test_gdn_prefill_kkt_rdna2.py
    tests/kernels/quantization/test_gdn_prefill_solve_wy_rdna2.py
    tests/kernels/quantization/test_gdn_prefill_delta_h_rdna2.py
    tests/kernels/quantization/test_gdn_prefill_o_rdna2.py
    tests/kernels/test_gdn_prefill_rdna2.py
    tests/kernels/attention/test_fa_rdna2_shape_sweep.py
    tests/kernels/attention/test_fa_rdna2_writer_layout.py
    tests/kernels/attention/test_rdna_v1_consume.py
    tests/kernels/attention/rdna/dsv4/test_kv_insert.py
    tests/kernels/attention/rdna/dsv4/test_attention_ops.py
    tests/kernels/mamba/test_precopy_mamba_align.py
    tests/kernels/moe/test_v620_moe_wna16_config.py
    tests/quantization/test_moe_wna16.py
    tests/model_executor/layers/test_fused_shared_expert.py
    # One process per file: upstream's nvidia/ and amd/ Qwen4Exp hc modules
    # register the same op name, so collecting both in one process fails.
    tests/models/qwen4_exp/test_config.py
    tests/models/qwen4_exp/test_hc_ops.py
    tests/models/qwen4_exp/test_ple.py
    tests/models/qwen4_exp/test_ple_amd.py
    tests/models/qwen4_exp/test_qsa_amd.py
    tests/models/qwen4_exp/test_qsa_pre_indexer.py
    tests/models/qwen4_exp/test_qsa_reference.py
    tests/compile/test_cudagraph_replay_inputs.py
    tests/compile/test_config.py
    tests/distributed/test_rdna_p2p.py
    tests/platforms/test_rocm_amdsmi_index.py
    tests/v1/attention/test_gdn_metadata_builder.py
    tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
    tests/v1/core/test_kv_cache_utils.py
    tests/v1/core/test_mamba_align_chunk_split.py
    tests/v1/core/test_mixed_batch_policy.py
    tests/v1/core/test_prefix_caching.py
    tests/v1/engine/test_prefill_cadence.py
    tests/v1/kv_offload/test_factory.py
    tests/v1/ple_offload/test_ple_quant_gather.py
    tests/v1/spec_decode/test_qwen4_exp.py
    tests/v1/worker/test_gpu_block_table.py
    tests/v1/worker/test_ple_offload_worker.py
)

SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
export HIP_VISIBLE_DEVICES=$GPU
export VLLM_ROCM_USE_AITER=0 VLLM_RDNA_FORCE_FP16=1 TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn GPU_MAX_HW_QUEUES=2
export TRITON_CACHE_DIR=$OUT/triton-cache

cd "$TREE"
: > "$OUT/summary.txt"
for f in "${FILES[@]}"; do
    name=$(echo "$f" | tr '/' '_')
    if [[ -n $SKIP && $f =~ $SKIP ]] || [[ -n $ONLY && ! $f =~ $ONLY ]]; then
        echo "SKIPPED  $f" >> "$OUT/summary.txt"
        continue
    fi
    if [[ ! -e $f ]]; then
        echo "MISSING  $f" | tee -a "$OUT/summary.txt"
        continue
    fi
    t0=$(date +%s)
    # Load _C/_rocm_C before collection: several RDNA skip guards read
    # torch._C._jit_get_all_schemas() at import time and would skip otherwise.
    timeout "$PER_FILE_TIMEOUT" "$VENV/bin/python" -c '
import sys
import vllm._custom_ops  # noqa: F401
import pytest
sys.exit(pytest.main(["-q", "-rs", "-p", "no:cacheprovider", "-o", "addopts=", sys.argv[1]]))
' "$f" > "$OUT/$name.log" 2>&1
    rc=$?
    tail_line=$(grep -E "passed|failed|error|skipped|no tests ran" "$OUT/$name.log" | tail -1)
    echo "rc=$rc $(( $(date +%s) - t0 ))s  $f  :: $tail_line" | tee -a "$OUT/summary.txt"
done
echo "=== unit $TAG done $(date -Is)" | tee -a "$OUT/summary.txt"
