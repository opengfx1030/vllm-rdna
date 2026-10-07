#!/usr/bin/env bash
# D=128 / D=256 FA-RDNA2 prefill kernel A/B on gfx1030 (single GPU, no server).
# Run only with no engine active; GPUs must be free.
set -uo pipefail
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
D=/home/chenco_adm/w4a8_runs/pr34_validate
SL=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
SC=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$SL:$SC/host-math/lib:$SC/rocm_sysdeps/lib:$SC/core/lib:$V/lib/python3.12/site-packages/torch/lib"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}"
cp "$D/_rocm_C.patched.so" "$T/vllm/_rocm_C.abi3.so"
cd "$T"
echo "=== D=128: gqa vs short/varlen/splitk (G=4 even, G=7 odd, MHA-ish) ==="
timeout 1200 "$V/bin/python" benchmarks/kernels/benchmark_fa_rdna2_prefill.py \
  --heads 32/8 28/4 16/4 2>&1 | tee "$D/bench_d128.log"
echo
echo "=== D=256: gqa vs varlen/splitk (Flash-Next G=12 even) ==="
timeout 1200 "$V/bin/python" benchmarks/kernels/benchmark_fa_rdna2_prefill.py \
  --head-size 256 --heads 24/2 --block-size 784 2>&1 | tee "$D/bench_d256.log"
