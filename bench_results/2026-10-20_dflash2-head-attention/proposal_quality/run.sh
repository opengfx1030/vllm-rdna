#!/usr/bin/env bash
set -euo pipefail

TREE=/home/chenco_adm/vllm-rdna-0.31.0-dflash2
MODEL=/home/chenco_adm/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
VENV=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0
RUN=/home/chenco_adm/w4a8_runs/port-v031/dflash2-headopt-eager-proposals-s16-spec2-long
GPUS=6,7,8,9
RECIPE=flashnext-dflash2
PLE=bf16
DFLASH_TOKENS=2
BENCH_EXTRA='--temperature 0'
PPL_PROBE=0
PREFIX_PROBE=0
CONC_PROBE=0
WARMUP=0
VLLM_RDNA_PHASE_TIMING=0
VLLM_RDNA_OP_TIMING=0
RESET_PREFIX_CACHE=1
VLLM_SERVER_DEV_MODE=1
EXTRA_ARGS='--monokernel --enforce-eager'
VLLM_DFLASH2_PROPOSAL_PROBE_MIN_POSITION=16384
export TREE MODEL VENV GPUS RECIPE PLE DFLASH_TOKENS BENCH_EXTRA
export PPL_PROBE PREFIX_PROBE CONC_PROBE WARMUP VLLM_RDNA_PHASE_TIMING
export VLLM_RDNA_OP_TIMING RESET_PREFIX_CACHE VLLM_SERVER_DEV_MODE EXTRA_ARGS
export VLLM_DFLASH2_PROPOSAL_PROBE_MIN_POSITION
unset TUNABLEOP_ROOT TUNABLEOP_PROFILE
unset PYTORCH_TUNABLEOP_RECORD_UNTUNED PYTORCH_TUNABLEOP_UNTUNED_FILENAME

mkdir -p "$RUN"
test "$(sha256sum "$TREE/vllm/_rocm_C.abi3.so" | cut -d' ' -f1)" = 53a8cb48a69e0cd115e53bc56c155d380b5919fdbd38a1099c4cd410c203855a
test "$(sha256sum "$TREE/vllm/v1/attention/backends/rdna_attn.py" | cut -d' ' -f1)" = 99c5fab1e5dd71bde3afbcc924bd63c9be2077fb9bb7cb635b40dd1cfd9972fa
cat /proc/sys/kernel/random/boot_id > "$RUN/boot-id.before"
sha256sum "$TREE/vllm/_rocm_C.abi3.so" \
    "$TREE/vllm/v1/attention/backends/rdna_attn.py" \
    "$TREE/vllm/v1/worker/gpu/spec_decode/dflash/utils.py" \
    "$TREE/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py" \
    "$TREE/vllm/v1/worker/gpu/spec_decode/dflash2/proposal_probe.py" \
    "$TREE/vllm/v1/worker/gpu/model_runner.py" \
    "$TREE"/tunableop/rocm7.14-rocblas5.5/tunableop_results?.csv > "$RUN/sha256.before"

run_arm() {
    local name=$1 port=$2 conc=$3
    local out=/home/chenco_adm/w4a8_runs/port-v031/serve-$name
    local proposals=$RUN/$name.jsonl
    test ! -e "$proposals"
    if ss -ltn "( sport = :$port )" | grep -q "$port"; then
        printf 'port %s already open\n' "$port" >&2
        exit 1
    fi
    export SPEC_CONFIG='{"method":"dflash","model":"/home/chenco_adm/w4a8_runs/dflash2/model","num_speculative_tokens":2,"attention_backend":"RDNA_ATTN"}'
    export VLLM_DFLASH2_PROPOSAL_PROBE_PATH=$proposals
    CELLS="16384:320:$conc:$conc" TAG=$name PORT=$port \
        CACHE=/home/chenco_adm/w4a8_runs/port-v031/cache-$name \
        bash "$TREE/tools/rdna/port_v031/serve_validate.sh" MTP=2 \
        > "$RUN/$name.driver.log" 2>&1
    test -s "$out/bench-16384x320-c$conc.json"
    test -s "$out/reset-prefix-bench-16384x320-c$conc.json"
    grep -q 'PREFIX RESET PASS before bench-' "$out/summary.txt"
    grep -q 'PROBE-AFTER-LOAD PASS' "$out/summary.txt"
    grep -q 'fault markers in serve.log: 0' "$out/summary.txt"
    grep -q 'new amdgpu dmesg faults: 0' "$out/summary.txt"
    test -s "$proposals"
    local pid
    pid=$(cat "$out/serve.pid")
    for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null && ! pgrep -g "$pid" > /dev/null && \
                ! ss -ltn "( sport = :$port )" | grep -q "$port"; then
            break
        fi
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null || pgrep -g "$pid" > /dev/null || \
            ss -ltn "( sport = :$port )" | grep -q "$port"; then
        printf 'owned server group %s or port %s remains alive\n' "$pid" "$port" >&2
        exit 1
    fi
    wc -l "$proposals"
}

run_arm eagerproposal-headopt-s16-dflash2-spec2-c1 18175 1
run_arm eagerproposal-headopt-s16-dflash2-spec2-c8 18176 8
sha256sum "$TREE/vllm/_rocm_C.abi3.so" \
    "$TREE/vllm/v1/attention/backends/rdna_attn.py" \
    "$TREE/vllm/v1/worker/gpu/spec_decode/dflash/utils.py" \
    "$TREE/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py" \
    "$TREE/vllm/v1/worker/gpu/spec_decode/dflash2/proposal_probe.py" \
    "$TREE/vllm/v1/worker/gpu/model_runner.py" \
    "$TREE"/tunableop/rocm7.14-rocblas5.5/tunableop_results?.csv > "$RUN/sha256.after"
cmp -s "$RUN/sha256.before" "$RUN/sha256.after"
cat /proc/sys/kernel/random/boot_id > "$RUN/boot-id.after"
cmp -s "$RUN/boot-id.before" "$RUN/boot-id.after"
printf 'same-boot S16 spec2 eager proposal diagnostic complete\n'
