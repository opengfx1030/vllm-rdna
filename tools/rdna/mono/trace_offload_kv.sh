#!/usr/bin/env bash
# Diagnostic LOAD_HOOK: arm bounded layer snapshots only for the KV proof.
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
: "${VLLM_RDNA_EXPERT_TRACE_ARM_FILE:?Set the same arm file in server and hook}"
arm=$VLLM_RDNA_EXPERT_TRACE_ARM_FILE
if [[ -e $arm ]]; then
    echo "FAIL: trace arm file already exists: $arm" >&2
    exit 2
fi
mkdir -p "$(dirname "$arm")"
touch "$arm"
trap 'rm -f -- "$arm"' EXIT
bash "$TREE/tools/rdna/mono/probe_offload_kv.sh" "$@"
