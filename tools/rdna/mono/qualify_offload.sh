#!/usr/bin/env bash
# Full-model LOAD_HOOK: strict KV parity first, then long workload coverage.
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
bash "$TREE/tools/rdna/mono/probe_offload_kv.sh"
bash "$TREE/tools/rdna/mono/qualify_offload_long.sh"
