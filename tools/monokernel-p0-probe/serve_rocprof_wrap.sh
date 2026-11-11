#!/usr/bin/env bash
# SERVE_SCRIPT for serve_validate.sh: preload the venv's rocprofiler-sdk tool
# (the env `rocprofv3 --kernel-trace --memory-copy-trace -f csv` sets) and exec
# tools/rdna/serve_rdna.sh. Kernel tracing only inside ROCPROF_PERIOD
# (start_delay:duration:repeat, seconds from each process's tool init).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${VENV:?}" "${ROCPROF_OUT:?}"
L=$VENV/lib/python3.12/site-packages/_rocm_sdk_core/lib
IFS=: read -r p_delay p_dur p_rep <<< "${ROCPROF_PERIOD:-120:400:1}"
export ROCPROFILER_REGISTER_LIBRARY=$L/librocprofiler-sdk.so.1 ROCPROFILER_LIBRARY_CTOR=1
export LD_PRELOAD=$L/rocprofiler-sdk/librocprofiler-sdk-tool.so.1:$L/librocprofiler-sdk.so.1
export ROCP_TOOL_LIBRARIES=$L/rocprofiler-sdk/librocprofiler-sdk-tool.so.1
export ROCPROF_OUTPUT_FILE_NAME=%pid% ROCPROF_OUTPUT_PATH=$ROCPROF_OUT ROCPROF_OUTPUT_FORMAT=csv
export ROCPROF_KERNEL_TRACE=1 ROCPROF_MEMORY_COPY_TRACE=1
export ROCPROF_COLLECTION_PERIOD=$((p_delay * 1000000000)):$((p_dur * 1000000000)):$p_rep
exec bash "$HERE/../rdna/serve_rdna.sh" "$@"
