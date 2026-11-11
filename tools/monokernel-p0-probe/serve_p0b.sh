#!/usr/bin/env bash
# P0(b): flashnext-mtp2 serve (production config, INT4 PLE) through the
# validation harness: gates + the four-metric bench cells, then a profiling
# LOAD_HOOK on steady decode at c=1 / c=8.
#   GPUS=2,3,4,5 PORT=18790 MODEL=... PLE_QUANT_DIR=... bash serve_p0b.sh
# Profile-only rerun on a warm cache:
#   TAG=mono-p0b-prof CACHE=$RUN/cache-mono-p0b CELLS=none PPL_PROBE=0 \
#     PREFIX_PROBE=0 CONC_PROBE=0 ... bash serve_p0b.sh
#
# PROF=torch (default): vLLM's torch profiler with GPU activities, plus
#   kineto_shim/sitecustomize.py on PYTHONPATH. Without the shim the workers
#   record CPU ops only: the API server imports torch at startup (venv
#   sitecustomize), its HIP runtime exports ROCPROFILER_REGISTER_LIBRARY into
#   the C environment, every spawned worker inherits it, and rocprofiler-register
#   then never discovers libkineto. Hook: prof_hook_torch.sh.
# PROF=rocprof: preload the venv rocprofiler-sdk tool into the server
#   (serve_rocprof_wrap.sh). Hung the engine on the first requests with
#   VLLM_RDNA_AR=1 (queue interposition vs the cross-GPU flag waits); kept for
#   reference only. Hook: prof_hook.sh.
# PROF=graphs: no profiler; graphcount_shim logs each captured graph's kernel
#   nodes and VLLM_RDNA_PHASE_TIMING logs forward / sample / draft GPU time.
# PROF=none: no profiler (baseline numbers).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
RUN=${RUN:-$HOME/w4a8_runs/monokernel-p0}
: "${MODEL:?}" "${PLE_QUANT_DIR:?}" "${GPUS:?}"
export RECIPE=${RECIPE:-flashnext-mtp2} GPUS PORT=${PORT:-18790} TAG=${TAG:-mono-p0b}
export OUT=$RUN/serve-$TAG CACHE=${CACHE:-$RUN/cache-$TAG} MODEL
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
case ${PROF:-torch} in
    torch)
        export LOAD_HOOK=${LOAD_HOOK:-$HERE/prof_hook_torch.sh}
        export PYTHONPATH=$HERE/kineto_shim${PYTHONPATH:+:$PYTHONPATH}
        export EXTRA_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/torchprof\",\"torch_profiler_with_stack\":false,\"torch_profiler_activities\":[\"CPU\",\"CUDA\"]}"
        ;;
    rocprof)
        export LOAD_HOOK=${LOAD_HOOK:-$HERE/prof_hook.sh}
        mkdir -p "$OUT/rocprof"
        export ROCPROF_OUT=$OUT/rocprof ROCPROF_PERIOD=${ROCPROF_PERIOD:-120:400:1}
        export SERVE_SCRIPT=$HERE/serve_rocprof_wrap.sh
        ;;
    graphs)
        # No profiler: log every captured graph's kernel nodes (graphcount_shim)
        # and per-step GPU phase times (VLLM_RDNA_PHASE_TIMING, CUDA events),
        # under marked c=1 / c=8 decode windows (prof_hook.sh, no idle).
        export LOAD_HOOK=${LOAD_HOOK:-$HERE/prof_hook.sh} PROF_IDLE_S=${PROF_IDLE_S:-0}
        export PYTHONPATH=$HERE/graphcount_shim${PYTHONPATH:+:$PYTHONPATH}
        export GRAPHCOUNT_DIR=$OUT/graphs VLLM_RDNA_PHASE_TIMING=${VLLM_RDNA_PHASE_TIMING:-50}
        ;;
    none) ;;
    *) echo "PROF=torch|graphs|rocprof|none" >&2; exit 2 ;;
esac
exec bash "$TREE/tools/rdna/port_v031/serve_validate.sh" \
    PLE=int4 PLE_QUANT_DIR="$PLE_QUANT_DIR"
