#!/usr/bin/env bash
# One Flash-Next serve (INT4 PLE) of this tree through serve_validate.sh,
# MonoKernel (--monokernel) off or on.
#   ARM=on TAG=m-on GPUS=6,7,8,9 PORT=18790 bash tools/rdna/mono/serve_2a.sh [KEY=v ...]
# ARM=on adds --monokernel (all parts, validated defaults) to EXTRA_ARGS;
# MONO_CFG adds rocm_mono_decode knobs, e.g. MONO_CFG='"k2_max_rows":8'.
# ARM=env sets VLLM_ROCM_MONO_DECODE=1 instead of the flag. CELLS /
# BENCH_EXTRA / PPL_PROBE ... pass through to serve_validate.sh. The LOAD_HOOK
# dumps greedy top-5 logprobs (logprob_parity.py) for the off/on comparison.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
RUN=${RUN:-$HOME/w4a8_runs/mono-layout}
: "${ARM:?ARM=on|off}" "${GPUS:?}"
export MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PLE_QUANT_DIR=${PLE_QUANT_DIR:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
export RECIPE=${RECIPE:-flashnext-mtp2} GPUS PORT=${PORT:-18790} TAG=${TAG:-2a-$ARM}
export OUT=$RUN/serve-$TAG CACHE=${CACHE:-$RUN/cache-2a-$ARM}
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
# Keep the Flash-Next mono recipe on the validated 0.92 memory target unless
# the caller explicitly overrides it (e.g. for a controlled memory study).
export GMEM=${GMEM:-0.92}
export LOAD_HOOK=${LOAD_HOOK:-$HERE/parity_hook.sh}
# ARM=on-t / off-t: serve timing marks (mono/rdna2/debug.py); off-t keeps
# today's modules under the same marks. gdn: the GDN part alone. cfg: the
# flag with exactly the MONO_CFG knobs.
X=${MONO_CFG:+,$MONO_CFG}
MK=--monokernel
case $ARM in
    on) KN=${MONO_CFG:-} ;;
    on-t) KN="\"timing\":true$X" ;;
    off-t) KN="\"k1\":false,\"gdn\":false,\"qsa\":false,\"timing\":true,\"modules\":true$X" ;;
    gdn) KN="\"k1\":false,\"k2\":false,\"qsa\":false$X" ;;
    cfg) KN=${MONO_CFG:-} ;;
    env) MK='' KN=${MONO_CFG:-}; export VLLM_ROCM_MONO_DECODE=1 ;;
    off) MK='' KN='' ;;
    *) echo "ARM=on|off|env|on-t|off-t|gdn|cfg" >&2; exit 2 ;;
esac
[[ -n $MK ]] && export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }$MK"
if [[ -n $KN ]]; then
    export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--additional-config {\"rocm_mono_decode\":{$KN}}"
fi
exec bash "$TREE/tools/rdna/port_v031/serve_validate.sh" \
    PLE=int4 PLE_QUANT_DIR="$PLE_QUANT_DIR" "$@"
