#!/usr/bin/env bash
# Run run_specs.sh once a previous serve has finished (its summary shows the
# final probe, a death or a boot failure), e.g. to queue a second stage.
#   AFTER=~/w4a8_runs/monokernel-2a/serve-x12-mtp2-cfg SPECS=... GPUS=... \
#       bash tools/rdna/mono/chain_specs.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${AFTER:?serve dir to wait for}"
until grep -qE "PROBE-AFTER-LOAD|DIED|FAILED" "$AFTER/summary.txt" 2>/dev/null; do sleep 30; done
sleep 30
exec bash "$HERE/run_specs.sh"
