#!/usr/bin/env bash
# Copy the small artefacts of finished serves (bench JSONs, summary, timers,
# parity, concurrency, load hook) from the box into a local results folder.
#   bash tools/rdna/mono/fetch_results.sh <dest dir> <serve tag> [<serve tag> ...]
# Tags are the serve dir names without "serve-", e.g. a12q-mtp2-cfg.
set -uo pipefail
DEST=${1:?dest}; shift
BOX=${BOX:-chenco_adm@192.168.1.84}
KEY=${KEY:-$HOME/.ssh/id_ed25519_ansible}
RUN=${RUN:-w4a8_runs/monokernel-2a}
mkdir -p "$DEST"
for t in "$@"; do
    mkdir -p "$DEST/serve-$t"
    rsync -az -e "ssh -i $KEY" \
        --include='bench-*.json' --include='summary.txt' --include='timing.txt' \
        --include='parity.json' --include='load_hook.txt' --include='concurrency.txt' \
        --exclude='*' "$BOX:$RUN/serve-$t/" "$DEST/serve-$t/"
done
