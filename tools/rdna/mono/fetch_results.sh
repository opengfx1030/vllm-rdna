#!/usr/bin/env bash
# Copy the small artefacts of finished serves (bench JSONs, summary, timers,
# parity, concurrency, load hook) from the box into a local results folder.
#   bash tools/rdna/mono/fetch_results.sh <dest dir> <serve tag> [<serve tag> ...]
# Tags are the serve dir names without "serve-", e.g. a12q-mtp2-cfg.
# Env: BUILD_USER, BUILD_HOST (required, or BOX=<user>@<build-host>), SSH_KEY, RUN.
set -uo pipefail
DEST=${1:?dest}; shift
BUILD_USER=${BUILD_USER:?set BUILD_USER (login on the build host)}
BUILD_HOST=${BUILD_HOST:?set BUILD_HOST (address of the build host)}
BOX=${BOX:-$BUILD_USER@$BUILD_HOST}   # i.e. <user>@<build-host>
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
RUN=${RUN:-w4a8_runs/monokernel-2a}
mkdir -p "$DEST"
for t in "$@"; do
    mkdir -p "$DEST/serve-$t"
    rsync -az -e "ssh -i $SSH_KEY" \
        --include='bench-*.json' --include='summary.txt' --include='timing.txt' \
        --include='parity.json' --include='load_hook.txt' --include='concurrency.txt' \
        --exclude='*' "$BOX:$RUN/serve-$t/" "$DEST/serve-$t/"
done
