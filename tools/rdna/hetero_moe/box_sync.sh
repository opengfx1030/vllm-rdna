#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copy this worktree to a work tree on par1-cs25 (CPU pytest, hipcc builds).
# No GPU is touched. Usage, from any directory of the worktree:
#
#   tools/rdna/hetero_moe/box_sync.sh                # rsync to ~/vllm-hx-wip
#   DEST=vllm-hx-probe tools/rdna/hetero_moe/box_sync.sh
#   SO_FROM=vllm-audit-build-moe tools/rdna/hetero_moe/box_sync.sh
#
# SO_FROM copies the prebuilt vllm/*.abi3.so of another box tree next to the
# synced sources (only files the work tree does not have yet), so a GPU run
# can import the compiled ops. Never use it for a measured run: those need an
# exact-SHA tree and a matching build (docs/rdna2/hetero-hx-2026-11-07.md).
set -euo pipefail

BOX=${BOX:-chenco_adm@par1-cs25}
KEY=${KEY:-$HOME/.ssh/id_ed25519_ansible}
DEST=${DEST:-vllm-hx-wip}
SO_FROM=${SO_FROM:-}

SRC=$(git rev-parse --show-toplevel)
SSH="ssh -i $KEY -o BatchMode=yes"

rsync -a --delete -e "$SSH" \
    --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude '*.so' \
    --exclude build --exclude .pytest_cache --exclude .ruff_cache \
    --exclude .venv --exclude node_modules \
    "$SRC/" "$BOX:$DEST/"

if [[ -n "$SO_FROM" ]]; then
    # shellcheck disable=SC2029  # DEST and SO_FROM are meant to expand locally
    $SSH "$BOX" "for f in \$HOME/$SO_FROM/vllm/*.abi3.so; do \
        [ -e \$HOME/$DEST/vllm/\$(basename \$f) ] || cp \$f \$HOME/$DEST/vllm/; done"
fi

SHA=$(git -C "$SRC" rev-parse --short HEAD)
# shellcheck disable=SC2029
$SSH "$BOX" "echo $SHA-dirty > \$HOME/$DEST/.sha"
echo "synced $SRC -> $BOX:~/$DEST (HEAD $SHA plus the working tree)"
