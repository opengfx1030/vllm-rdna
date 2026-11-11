#!/usr/bin/env bash
# Mirror this worktree to the build box (dedicated tree, no --delete).
#   bash tools/rdna/rdna_hint_flex/sync.sh [build]
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
DEST=${DEST:-chenco_adm@par1-cs25:/home/chenco_adm/vllm-rdna-hi2/}
rsync -az --exclude='.git/' --exclude='.deps/' --exclude='build/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.abi3.so' \
  --exclude='.cache/' --exclude='rust/target/' \
  -e "ssh -i $HOME/.ssh/id_ed25519_ansible" "$TREE/" "$DEST"
if [[ ${1:-} == build ]]; then
  ssh -i "$HOME/.ssh/id_ed25519_ansible" "${DEST%%:*}" \
    "cd ${DEST#*:} && MAX_JOBS=24 bash tools/rdna/mono/build_tree.sh > \$HOME/w4a8_runs/hi2-support/build_last.log 2>&1; rc=\$?; grep -E ' error|error:' \$HOME/w4a8_runs/hi2-support/build_last.log | head -20; tail -2 \$HOME/w4a8_runs/hi2-support/build_last.log; exit \$rc"
fi
