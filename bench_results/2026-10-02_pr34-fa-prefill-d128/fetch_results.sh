#!/usr/bin/env bash
# Pull PR34 validation artifacts from par1-cs25 into this directory.
set -uo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)"
REMOTE=/home/chenco_adm/w4a8_runs/pr34_validate
rsync -az --exclude='*.so' --exclude='*.pt' --exclude='*.o' --exclude='*.cubin' \
  -e "ssh -i $HOME/.ssh/id_ed25519_ansible" \
  "chenco_adm@par1-cs25:$REMOTE/" "$DEST/remote/"
echo "fetched -> $DEST/remote"
ls "$DEST/remote" | head -40
