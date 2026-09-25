#!/bin/bash
# SYNC PROTOCOL: capture validated work from .176 build server back to local,
# commit it, and push to GitHub. Run this AFTER every bench/validate cycle.
#
# Why this exists: the .176 build server is the place where builds actually run
# and benchmarks actually validate. Its working tree accumulates uncommitted
# build edits (generated files, _so binaries, etc.) that should be captured
# before the next clean rebuild or before someone investigates a regression.
#
# This script:
#   1. rsyncs the validated state from .176 back to local
#   2. reports any uncommitted/untracked changes on local
#   3. prompts to commit if any
#   4. pushes to origin/rdna_extras

set -euo pipefail

LOCAL=/Users/kletorch/Projects/infrastructure/gfx1030_optimized/opengfx1030_vllm-rdna
REMOTE=chenco_adm@192.168.1.176
REMOTE_PATH=/home/chenco_adm/opengfx1030_vllm-rdna
SSH_KEY=~/.ssh/id_ed25519_ansible
BRANCH=rdna_extras

echo "=== STEP 1: rsync .176 validated state -> local ==="
rsync -a --update --delete \
  -e "ssh -i $SSH_KEY" \
  --exclude='.git/' --exclude='.deps/' --exclude='build/' --exclude='__pycache__/' \
  --exclude='*.pyc' --exclude='*.abi3.so' \
  --exclude='.cache/' --exclude='.triton/' --exclude='.vllm-cache/' --exclude='.tmp/' \
  "$REMOTE:$REMOTE_PATH/" "$LOCAL/" 2>&1 | tail -5

cd "$LOCAL"
echo
echo "=== STEP 2: check local working tree ==="
git status --short | head -30
NEW_UNCOMMITTED=$(git status --porcelain | wc -l | tr -d ' ')
echo "  $NEW_UNCOMMITTED changed files"

if [ "$NEW_UNCOMMITTED" -gt 0 ]; then
  echo
  echo "=== STEP 3: review changes (M = modified, ?? = untracked) ==="
  git diff --stat | tail -20
  echo
  echo "Continue and commit? (yes/no)"
  read -r ans
  if [ "$ans" = "yes" ]; then
    git add -A
    git commit -m "sync: validated state from .176 build server

Auto-captured by scripts/sync_remote_to_local.sh. Captures any
uncommitted build edits, generated files, or working-tree
modifications that the .176 build server has accumulated since
the last sync. See journal/2026-09-05-gdn-decode-profiling.md
for the protocol rationale."
    echo "  committed"
  else
    echo "  aborted"
    exit 0
  fi
else
  echo "  working tree clean"
fi

echo
echo "=== STEP 4: compare with origin/rdna_extras ==="
git log --oneline origin/$BRANCH..HEAD
AHEAD=$(git rev-list --count origin/$BRANCH..HEAD)
echo "  $AHEAD commits ahead of origin"

if [ "$AHEAD" -gt 0 ]; then
  echo
  echo "Push to origin/$BRANCH? (yes/no)"
  read -r ans
  if [ "$ans" = "yes" ]; then
    git remote set-url origin https://github.com/opengfx1030/vllm-rdna.git
    git push origin $BRANCH 2>&1 | tail -5
    git remote set-url origin git@github.com:opengfx1030/vllm-rdna.git
  else
    echo "  skipped push"
  fi
fi

echo
echo "=== STEP 5: also commit .176 mirror working tree ==="
ssh -i $SSH_KEY $REMOTE "cd $REMOTE_PATH && \
  git config user.email 'kletorch@users.noreply.github.com' && \
  git config user.name 'kletorch' && \
  git add -A && \
  if ! git diff --cached --quiet; then \
    echo 'committing .176 mirror working tree'; \
    git commit -m 'sync: capture .176 working tree (matches local rdna_extras)'; \
  else \
    echo '.176 working tree already clean'; \
  fi" 2>&1 | tail -5

echo
echo "=== DONE ==="
echo "  local:    $LOCAL  -> $(git log --oneline -1)"
echo "  origin:   $(git log --oneline origin/$BRANCH -1)"
echo "  .176:     $(ssh -i $SSH_KEY $REMOTE 'cd $REMOTE_PATH && git log --oneline -1' 2>&1 | head -1)"
