#!/bin/bash
# SYNC PROTOCOL: capture validated work from a remote build/bench box back to a
# local checkout, commit it, and push to the fork. Run this after every
# bench/validate cycle.
#
# Why this exists: the remote build box is where builds actually run and
# benchmarks actually validate. Its working tree accumulates uncommitted build
# edits (generated files, _so binaries, etc.) that should be captured before the
# next clean rebuild or before someone investigates a regression.
#
# Required env (there are no baked-in host paths):
#   LOCAL=<local checkout>     REMOTE=<user@host>     REMOTE_PATH=<remote tree>
# Optional env:
#   SSH_KEY=~/.ssh/id_ed25519_ansible   BRANCH=rdna_extras
#   SYNC_GIT_NAME / SYNC_GIT_EMAIL      identity for the remote mirror commit
#
# This script:
#   1. rsyncs the validated state from the remote box back to local
#   2. reports any uncommitted/untracked changes on local
#   3. prompts to commit if any
#   4. pushes to origin/<BRANCH>

set -euo pipefail

LOCAL=${LOCAL:?set LOCAL=/path/to/local/checkout}
REMOTE=${REMOTE:?set REMOTE=user@host}
REMOTE_PATH=${REMOTE_PATH:?set REMOTE_PATH=/path/to/remote/tree}
SSH_KEY=${SSH_KEY:-~/.ssh/id_ed25519_ansible}
BRANCH=${BRANCH:-rdna_extras}

echo "=== STEP 1: rsync remote validated state -> local ==="
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
    git commit -m "sync: validated state from remote build box

Auto-captured by tools/rdna/sync_remote_to_local.sh. Captures any
uncommitted build edits, generated files, or working-tree
modifications that the remote build box accumulated since the last sync."
    echo "  committed"
  else
    echo "  aborted"
    exit 0
  fi
else
  echo "  working tree clean"
fi

echo
echo "=== STEP 4: compare with origin/$BRANCH ==="
git log --oneline origin/$BRANCH..HEAD
AHEAD=$(git rev-list --count origin/$BRANCH..HEAD)
echo "  $AHEAD commits ahead of origin"

if [ "$AHEAD" -gt 0 ]; then
  echo
  echo "Push to origin/$BRANCH? (yes/no)"
  read -r ans
  if [ "$ans" = "yes" ]; then
    git remote set-url origin https://github.com/opengfx1030/vllm-rdna.git
    git push origin "$BRANCH" 2>&1 | tail -5
    git remote set-url origin git@github.com:opengfx1030/vllm-rdna.git
  else
    echo "  skipped push"
  fi
fi

echo
echo "=== STEP 5: also commit remote mirror working tree ==="
ssh -i "$SSH_KEY" "$REMOTE" "cd $REMOTE_PATH && \
  ${SYNC_GIT_NAME:+git config user.name '$SYNC_GIT_NAME' &&} \
  ${SYNC_GIT_EMAIL:+git config user.email '$SYNC_GIT_EMAIL' &&} \
  git add -A && \
  if ! git diff --cached --quiet; then \
    echo 'committing remote mirror working tree'; \
    git commit -m 'sync: capture remote mirror working tree (matches local fork branch)'; \
  else \
    echo 'remote working tree already clean'; \
  fi" 2>&1 | tail -5

echo
echo "=== DONE ==="
echo "  local:    $LOCAL  -> $(git log --oneline -1)"
echo "  origin:   $(git log --oneline origin/$BRANCH -1)"
echo "  remote:   $(ssh -i "$SSH_KEY" "$REMOTE" "cd $REMOTE_PATH && git log --oneline -1" 2>&1 | head -1)"
