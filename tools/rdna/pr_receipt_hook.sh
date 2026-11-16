#!/usr/bin/env bash
# LOAD_HOOK of tools/rdna/pr_receipt.sh: the quality collects of one boot.
#
# serve_validate.sh runs it after the bench cells with URL, PORT, SERVED, MODEL
# and OUT exported; pr_receipt.sh adds RECEIPT_PY (python of the boot's venv),
# RECEIPT_TOOLS (tree holding these tools), QUALITY_WINDOWS (topk_kl.py windows
# file) and QUALITY_LIMIT / QUALITY_TOP / QUALITY_PASSAGES. Writes
# $OUT/topk.npz (topk_kl.py collect) and $OUT/quality.json (quality_ab.py
# collect), and ends with a PASS or FAIL line as serve_validate.sh expects.
set -uo pipefail
: "${URL:?URL required}" "${SERVED:?SERVED required}" "${OUT:?OUT required}"
: "${RECEIPT_PY:?RECEIPT_PY required}" "${RECEIPT_TOOLS:?RECEIPT_TOOLS required}"
: "${QUALITY_WINDOWS:?QUALITY_WINDOWS required}"
TOOLS=$RECEIPT_TOOLS/tools/rdna/port_v031
rc=0
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "topk_kl collect: ${QUALITY_LIMIT:-0} windows per domain, top ${QUALITY_TOP:-20}"
if "$RECEIPT_PY" "$TOOLS/topk_kl.py" collect --url "$URL" --model "$SERVED" \
    --windows "$QUALITY_WINDOWS" --top "${QUALITY_TOP:-20}" \
    --limit "${QUALITY_LIMIT:-0}" --out "$OUT/topk.npz" > "$OUT/topk-collect.log" 2>&1; then
    say "topk_kl collect done"
else
    say "topk_kl collect FAILED (see topk-collect.log)"
    rc=1
fi

say "quality_ab collect: ${QUALITY_PASSAGES:-6} passages"
if "$RECEIPT_PY" "$TOOLS/quality_ab.py" collect --url "$URL" --model "$SERVED" \
    --passages "${QUALITY_PASSAGES:-6}" --out "$OUT/quality.json" \
    > "$OUT/quality-collect.log" 2>&1; then
    say "quality_ab collect done"
else
    say "quality_ab collect FAILED (see quality-collect.log)"
    rc=1
fi

if ((rc == 0)); then
    echo "PASS pr_receipt quality collects"
else
    echo "FAIL pr_receipt quality collects"
fi
exit "$rc"
