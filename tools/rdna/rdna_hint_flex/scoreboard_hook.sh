#!/usr/bin/env bash
# LOAD_HOOK of scoreboard.sh: the quality collection of one arm, against the
# server serve_validate.sh booted. Runs, in this order (raw results in $OUT/quality/):
#
#   decode.json  decode_kl.py   32 passages x 512 greedy tokens, 8 concurrent (decode path)
#   qab.json     quality_ab.py  6 passages of ~15k tokens: long-context PPL + greedy text
#   topk.npz     topk_kl.py     64 windows x 2048 tokens, top-20 prompt logprobs (prefill path)
#   spec.txt     spec_accept.py when the arm has MTP > 0 (acceptance at c=1 and c=4)
#
# Every arm must see identical text: the passages and the token windows are cut from
# a corpus frozen at $OUT_ROOT/corpus.txt (the docs of the tree that FIRST ran it);
# delete corpus.txt and windows.json only when you also re-run the baseline.
# Env from serve_validate.sh: URL PORT SERVED MODEL OUT; from scoreboard.sh: VENV
# OUT_ROOT ARM_MTP. Prints PASS / FAIL as the last line.
set -uo pipefail

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=${VENV:?}/bin/python
KIT=$TREE/tools/rdna/port_v031
Q=$OUT/quality
mkdir -p "$Q"
fail=0

# Sizes. SB_SMOKE=1 shrinks everything to check the harness; use a separate OUT_ROOT
# for it, because corpus.txt and windows.json are frozen on first use.
if [[ ${SB_SMOKE:-0} == 1 ]]; then
    N_DEC=2 DEC_TOK=32 DEC_CH=3000 N_LONG=1 LONG_CH=12000 N_WIN=4
else
    N_DEC=32 DEC_TOK=512 DEC_CH=6000 N_LONG=6 LONG_CH=60000 N_WIN=64
fi

CORPUS=$OUT_ROOT/corpus.txt
if [[ ! -s $CORPUS ]]; then
    find "$TREE/docs" -name '*.md' | sort | xargs cat > "$CORPUS.tmp" && mv "$CORPUS.tmp" "$CORPUS"
fi
WINDOWS=$OUT_ROOT/windows.json
if [[ ! -s $WINDOWS ]]; then
    "$PY" "$KIT/topk_kl.py" windows --tokenizer "$MODEL" --out "$WINDOWS" \
        --domain "docs=$CORPUS@0" --count "$N_WIN" --seq-len 2048 || fail=1
fi

step() { # step NAME cmd...
    local name=$1
    shift
    echo "[$(date +%T)] $name"
    "$@" || { echo "[$(date +%T)] $name FAILED"; fail=1; }
}

step decode_kl "$PY" "$KIT/decode_kl.py" collect --url "$URL" --model "$SERVED" \
    --files "$CORPUS" --out "$Q/decode.json" --passages "$N_DEC" \
    --passage-chars "$DEC_CH" --gen-tokens "$DEC_TOK" --workers 8
step quality_ab "$PY" "$KIT/quality_ab.py" collect --url "$URL" --model "$SERVED" \
    --files "$CORPUS" --out "$Q/qab.json" --passages "$N_LONG" \
    --passage-chars "$LONG_CH" --gen-tokens 96
step topk_kl "$PY" "$KIT/topk_kl.py" collect --url "$URL" --model "$SERVED" \
    --windows "$WINDOWS" --top 20 --out "$Q/topk.npz"
if [[ ${ARM_MTP:-0} != 0 ]]; then
    : > "$Q/spec.txt"
    for c in 1 4; do
        echo "[$(date +%T)] spec_accept c=$c"
        "$PY" "$TREE/tools/rdna/rdna_hint_flex/spec_accept.py" --port "$PORT" \
            --model "$SERVED" --conc "$c" --max-tokens 256 --rounds 2 --expect-spec \
            2>&1 | tee -a "$Q/spec.txt" || fail=1
    done
fi
((fail == 0)) && echo "PASS: quality collection in $Q" || echo "FAIL: see above"
