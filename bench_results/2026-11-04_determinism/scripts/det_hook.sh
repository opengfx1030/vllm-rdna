#!/usr/bin/env bash
# serve_validate.sh LOAD_HOOK for one determinism arm. serve_validate.sh exports
# PORT, SERVED, MODEL and OUT; run_det_arms.sh exports DET, ARM and DET_PHASES
# (plus DTRACE_DIR for eager trace arms). Ends with PASS or FAIL.
#   ident : IDENT_SEQ windows x IDENT_REPEATS, interleaved (dclient.py)
#   trace : mix:28,docs:0 x 2 interleaved; dumps layer 0/1 MoE outputs (rank 0)
#   noise : +-1 ulp on a random NOISE_FRAC of every MoE block output (same on
#           all ranks), mix:28 x 2 and docs:0 x 2; dumps every decoder layer
#           output and router logits on rank 0 for the mix:28 passes
#   elem  : +1 ulp at layer 0 MoE output [row 0, col 0], mix:28 x 1
#   perf  : PERF_CELLS through vllm bench serve, each inside a KFD
#           evicted_ms window (kfd_evictions.py ok / INVALID)
set -uo pipefail
: "${PORT:?}" "${SERVED:?}" "${OUT:?}" "${DET:?}" "${ARM:?}"
PY=${PY:-$HOME/Apps/vllm/venv-7.14.0_0.32.0/bin/python}
BASE=http://127.0.0.1:$PORT
WIN=$HOME/w4a8_runs/w4a8-quality-2026-11-04/windows.json
KFD=$HOME/w4a8_runs/rdna_ar_wedge/kfd_evictions.py
ROOT_PID=$(cat "$OUT/serve.pid")
rc=0
say() { echo "[$(date +%T)] $*"; }
client() {  # client <name> <seq> <repeats> <order>
    "$PY" "$DET/dclient.py" --url "$BASE" --model "$SERVED" --windows "$WIN" \
        --seq "$2" --repeats "$3" --order "$4" --out "$OUT/$1.npz" \
        > "$OUT/$1.log" 2>&1 || { say "client $1 FAILED"; rc=1; }
    grep -E "vs pass0" "$OUT/$1.log" | sed 's/^/    /' | head -40
}
ctl() {  # ctl <dump_names content> <perturb content>
    [[ -n ${DTRACE_DIR:-} ]] || return 0
    printf '%b' "$1" > "$DTRACE_DIR/dump_names.txt"
    printf '%b' "$2" > "$DTRACE_DIR/perturb.txt"
}
for phase in ${DET_PHASES:-ident}; do
    say "$phase start"
    case $phase in
        ident)
            client ident "${IDENT_SEQ:-mix:28,docs:0}" "${IDENT_REPEATS:-3}" interleave
            ;;
        trace)
            ctl '*layers.0.mlp\n*layers.1.mlp\n*layers.0.mlp.experts\n*layers.0.mlp.gate\n' ''
            client trace mix:28,docs:0 2 interleave
            ctl '' ''
            ;;
        noise)
            frac=${NOISE_FRAC:-0.3}
            [[ -s $DET/noise_frac.txt ]] && frac=$(cat "$DET/noise_frac.txt")
            say "noise frac $frac"
            ctl '*layers.[0-9]\n*layers.[0-9][0-9]\n*layers.*.mlp.gate\n' \
                "*layers.*.mlp noise $frac 1\n"
            client noise_mix mix:28 2 block
            ctl '' "*layers.*.mlp noise $frac 2\n"
            client noise_docs docs:0 2 block
            ctl '' ''
            ;;
        elem)
            ctl '' '*layers.0.mlp elem 0 0 1\n'
            client elem mix:28 1 block
            ctl '' ''
            ;;
        perf)
            for cell in ${PERF_CELLS:-16384:256:1:4 1024:512:8:32}; do
                IFS=: read -r in out conc nreq <<< "$cell"
                name=perf-${in}x${out}-c${conc}
                "$PY" "$KFD" "$ROOT_PID" --save "$OUT/$name.kfd0.json" > /dev/null 2>&1
                "$PY" -m vllm.entrypoints.cli.main bench serve \
                    --backend openai --endpoint /v1/completions \
                    --base-url "$BASE" --model "$SERVED" --tokenizer "$MODEL" \
                    --dataset-name random --random-input-len "$in" \
                    --random-output-len "$out" --num-prompts "$nreq" \
                    --max-concurrency "$conc" --ignore-eos --temperature 0 \
                    --request-rate inf --seed 4242 --save-result --result-dir "$OUT" \
                    --result-filename "$name.json" > "$OUT/$name.log" 2>&1 ||
                    { say "bench $name FAILED"; rc=1; }
                kfd=$("$PY" "$KFD" "$ROOT_PID" --since "$OUT/$name.kfd0.json" 2>&1 | tail -1)
                say "$name kfd: $kfd"
                python3 "$HOME/vllm-w4a8q-4a3d/tools/rdna/port_v031/bench_metrics.py" \
                    "$OUT/$name.json" | sed 's/^/    /'
            done
            ;;
        *) say "unknown phase $phase"; rc=1 ;;
    esac
    say "$phase done rc=$rc"
done
if ((rc == 0)); then echo "PASS det_hook $ARM"; else echo "FAIL det_hook $ARM"; fi
exit "$rc"
