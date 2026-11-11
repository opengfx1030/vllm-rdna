#!/usr/bin/env bash
# Compile-time variant sweep through the 16-layer M1 bench, one build dir per
# variant. Each argument is NAME or NAME:ENV=v,ENV=v (MONO_FFN_DEF_X=v becomes
# -DMONO_FFN_X=v; MONO_FFN_PREFETCH etc. are runtime env).
#   GPU=2 TAG=r4 bash tools/rdna/mono/sweep.sh base u4:MONO_FFN_DEF_UNR=4
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-sweep}
ROOT=${ROOT:-$HOME/w4a8_runs/monokernel-2a/$TAG}
WIDTHS=${WIDTHS:-3,6,12,24}
ITERS=${ITERS:-10}
mkdir -p "$ROOT"
for spec in "$@"; do
    name=${spec%%:*}
    envs=""
    [[ $spec == *:* ]] && envs=${spec#*:}
    out=$ROOT/$name
    mkdir -p "$out"
    (
        IFS=, read -r -a kv <<< "$envs"
        for e in "${kv[@]}"; do [[ -n $e ]] && export "$e"; done
        export MONO_FFN_VERBOSE=1
        OUT=$out timeout 3000 bash "$HERE/run_m1.sh" --widths "$WIDTHS" --layers 16 \
            --iters "$ITERS" --json "$out/m1.json" > "$out/m1.log" 2>&1
    )
    echo "== $name ($envs) rc=$?"
    grep -E "mono_ffn\[" "$out/m1.log" | sort -u | sed 's/^/   /'
    python3 - "$out/m1.json" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("   no result:", e); sys.exit()
for M, v in r["widths"].items():
    t = v["timing"]
    c = v["correctness"]
    rel = max(x["mono_vs_gold(own routing)"]["max_rel"] for x in c.values())
    rd = ",".join(x["mono_routing_differs_from_gold"] + "|" + x["ref_routing_differs_from_gold"] for x in c.values())
    rrel = max(x["ref_vs_gold(own routing)"]["max_rel"] for x in c.values())
    print(f"   M={M:>2} today {t['today_block']['median_us_per_layer']:6.1f}  mono "
          f"{t['mono_kernel']['median_us_per_layer']:6.1f}  x{t['speedup']:.3f}  "
          f"rel {rel:.1e} (today {rrel:.1e}) route-diff mono|today {rd}  G={t['sticky_err_after'][3]} "
          f"err={t['sticky_err_after'][0]}  {t['mono_phase_us']}")
    print(f"        GB/s {t.get('mono_phase_GBps')}  experts used {t.get('local_experts_used')}")
PY
done
