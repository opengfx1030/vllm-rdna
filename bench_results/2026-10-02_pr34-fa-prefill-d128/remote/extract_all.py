#!/usr/bin/env python3
"""Regenerate the combined PR34 matrix CSV from the saved bench JSONs."""
import csv
import glob
import json
import os
import re

D = "/home/chenco_adm/w4a8_runs/pr34_validate"


def r1(v, nd=1):
    return round(v, nd) if isinstance(v, (int, float)) else ""


rows = []
for jf in sorted(glob.glob(f"{D}/*/cells/*/*.json")):
    parts = jf.split("/")
    tag, cell = parts[-4], parts[-2]
    m = re.match(r"c(\d+)_(\d+)", cell)
    n, inn = (m.group(1), m.group(2)) if m else ("?", "?")
    d = json.load(open(jf))
    dl = f"{D}/{tag}/driver.log"
    arm = so = recipe = mode = "?"
    if os.path.exists(dl):
        t = open(dl).read()
        mm = re.search(r"ARM=(\w+) so=(\w+) recipe=(\S+) mode=(\S+)", t)
        if mm:
            arm, so, recipe, mode = mm.groups()
    g = lambda k: d.get(k) if d.get(k) is not None else ""
    ttft = d.get("mean_ttft_ms") or 0
    tpot = d.get("mean_tpot_ms") or 0
    rows.append({
        "tag": tag, "cell": f"c{n}_{inn}", "arm": arm, "so": so,
        "recipe": recipe, "mode": mode,
        "completed": g("completed"), "failed": g("failed"),
        "output_tok_s": r1(g("output_throughput"), 2),
        "total_tok_s": r1(g("total_token_throughput"), 2),
        "ttft_ms": r1(ttft), "median_ttft_ms": r1(g("median_ttft_ms")),
        "tpot_ms": r1(tpot, 2),
        "per_req_decode_tok_s": r1(1000.0 / tpot, 2) if tpot else "",
        "prefill_tok_s_per_req": r1(int(inn) / (ttft / 1000.0)) if ttft and inn.isdigit() else "",
        "median_itl_ms": r1(g("median_itl_ms")),
        "p99_itl_ms": r1(g("p99_itl_ms"), 2),
        "accept_pct": r1(g("spec_decode_acceptance_rate"), 2),
        "accept_len": r1(g("spec_decode_acceptance_length"), 3),
    })

cols = list(rows[0].keys()) if rows else []
out = f"{D}/matrix_combined.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}: {len(rows)} rows")
