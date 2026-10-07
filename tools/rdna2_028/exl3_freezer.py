import csv, datetime, json, shutil, sys
from pathlib import Path

T = Path(sys.argv[1])
W = Path(sys.argv[2])
ROWS = T / "tunableop/rocblas-f30bb442e9b5"
BK = W / "repo-rows-premerge"

BK.mkdir(parents=True, exist_ok=True)
for f in list(ROWS.glob("tunableop_results*.csv")) + [ROWS / "provenance.json"]:
    if f.exists():
        shutil.copy(f, BK / f.name)

pre = sum(1 for l in (ROWS / "tunableop_results0.csv").read_text().splitlines() if l.startswith("GemmTunableOp"))
curated0 = (W / "curated/tunableop_results0.csv").read_text()
for r in range(4):
    (ROWS / f"tunableop_results{r}.csv").write_text(curated0)
post = sum(1 for l in (ROWS / "tunableop_results0.csv").read_text().splitlines() if l.startswith("GemmTunableOp"))
print(f"FROZEN {pre} -> {post} rows/rank (backup at {BK})")

def keys(path):
    s = set()
    for row in csv.reader(Path(path).open()):
        if len(row) >= 4 and row[0].startswith("GemmTunableOp"):
            s.add(row[1])
    return s

prior = keys(BK / "tunableop_results0.csv")
adopted = (W / "adopted_new_keys.txt").read_text().splitlines() if (W / "adopted_new_keys.txt").exists() else []
dropped = (W / "dropped_keys.txt").read_text().splitlines() if (W / "dropped_keys.txt").exists() else []
existing_dropped = [k for k in dropped if k in prior]
new_keys = (W / "new_keys.txt").read_text().splitlines() if (W / "new_keys.txt").exists() else []

p = json.loads((ROWS / "provenance.json").read_text())
p["rows_per_rank"] = post
p["campaign"] = {
    "date": datetime.date.today().isoformat(),
    "arm": "exl3-27b-mul1-m0 (Qwen3.8-27B-exl3-3.00bpw, mul1, TP=4, FA-RDNA2, FULL_AND_PIECEWISE)",
    "capture_method": "tools/rdna2_028/exl3_capture.sh MODE=capture with PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 + PYTORCH_TUNABLEOP_UNTUNED_FILENAME; record_untuned DISABLES the results lookup, so cells run on rocBLAS heuristics (shape discovery only).",
    "captured_shapes": len(new_keys),
    "mtp2_status": "NOT CAPTURED: EXL3 MTP head is not wired in this tree (qwen3_5_mtp.py registers only fc.weight for mtp.fc; the checkpoint's fc.suh/trellis/mul1 have nowhere to load -> ValueError at boot). Validated EXL3 serving config is MTP=0; MTP=2 needs an EXL3+MTP loader fix.",
    "tune_method": "tools/rdna2_028/tune_prod_shapes.py --mode tune, novel keys only, 4 GPUs in parallel, >=10 iterations, 25 ms/solver, numerical check 0.01/0.01.",
    "curate": {
        "tool": "tools/rdna2_028/curate_tunableop_rows.py",
        "adopt_margin": 0.03,
        "drop_margin": 0.03,
        "new": len(adopted),
        "dropped": len(dropped),
        "existing_dropped": len(existing_dropped),
        "rows_before": pre,
        "rows_after": post,
    },
}
p["validated"] = (
    datetime.date.today().isoformat()
    + ": EXL3 27B expansion. " + str(len(adopted)) + " new rows adopted (21 novel EXL3 K=5120/1536/4352 shapes + 8 small-M decode extensions, each >=3% faster than both current and the rocBLAS heuristic; best 2.35x). "
    + str(len(existing_dropped)) + " previously-shipped rows dropped (fresh measurement 4-7% slower than the heuristic). "
    + "EXL3 27B rocBLAS fp16 GEMMs are K=5120/1536/4352 and collide with none of the Flash-Next K=2560/320/10240 set (0 collisions, 34 novel captured). "
    + "MTP=2 not captured (EXL3 MTP head unwired). Lookup-hit proof 783/783. Prior set preserved at repo-rows-premerge/."
)
(ROWS / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")
print(f"provenance updated: adopted={len(adopted)} dropped={len(dropped)} existing_dropped={len(existing_dropped)}")
