import csv, json
from pathlib import Path
W = Path("/home/chenco_adm/w4a8_runs/exl3-tunableop")
dec = json.load(open(W / "curated/decisions.json"))
src = set()
for r in csv.reader(open("/home/chenco_adm/vllm-rdna-0.28.0/tunableop/rocblas-f30bb442e9b5/tunableop_results0.csv")):
    if len(r) >= 4 and r[0].startswith("GemmTunableOp"):
        src.add(r[1])
ed = [k for k, v in dec.items() if v == "dropped" and k in src]
print("EXISTING keys DROPPED:", len(ed))
ba = {}
for r in csv.DictReader(open(W / "curated/before_after.csv")):
    ba[r["shape"]] = r
for k in ed:
    r = ba.get(k, {})
    print("  %s  heur=%s cur=%s cur_solver=%s" % (k, r.get("heur_ms"), r.get("current_ms"), r.get("current_solver")))
print()
cap = Path("/home/chenco_adm/w4a8_runs/_captures-exl3/exl3-m0/shapes_exl3-m0.txt")
novel = set(l.split(",")[1] for l in cap.read_text().splitlines() if l.startswith("GemmTunableOp_Half"))
adopted = sorted(k for k in dec if dec[k] == "new" and k in novel)
print("Adopted EXL3 novel keys (%d):" % len(adopted))
for k in adopted:
    r = ba.get(k, {})
    try:
        g = float(r.get("new_ms")) / float(r.get("heur_ms"))
    except Exception:
        g = None
    print("  %s  heur=%s new=%s solver=%s speedup=%s" % (k, r.get("heur_ms"), r.get("new_ms"), r.get("new_solver"), g))
