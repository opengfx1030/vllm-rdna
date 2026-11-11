#!/usr/bin/env bash
# Refresh the PR134 serving evidence on the current base.
#
# The campaign on the current base (core 13615214, base 12a46f40b4) produced
# four arms: FP16-A, INT4-A, INT4-B, FP16-B, all one boot, one config and one
# source/binary manifest. This script folds them into the evidence tree under
# the same core-tagged names the remote uses, so a local directory always says
# which core produced it.
#
# Usage: refresh_evidence.sh <staging-dir>
#   <staging-dir> holds the pulled raw arms (see collect_serving_arms.sh).
set -euo pipefail

staging=${1:?usage: $0 <staging-dir>}
root=/Users/kletorch/Projects/infrastructure/gfx1030_optimized/vllm-pr-int4fp16
serving=$root/bench_results/2026-10-24-pr134-int4fp16/serving
tools=$root/benchmarks/kernels/pr134
cd "$root"

# ---------------------------------------------------------------- rename ----
# The old names carried no core, so three different campaigns' arms sat under
# names that did not say which core produced them. Give every arm the core
# tag the remote uses.
rename_arm() {
  local from=$1 to=$2
  if [[ -d $serving/$from && ! -e $serving/$to ]]; then
    git mv "$serving/$from" "$serving/$to"
    echo "renamed $from -> $to"
  fi
}
rename_arm 27b-fp16-hip6789 27b-fp16-corec495d204-hip6789
rename_arm 27b-int4-hip6789 27b-int4-corec495d204-hip6789
rename_arm 27b-fp16-hip6789-b 27b-fp16-coree44311ef-hip6789-b
rename_arm 27b-int4-hip6789-b 27b-int4-coree44311ef-hip6789-b
rename_arm 27b-fp16 27b-fp16-coree44311ef-hip6789-r1
rename_arm 27b-fp16-r2 27b-fp16-coree44311ef-hip6789-r2
rename_arm 27b-fp16-r3-hip25 27b-fp16-coree44311ef-hip25-r3

# The committed ABBA artifact was built from the pre-rebase campaign, before
# the A arms were overwritten by the rebased one. Give it the core tag of the
# campaign it actually describes.
for f in abba-counterbalanced-hip6789.json \
         paired-comparison-hip6789.json \
         report-16k-hip6789.md; do
  if [[ -e $serving/$f ]]; then
    case $f in
      abba-*) to=abba-counterbalanced-coree44311ef-hip6789.json ;;
      paired-*) to=paired-comparison-corec495d204-hip6789.json ;;
      report-*) to=report-16k-corec495d204-hip6789.md ;;
    esac
    git mv "$serving/$f" "$serving/$to"
    echo "renamed $f -> $to"
  fi
done

# ---------------------------------------------------------------- curate ----
# Curate each pulled raw arm into the tree. The raw arm carries the
# rebuildable caches and the 1 MB kernel ring buffers; the curated arm keeps
# the metrics, the gates, the kernel delta and the manifest that pinned the
# binary.
curate() {
  local raw=$1 name=$2
  if [[ ! -d $raw ]]; then
    echo "skip $name: no raw arm at $raw" >&2
    return 0
  fi
  # Re-curate from scratch every time: the staged raw arm is the source of
  # truth, and an earlier curation may predate a file that is now pulled (the
  # manifest, the serve log). Curation is idempotent, so re-running is safe.
  python3 "$tools/curate_arm.py" --force "$raw" "$serving/$name"
  echo "curated $name"
}

curate "$staging/pre-fp16-coree443-hip6789" 27b-fp16-coree44311ef-hip6789
curate "$staging/pre-int4-coree443-hip6789" 27b-int4-coree44311ef-hip6789
curate "$staging/pre-fp16-coree443-hip6789-b" 27b-fp16-coree44311ef-hip6789-b
curate "$staging/pre-int4-coree443-hip6789-b" 27b-int4-coree44311ef-hip6789-b
curate "$staging/fp16-core13615214-hip6789" 27b-fp16-core13615214-hip6789
curate "$staging/int4-core13615214-hip6789" 27b-int4-core13615214-hip6789
curate "$staging/int4-core13615214-hip6789-b" 27b-int4-core13615214-hip6789-b
curate "$staging/fp16-core13615214-hip6789-b" 27b-fp16-core13615214-hip6789-b

# -------------------------------------------------------------- artifacts ---
# The counterbalanced artifact for the pre-rebase campaign, rebuilt from the
# four arms with the committed generator (it reproduces the previous values
# exactly; see the generator's commit message).
python3 "$tools/abba_report.py" \
  "$serving/27b-fp16-coree44311ef-hip6789" \
  "$serving/27b-int4-coree44311ef-hip6789" \
  "$serving/27b-int4-coree44311ef-hip6789-b" \
  "$serving/27b-fp16-coree44311ef-hip6789-b" \
  -o "$serving/abba-counterbalanced-coree44311ef-hip6789.json" >/dev/null
echo "wrote abba-counterbalanced-coree44311ef-hip6789.json"

# The current campaign.
python3 "$tools/abba_report.py" \
  "$serving/27b-fp16-core13615214-hip6789" \
  "$serving/27b-int4-core13615214-hip6789" \
  "$serving/27b-int4-core13615214-hip6789-b" \
  "$serving/27b-fp16-core13615214-hip6789-b" \
  -o "$serving/abba-counterbalanced-core13615214-hip6789.json" >/dev/null
echo "wrote abba-counterbalanced-core13615214-hip6789.json"

# The paired comparison and the 16k report, for the rebased campaign and for
# the current one.
python3 "$tools/compare_arms.py" \
  "$serving/27b-fp16-corec495d204-hip6789" \
  "$serving/27b-int4-corec495d204-hip6789" \
  -o "$serving/paired-comparison-corec495d204-hip6789.json" >/dev/null
python3 "$tools/report_16k.py" \
  "$serving/27b-fp16-corec495d204-hip6789" \
  "$serving/27b-int4-corec495d204-hip6789" \
  -o "$serving/report-16k-corec495d204-hip6789.md" >/dev/null
python3 "$tools/compare_arms.py" \
  "$serving/27b-fp16-core13615214-hip6789" \
  "$serving/27b-int4-core13615214-hip6789" \
  -o "$serving/paired-comparison-core13615214-hip6789.json" >/dev/null
python3 "$tools/report_16k.py" \
  "$serving/27b-fp16-core13615214-hip6789" \
  "$serving/27b-int4-core13615214-hip6789" \
  -o "$serving/report-16k-core13615214-hip6789.md" >/dev/null
echo "wrote paired comparisons and 16k reports"

# The FP16 baseline is what every INT4 delta is measured against, so each
# rebase has to be shown not to have moved it.
python3 "$tools/compare_baselines.py" \
  "$serving/27b-fp16-coree44311ef-hip6789" \
  "$serving/27b-fp16-corec495d204-hip6789" \
  -o "$serving/baseline-reproducibility-rebase1.json" >/dev/null
python3 "$tools/compare_baselines.py" \
  "$serving/27b-fp16-corec495d204-hip6789" \
  "$serving/27b-fp16-core13615214-hip6789" \
  -o "$serving/baseline-reproducibility-rebase2.json" >/dev/null
echo "wrote baseline reproducibility artifacts"

# The GPU placement proof.
if [[ -f $staging/gpu-placement.json ]]; then
  cp "$staging/gpu-placement.json" "$serving/gpu-placement.json"
  echo "wrote gpu-placement.json"
fi

echo "refresh complete"
