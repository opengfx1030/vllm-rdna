#!/usr/bin/env bash
# Pull PR134 serving evidence from par1-cs25 into the local bench_results tree.
#
# Usage: pull_serving_evidence.sh <remote-arm-dir> <local-name> [more arms...]
# Example: pull_serving_evidence.sh serve-27b-fp16-coree443-r3 27b-fp16-r3 \
#                    serve-27b-int4-coree443-r3 27b-int4-r3
set -euo pipefail

readonly results=/home/chenco_adm/pr134-results
readonly local_root=/Users/kletorch/Projects/infrastructure/gfx1030_optimized/vllm-pr-int4fp16/bench_results/2026-10-24-pr134-int4fp16/serving
readonly key=~/.ssh/id_ed25519_ansible
readonly host=par1-cs25

if (( $# < 2 )); then
    echo "usage: $0 <remote-arm-dir> <local-name> [<remote-arm-dir> <local-name> ...]" >&2
    exit 2
fi

while (( $# >= 2 )); do
    remote=$1
    local_name=$2
    shift 2
    target=$local_root/$local_name
    if [[ -e $target ]]; then
        echo "target exists, refusing to overwrite: $target" >&2
        exit 3
    fi
    mkdir -p "$target"
    echo "pulling $remote -> $target"
    # Copy the result artifacts. The triton/inductor caches are large and are
    # rebuildable, so they are excluded; the kernel logs are kept because they
    # carry the fault attribution.
    scp -i "$key" -q -r "$host:$results/$remote/." "$target/"
    if [[ -d $target/cache ]]; then
        echo "  dropping rebuildable cache dir ($(du -sh "$target/cache" | cut -f1))"
        rm -rf "$target/cache"
    fi
    echo "  files: $(find "$target" -type f | wc -l | tr -d ' ')"
done

echo "done"
