#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build a NEW directory of symlinks for serving an HI2 Inkling directory, without
# writing anything into the quantizer's output.
#
#   bash tools/inkling_hi2/prep_serve_dir.sh SRC OUT
#   MTP_STOCK=~/hfcache/inkling2-mini/mtp.safetensors \
#     bash tools/inkling_hi2/prep_serve_dir.sh \
#       ~/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5 \
#       ~/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5
#
# SRC is the quantizer's output: config.json (with quantization_config), the
# safetensors index and the shards it lists, the tokenizer/processor files, and
# the unindexed vision shard.  OUT receives one absolute symlink per file plus a
# ``.prep_serve_dir`` marker; SRC is only read.  The stock MTP file (which the
# quantizer does not write) is linked next to them when it exists; with
# REQUIRE_MTP=1 a missing one is an error.
#
# Check the result (CPU, header only, no tensor data):
#   $VENV/bin/python tools/inkling_hi2/validate_dir.py --model OUT
#
# Exit codes: 0 the directory was built and has every required file;
# 1 a required source file is missing; 2 bad usage / OUT already exists.
set -euo pipefail

die() { echo "prep_serve_dir: $*" >&2; exit 2; }
log() { echo "prep_serve_dir: $*" >&2; }

[[ $# -eq 2 ]] || die "usage: $0 SRC OUT  (env: MTP_STOCK=... FORCE=1 REQUIRE_MTP=1)"

src=$1
out=$2
[[ -d $src ]] || die "source directory '$src' does not exist"
src=$(cd "$src" && pwd -P)
if [[ -e $out || -L $out ]]; then
    if [[ ${FORCE:-0} != 1 ]]; then
        die "output '$out' already exists (FORCE=1 replaces a directory this script built)"
    fi
    [[ -f $out/.prep_serve_dir ]] ||
        die "refusing FORCE=1: '$out' has no .prep_serve_dir marker"
    rm -rf -- "$out"
fi
case $out in
    "$src" | "$src"/*) die "OUT must be outside SRC" ;;
esac

python=${PYTHON:-python3}
"$python" - "$src" "$out" "${MTP_STOCK:-}" "${REQUIRE_MTP:-0}" <<'PYEOF'
import json
import os
import sys

src, out, mtp_arg, require_mtp = sys.argv[1:5]

cfg_path = os.path.join(src, "config.json")
idx_path = os.path.join(src, "model.safetensors.index.json")
for path in (cfg_path, idx_path):
    if not os.path.isfile(path):
        sys.exit(f"prep_serve_dir: {path} is missing")

cfg = json.load(open(cfg_path))
qcfg = cfg.get("quantization_config") or {}
if qcfg.get("quant_method") != "rdna_hint_flex":
    sys.exit(
        "prep_serve_dir: config.json is not an HI2 output "
        f"(quant_method={qcfg.get('quant_method')!r})"
    )
if tuple(cfg.get("architectures") or ()) != ("Inkling2LoopForConditionalGeneration",):
    print(f"prep_serve_dir: warning: architectures={cfg.get('architectures')}", file=sys.stderr)

index = json.load(open(idx_path))
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    sys.exit("prep_serve_dir: the index has no weight_map")

# Required: the index and the two files the tokenizer loader needs.  Optional:
# the rest of the side files; whichever exists is linked, the others are
# reported (the stock model's hf_quant_config.json is deliberately not copied).
required = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
] + sorted(set(weight_map.values()))
optional = (
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
    "video_preprocessor_config.json",
    "preprocessor_config.json",
    "tokenizer.model",
    "README.md",
    "tiktoken",
)
wanted = list(required) + [name for name in optional if os.path.exists(os.path.join(src, name))]
# The unindexed vision shard (or another side file) is part of the serve dir.
wanted += sorted(
    name
    for name in os.listdir(src)
    if name.endswith(".safetensors") and name not in set(weight_map.values())
)
missing = [name for name in required if not os.path.exists(os.path.join(src, name))]
if missing:
    sys.exit(f"prep_serve_dir: missing in the quantizer output: {missing}")

os.makedirs(out, exist_ok=False)
for name in dict.fromkeys(wanted):
    os.symlink(os.path.join(src, name), os.path.join(out, name))

mtp = mtp_arg or os.path.expanduser("~/hfcache/inkling2-mini/mtp.safetensors")
if os.path.isfile(mtp):
    os.symlink(os.path.abspath(mtp), os.path.join(out, "mtp.safetensors"))
    print(f"prep_serve_dir: linked mtp.safetensors -> {os.path.abspath(mtp)}", file=sys.stderr)
elif require_mtp == "1":
    sys.exit(f"prep_serve_dir: MTP file {mtp!r} is missing and REQUIRE_MTP=1")
else:
    print(f"prep_serve_dir: warning: no MTP file at {mtp!r}; not linked", file=sys.stderr)
PYEOF

printf 'source=%s\ncreated=%s\n' "$src" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$out/.prep_serve_dir"
log "built $out"
find "$out" -mindepth 1 -maxdepth 1 -printf '%f -> %l\n' | sort
du -shL -- "$out" | sed 's/^/prep_serve_dir: total /'
