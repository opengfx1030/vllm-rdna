#!/usr/bin/env bash
# Builds two variants: default WGP mode and -mcumode. Dumps code-object metadata.
set -euo pipefail
cd "$(dirname "$0")"
ROCM=${ROCM_PATH:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM/bin/hipcc}
READELF=${READELF:-$ROCM/llvm/bin/llvm-readelf}
mkdir -p build
# Pinned flags. -ffp-contract=fast stated explicitly; NO -ffast-math.
COMMON=(--offload-arch=gfx1030 -O3 -ffp-contract=fast -std=c++17 -Wall)
for v in wgp cumode; do
  extra=(); [[ $v == cumode ]] && extra=(-mcumode)
  out=build/$v; mkdir -p "$out"
  echo "== build $v: $HIPCC ${COMMON[*]} ${extra[*]}"
  # --save-temps keeps the device .s (with .amdhsa_* directives and metadata).
  (cd "$out" && "$HIPCC" "${COMMON[@]}" "${extra[@]}" --save-temps ../../probe.hip -o probe) \
     > "$out/build.log" 2>&1 || { cat "$out/build.log"; exit 1; }
  echo "${COMMON[*]} ${extra[*]}" > "$out/flags.txt"
  # Extract the device code object and dump its notes (msgpack metadata).
  co=$(ls "$out"/*gfx1030*.out 2>/dev/null | head -1 || true)
  if [[ -z "$co" ]]; then
    (cd "$out" && "$ROCM/llvm/bin/clang-offload-bundler" --list --type=o --input=probe >/dev/null 2>&1 || true)
    "$ROCM/bin/roc-obj-ls" "$out/probe" > "$out/roc-obj-ls.txt" 2>&1 || true
  fi
  if [[ -n "$co" ]]; then
    "$READELF" --notes "$co" > "$out/notes.txt"
    # Concise per-kernel summary (names, sgpr/vgpr, lds, scratch, wavefront, denorm).
    grep -E '\.name:|\.sgpr_count|\.vgpr_count|\.group_segment_fixed_size|\.private_segment_fixed_size|\.wavefront_size|\.sgpr_spill|\.vgpr_spill|\.max_flat_workgroup_size' \
      "$out/notes.txt" > "$out/kernel_meta.txt" || true
  fi
  s=$(ls "$out"/*gfx1030*.s 2>/dev/null | head -1 || true)
  if [[ -n "$s" ]]; then
    grep -E '^\s*\.amdhsa_(kernel|next_free_vgpr|next_free_sgpr|group_segment_fixed_size|workgroup_processor_mode|float_denorm_mode|float_round_mode|dx10_clamp|ieee_mode|private_segment_fixed_size)|; (NumVgprs|NumSgprs|ScratchSize|Occupancy|LDSByteSize)' \
      "$s" > "$out/amdhsa_directives.txt" || true
  fi
  echo "   -> $out/probe ; metadata: $out/kernel_meta.txt $out/amdhsa_directives.txt"
done
