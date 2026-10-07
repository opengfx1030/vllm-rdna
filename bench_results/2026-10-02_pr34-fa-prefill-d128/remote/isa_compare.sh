#!/usr/bin/env bash
# Compile fa_rdna2 for each arm with --save-temps and extract the <256,2,8>
# prefill GQA kernel body, then compare instruction streams (D=256 even group
# must be byte-identical between baseline and PR34).
set -uo pipefail
T=/home/chenco_adm/vllm-rdna-0.28.0
BD=$T/build/temp.linux-x86_64-cpython-312
D=/home/chenco_adm/w4a8_runs/pr34_validate
ASM_NAME=fa_rdna2-hip-amdgcn-amd-amdhsa-gfx1030.s

compile_save_temps() {
  cd "$BD"
  local CMD NEW
  CMD=$(ninja -t commands _rocm_C 2>/dev/null | grep "fa_rdna2.hip.o" | head -1)
  NEW=$(printf "%s" "$CMD" | sed "s#bin/clang++#bin/clang++ --save-temps#; s#-o CMakeFiles/_rocm_C.dir/csrc/rocm/fa_rdna2.hip.o#-o $D/fa_probe.o#; s#-MD -MT [^ ]* -MF [^ ]*##")
  eval "$NEW" >/dev/null 2>&1 || { echo "compile failed for $1"; return 1; }
  cp "$BD/$ASM_NAME" "$D/isa_$1.s"
}

extract() {  # $1=arm file  $2=symbol-substring
  awk -v sym="$2" '
    index($0, sym) && $0 ~ /^_Z/ {f=1}
    f {print}
    f && /^[[:space:]]*\.Lfunc_end/ {exit}
  ' "$1" \
  | sed -E 's/;[^\n]*$//; s/[[:space:]]+$//; /^[[:space:]]*\.loc/d; /^[[:space:]]*\.cfi/d; s/_Z[0-9A-Za-z_]+/SYM/g' \
  | sed '/^[[:space:]]*$/d'
}

ARM=baseline bash "$D/build_arm.sh" >/dev/null 2>&1 || { echo "baseline build failed"; exit 1; }
compile_save_temps baseline || exit 1
ARM=patched bash "$D/build_arm.sh" >/dev/null 2>&1 || { echo "patched build failed"; exit 1; }
compile_save_temps patched || exit 1

extract "$D/isa_baseline.s" "fa_prefill_paged_varlen_gqa_kernel_256ILi2ELi8E" > "$D/k256_baseline.s"
extract "$D/isa_patched.s"  "fa_prefill_paged_varlen_gqa_kernelILi256ELi2ELi8E" > "$D/k256_patched.s"
echo "baseline k256 lines: $(wc -l < "$D/k256_baseline.s")"
echo "patched  k256 lines: $(wc -l < "$D/k256_patched.s")"
if diff -q "$D/k256_baseline.s" "$D/k256_patched.s" >/dev/null; then
  echo "D256_EVEN_IDENTICAL=yes"
else
  echo "D256_EVEN_IDENTICAL=no"
  diff "$D/k256_baseline.s" "$D/k256_patched.s" | head -30
fi
