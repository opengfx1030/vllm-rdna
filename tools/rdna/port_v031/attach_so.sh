#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Attach an exact-SHA build's extension artifacts to a plain git-archive tree.
#   bash attach_so.sh <build-dir> <tree> [--force]
#
# A plain `git archive` tree has no compiled extensions, and the venv's
# installed `_rocm_C` may belong to a different commit (custom-op schemas are
# baked into the build, so using it raised e.g. "_rocm_C::rdna_se_gate_up_silu()
# expected at most 3 argument(s) but received 4"). Benchmarks and proofs must
# run with the `.so` files built from their exact SHA. A Python-only commit
# reuses its parent's build after checking that the build-relevant sources are
# byte-identical.
set -uo pipefail
B=$(readlink -f "$1")
T=$(readlink -f "$2")
FORCE=${3:-}
[[ -d $B/vllm && -f $B/vllm/_rocm_C.abi3.so ]] || {
    echo "attach_so: $B/vllm/_rocm_C.abi3.so missing" >&2
    exit 1
}
[[ -d $T/vllm/model_executor ]] || {
    echo "attach_so: $T does not look like an extracted vLLM tree" >&2
    exit 1
}
if [[ -f $T/vllm/_rocm_C.abi3.so && $FORCE != --force ]] &&
    ! cmp -s "$B/vllm/_rocm_C.abi3.so" "$T/vllm/_rocm_C.abi3.so"; then
    echo "attach_so: $T already has a different _rocm_C build; pass --force to replace it" >&2
    exit 1
fi
mkdir -p "$T/vllm/third_party"
for f in "$B"/vllm/*.abi3.so "$B"/vllm/vllm-rs "$B"/vllm/_version.py; do
    [[ -e $f ]] || continue
    rm -f "$T/vllm/$(basename "$f")"
    cp -p "$f" "$T/vllm/"
done
rm -rf "$T/vllm/third_party/triton_kernels"
cp -rp "$B/vllm/third_party/triton_kernels" "$T/vllm/third_party/" || exit 1
echo "attach_so: $T/vllm/_rocm_C.abi3.so $(md5sum "$T/vllm/_rocm_C.abi3.so" | cut -c1-12)"
