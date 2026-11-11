// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Development shim (tools/rdna/hetero_moe/build_hx_dev.sh): registers the hx_*
// ops into a fragment of the _rocm_C library so that a one-translation-unit
// build of csrc/rocm/hetero_moe/hx_ops.cu can be loaded with
// torch.ops.load_library() into a process whose vllm._rocm_C was built before
// the hx ops existed. Production builds register the same function from
// csrc/rocm/torch_bindings.cpp and never contain this file.
#include <torch/library.h>

void hx_register_ops(torch::Library& m);

TORCH_LIBRARY_FRAGMENT(_rocm_C, m) { hx_register_ops(m); }
