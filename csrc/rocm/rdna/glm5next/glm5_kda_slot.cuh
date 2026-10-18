// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// vLLM state slots (vllm/v1/attention/backends/utils.py):
//   NULL_BLOCK_ID = 0
//   PAD_SLOT_ID   = -1
// fused_recurrent_kda skips state_idx <= 0 and stores only when the
// index is > 0. Both Glm5Next KDA kernels use this predicate so slot 0
// and a padding slot leave conv/recurrent state untouched.
__host__ __device__ __forceinline__ bool glm5_kda_slot_is_empty(long slot,
                                                                long num_slots) {
  return slot <= 0 || slot >= num_slots;
}
