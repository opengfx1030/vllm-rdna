#include <torch/extension.h>
#include "ops.h"

#ifndef PR134_FP16_ONLY
void fa_rdna2_paged_int4_fp16_block(torch::Tensor, torch::Tensor, torch::Tensor,
                                    torch::Tensor, torch::Tensor, torch::Tensor,
                                    torch::Tensor, torch::Tensor, int64_t,
                                    int64_t, int64_t, double, bool,
                                    torch::Tensor);
#endif
void reshape_and_cache_int4_fp16_rdna2(torch::Tensor, torch::Tensor,
                                       torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &fa_rdna2_decode_paged);
  m.def("prefill", &fa_rdna2_prefill_paged_varlen);
  m.def("prefill_gqa", &fa_rdna2_prefill_paged_varlen_gqa);
  m.def("write_fp16", &reshape_and_cache_flash_rdna2);
#ifndef PR134_FP16_ONLY
  m.def("packed", &fa_rdna2_paged_int4_fp16_block);
#endif
  m.def("write_packed", &reshape_and_cache_int4_fp16_rdna2);
}
