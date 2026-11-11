#include <torch/extension.h>
void reshape_and_cache_int4_fp16_rdna2(torch::Tensor, torch::Tensor,
                                       torch::Tensor, torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("write_packed", &reshape_and_cache_int4_fp16_rdna2);
}
