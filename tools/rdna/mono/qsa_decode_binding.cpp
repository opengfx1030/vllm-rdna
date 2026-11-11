// pybind11 entry points for the standalone (JIT) build of
// csrc/rocm/mono_qsa_decode_rdna2.cu, used by qsa_bench.py.
#include <torch/extension.h>

void mono_qsa_decode_rdna2(const torch::Tensor& q, const torch::Tensor& k_cache,
                      const torch::Tensor& v_cache, const torch::Tensor& indices,
                      const torch::Tensor& block_table, const torch::Tensor& token_to_req,
                      torch::Tensor& out, torch::Tensor& scratch, torch::Tensor& counters,
                      int64_t splits);
int64_t mono_qsa_decode_rdna2_splits(int64_t M, int64_t KVH, int64_t W);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &mono_qsa_decode_rdna2);
  m.def("splits", &mono_qsa_decode_rdna2_splits);
}
