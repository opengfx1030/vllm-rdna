// pybind11 entry points for the standalone (JIT) build of
// csrc/rocm/mono_k2_rdna2.cu, used by m1_bench.py.
#include <torch/extension.h>

int64_t mono_k2_rdna2_workspace_bytes(int64_t H, int64_t R, int64_t ND,
                                       int64_t E, int64_t Ise, int64_t I);
void mono_k2_rdna2(const torch::Tensor& res, const torch::Tensor& blk,
                    const torch::Tensor& inj, const torch::Tensor& norm_w,
                    const torch::Tensor& wd, const torch::Tensor& wu,
                    const torch::Tensor& wr, const torch::Tensor& se_w1,
                    const torch::Tensor& se_w2, const torch::Tensor& se_wg,
                    const torch::Tensor& w13, const torch::Tensor& s13,
                    const torch::Tensor& z13, const torch::Tensor& w2,
                    const torch::Tensor& s2, const torch::Tensor& z2,
                    torch::Tensor& res_out, torch::Tensor& inj_out,
                    torch::Tensor& out, torch::Tensor& workspace,
                    int64_t expert_base, int64_t zero_offset, int64_t num_wg,
                    double eps, double timeout_ms);
torch::Tensor mono_k2_rdna2_error(const torch::Tensor& workspace);
torch::Tensor mono_k2_rdna2_workspace_alloc(int64_t nbytes, int64_t device);
int64_t mono_k2_rdna2_mem_flags(const torch::Tensor& t);
void mono_k2_rdna2_e1a(const torch::Tensor& res, const torch::Tensor& blk,
                    const torch::Tensor& inj, const torch::Tensor& norm_w,
                    const torch::Tensor& wd, const torch::Tensor& wu,
                    const torch::Tensor& wq, const torch::Tensor& wba,
                    const torch::Tensor& convw, const torch::Tensor& convb,
                    torch::Tensor& conv_state, torch::Tensor& ssm_state,
                    const torch::Tensor& alog, const torch::Tensor& dtb,
                    const torch::Tensor& gnorm_w, const torch::Tensor& wo,
                    const torch::Tensor& sidx, torch::Tensor& res_out,
                    torch::Tensor& inj_out, torch::Tensor& out, torch::Tensor& workspace,
                    int64_t num_wg, double eps, double gn_eps, int64_t gn_sigmoid,
                    double scale, double timeout_ms);
void mono_k1_rdna2(const torch::Tensor& res, const torch::Tensor& blk,
                   const torch::Tensor& inj, const torch::Tensor& norm_w,
                   const torch::Tensor& wd, const torch::Tensor& wu,
                   torch::Tensor& res_out, torch::Tensor& inj_out,
                   torch::Tensor& out, torch::Tensor& workspace, int64_t num_wg,
                   double eps, double timeout_ms);
torch::Tensor mono_k2_rdna2_phase_ts(const torch::Tensor& workspace);
torch::Tensor mono_k2_rdna2_topk(const torch::Tensor& workspace);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_bytes", &mono_k2_rdna2_workspace_bytes);
  m.def("mono_ffn", &mono_k2_rdna2);
  m.def("mono_hc", &mono_k1_rdna2);
  m.def("workspace_alloc", &mono_k2_rdna2_workspace_alloc);
  m.def("mem_flags", &mono_k2_rdna2_mem_flags);
  m.def("mono_gdn", &mono_k2_rdna2_e1a);
  m.def("error", &mono_k2_rdna2_error);
  m.def("phase_ts", &mono_k2_rdna2_phase_ts);
  m.def("topk", &mono_k2_rdna2_topk);
}
