#!/usr/bin/env python
"""Standalone hipcc probe for gdn_prefill_prep_rdna2: screens variants in
seconds instead of a ~30 s-2 min ROCm extension rebuild.

It extracts the device-code namespace out of
``csrc/rocm/gdn_prefill_prep_rdna2.cu`` (everything between ``namespace {``
and ``}  // namespace``), appends a small host harness, compiles it with
``hipcc -O3 --offload-arch=gfx1030`` into ``/tmp/probe_prep`` and runs it.

The harness mirrors ``/tmp/bench_prep.py``: same varlen single-sequence
geometry, same buffers reused across iterations, 1 s clock warm, then
min-of-5 batched reps of 60 iterations.

    HIP_VISIBLE_DEVICES=2 python tools/rdna2/probe_gdn_prefill_prep_rdna2.py
"""

import os
import re
import subprocess
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "rocm",
                   "gdn_prefill_prep_rdna2.cu")
BIN = "/tmp/probe_prep_rdna2"

HARNESS = r"""
}  // namespace

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define HC(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void clock_warm_kernel(float* p, int n) {
  float a = p[threadIdx.x];
  for (int i = 0; i < 200000; ++i) a = fmaf(a, 1.000001f, 0.5f);
  if (a == 12345.0f) p[threadIdx.x] = a;
}

int main(int argc, char** argv) {
  long L = (argc > 1) ? atol(argv[1]) : 8192;
  int n_seq = (argc > 2) ? atoi(argv[2]) : 1;
  int reps = (argc > 3) ? atoi(argv[3]) : 5;
  const long H = 4, HV = 12, K = 128, V = 128, BT = 64;
  const long qkv_dim = 2 * H * K + HV * V;
  const long NT = (L + BT - 1) / BT;

  std::vector<__half> h_x(L * qkv_dim), h_a(L * HV), h_b(L * HV);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (auto& x : h_x) x = __float2half(d(rng));
  for (auto& x : h_a) x = __float2half(d(rng));
  for (auto& x : h_b) x = __float2half(d(rng));

  __half *d_x, *d_a, *d_b, *d_q, *d_k, *d_v;
  float *d_A, *d_dt, *d_g, *d_be;
  int *d_cu, *d_ci;
  HC(hipMalloc(&d_x, L * qkv_dim * sizeof(__half)));
  HC(hipMalloc(&d_a, L * HV * sizeof(__half)));
  HC(hipMalloc(&d_b, L * HV * sizeof(__half)));
  HC(hipMalloc(&d_q, L * H * K * sizeof(__half)));
  HC(hipMalloc(&d_k, L * H * K * sizeof(__half)));
  HC(hipMalloc(&d_v, L * HV * V * sizeof(__half)));
  HC(hipMalloc(&d_A, HV * sizeof(float)));
  HC(hipMalloc(&d_dt, HV * sizeof(float)));
  HC(hipMalloc(&d_g, L * HV * sizeof(float)));
  HC(hipMalloc(&d_be, L * HV * sizeof(float)));
  HC(hipMalloc(&d_cu, (n_seq + 1) * sizeof(int)));
  HC(hipMalloc(&d_ci, NT * 2 * sizeof(int)));
  HC(hipMemcpy(d_x, h_x.data(), L * qkv_dim * sizeof(__half), hipMemcpyHostToDevice));
  HC(hipMemcpy(d_a, h_a.data(), L * HV * sizeof(__half), hipMemcpyHostToDevice));
  HC(hipMemcpy(d_b, h_b.data(), L * HV * sizeof(__half), hipMemcpyHostToDevice));
  {
    std::vector<float> A(HV), dt(HV);
    for (int i = 0; i < HV; ++i) { A[i] = d(rng); dt[i] = d(rng); }
    HC(hipMemcpy(d_A, A.data(), HV * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_dt, dt.data(), HV * sizeof(float), hipMemcpyHostToDevice));
  }
  {
    std::vector<int> cu(n_seq + 1), ci(NT * 2);
    long per = L / n_seq, off = 0;
    long k = 0;
    for (int s = 0; s < n_seq; ++s) {
      long len = (s == n_seq - 1) ? (L - off) : per;
      cu[s] = (int)off;
      for (long t = 0; t < len; t += BT) { ci[2 * k] = s; ci[2 * k + 1] = (int)(t / BT); ++k; }
      off += len;
    }
    cu[n_seq] = (int)L;
    HC(hipMemcpy(d_cu, cu.data(), (n_seq + 1) * sizeof(int), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ci, ci.data(), NT * 2 * sizeof(int), hipMemcpyHostToDevice));
  }

  float* d_warm; HC(hipMalloc(&d_warm, 4096 * sizeof(float)));
  dim3 grid((unsigned)(H + HV), (unsigned)NT);
  const long sX = qkv_dim, sA = HV, sQ = H * K, sV = HV * V;

  auto launch = [&]() {
    gdn_prefill_prep_rdna2_kernel<true, float, float><<<grid, PREP_THREADS, 0, 0>>>(
        d_x, d_a, d_b, d_A, d_dt, d_q, d_k, d_v, d_g, d_be,
        sX, sA, sA, sQ, sQ, sV, L, H, HV, d_cu, d_ci, NT);
  };
  auto warm = [&]() { clock_warm_kernel<<<72, 256, 0, 0>>>(d_warm, 4096); };

  // 1 s of clock warm, then min of `reps` batched (60-iter) timings.
  for (int i = 0; i < 40; ++i) warm();
  HC(hipDeviceSynchronize());
  for (int i = 0; i < 20; ++i) launch();
  HC(hipDeviceSynchronize());

  const int ITERS = 60;
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    for (int i = 0; i < 10; ++i) warm();
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    HC(hipEventRecord(e0, 0));
    for (int i = 0; i < ITERS; ++i) launch();
    HC(hipEventRecord(e1, 0));
    HC(hipEventSynchronize(e1));
    float ms; HC(hipEventElapsedTime(&ms, e0, e1));
    best = fmin(best, ms / ITERS * 1000.0);
  }
  printf("L=%ld n_seq=%d  hip_us=%.1f\n", L, n_seq, best);
  return 0;
}
"""


def extract() -> str:
    text = open(SRC).read()
    m = re.search(r"\nnamespace \{\n", text)
    if not m:
        sys.exit("could not find 'namespace {' in " + SRC)
    body = text[m.end():]
    body = body[: body.index("}  // namespace")]
    return body


def main():
    src = "/tmp/probe_prep_rdna2.cpp"
    with open(src, "w") as f:
        f.write('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n')
        f.write("namespace {\n")
        f.write(extract())
        f.write(HARNESS)
    cmd = ["hipcc", "-O3", "--offload-arch=gfx1030", "-o", BIN, src]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        sys.exit("compile failed")
    args = [BIN] + sys.argv[1:]
    print("$", " ".join(args))
    subprocess.run(args)


if __name__ == "__main__":
    main()
