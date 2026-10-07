#!/usr/bin/env python3
"""Cross-build bitwise probe: D=256 even-group gqa prefill, Flash-Next shape.

The op name is identical in the baseline and PR34 builds, so swapping
vllm/_rocm_C.abi3.so and re-running this in a fresh process compares the two
kernels on byte-identical inputs. Output written to sys.argv[1].
"""
import sys

import torch

sys.path.insert(0, "/home/chenco_adm/vllm-rdna-0.28.0/tests/kernels/attention")
from test_fa_rdna2_writer_layout import _fill_cache  # noqa: E402

from vllm.v1.attention.ops import fa_rdna2_backend as fa  # noqa: E402

D, H_q, H_kv, bs = 256, 24, 2, 16
seq_lens_l, q_lens = [37, 1000, 2000], [37, 1000, 500]
kc, vc, bt, per_seq_kv = _fill_cache(seq_lens_l, H_kv, D, bs, seed=11,
                                     layout="dense")
cu_l = [0]
for n in q_lens:
    cu_l.append(cu_l[-1] + n)
torch.manual_seed(11)
Q = torch.randn(cu_l[-1], H_q, D, dtype=torch.float16, device="cuda")
cu = torch.tensor(cu_l, dtype=torch.int32, device="cuda")
seqlens = torch.tensor(seq_lens_l, dtype=torch.int32, device="cuda")
out = fa.fa_rdna2_prefill_paged_varlen_gqa(Q, kc, vc, bt, cu, seqlens, bs, 1, 0)
torch.save(out.cpu(), sys.argv[1])
print(f"shape={tuple(out.shape)} dtype={out.dtype} "
      f"sum={float(out.float().sum()):.6f}")
