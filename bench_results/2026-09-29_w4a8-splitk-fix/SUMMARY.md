# W4A8 split-K K_STEP-alignment fix — op-level evidence (2026-09-29)

Commit: `a641d6565` (`fix(rdna2): K_STEP-align prefill split-K (W4A8 fallback
NaN) + W4A8 selector guard`), branch `w4a8-wiring`.

## Verdict

The W4A8 sdot4 fast-path GEMM (`w4a8_gemm_rdna2`) is **numerically correct**.
The failure was in its W4A16 fallback `gptq_gemm_rdna2_prefill`: `compute_split_k`
could return a split whose `k_per_split = size_k/split` was not a multiple of
`K_STEP = 32`. The kernels walk K in K_STEP-wide chunks and never clamp the final
chunk, so the tail chunk read past the split's LDS row / global K range.

Root cause: `csrc/rocm/q_gemm_rdna2_prefill.cu` `compute_split_k` (doubling from
split=1 only considered powers of two and never checked K_STEP alignment).

Fix:
- `q_gemm_rdna2_prefill.cu`: enumerate every usable split (divisor of `size_k`
  with `(size_k/split) % K_STEP == 0`), not just powers of two.
- `w4a8_sdot4_rdna2.cu` `w4a8_lds_fits` + `pick_split_k` and the Python mirror
  `_w4a8_lds_fits`: same K_STEP-alignment guard.

Rebuild: `tools/rdna2_028/build_rocm_c_incr.sh`, `ninja_rc=0`.

## In-model shape sweep (probe_w4a8_op.py --mode shapes)

15 distinct fast-path shapes extracted from the serve logs (all group=32),
m ∈ {225,2001,2048}, k ∈ {1536,4352,5120}, n ∈ {3584,4096,5120,8704}.
`w4a8/ref` = W4A8 vs an fp32 dequant reference; `gptq/ref` = W4A16 prefill vs
the same reference. Both GPTQ (uint4b8) and AWQ (uint4) packed weights.

| shape | AWQ w4a8/ref | AWQ gptq/ref | GPTQ w4a8/ref | GPTQ gptq/ref |
|---|---:|---:|---:|---:|
| (225,1536,5120,32)  | 0.0054 | 0.0205 | 0.0053 | 0.0210 |
| (225,4352,5120,32)  | 0.0054 | **0.0206** (was NaN) | 0.0054 | **0.0211** (was NaN) |
| (225,5120,3584,32)  | 0.0054 | 0.0207 | 0.0054 | 0.0211 |
| (225,5120,4096,32)  | 0.0053 | 0.0205 | 0.0054 | 0.0210 |
| (225,5120,8704,32)  | 0.0054 | 0.0206 | 0.0054 | 0.0210 |
| (2001,*) and (2048,*) | 0.0053-0.0054 | 0.0206-0.0207 | 0.0053-0.0054 | 0.0210-0.0211 |

Before: `(225,4352,5120,32)` was NaN (both variants). After: 0 NaN, all rows
`w4a8/ref ≈ 0.0054` (int8 activation-quant error only). `before_shapes.txt` /
`after_shapes.txt`.

## k sweep (probe_w4a8_op.py --mode bisect-k, m=2001 n=5120 g=32, k=512..6144 step 64)

| metric | before | after |
|---|---:|---:|
| NaN rows (of 88) | 41 | **0** |
| max `w4a8/ref` | 0.0054 | 0.0054 |
| max `gptq/ref` | NaN / 0.17 | **0.0208** (k=4160: 0.1735 → 0.0207) |

`before_bisectk.txt` / `after_bisectk.txt`.

## Unit tests

`tests/kernels/quantization/test_rdna2_w4a8.py`: **18 passed** (`after_pytest.txt`).

## Real AWQ checkpoint weights

`probe_real_weights.py` on Qwen3.8-27B-AWQ-INT4 tensors — w4a8 vs fp32 ref:
- o_proj (k=6144, n=5120, m=2001): 0.00536
- down_proj TP shard (k=4352, n=5120, m=2001): 0.00535

## TP=4 validation — NOT run

A concurrent TP=4 FULL_AND_PIECEWISE `RDNA_AR=0` baseline hard-reset the chassis
(IPMI PCI SERR `e68` @ 14:50:42, reboot; already documented in `2fb0204b6`).
The coordination guard says stop on a new SERR, and the baseline precondition
("TP=4 baseline finished") is unmet, so the TP=4 W4A8=1 arm was not launched.
Single-GPU op-level work after the reboot produced no new SERR (last SEL entry
remained `e68`).

## Reproduce

```bash
# op-level sweep (single GPU)
HIP_VISIBLE_DEVICES=0 python tools/rdna2_028/probe_w4a8_op.py --mode shapes
HIP_VISIBLE_DEVICES=0 python tools/rdna2_028/probe_w4a8_op.py \
  --mode bisect-k --m 2001 --n 5120 --group 32 --k-lo 512 --k-hi 6144 --k-step 64
# rebuild
bash tools/rdna2_028/build_rocm_c_incr.sh
```
