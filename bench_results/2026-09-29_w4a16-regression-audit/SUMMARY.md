# W4A16 regression audit — W4A8 era (2026-09-29)

Branch `w4a8-wiring`. Box `par1-cs25`, tree `vllm-rdna-0.28.0`, venv
`venv-7.14.0_0.28.0`. Scope: W4A16 must be unchanged by the W4A8 work except
where the W4A8-era fix repaired the split-K NaN. W4A8 stays opt-in.

## 1. Split selection (`compute_split_k`, dense W4A16 prefill)

`a641d6565` replaced the powers-of-two split search with a full enumeration of
K_STEP-aligned splits. It repaired the NaN but also re-tuned 13 shapes whose
legacy split was already valid.

Minimal fix (this audit): keep the legacy powers-of-two split whenever it is
usable (`size_k % split == 0 && (size_k/split) % K_STEP == 0`); fall back to the
enumeration only when it is not. Ground truth from the fixed kernel debug log
(`VLLM_RDNA2_PREFILL_DEBUG=1`) matches the model for all 60 production shapes:

- **8 repairs** (old invalid → NaN/garbage): `(225,4352,{3584,4096,5120,8704})`,
  `(2001,4352,8704)`, `(2001,8704,8704)`, `(2048,4352,8704)`, `(2048,8704,8704)`.
- **13 retunes reverted** (old valid, new chose non-pow2): see `split_table.txt`.
- **0 shape where the fix deviates from the legacy split on a legacy-valid shape.**

Same-build old-vs-fixed op timing (`old_build` vs `fixed_build2`): every
legacy-valid shape within ±1.1%; the 8 repairs 3.9–9.3% faster (old produced
NaN/garbage so it is not a valid timing baseline).

Same-session forced-split A/B on the fixed `.so` (`VLLM_RDNA2_PREFILL_FORCE_SPLIT_K`)
isolates the split choice: the W4A8 enumeration was **slower** on
`(2001/2048,1536,3584)` by ~26%, `(2001,5120,4096)` by ~16%,
`(2001,6144,4096)` by ~16%; it was ~2–4% faster on several others. The policy is
"repair, don't re-tune", so all are reverted to the legacy choice.

Artifacts: `split_table.txt`, `{new,old_build,fixed_build2}_prefill_timing.csv`,
`fixed_build2_splits.log` (+ `old_build_splits.log`), `forced_ab.csv`,
`forced_batch{1,2}.csv`.

## 2. Python selector (`rdna2_w4a16.py`)

Differential test of the extracted `_rdna2_w4a16_select_kernel` before vs after
over 4400 `(m,k,n,is_awq)` cases with `w4a8=False`: **0 diffs**. The
`w4a8_prefill` arm never fires with the env unset; `self._w4a8` resolves to
`False` unless `VLLM_RDNA2_W4A8_SDOT4=="1"`. The four W4A16 call sites
(`awq_prefill`/`prefill`/`exllama`/`rdna2_decode`) are textually unchanged.
`VLLM_DISABLED_KERNELS` still bypasses the whole `RDNA2W4A16LinearKernel` class
(registry check in `kernels/linear/__init__.py`, unchanged in the W4A8 range).

## 3. Ops surface

`vllm/_custom_ops.py`, `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp`,
`CMakeLists.txt`: W4A8 additions are purely additive. `gptq_gemm_rdna2_prefill`
and `moe_gptq_gemm_rdna2` signatures are untouched; no `csrc/rocm/q_gemm_rdna2.cu`
or MoE W4A16 file changed in `4d94aafd3^..HEAD`.

## 4. Launcher defaults

Flipped `W4A8` default `1 → 0` in `scripts/serve_gfx1030_27b_dense.sh`,
`tools/rdna2_028/m27b_matrix.sh`, `tools/rdna2_028/w4a8_ab.sh`. Flash-Next
launchers (`serve_gfx1030_flashnext.sh`, `serve_gfx1030_flashnext_mtp.sh`) never
set `VLLM_RDNA2_W4A8_SDOT4`. `arm_ladder.sh`/`fp_aroff_27b.sh` were already 0;
`flashnext_w4a8_arm.sh` requires it explicitly.

## 5. Proof runs

Tests (gfx1030, fixed `.so`):

| suite | result |
|---|---|
| `test_rdna2_w4a8.py` | 18 passed |
| `test_rdna2_w4a16.py` | 18 passed |
| `test_rdna2_w4a16_selection.py` | 14 passed |
| `test_rdna2_moe_w4a16.py` | 36 failed / 24 passed / 49 xfailed — **identical on the pre-fix `.so`** → pre-existing, unrelated to this change |

In-model Flash-Next, **W4A8=0 (default path)**, TP=4, FULL_AND_PIECEWISE, FA-RDNA2,
RDNA_AR one-shot 64 KiB (`2026-09-29_w4a16-audit-default2`):

- coherence 4 OK / 0 BAD (France→Paris, 2+2→4)
- 16k/1k c=1: 1/1 ok, **31.33 tok/s** output, TTFT 9.12 s (prefill 1795.7 tok/s),
  TPOT 23.03 ms, total 532.6 tok/s
- `W4A8 sdot4 path active` count in the serve log: **0**
- no new PCI SERR (15 before and after)

A first identical run (`2026-09-29_w4a16-audit-default`) crashed with a
host-SIGSEGV worker death (`exit -11`) on the first inference step after a fresh
inductor cache; a clean re-run with the same `.so` and config passed. No GPU
page fault (dmesg), no SERR, no reboot — transient, not reproduced.
