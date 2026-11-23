# Inkling-2 csrc archdiff receipt

Code-generation diff of the csrc change of `feat/inkling2-merge`.
No kernel was run and no GPU was touched.

## Revisions and method

- HEAD: `0fd3e424dc` (`feat/inkling2-merge`), exported with `git archive`.
- BASE: `9031c79a047b54d19e96b2105c74e7c45fa3be1e` = `origin/rdna_extras` =
  `git merge-base 0fd3e424dc origin/rdna_extras` (fetched before the run).
- Method: `tools/rdna/archdiff.sh` with BASE and HEAD as source tree directories (the `git archive` exports),
  CPU only (`HIP_VISIBLE_DEVICES=` empty), `nice -n 19`, `JOBS=8`,
  `~/w4a8_runs/audit/build.lock` taken by the script, `KEEP_BUILD=1`.
- Arches compiled and diffed: gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1200.
- Settings: `EXPECT='_kernel|inkling2|_rdna2'`,
  `MAY_DIFFER='gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1200'`.
- Allowed `__LINE__` shifts (derived from the source diff):
  +212 +216 +347 +371 +764 +1072 +1112 +1216 +1355 +2052 +2122 +2326.

```bash
D=$HOME/w4a8_runs/inkling2-work/Inkling2Archdiff
cd $D/src-head4
BASE=$D/src-base2 HEAD=$D/src-head4 WORK=$D/work-final JOBS=8 KEEP_BUILD=1 \
EXPECT='_kernel|inkling2|_rdna2' \
MAY_DIFFER='gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1200' \
HIP_VISIBLE_DEVICES= nice -n 19 bash tools/rdna/archdiff.sh
```

## Result

Exit status 0: "PASS (exit 0): 300 differing function(s), all expected by EXPECT / MAY_DIFFER".
Unexpected differences: 0. Contract platforms not run: none.

The gate is the table, not `EXPECT`: its counts do not depend on `EXPECT`.
Every one of the 121 functions of BASE is identical on all six arches (CHANGED 0, only-in-base 0, identical after
masking 0, line-constant-only 0).
Every difference is a function that exists only in HEAD.

| arch | functions base / head | identical | CHANGED | only in base | only in head | empty stubs | new, with code | verdict |
|---|---|---|---|---|---|---|---|---|
| gfx90a | 121 / 347 | 121 | 0 | 0 | 226 | 224 | 2 | expected only |
| gfx942 | 121 / 347 | 121 | 0 | 0 | 226 | 224 | 2 | expected only |
| gfx950 | 121 / 347 | 121 | 0 | 0 | 226 | 224 | 2 | expected only |
| gfx1030 | 121 / 411 | 121 | 0 | 0 | 290 | 0 | 290 | expected only |
| gfx1100 | 121 / 347 | 121 | 0 | 0 | 226 | 224 | 2 | expected only |
| gfx1200 | 121 / 347 | 121 | 0 | 0 | 226 | 224 | 2 | expected only |

The 300 differing functions are the 290 of gfx1030 plus the 2 new arch-neutral prefill helpers on each of the other
five arches (see below).

## Changed sources and objects

Changed under csrc and CMakeLists.txt, BASE to HEAD (12749 insertions, 1 deletion, a comment line in CMakeLists.txt):

| file | lines added |
|---|---|
| `csrc/rocm/inkling2_attn_rdna2.cu` | 2122 |
| `csrc/rocm/inkling2_dspark_rdna2.cu` | 2052 |
| `csrc/rocm/inkling2_fuse_attn_rdna2.cu` | 1112 |
| `csrc/rocm/inkling2_glue_rdna2.cu` | 2326 |
| `csrc/rocm/inkling2_linear_rdna2.cu` | 1355 |
| `csrc/rocm/inkling2_vit_rdna2.cu` | 1216 |
| `csrc/rocm/moe_lowp_dot2_rdna2.cu` | 764 |
| `csrc/rocm/moe_prefill_bm_rdna2.cu` | 1072 |
| `csrc/rocm/ops.h` | 371 |
| `csrc/rocm/torch_bindings.cpp` | 347 |
| `CMakeLists.txt` | 12 (+12 / -1) |

Per arch 13 device-code objects were compiled and compared:

- 8 objects exist only on the HEAD side: the 8 new `.cu` files above.
- 5 are existing TUs that include `ops.h`: `gemm_w8a8_fp8_dense_rdna2`, `moe_w4a8_rdna2`, `moe_w8a16_rdna2`,
  `q_gemm_w8a16_fp8_rdna2`, `qsa_rdna2`. Their functions are the 121 identical ones.
- 3 more TUs that include `ops.h` are host-only and not compared: `attention_dispatch`, `torch_bindings.cpp`,
  `skinny_gemms_int4_api`.

## gfx1030: the 290 new functions

All 290 are only-in-head with real code, in 44 families, under these Inkling names:

| namespace or prefix | functions | families (functions each) |
|---|---|---|
| `i2_*` (attn) | 30 | combine 4, decode_fused 4, decode 4, kv_prep 4, sconv 8, state_write 2, tile 4 |
| `inkling2_dspark::` | 34 | 15 families, among them gemv 10, head_logits 6, embed_block 4 |
| `inkling2_fuse::` | 28 | attn_res_fused_kernel 28 |
| `inkling2_glue::` | 66 | attn_res 24, gate_mix 18, moe_route 8, gate_mix_split 6, rms_norm 4, gate_down 3, copy_states 1, embed_lookup 1, swiglu 1 |
| `inkling2_linear::` | 103 | gemv 87, gemm 12, gemm_f32w 2, argmax_partial 1, argmax_reduce 1 |
| `lowp_dot2::` | 8 | moe_gemm_mxfp8_kernel_rdna2 (BM 1/2/4/8) 4, moe_gemm_nvfp4_kernel_rdna2 (BM 1/2/4/8) 4 |
| `moe_prefill_bm::` | 5 | moe_prefill_act_quant 1, moe_prefill_cta 1, nvfp4_prefill 1, mxfp8_prefill 1, nvfp4_i8_prefill 1 |
| `vllm::inkling2_vit::` | 16 | gemm 6, gather_patches 4, attention, embed, layernorm, pool, rope_qk, rope_tables 1 each |

Why gfx1030 changes, and what is new since the receipt of `91c2ffd64a`:

- Nothing of BASE changes on gfx1030. Everything above is new code of the Inkling-2 port; the device code of the 121
  BASE functions is bit-identical.
- New families: `inkling2_fuse::attn_res_fused_kernel` (the fused chain of a loop pass,
  `inkling2_fuse_attn_rdna2.cu`), `i2_decode_fused_kernel` (the fused sliding decode in `inkling2_attn_rdna2.cu`),
  `inkling2_glue::gate_mix_split_kernel` (decode glue) and the five `moe_prefill_bm::` kernels (the prefill expert
  kernels, `moe_prefill_bm_rdna2.cu`).
- Changed code that has no BASE counterpart (archdiff lists it as new and cannot show the change itself):
  `i2_combine_kernel` went from 3075 to 3123 instructions against the receipt of `91c2ffd64a`;
  the `lowp_dot2::` kernels now total 7503 (NVFP4) and 7739 (MXFP8) instructions, against 7141 and 15385 in the
  first receipt (`a16d844cea`, before the MXFP8 rewrite).

## Other arches: empty stubs and the two prefill helpers

- On gfx90a, gfx942, gfx950, gfx1100 and gfx1200 the arch-specific new kernels compile to 224 empty stubs
  (`s_endpgm` or `s_nop` only: `inkling2_fuse` 28, `inkling2_glue` 66, `inkling2_linear` 103, `inkling2_vit` 16,
  `lowp_dot2` 8, `moe_prefill_bm` 3). They are listed as one-sided and are not counted as differences.
- Two new functions have code on all arches: `moe_prefill_bm::moe_prefill_act_quant_kernel` and
  `moe_prefill_bm::moe_prefill_cta_kernel`. They sit after the `#endif` of the `__HIP__RDNA2__` guard in
  `moe_prefill_bm_rdna2.cu` (the guard wraps only the three dot2 prefill kernels). They are new, under an expected
  name, and touch no BASE function.

| arch | `moe_prefill_act_quant_kernel` instructions | `moe_prefill_cta_kernel` instructions |
|---|---|---|
| gfx90a | 269 | 845 |
| gfx942 | 265 | 842 |
| gfx950 | 265 | 842 |
| gfx1030 | 255 | 584 |
| gfx1100 | 290 | 690 |
| gfx1200 | 294 | 678 |

- Whether the host entry points that launch these two helpers are reachable on a non-gfx1030 arch was not checked
  by this run (archdiff covers device code only).

## Not covered

- Execution on hardware, host code, kernel descriptors and HSA metadata (register and LDS counts).
- Flag or target-list effects of the `CMakeLists.txt` change: only TUs whose sources or headers changed are built.
  The change adds the new sources to the ROCm target and compiles the Inkling-2 vision and linear kernels with `-O3`;
  it does not touch an existing source.
- The `EXPECT` regex is broad on purpose: several new kernels live in the anonymous namespace (`i2_*`) or in other
  namespaces. It cannot hide a change of a BASE function, because those are counted by the table above.

## Artifacts

On the box, in `~/w4a8_runs/inkling2-work/Inkling2Archdiff/`:

- `work-final/results/archdiff.md`, `archdiff.json`, `plan.json`, `source.diff`, `info.txt`.
- sha256 of `archdiff.json`: `0816728b53a711ed25cb355eedab2a1ec88725decef81369609c02e032ff6607`.
- sha256 of `archdiff.md`: `2b31e1e5be8f300276c3314a488857905d2a7101cf29bda08b5404d26cbf77f2`.
- Log `archdiff-final.log`, runner `run_final.sh`, sources `src-base2/` and `src-head4/`; `work-final/build` and the
  object directories are kept for a `COMPARE_ONLY=1` re-judge.
- Phases: configure 1754 s (the script was queued on the build lock), plan-base 6 s, build-base 203 s,
  sync-head 65 s, build-head 321 s, compare 16 s.
