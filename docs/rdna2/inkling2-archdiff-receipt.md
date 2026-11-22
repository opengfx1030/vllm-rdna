# Inkling-2 csrc archdiff receipt

Code-generation diff of the csrc change (no kernel was run, no GPU was touched).

- HEAD: `91c2ffd64a` (csrc identical to `feat/inkling2-merge@0722fa5075`; `git diff` of csrc and CMakeLists.txt against it is empty)
- BASE: `703349599a6d3ae649aa199f6c60fce5bd12d4a5` = `git merge-base 91c2ffd64a origin/rdna_extras`
  (origin/rdna_extras fetched at `9031c79a04`; the merge-base is the same as in the earlier receipt)
- Method: `tools/rdna/archdiff.sh`, BASE and HEAD as `git archive` exports of those commits (tree-dir mode), CPU only
- Arches compiled and diffed: gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1200
- `EXPECT='_kernel|inkling2|_rdna2'`, `MAY_DIFFER='gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1200'`
- JOBS=8, `nice -n 19`, `~/w4a8_runs/audit/build.lock` taken by the script
- Allowed `__LINE__` shifts (derived from the source diff): +159 +164 +279 +296 +620 +1216 +1355 +1787 +2052 +2118
- Exit status: 0, "PASS (exit 0): 247 differing function(s), all expected by EXPECT / MAY_DIFFER"

Changed sources: CMakeLists.txt, ops.h (+296), torch_bindings.cpp (+279),
inkling2_{attn,dspark,glue,linear,vit}_rdna2.cu, moe_lowp_dot2_rdna2.cu.
Built per arch: 11 objects, 8 of them device-code objects. The 6 new .cu objects exist only on the HEAD side.
The other 5 are the existing ROCm TUs that include ops.h; 3 more TUs that include ops.h are host-only and not compared.

| arch | functions base / head | identical | after masking | line-const-only | CHANGED | only in base | only in head | of which empty stubs | verdict |
|---|---|---|---|---|---|---|---|---|---|
| gfx90a | 121 / 308 | 121 | 0 | 0 | 0 | 0 | 187 | 187 | no difference |
| gfx942 | 121 / 308 | 121 | 0 | 0 | 0 | 0 | 187 | 187 | no difference |
| gfx950 | 121 / 308 | 121 | 0 | 0 | 0 | 0 | 187 | 187 | no difference |
| gfx1030 | 121 / 368 | 121 | 0 | 0 | 0 | 0 | 247 | 0 | expected only |
| gfx1100 | 121 / 308 | 121 | 0 | 0 | 0 | 0 | 187 | 187 | no difference |
| gfx1200 | 121 / 308 | 121 | 0 | 0 | 0 | 0 | 187 | 187 | no difference |

Pre-existing functions: 121 of 121 identical on every arch; CHANGED = 0 and only-in-base = 0 on every arch.
Unexpected differences: 0. Contract platforms not run: none.

gfx1030, 247 new functions in 44 families, all only-in-head with real code:

- i2_* (26): i2_combine 4, i2_decode 4, i2_kv_prep 4, i2_sconv 8, i2_state_write 2, i2_tile 4
- inkling2_dspark (34): 15 families, among them gemv 10, head_logits 6, embed_block 4
- inkling2_glue (60): attn_res 24, gate_mix 18, moe_route 8, rms_norm 4, gate_down 3, copy_states 1, embed_lookup 1, swiglu 1
- inkling2_linear (103): gemv 87, gemm 12, gemm_f32w 2, argmax_partial 1, argmax_reduce 1
- lowp_dot2 (8): moe_gemm_mxfp8_kernel_rdna2 (BM 1/2/4/8), moe_gemm_nvfp4_kernel_rdna2 (BM 1/2/4/8)
- vllm::inkling2_vit (16): gemm 6, gather_patches 4, attention, embed, layernorm, pool, rope_qk, rope_tables 1 each

On the other five arches the same new objects compile to 187 empty stubs (`s_endpgm`/`s_nop` only),
listed as one-sided but not counted as differences.

Not covered: execution on hardware; host code; kernel descriptors and HSA metadata (register and LDS counts);
flag or target-list effects of the CMakeLists.txt change (only TUs whose sources or headers changed are built).

Notes:

- The merge-base `703349599a` has no `moe_lowp_dot2_rdna2.cu`. The MXFP8 M=1/2/4/8 gfx1030 kernels therefore
  show as new (only-in-head), not as changed-versus-earlier-draft. The earlier draft's receipt is
  `a16d844cea` (its own archdiff).
- `EXPECT` is broad on purpose (some new kernels live in anonymous or other namespaces). It cannot mask a
  regression: the counts in the table do not depend on EXPECT (121 base functions, 121 identical).

Artifacts (box, `~/w4a8_runs/inkling2-work/Inkling2Archdiff/work-spec/results/`):
`archdiff.md`, `archdiff.json` (sha256 `6cf8381842d93f75eebb08d4956e79dcc6e95dcc5a69c7c57326144e9a2f3200`),
`plan.json`, `source.diff`, `info.txt`; log `archdiff-spec.log`.
