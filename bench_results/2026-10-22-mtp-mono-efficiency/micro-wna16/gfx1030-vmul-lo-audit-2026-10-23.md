# gfx1030 `v_mul_lo_u32` audit

## Finding

The Strata-style 24-bit multiply optimization is already present in our RDNA2
W4A8 DOT4 source. `csrc/rocm/w4a8_sdot4_rdna2.cuh::as_i24` sign-extends the
24-bit activation group sum before the zero-fold multiply; both the dense
W4A8 GEMM and fused W4A8 MoE GEMM use that helper. The comment documents the
bound (`|asum| < 2^23`, group size at most 65535). Current local and remote
source hashes for the shared header match:

```text
fde5a319d658d6eab4d82e5cb7436da575fa1de819e6986722a085138291cdb2
```

The archived gfx1030/ROCm 7.14 compiler assembly for those W4A8 translation
units shows the expected `v_mul_i32_i24_sdwa` zero-fold instructions. Its
remaining three `v_mul_lo_u32` instructions in each translation unit are at
the activation-quantizer's index/address calculation (header lines 453/482),
not in the DOT4 GEMM inner loop:

| Assembly bundle | `v_dot4c_i32_i8` | `v_mul_i32_i24` | `v_mul_lo_u32` | `v_perm_b32` |
| --- | ---: | ---: | ---: | ---: |
| Fused MoE W4A8 | 3,360 | 180 | 3 | 50 |
| Dense W4A8 | 26,880 | 1,440 | 3 | 104 |

The audited build output is preserved remotely under
`/home/chenco_adm/w4a8_runs/2026-09-29_w4a8-moe-build/` (`moe_asm.s`,
`dense_asm.s`). Its generated source path is the v0.28 build tree; the shared
header hash matches the current source, but this is not a fresh full rebuild
of the current tree.

## Linked extension check and limitation

The installed extension inspected on par1-cs25 was
`/home/chenco_adm/vllm-rdna-0.31.0-mono/vllm/_rocm_C.abi3.so`, SHA-256
`fd43c214da6c2d4d788fd5fda23ecb65ba8f76c1c61a168d93e8b57753fb77ce`, dated
October 11. Its extracted gfx1030 code object had 24 total `v_mul_lo_u32`
instructions; all were in `__assert_fail` or
`__ockl_fprintf_append_string_n`, with none in a serving kernel. It contained
no W4A8 DOT4 kernels, so this older linked extension cannot validate the
current W4A8 object. We cleaned the temporary extraction after inspection.

## Decision

No source change is indicated by this audit: the low-rate multiply in the
W4A8 DOT4 accumulator setup is already replaced by a 24-bit multiply, and
remaining source-build occurrences are quantizer address arithmetic. A fresh
full build/disassembly would be needed to make a claim about every object in
the current extension. The audit does not imply a serving-performance gain.
