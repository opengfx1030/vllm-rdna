# Plan: Real Packed Trellis (Option D)

**Status**: planned, not started. Blocked on completion of A and C.
**Author**: Sisyphus / chenco
**Last updated**: 2026-09-05

## Context

EXL3 3bpw on gfx1030 stores the trellis as `int16` per element. A 16x16 tile
contains `NW = 8 * bits = 24` uint32 words = 48 int16 elements = 96 bytes per
tile. The v2 decode kernel reads these as uint32 (2 int16 per uint32) and
uses `fshift(tw[i1], tw[i0], s1)` for window extraction across all 128
distinct (i0, i1, s1) triples fired by the kernel.

**Audit result** (`/tmp/audit_bits.py`, 2026-09-05): all 32 bits of every
uint32 word are used by some shift in the kernel. The 96-byte storage is at
the information-theoretic minimum for this kernel's access pattern. A naive
"drop the high int16" packing breaks because shifts with s1 ≥ 16 read
`tw[i1]`'s high half — verified by op-level parity test
(`maxdiff` up to 22.7 vs the full kernel).

To get a real 2× storage reduction we need to **rewrite the kernel** to read
16-bit windows directly (without uint32 pairs and fshift), then store 24
windows × 16 bits = 48 bytes per tile (single uint16 per window).

## Goal

Reduce EXL3 trellis VRAM by ~1.9 GiB on Ornith-1.5-9b-exl3-3bpw (2.527 GiB
trellis + 0.6 GiB CG-PATH copies → 1.26 GiB trellis + 0.3 GiB CG-PATH
copies) by changing the in-memory storage from 96 bytes/tile (int16 pair
read as uint32) to 48 bytes/tile (explicit uint16 per window), AND
maintaining bit-exact outputs AND neutral-or-positive decode perf.

## Approach

### Phase 0: Information gathering (0.5 day)

The previous attempt's failure (`maxdiff 22.7`) was a layout design error,
not a kernel design error. The audit script proves all 32 bits per uint32
are used. To get 2× packing we MUST change the kernel's read pattern.

**Concrete design questions to answer before writing code**:

1. **How does the kernel's bit-shifting math map to the new uint16 layout?**
   - The current kernel: `tw[w]` is uint32 from int16 pair; `fshift` does
     `(tw[i0] << 32 | tw[i1]) >> s1` to read a window that may straddle
     a uint32 boundary.
   - The new kernel: `tw[w]` is uint16; need explicit "load window X"
     primitive that reads uint16[X] without bit-shifting across boundaries.
   - The 24 windows per tile live at bit positions
     `(p+1)*bits - 16 .. (p+1)*bits - 1` (mod 768 bits) in the packed
     stream, where p ∈ [0, 256). Two consecutive windows p, p+1 OVERLAP by
     10 bits (16-bit windows, 6 bits apart). So the bitstream is NOT a
     simple concatenation of windows — they're interleaved in a 6-bit
     tail-biting pattern.

2. **What does the kernel actually compute?**
   The kernel computes `decode_3inst<cb>(window) * activation[k]` for 4
   windows per grain (32 grains per tile). It doesn't actually NEED the
   bitstream order — what it needs is: for each grain, the 4 window
   values at the bit positions dictated by the formula. As long as the
   kernel can produce those 4 windows from the storage layout, the
   encoding doesn't matter.

3. **What storage layout enables the kernel?**
   The kernel currently reads uint32 because the bit-streaming math is
   simpler in 32-bit units. With an explicit uint16-per-window layout,
   each window is at a known offset in the buffer; the kernel reads
   uint16 directly without bit-shifting. The trade-off is more
   indexing math in the kernel (vs. simple uint32 indexing), but the
   indexing is cheap (1-2 ops).

4. **How do we preserve the bit-streaming semantics?**
   The encoder writes bits in the tail-biting 6-bit stride pattern. If
   the decoder reads windows explicitly (each at its own offset), we
   need to compute that offset from the formula. The encoding IS the
   offset computation; if we pre-compute the offsets per window, the
   kernel becomes a direct uint16 lookup table indexed by tpos.

### Phase 1: Layout design (1 day)

**Goal**: define a uint16-packed storage format that the v2 kernel can
read bit-exactly (matching the current kernel's outputs).

Approach:
- Keep the on-disk int16 layout unchanged (matches safetensors). The
  packing is purely an in-memory reshape.
- Loader reads the on-disk int16, builds a uint16 buffer of half the
  size where each element is a 16-bit window VALUE (not the
  bit-streaming representation). The transformation is: for each
  (k_tile, n_tile, window_index) in [0, K/16) × [0, N/16) × [0, 24),
  extract the 16-bit window from the bitstream and store it.
- Cost: ~10-20 lines of Python at load time (CPU loop over 768 bits
  per tile × 200 tiles × layers). Acceptable: load time is 6 sec,
  this adds <1 sec.

Wait — this is wrong. The kernel doesn't just READ 16-bit windows; it
needs the windows at SPECIFIC bit positions determined by the
formula. The formula picks bit positions p, p+1, p+2, ... based on
grain index. These don't correspond to "the 24 windows of this tile"
in a 1:1 mapping. Instead, the formula reads overlapping windows in
the tail-biting stream.

Re-approach: the packing doesn't simplify unless we ALSO change the
kernel math. The current kernel math reads bits 0-15 of uint32[X] and
bits 0-15 of uint32[X+1] (effectively) to recover one window. To
read windows directly, we'd need a pre-decoded representation that
stores each unique window value ONCE.

Alternative: keep the uint32-style reading but accept the storage
shape. Can we pack uint32 differently?

Actually — let me reconsider. The audit says all 32 bits of every
uint32 are used. But the 32-bit "use" comes from a SINGLE read of a
uint32 in the kernel code, where the bits may come from DIFFERENT
windows (one window in bits 0-15, another in bits 16-31). If we
REDESIGN the storage to keep each 16-bit window contiguous (not split
across uint32 boundaries), the kernel reads uint16[X] and gets a
full window.

The math change: instead of "tw[i0] | tw[i1]" with bit-shifting, we
read uint16[window_index] and that's the window. The 24 windows are
stored contiguously as uint16[24].

But which uint16 index corresponds to "window p" for the given
grain? It depends on the formula. The formula picks window positions
`p = (c%8)<<5 | (off(r) mod 32)` (for bits=3). For each tile, there
are 256 distinct p values. The 24-window storage would need to
contain all 256 windows, not 24. So 24-window storage isn't enough.

Hmm. Let me re-think.

Actually the bit-stream holds 768 bits per tile = 48 16-bit windows
(768/16). The kernel reads `tw[w]` for w in [0, 24) — that's 24
uint32 reads = 24 × 32 = 768 bits. So the kernel reads ALL 768 bits
of the bit-stream. Not 24 windows — 768 raw bits that it then
interprets as windows via bit-shifting.

So the storage holds 768 bits per tile (one 6-bit-stride bit-stream).
The kernel reads those 768 bits and does window extraction. There's no
shortcut: you need all 768 bits.

The 96-byte/tile storage is OPTIMAL for this kernel. To save bytes,
we must compress the 768-bit stream. The encoder wrote each window's
16 bits at bit position `(p+1)*bits - 1` (with bits=3, that's `3p+2`
down to `3p-13`). So each window's 16 bits span a 16-bit range in
the bitstream. Two consecutive windows overlap by 10 bits (because
they're 6 bits apart).

Wait — let me re-check. `b0 = tpos * 2 * bits + bits - 16 + 256 * bits`.
For bits=3: `b0 = tpos * 6 + 3 - 16 + 768 = tpos*6 + 755`. Two
consecutive tpos: b0(tpos) = 6*tpos + 755, b0(tpos+1) = 6*tpos + 761.
Difference = 6. So consecutive windows START 6 bits apart, and each
window is 16 bits long → they OVERLAP by 10 bits.

So the bitstream is structured: window[0] at bits [755, 770], window[1]
at bits [761, 776], window[2] at bits [767, 782], etc. With 128 windows
(tpos in [0, 128)) and 6-bit stride, the windows cover the full 768-bit
stream with overlaps.

**No further compression possible without breaking the encoding.**

The encoder writes 16-bit windows with 6-bit stride and 10-bit overlap.
The storage is 768 bits = 96 bytes per tile. This is the minimum.

**Conclusion**: Option D as originally conceived (2× packed trellis
storage) is **NOT ACHIEVABLE** without changing the encoding scheme
itself, which would mean re-encoding all model checkpoints.

This is a significant conclusion and changes the framing of Option D.

### Revised Option D scope: re-encode-aware kernel

If we want a real 2× storage win, the encoding must change. Two
possibilities:

**D-a: Custom encoding for new models.** Re-encode EXL3 checkpoints
with a 16-bit-per-window layout (no overlap), giving 24 windows ×
16 bits = 48 bytes per tile. Kernel reads uint16 directly, no
fshift. Migration cost: re-quantize every existing model OR ship
a converter. Decoder-only changes (no encoder changes). Storage
saving: 50% of trellis (~1.26 GiB primary + ~0.6 GiB CG-PATH).

**D-b: Streaming decode, no per-tile storage.** Stream the bit-stream
from host memory through a small on-device cache. Trellis never
fully resident on GPU. Storage: tiny per-tile cache (~few MB).
Migration cost: PCI-e latency on every decode (kills perf).

**D-c: Different quantization scheme entirely.** Marlin / QuaRot /
SmoothQuant — outside EXL3 scope, would replace the quantization
format. Out of scope for this plan.

### Decision

Given the audit conclusion, **Option D is RECLASSIFIED as "not
feasible without re-quantization"**. The 96-byte/tile storage is
optimal for the current encoding scheme.

**Recommendation**: focus engineering effort on Option A (done,
−2.45 GiB) and Option C (env var, free). If more memory savings
are needed, investigate re-quantization (D-a) — but that requires
rebuilding model artifacts, which is out of scope for inference-
side optimization.

### Phase 2: Validation gate (skipped — see Decision above)

If D-a becomes desired, validation gates would be:
1. Encode-side: write a script that takes an EXL3 checkpoint and
   produces a `*-packed.exl3` variant with the new encoding.
2. op-level: new `exl3_gemm_rdna2_v3` kernel reads uint16 layout;
   `maxdiff ≤ 1e-3` vs the full kernel on real weights across
   multiple layer shapes (q_proj, gate_proj, fused in_proj_qkv,
   down_proj).
3. Server: end-to-end golden probes + ITL parity + bench matrix.
4. Migration: convert one model end-to-end (Ornith + Qwen3.8),
   bench, compare.

## Risks

- Re-quantization (D-a) requires offline encoding work; the EXL3
  encoder (exllamav3) is CUDA-only and would need an AMD port or
  a CPU fallback. Months of work, not a quick win.
- Streaming decode (D-b) kills perf via PCIe latency.
- D-c (different quantization) is a research project.

## Conclusion

**Option D is not viable short-term** for the current EXL3 3bpw
encoding. The 96-byte/tile storage is at the information-theoretic
minimum for the tail-biting bit-stream layout. To go further
requires re-quantization, which is months of engineering work.

The realistic memory budget for vLLM on gfx1030 with EXL3 3bpw:
- Pre-Option A: ~27.3 GiB (KV 15.91 + weights 11.0 + graph 0.66 + activation 0.37, peak 9.8 GiB during load)
- Post-Option A: ~25 GiB steady (KV 18.36 + weights 8.55 + ...)
  - 591,688 tokens cached simultaneously (vs 512,727 before)
  - 2.96× the 200k max-model-len (vs 2.56× before)
- Post-Option C (env var): peak drops further; steady-state unchanged

If more memory is needed: re-quantization to a packed format (D-a)
is the only path, but it's a separate project.

## References

- Audit: `/tmp/audit_bits.py` — enumerated every (i0, i1, s1) triple
- Failed parity test: `/tmp/exl3_packed_test.py` — `maxdiff 22.7`
- Option A journal entry: see journal.md 2026-09-05
- Original commit (stub): `c3e525859`
