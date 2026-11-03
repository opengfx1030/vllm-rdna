# Progress: Qwen4 monokernel design (agent Qwen4MonoDesign)

State at 2026-11-03, end of the design turn.

## Done

- Read the shipped monokernel end to end: `mono_decode_rdna2.py`,
  `mono/rdna2/{k1,k2,gdn,qsa,layout,__init__}.py`,
  `csrc/rocm/mono_{k2,gdn,qsa_decode}_rdna2.cu`, `rdna_allreduce.cuh`, the
  `--monokernel` tests, `docs/rdna/monokernel/*`, the MoE offload docs and the
  stall profile.
- `docs/rdna2/qwen4/monokernel-design.md`: architecture, gap table, design,
  Wave32/LDS/VGPR budgets, TP4/TP2 roofline tables, test plan, open questions.
- `feat/qwen4-prep` (worktree `/Users/kletorch/Projects/infrastructure/gfx1030_optimized/vllm-qwen4-prep`):
    - `vllm/models/qwen4_prep/__init__.py`: architecture gate (Qwen4, not
    Qwen4Exp). Not imported by serving.
    - `vllm/models/qwen4_prep/mono_plan.py`: config -> stage graph, row caps,
    grid, traffic roofline; `flash_next_plan()` reproduces today's plan.
    - `tools/rdna/qwen4/test_mono_plan.py`: 14 CPU tests, green:
    `python3 -m pytest tools/rdna/qwen4/test_mono_plan.py -q`.
    (Placed under tools/, not tests/: an untethered `tests/` file is rejected
    by `check_test_tethering.py` and there is no CI job for pre-release Qwen4
    scaffolding.)
    - ruff 0.17.3 check + format clean on the three files.

## Findings a resume should not re-derive

- K1/K2 share `mono_k2_rdna2.cu`; K1 is `hc_only` (P1–P3). K2 is one launch,
  144 WGs x 256 threads, barrier on a generation counter, 2 s bound, NaN
  poison, fixed-point cross-WG sums.
- QSA HIP sparse decode gate: head_dim 256, group <= 8, FP16 KV, 3..64 rows.
  TP2 Flash-Next is group 12 (12 Q / 1 KV per rank), so it falls back there;
  TP4 is group 3.
- TP2 Flash-Next needs the expert cache: 256 local experts x 2.55 MB x 48
  layers ~= 31.3 GB resident.
- MTP4 (leak: 4 draft tokens) gives widths 5/10/20/40. A 40-row MoE K2 with
  today's buffers (`s_lora` 25.6 KiB + `s_urow` 10.2 KiB) goes past 32 KiB
  static LDS, halving the grid to 72 WGs; the dense K2 (new) stays under
  32 KiB.
- The model class builds `Qwen4ExpSparseMoeBlock` for every layer
  (`amd/model.py`, ~line 265), so a dense Qwen4 needs a model-side MLP too,
  not only a dense K2.
- vLLM upstream has no Qwen4 model class; all "Qwen4" there is `qwen4_exp`.

## Next (after release)

W1 plan plumbing, W2 K2 parameterisation, W3 dense K2 + dense MLP, W4 40-row
support, W5 QSA templates / no-indexer FA, W6 GDN `kMaxT`, W7 expert cache
> 256, W8 recipe. See section 3.7 of the design.
