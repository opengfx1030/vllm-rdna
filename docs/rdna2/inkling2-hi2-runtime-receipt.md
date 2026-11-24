# Inkling-2-Mini HI2 runtime: CPU receipt

Branch `feat/inkling2-hi2-runtime` (based on `origin/rdna_extras`, which carries the
PR #195 Inkling-2-Mini code), 2026-11-24.  Every GPU was leased for the whole
session, so this receipt records exactly what ran on CPU, with the real quantizer
output as the input, and the commands that still need a GPU.

**Change class.** Python model/runtime wiring plus tools; **no csrc, CMake, setup or
C++ change**.  The HI2 expert path calls the fork's existing RDNA2 ops
(`rdna_hint_flex_moe_transform_rdna2`, `rdna_hint_flex_moe_gemm_rdna2`,
`inkling2_glue_swiglu`); the arch gates and launches are the ops' own (gfx1030 only,
current stream).  The CUDA / CDNA / CPU paths are untouched because no compiled code
changed; the pure-torch reference and the loader run everywhere torch is available.

Files added by this change set: `vllm/models/inkling2_loop/amd/{hi2,loader}.py`
changes plus `hi2.py` decoding/reference, `amd/moe.py` HI2 experts
(`_forward_chunk_hi2`, `forward_reference`, `_forward_hip_hi2`), `amd/glue.py`
(`hi2=True` workspace), `amd/model.py` loader wiring, three test modules, and the
tools below.

## What ran on CPU (all on the box, `par1-cs25`, no lease)

| Step | Command (abbreviated) | Exit | Result |
| --- | --- | --- | --- |
| CPU gate | `pytest test_hi2_loader.py test_loader.py test_hi2_kernel_gpu.py test_hi2_reference.py -q` | 0 | 33 passed, 5 skipped (the 5 GPU tests) |
| Pre-fix check | the same gate on the 5 new regression tests, old product files | 1 | 5 failed, 16 deselected (the tests can fail) |
| Mutants | `python tools/inkling_hi2/mutants.py --tree <scratch copy>` | 0 | baseline 27 passed; 21/21 killed (20-battery on the pre-B4 tree + the changed/new 2 targeted on the final tree) |
| Real-dir load | `python tools/inkling_hi2/check_real_dir.py --model <bpw3.5> --stock <stock>` | 0 | 1125 text tensors loaded, 0 missing parameters, lm_head == stock |
| TP suite | `pytest tests/models/inkling2_loop/test_tensor_parallel.py -q` with the workaround below | 0 | 156 passed |
| Serve dir | `bash tools/inkling_hi2/prep_serve_dir.sh <quantizer out> <serve dir>` | 0 | 15 symlinks + MTP; source untouched |
| Validate dir | `python tools/inkling_hi2/validate_dir.py --model <serve dir>` | 0 | VALID (see census below) |
| archdiff | `archdiff_report.py plan --base-tree base-rdna_extras --head-tree tree` | 0 | `units: []`, `csrc.diff` 0 bytes |
| Diff scan | `pr_receipt_scan.py base-rdna_extras tree` | 0 | 5 added `tolist` lines, all CPU reference; 0 per-step, 0 csrc |
| Receipt dry run | `DRY_RUN=1 BASE_TREE=HEAD_TREE=tree pr_receipt.sh ... RECIPE=inkling2-hi2-tp2` | 0 | 4 boots, 8 speed cells, 2 exactness cells, quality 0 |
| Recipe assembly | `PRINT=1 serve_rdna.sh RECIPE=inkling2-hi2-tp2 MODEL=...` | 0 | command below |

The CPU gate is the six-layer HI2 test set: format declaration/name map/TP slice
(`test_hi2_loader.py`, 23), the legacy loader (`test_loader.py`, 6), the kernel tests
collected and skipped off a GPU (5), and the reference against the quantizer's own
`hi2.format.dequantize` on a tiny HI2 export of the quantizer's fixture (4, fp16
exact).  `test_hi2_reference.py` needs the quantizer on `PYTHONPATH`
(`HI2_SRC=$HOME/w4a8_runs/inkling-hi2/tree`) and reuses its fixture with
`INKLING2_TINY_OUT`; without the quantizer the 4 tests skip (do not read a skip as a
pass).

### Verifier round (fix-first -> fixed)

`InklingRuntimeVerify` loaded the real directory on CPU and found four start-up
blockers; each is fixed on the branch and covered by a CPU regression test (the
B1-B3 tests fail on the pre-fix overlay, the "Pre-fix check" row; the B4 test
exercises the exact call shape that crashed):

1. **TP=2 workspace (`B1`).** The w2 output reused the gate/up scratch and the
   workspace required `2 * intermediate == hidden`; the real TP=2 rank has hidden
   2304 and intermediate 576. The w2 GEMM now writes `moe_hidden_view()` (its own
   `[T*K, H]` fp16 buffer) and the requirement is gone; tested at TP=1 and TP=2.
2. **Block-64 routing (`B2`).** `sorted_ids64` was `numel + E*(64-1)` long, not a
   multiple of 64, but the HIP GEMM checks `numel % 64 == 0`. The buffer and the
   view are rounded up (the extra rows are pad ids the GEMM never reads); tested for
   every chunk size 1..1024 at TP=1 and TP=2.
3. **`lm_head.weight` (`B3`).** The exporter keeps the stock top-level name; the
   text model calls it `model.llm.unembed.weight`. `serving_checkpoint_name()`
   aliases it in `load_weights`; `check_real_dir.py` loads the real directory and
   compares the head with the stock unembed.
4. **w2 top-k reduction (`B4`, introduced by the B1 fix).** The serving path kept
   the old 3-D broadcast on the now 2-D `[T*K, H]` buffer and raised for every
   chunk size at both TP sizes. `_sum_topk_rows()` takes the 2-D workspace view and
   the workspace's fp32 `[T, K]` `topk_w`, does the `[T, K, H]` multiply and sums
   over K into `[T, H]`; the CPU test runs it on the exact workspace buffer and
   `topk_w` view, and a mutant (view as `T*K` rows) is killed.

Smaller items: a runtime `on_gfx10x()` guard on both HI2 kernel paths (wider than
the gfx1030-only kernels: documented residual); `validate_stack` refuses uneven
expert rows for every part of every expert, so the gate/up stack at TP=1 is covered
too; the serving path rounds the per-expert products to fp16 before the top-k sum
(fp32 accumulator, fp16 output), while the eager and reference paths multiply fp32
products (the plan's risk 4 says so); the alignment op's
zero-element `expert_map` placeholder is the only per-call allocation (as in the
generic MoE path).

### `test_tensor_parallel.py` and the venv's LLVM fault

The 18 errors are an environment fault, not this branch: the engine's model-registry
subprocess loads two LLVMs and aborts with `spirv-expand-step registered more than
once`.  With an empty `VLLM_CACHE_ROOT` the base commit fails the same way; it
passes only because a cached modelinfo entry exists for its `model.py` hash.  Run
the suite with the venv's ROCm libraries on `LD_LIBRARY_PATH` and an empty cache:

```bash
site=$($VENV/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
LD_LIBRARY_PATH=$site/_rocm_sdk_libraries/lib:$site/_rocm_sdk_core/lib/host-math/lib:\
$site/_rocm_sdk_core/lib/rocm_sysdeps/lib:$site/_rocm_sdk_core/lib:$site/torch/lib \
VLLM_CACHE_ROOT=<empty dir> python -m pytest tests/models/inkling2_loop/test_tensor_parallel.py -q
```

### Real 3.5 bpw directory (header-only census)

Input `~/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5`, serve dir built by links at
`~/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5`:

- config: `architectures ["Inkling2LoopForConditionalGeneration"]`, quant
  `rdna_hint_flex` 3.2, `moe_layout stacked`; 24 layers, hidden 2304, inter 1152, 128
  experts; both shards (`model-00001-of-00001`, `model-vision`) indexed.
- 48/48 expert groups have all ten fields; `col_scale` present in all 48 (smoothed).
- bytes: experts 10.08 GiB (codes 8.99, blk_params 0.86, offsets 0.11, constants
  0.08, col_scale 0.02, row flags 0.03), dense 3.59, vision 0.59, MTP 0.59; indexed
  total 13.67 GiB.
- TP=2 per-rank bound: experts 5.04 GiB + dense 3.59/2 (sharded) to 3.59
  (replicated) + vision 0.59 + MTP 0.59 = **8.02 to 9.81 GiB** before KV, against
  29.98 GiB per V620.
- the stock `mtp.safetensors` is linked but not indexed; `moe_layout` and the MTP
  are not loaded by the language-only recipe (`MTP=0`, `--language-model-only`).

### archdiff: empty by construction, and that is the finding

`archdiff_report.py plan` compares the csrc sources of `origin/rdna_extras` and this
branch and returns `"units": []` with a zero-byte `csrc.diff`: no translation unit
changed, so there is nothing to compile or disassemble for any arch.  The full
`BASE=origin/rdna_extras HEAD=feat/inkling2-hi2-runtime bash tools/rdna/archdiff.sh`
therefore has no TU to build (it would print nothing to diff); it needs no GPU.  Any
future change to `csrc/` on this branch must run it with all six arches.

### pr_receipt scan and dry run

The scan against `origin/rdna_extras` lists five added `tolist()` lines, all in
`amd/hi2.py` at the tensor-parallel slice and reference-decode helpers (CPU paths:
`_select_cols`, `dequantize_stored_rows`, `reference_moe`).  None is in the per-step
serving path; the HI2 kernels receive precomputed index tensors.  The
`DRY_RUN=1` receipt (run A/A on the same tree, because the base tree of a real A/B
run must itself be built) exits 0 and prints the production plan: boots XA/XB exact,
A1/B1 speed, 8 speed cells (1024:512:1/4/8, 16384:1024:1/4/8, 32768:2048:1/4), 2
exactness cells, stall probe on.

### Recipe assembly (CPU, no launch)

`PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=inkling2-hi2-tp2 MODEL=<serve dir>
VENV=<venv> HIP_VISIBLE_DEVICES=<slot>` assembles:

```text
python -m vllm.entrypoints.cli.main serve --model <serve dir> \
  --served-model-name inkling2-hi2 --host 127.0.0.1 --port 18225 \
  --tensor-parallel-size 2 --dtype float16 --max-model-len 1048576 \
  --max-num-seqs 8 --max-num-batched-tokens 320 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --enable-prefix-caching --language-model-only --limit-mm-per-prompt '{"image":0}'
```

## What still needs a GPU (ordered; details in tools/inkling_hi2/README.md)

1. Build the head tree on the box (no csrc change: the existing `_rocm_C` of the
   worktree is enough; this branch adds no compiled code).
2. `pytest tests/models/inkling2_loop/test_hi2_kernel_gpu.py -m gpu` (5 tests; the
   only correctness evidence for the grouped GEMMs; tolerances in the file).
3. Boot with the recipe above; greedy smoke (`smoke.sh`), compare against the stock
   model's greedy ids on the same prompts.
4. KL probe on the quantizer's held-out windows (`kl_probe.py`,
   `~/w4a8_runs/inkling-hi2/stage4/eval/holdout_ids.npz`, reference KL 0.0996 /
   top-1 0.938): gate KL <= 0.15, top-1 >= 0.90.
5. Bench cells (`bench.py`), quiet window, KV/peak memory; record in this doc.
6. Only then vision and MTP/DSpark (the recipe is language-only, `MTP=0`).
7. For a real `pr_receipt` A/B: build the base tree, then
   `BASE_TREE=... HEAD_TREE=... VENV=... RECIPE=inkling2-hi2-tp2 MODEL=<serve dir>
   GPUS=<leased HIP ids> bash tools/rdna/pr_receipt.sh` (exit 0: gates pass; speed
   verdicts are reported, not gated).  The `DRY_RUN=1` output above is its plan.

## Known limits of this receipt

- No GPU run happened: kernel numerics, graph capture, real decode speed and the KL
  of the 3.5 bpw directory are unmeasured; the 8.02-9.81 GiB per-rank figure is a
  header census, not a measured `Model loading took`.
- The 5 GPU tests are skipped in the CPU gate; a green CPU run says nothing about
  the HIP kernels.
- `pr_receipt_scan` signed off the reference-path `tolist` lines as non-per-step;
  it never gates, it only lists.

## Reproduce

```bash
TREE=<this worktree>
H=$HOME/w4a8_runs/inkling-hi2/tree                 # quantizer checkout on the box
V=$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340/bin/python
S=$HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5
O=$HOME/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5

bash $TREE/tools/inkling_hi2/prep_serve_dir.sh "$S" "$O"
PYTHONPATH=$TREE $V $TREE/tools/inkling_hi2/validate_dir.py --model "$O"
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
  PYTHONPATH=$H:$HOME/w4a8_runs/pydeps/tfm-main:$TREE INKLING2_TINY_OUT=$HOME/w4a8_runs/inkling-hi2-runtime/tiny \
  $V -m pytest $TREE/tests/models/inkling2_loop/test_hi2_loader.py \
    $TREE/tests/models/inkling2_loop/test_loader.py \
    $TREE/tests/models/inkling2_loop/test_hi2_kernel_gpu.py \
    $TREE/tests/models/inkling2_loop/test_hi2_reference.py -q
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
  PYTHONPATH=$H:$HOME/w4a8_runs/pydeps/tfm-main INKLING2_TINY_OUT=$HOME/w4a8_runs/inkling-hi2-runtime/tiny \
  $V $TREE/tools/inkling_hi2/mutants.py --tree <scratch copy>
PRINT=1 bash $TREE/tools/rdna/serve_rdna.sh RECIPE=inkling2-hi2-tp2 MODEL=$O VENV=$V \
  HIP_VISIBLE_DEVICES=<leased HIP ids>
```
