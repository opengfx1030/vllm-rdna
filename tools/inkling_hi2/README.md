# HI2 Inkling-2-Mini runtime: load/run kit

The quantizer (`rdna-hint-flex-quantizer-inkling`, `feat/inkling2-source`) writes a
3.5 bpw Inkling-2-Mini directory: routed experts in HI2 (INT2/INT4 rows, rotated and
smoothed), everything else bf16, Hugging Face text names plus the quantizer's stacked
expert containers (`...mlp.experts.gate_up.hi2_*`, `...experts.down.hi2_*`). This
worktree (`feat/inkling2-hi2-runtime`) loads and runs that directory on two V620s:

| Layer | Where |
| --- | --- |
| format declaration (`rdna_hint_flex`, v3.2, `moe_stacks`) | `vllm/models/inkling2_loop/amd/loader.py` (`hi2_expert_formats`) |
| HF name map (incl. `layers.N.mlp.experts` containers) | `amd/loader.py` (`hf_to_serving`), `amd/hi2.py` (`parse_tensor_name`) |
| per-expert packed stack, TP slice (rows of gate/up, columns of down) | `amd/hi2.py` (`canonicalize_stack`, `slice_stack`, `validate_stack`) |
| float64 decode + routed MoE reference | `amd/hi2.py` (`decode_part`, `expert_weights`, `reference_moe`) |
| grouped expert GEMMs (transform + INT2/INT4 GEMM, block 64) | `amd/moe.py` (`_forward_chunk_hi2`) |
| serving path over workspace buffers (no device allocation) | `amd/moe.py` (`_forward_hip_hi2`), `amd/glue.py` (`hi2=True`) |

The activation is Inkling's soft-capped two-argument SwiGLU, `tanh(g/4)*4 *
sigmoid(g) * tanh(u/25)*25`, in fp32; the router is fp32, the top-k sum fp32. The
fused generic op `rdna_hint_flex_moe_rdna2` implements SiLU, so this path drives
`rdna_hint_flex_moe_transform_rdna2` + `rdna_hint_flex_moe_gemm_rdna2` and
`inkling2_glue_swiglu`; no new HIP kernel, no csrc change, no rebuild needed.

## 0. Protocol (AGENTS.md: "Two-slot box operation 2026-11-15")

Six V620s in two leases; this kit is the TP=2 one on a2:

| Lease | HIP | V620 PCI bus | Use |
| --- | --- | --- | --- |
| a2 | 0, 1 | `d1`, `d4` | this kit (TP=2) |
| b4 | 2, 3, 4, 5 | `d9`, `dc`, `df`, `e2` | TP4 work (not this recipe) |

`SLOT` selects the lease (`a2` default; `b4` is refused by the TP=2 launchers).
`HIP_VISIBLE_DEVICES` and `EXPECT_PCI_BUSES` default from `SLOT` and must be given
together; `tools/inkling_hi2/slot.py` resolves the HIP indexes to their live PCI
buses (clearing the visibility variables so it sees the physical enumeration) and
refuses a moved bus, a missing index or a non-V620 before the boot and again
immediately before the launch. The HTTP clients (`kl_probe.py`, `bench.py`) cannot
see the GPUs and check the declared env pair against the same table, recording the
slot in their JSON.

Every GPU command carries `HIP_VISIBLE_DEVICES=<leased HIP list>` on its own command
line; CPU-only commands use
`env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= <cmd>`.
Leases live in `~/w4a8_runs/lanes/lease/README.md` (holder files `<owner>.a2` /
`<owner>.b4`, requests, release trap; kill the holder's `sleep` on release). Boots
take `flock -w 900 ~/w4a8_runs/boot.lock` and the two slots boot in parallel. A
measured window touches `~/w4a8_runs/lanes/quiet-a2/<Agent>` or
`quiet-b4/<Agent>` (<= 15 min, re-touched every 5 min), waits until no file in its
slot is newer than 6 min, then waits for `flock -n` on the boot lock and re-checks
it before each cell; `MEASURE_LOCK=` is empty (one slot cannot block the other).
`serve.sh` does the quiet wait (`WAIT_QUIET=1`) and the boot lock; `smoke.sh`
touches its quiet file after the boot and re-checks the boot lock before each
probe. A window with a KFD `evicted_ms` jump or a thermal alarm is invalid and is
rerun; no measured window while a `docker buildx build` runs.

## 1. Preconditions (CPU only, no lease)

```bash
TREE=/path/to/vllm-inkling2-hi2-runtime
V=$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340/bin/python

# the new Python path imports against the tree's built _rocm_C
cd $TREE && env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
    PYTHONPATH=$TREE $V -c "import vllm._custom_ops, vllm.models.inkling2_loop.amd.moe"

# CPU gates: loader (23), the slot/HIP->PCI gate (7), quantizer reference on the
# tiny HI2 directory (4; the fixture is built by the quantizer's own pipeline,
# ~3.7 min), legacy loader (6), plus the GPU kernel tests collected-and-skipped on
# CPU (5). H is the quantizer tree; it provides the `hi2` package the reference
# test imports.
H=$HOME/w4a8_runs/inkling-hi2/tree
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$H:$TREE:$HOME/w4a8_runs/pydeps/tfm-main $V -m pytest \
  tests/models/inkling2_loop/test_hi2_loader.py \
  tests/models/inkling2_loop/test_hi2_slots.py \
  tests/models/inkling2_loop/test_loader.py \
  tests/models/inkling2_loop/test_hi2_kernel_gpu.py \
  tests/models/inkling2_loop/test_hi2_reference.py -q
# 40 passed, 5 skipped once the fixture is cached; INKLING2_TINY_OUT=<dir>
# reuses the <dir>/stock + <dir>/hi2 pair built below instead of rebuilding it.
# The slot test needs no torch: it fakes the six-V620 PCI table.

# mutant battery: the loader / name map / TP slice / reference tests must kill
# every mutant (21; exit 0). Use a scratch copy of the tree, not the checkout.
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$H:$HOME/w4a8_runs/pydeps/tfm-main INKLING2_TINY_OUT=$HOME/w4a8_runs/inkling-hi2-runtime/tiny \
$V $TREE/tools/inkling_hi2/mutants.py --tree <scratch copy>

# the real quantizer output, loaded on CPU (the first-boot proxy): no GPU
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$TREE $V $TREE/tools/inkling_hi2/check_real_dir.py \
  --model $HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5 \
  --stock $HOME/hfcache/inkling2-mini
# expected: 1125 text tensors, 0 missing parameters, 480 expert buffers,
# lm_head equals the stock unembed's first 200058 rows, 643 dense tensors bit-equal
```

Build the tiny HI2 fixture once for the GPU kernel test (it is the same fixture the
reference test builds in its tmp dir; keep it under `~/w4a8_runs`):

```bash
H=$HOME/w4a8_runs/inkling-hi2/tree   # the quantizer tree (feat/inkling2-source)
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$H:$HOME/w4a8_runs/pydeps/tfm-main:$TREE INKLING2_TOOLS=$TREE/tools/rdna/inkling2 \
$V $TREE/tools/inkling_hi2/make_tiny_hi2.py --out $HOME/w4a8_runs/inkling-hi2-runtime/tiny
# expected: "quantized 144 expert tensors" and .../tiny/hi2
```

## 2. Ordered GPU checklist

One lease of two V620s: the a2 slot (HIP 0,1; PCI `d1 d4`) from Main. b4
(HIP 2,3,4,5; `d9 dc df e2`) is the TP4 lease and is not used by this TP=2 kit.
Each command repeats `HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4` on its own
line; `serve.sh`/`smoke.sh` resolve the pair to the live PCI buses before the boot
and before the measured cells and refuse a moved or foreign device. Every step
prints a `RESULT ... PASS` or a numeric record; stop at the first failure and keep
the log.

1. **Kernel unit test** (no server). `HI2_TINY_DIR` set to the fixture's `hi2/`:

   ```bash
   cd $TREE && HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 PYTHONPATH=$TREE \
   HI2_TINY_DIR=$HOME/w4a8_runs/inkling-hi2-runtime/tiny/hi2 \
   $V -m pytest tests/models/inkling2_loop/test_hi2_kernel_gpu.py -q
   ```

   Verifies: the HI2 collector reads the real stacked layout; block-64 transform+GEMM
   +`inkling2_glue_swiglu` match `hi2.reference_moe` (float64 decode, fp32 matmul) at
   1/8/64 tokens (rel err printed, gate 5e-2); the two TP=2 rank outputs sum to the
   TP=1 output (5e-2); `forward_hip` matches the eager kernel and allocates **0**
   bytes after warm-up. Expected: 3 passed (this file's gate), e.g.
   `HI2 M=1: rel err 3e-3`. A `5e-2` breach is a numerics finding, not a flake.

2. **Stock reference server** (same lease, TP=2, language-only, 8192 context) so the
   probes have a baseline:

   ```bash
   MODEL=$HOME/hfcache/inkling2-mini SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   VENV=$(dirname $(dirname $V)) PORT=18231 MAXLEN=8192 MAXBAT=256 SEQS=1 \
   bash $TREE/tools/inkling_hi2/serve.sh > ~/w4a8_runs/inkling-hi2-runtime/logs/stock.log 2>&1 &
   ```

   Verifies: the recipe boots an NVFP4 stock directory unchanged (no regression from
   the loader changes). Wait for `/health`; record "Model loading took" and the
   `max_model_len=8192`.

3. **HI2 server**: same command with `MODEL=$HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5`
   and `PORT=18230`, log `logs/serve.log`; the env pair stays
   `SLOT=a2 HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4`. Verifies: all 24 layers
   find their `gate_up`/`down` stacks (`HI2 expert groups incomplete` and
   `unmapped ... tensor` must NOT appear); the engine reports the model memory
   (expect ~7.1-8.6 GiB/rank for the weights, versus 18.68 GiB for the stock NVFP4
   at TP=1); FULL_AND_PIECEWISE graph capture succeeds; `VLLM_INKLING2_PATH` is
   unset (native HIP path) and the log shows "Inkling-2 HIP serving path: workspace
   ...". If capture fails, rerun with `EAGER=1` and record the difference (FULL
   graphs are a production requirement; an eager-only boot is a finding).

4. **Greedy smoke** with the HI2 server (it boots its own) and the stock server as
   the reference:

   ```bash
   MODEL=$HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5 SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   VENV=$(dirname $(dirname $V)) STOCK_URL=http://127.0.0.1:18231 \
   bash $TREE/tools/inkling_hi2/smoke.sh
   ```

   Verifies: `/v1/models` name and context; the greedy chat answer names Paris
   (`RESULT greedy PASS`); the greedy prefix comparison against the stock server
   prints the first-divergence length (3.5 bpw top-1 is 93 %, so short prefixes, not
   full streams, are expected). The smoke touches `lanes/quiet-a2/inkling-hi2-smoke`
   before each probe and waits for the boot lock.

5. **KL vs stock on the quantizer's held-out windows** (mixed set, first 32 of 96,
   2048 tokens each; the same windows the published 0.0996 nats were measured on):

   ```bash
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/kl_probe.py \
     --hi2-url http://127.0.0.1:18230 --stock-url http://127.0.0.1:18231 \
     --ids $HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy \
     --windows 32 --topk 100 --max-kl 0.15 --min-top1 0.90 \
     --reference-eval $HOME/w4a8_runs/inkling-hi2/stage4/eval/bpw3.5_mixed.json \
     --out ~/w4a8_runs/inkling-hi2-runtime/kl_hi2.json
   ```

   Verifies: mean truncated KL near the full-logit reference (0.0996; expect the
   same order, gate 0.15), top-1 >= 0.90 (reference 0.930), per-window spread and
   prompt-token counts equal between the servers. Repeat with
   `heldout_wiki_ids.npy` (17 windows, reference 0.0468) and
   `heldout_pydocs_ids.npy` (0.0305) — those are stricter relative checks than the
   mixed set. Both servers must have `--max-logprobs 128` (serve.sh passes it).
   A measured window is <= 15 min: split a long set into `--windows` chunks and
   re-touch the quiet file (and re-check the boot lock) between chunks.

6. **Bench cells**, same lease, stock then HI2:

   ```bash
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/bench.py --url http://127.0.0.1:18231 \
     --label stock --log ~/w4a8_runs/inkling-hi2-runtime/logs/stock.log \
     --out ~/w4a8_runs/inkling-hi2-runtime/bench_stock.json
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/bench.py --url http://127.0.0.1:18230 \
     --label hi2 --log ~/w4a8_runs/inkling-hi2-runtime/logs/serve.log \
     --baseline ~/w4a8_runs/inkling-hi2-runtime/bench_stock.json \
     --out ~/w4a8_runs/inkling-hi2-runtime/bench_hi2.json
   ```

   Verifies: prefill prompt-tokens/s, decode wall per step at batch 1 and 4, and the
   engine's model-load/KV/workspace lines. No pass/fail: record the ratios. Known
   expectation [inference]: the block-64 HI2 GEMM pads an 8-pair decode to 64 rows,
   so batch-1 decode can be slower than the stock NVFP4 kernel; the memory cell must
   show the HI2 weights smaller. A decode step slower than ~10x stock is a finding.

7. **Graph-mode A/B** (only if step 3 needed it): rerun step 4/5 with `EAGER=1`; the
   numbers from the eager boot are not the serving record, they bound the graph
   contribution.

## 3. What is deliberately not covered by this kit

* Vision (`--limit-mm-per-prompt {"image":0}`): the tower is bf16 and carried
  bit-exact by the loader, but the HI2 serving path is validated on text first.
* MTP / DSpark (`DSPARK=0`): the draft file beside the output is not served; the
  recipe's draft needs its own conversion and a separate probe.
* Long context: `MAXLEN=8192` first; the TP=2 recipe's 720k plan applies to the
  stock model and must be re-measured with the HI2 KV (same 16 KiB/token/rank).
* TP=4 and CUDA: the kernel test is gfx1030-only, as the fork requires.
