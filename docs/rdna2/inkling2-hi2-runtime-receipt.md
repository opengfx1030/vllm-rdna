# Inkling-2-Mini HI2 runtime: CPU receipt

Branch `feat/inkling2-hi2-runtime` (cut from `origin/rdna_extras` at `1fdcb3a66e`,
which carries the merged Inkling-2-Mini runtime, `vllm/models/inkling2_loop/`, and
**merged up to `41ef6ba1e0` in `bae1b21db0`** on 2026-11-25), 2026-11-24/25.  From
the merge the PR base is `41ef6ba1e0`, so every "base" below is that commit; the
branch's csrc is byte-identical to it (the merge brought the base's 5-TU csrc delta
into the branch history, and the box `_rocm_C` was rebuilt from the merged tree).
The CPU evidence
below was produced without a GPU; the exploratory a2 runs of 2026-11-25 (kernel
test, smoke + KL, bench, a long serving run) are recorded in their own section and
are **not** the PR contract receipt.  The contract's gates, their exact commands,
the a2 time each needs, and the gates that cannot be satisfied as written are in
"PR contract" below.

**Change class.** Python model/runtime wiring plus tools; against the merged base
`41ef6ba1e0` there is **no csrc, CMake, setup or C++ change**.  The HI2 expert
path calls the fork's existing RDNA2 ops
(`rdna_hint_flex_moe_transform_rdna2`, `rdna_hint_flex_moe_gemm_rdna2`,
`inkling2_glue_swiglu`); the arch gates and launches are the ops' own (gfx1030 only,
current stream).  The CUDA / CDNA / CPU paths are untouched because no compiled code
changed; the pure-torch reference and the loader run everywhere torch is available.

Files added by this change set: `vllm/models/inkling2_loop/amd/{hi2,loader}.py`
changes plus `hi2.py` decoding/reference, `amd/moe.py` HI2 experts
(`_forward_chunk_hi2`, `forward_reference`, `_forward_hip_hi2`), `amd/glue.py`
(`hi2=True` workspace), `amd/model.py` loader wiring, three test modules (the
launcher module is `test_hi2_slots.py`, 40 tests), and the tools below
(`tools/inkling_hi2/lease.sh` and `quiet_touch.py` carry the lease and the quiet
touch; `mutants_launchers.py` is the 13-mutant battery; `gpu_temps.py` gained the
watcher flags; the recipe pins a2 by PCI bus).

## What ran on CPU (all on the box, `par1-cs25`, no lease)

| Step | Command (abbreviated) | Exit | Result |
| --- | --- | --- | --- |
| CPU gate | `pytest test_hi2_loader.py test_hi2_slots.py test_loader.py test_hi2_kernel_gpu.py test_hi2_reference.py -q` | 0 | 74 passed, 5 skipped (the 5 GPU tests; 119.5 s on the post-fix tree) |
| Pre-fix check | the same gate on the 5 new regression tests, old product files | 1 | 5 failed, 16 deselected (the tests can fail) |
| Kit pre-fix | the new `test_hi2_slots.py` inside a scratch copy of `git archive f34d6cf6bc tools/` (old kit and recipes) | 1 | 32 failed, 5 passed (the protocol and KL tests can fail) |
| Mutants | `python tools/inkling_hi2/mutants.py --tree <scratch copy>` | 0 | baseline 27 passed; all 21 killed: the 20-mutant battery on the pre-B4 tree plus the changed `hi2_uneven_rows_check_off` and the new `moe_w2_reduce_rows_as_tokens` on the final tree |
| Launcher mutants | `python tools/inkling_hi2/mutants_launchers.py` | 0 | baseline 15 passed; all 16 killed (lease gate/owner/VRAM, the foreign-request queue, the stale-holder drop and the live-holder VRAM re-check, boot-lock release, the pre-launch and post-/health device checks, the server env filters, the temperature threshold, the missing-eviction window, the b4 client refusal, the bench evidence) |
| Real-dir load | `python tools/inkling_hi2/check_real_dir.py --model <bpw3.5> --stock <stock>` | 0 | 1125 text tensors loaded, 0 missing parameters, lm_head == stock |
| TP suite | `pytest tests/models/inkling2_loop/test_tensor_parallel.py -q` with the workaround below | 0 | 156 passed |
| Serve dir | `bash tools/inkling_hi2/prep_serve_dir.sh <quantizer out> <serve dir>` | 0 | 15 symlinks + MTP; source untouched |
| Validate dir | `python tools/inkling_hi2/validate_dir.py --model <serve dir>` | 0 | VALID (see census below) |
| archdiff | `archdiff_report.py plan` on clean archives of `41ef6ba1e0` (base) and `bae1b21db0` (head, the merge) | 0 | `empty`: `changed_files []`, `units []`, `source.diff` 0 bytes |
| Diff scan | `pr_receipt_scan.py base-rdna_extras tree` (base `41ef6ba1e0`, head = the merged tree) | 0 | 5 added `tolist` lines, all CPU reference; 0 per-step, 0 C findings (the 24 generated paged-attn units are head-only files) |
| Receipt dry run | `DRY_RUN=1 BASE_TREE=HEAD_TREE=tree pr_receipt.sh ... RECIPE=inkling2-hi2-tp2` | 0 | 4 boots, 8 speed cells, 2 exactness cells, quality 0 |
| Recipe assembly | `PRINT=1 serve_rdna.sh RECIPE=inkling2-hi2-tp2 MODEL=...` | 0 | command below |

Everything above was re-run on the **merged** tree (`bae1b21db0` / docs commit
`ad17998e4e`, csrc = `41ef6ba1e0`, rebuilt `_rocm_C` md5 `3da91a1a5350`): CPU gate
74 passed / 5 skipped (119.5 s), `mutants.py` 21/21 killed, `mutants_launchers.py`
16/16 killed, `check_real_dir.py` PASS (1125 tensors, 0 missing, 480 expert
buffers, 643 dense tensors bit-equal to stock), the stock-recipe `DRY_RUN=1`
receipt rc 0 with **identical base/head launch plans** and the six boots / 8 speed
cells / quality / stall plan, archdiff `empty` against `41ef6ba1e0`, and the scan's
5 `tolist` lines with 0 C findings.  The old kit's pre-fix rows above were measured
before the merge and were not re-run.

The CPU gate is the six-layer HI2 test set: format declaration/name map/TP slice
(`test_hi2_loader.py`, 23), the slot/HIP->PCI gate, the lease carrier and the
launcher protocol (`test_hi2_slots.py`, 40, on a fake visible-device table, a fake
torch and a stub HTTP server plus script checks), the legacy loader
(`test_loader.py`, 6), the kernel tests collected and skipped off a GPU (5), and the
reference against the quantizer's own `hi2.format.dequantize` on a tiny HI2 export of
the quantizer's fixture (4, fp16 exact).  `test_hi2_reference.py` needs the quantizer
on `PYTHONPATH` (`HI2_SRC=$HOME/w4a8_runs/inkling-hi2/tree`) and reuses its fixture
with `INKLING2_TINY_OUT`; without the quantizer the 4 tests skip (do not read a skip
as a pass).

### GPU kit on the six-V620 two-slot box (2026-11-24)

The kit follows AGENTS.md "Two-slot box operation 2026-11-15": `SLOT=a2` (HIP 0,1;
PCI `d1 d4`) is the TP=2 lease and the default, `b4` (HIP 2,3,4,5; `d9 dc df e2`) is
refused by the TP=2 launchers.  `serve.sh`/`smoke.sh` clear
`ROCR_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`; `slot.py` then enumerates the GPUs
with exactly the server's environment (`HIP_VISIBLE_DEVICES` only), so it can
neither miss a card the server sees nor open one the slot does not own, and refuses
a moved bus, a missing/extra device, a non-V620 or a W7800 before the boot and
before every measured cell.  The HTTP clients check the declared pair and record
it; `MEASURE_LOCK=` stays empty (cross-slot lock off).

Booter protocol: another `OWNER`'s `<holder>.a2` file refuses the boot; the booter
waits out `quiet-a2`, takes `boot.lock`, re-checks the quiet directory *under* the
lock, holds the lock until `/health` (`READY_SECONDS=1800`) and then releases it —
the server keeps running.  `smoke.sh` leaves that carrier to `serve.sh`, re-touches
its quiet file every 5 min (<= 15 min) and journals junction temperatures
(`gpu_temps.py`) and the vLLM tree's KFD evictions (`kfd_evictions.py` with
`SERVER_PID_FILE`); a 95 C sample or an `evicted_ms` jump makes the window `INVALID`
(non-zero exit), never a pass.  `bench.py --server-pid` samples the same two
signals and records them in its JSON.

Kit protocol round (P1-P3), each covered by the then-15-test `test_hi2_slots.py`
(11 of those 15 fail on `f34d6cf6bc`; the module now has 32, see P4-P7 below):

1. **P1 lease holder.** `OWNER` defaults to `HI2InklingRuntime` (was
   `HI2Inkling`); any other `<holder>.a2` in `~/w4a8_runs/lanes/lease` refuses
   `serve.sh`/`smoke.sh` with the holder's name, instead of silently double-booting
   the slot.
2. **P2 device check.** The checker no longer pops the visibility variables (that
   made it resolve the *physical* six cards while the server saw two); it runs with
   `HIP_VISIBLE_DEVICES` as the only filter and refuses an environment that still
   carries `ROCR_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` or a process that sees
   more/fewer devices than the slot.  On the box's torch 2.12+rocm7.14,
   `torch.accelerator.device_count()` returns 0 (even with the accelerator
   available) until the accelerator is initialized, so the enumeration calls
   `torch.cuda.init()` first — with `HIP_VISIBLE_DEVICES` as the only filter this
   initializes exactly the slot's cards (observed: `0xd1`/`0xd4` for a2).  The recipe
   now defaults to the a2 pair
   (`HIP_VISIBLE_DEVICES=0,1`; the stock recipe's example was the retired b4 pair).
3. **P3 boot lock and window validity.** `serve.sh` holds `flock boot.lock` until
   the server answers `/health` (the old `flock -w 900 true` released it before the
   boot), re-checks the quiet directory under the lock, and closes the lock fd in
   the server child.  `smoke.sh` re-touches quiet every 5 min and fails the
   measured window on a hot or evicted GPU; `bench.py` records the same and exits 3
   on an invalid window.

### Launcher carrier round (P4-P7, 2026-11-24)

1. **P4 lease carrier (`tools/inkling_hi2/lease.sh`, new).** `lease_acquire` types
   `<slot>.request` (release trap armed before it), waits until no foreign holder
   exists and the least-free leased card has 28 GiB (`LEASE_WAIT_SECONDS=1800`),
   then writes `$OWNER.$SLOT`; the trap removes it on EXIT/TERM.  A foreign holder
   or an older requester in the queue is refused with the name; `serve.sh` writes
   the request, `smoke.sh` only checks the holder.
2. **P5 boot lock through readiness.** `BOOT_LOCK` is taken with `flock -w 900`
   before the launch and released only after `/health`; the later cleanup no longer
   touches a closed fd.  The quiet directory is re-checked under the lock;
   `smoke.sh` re-checks the lock (non-blocking) before each measured cell and
   re-touches the quiet file.
3. **P6 window validity.** The temperature watcher's mean rule is armed only when
   the samples span the mean window (a 5 s mean never fires on one sample); the
   peak rule is unconditional; the smoke parses both the profile CSV and the
   watcher CSV; a missing pre-window KFD context or any eviction is `INVALID`;
   `bench.py` requires `--server-pid` for a valid window and exits 3 on `INVALID`;
   `kl_probe` refuses b4.
4. **P7 device litmus.** `serve.sh` checks the devices three times -- before the
   quiet wait, immediately before the launch, and after `/health` (the live server
   must still see the leased cards; a mismatch kills it and the boot fails) -- and
   `smoke.sh` checks once before `serve.sh` and once per cell.  The child sees
   `HIP_VISIBLE_DEVICES` only; the HTTP clients refuse b4 and record the slot.  The
   recipe pins the a2 pair by PCI bus (`PCI_BUSES="d1 d4"`).

The 40 tests in the module cover these (38 protocol/launcher plus 2 for the KL
parser); the launcher mutant battery kills 13/13 (`mutants_launchers.py`), and 32
of the 37 fail on the pre-round kit (`f34d6cf6bc`), including both KL tests (the old
probe raises `AttributeError` on the `None` at position 0 and has no
`score_window`).  Residual: no CPU test can prove the real `/sys/class/drm` VRAM
coresidency or the real `flock`/quiet timing on the box; they run only against
tmpfs paths and the fake torch.

### Probe round after the first HI2 boot (a2, 2026-11-25 03:29)

The first a2 boot of `bpw3.5` on the branch passed model load (6.87 GiB), the
native HIP path, FULL+PIECEWISE capture and `/health`; the first greedy request was
coherent (0.81 s) but `smoke.sh` failed only because the France probe capped
generation at 32 tokens and the reasoning model was cut before naming Paris.  The
probe now uses `GREEDY_TOKENS=160`, accepts the city anywhere in the
reasoning+answer text, runs the same request on `STOCK_URL` and prints both outputs
and the first-divergence character position.  A second capsule probe re-sends one
request twice and requires a non-empty deterministic repeat (it does not judge
wording).  A failed Python probe aborts `smoke.sh` with a non-zero exit; the
"smoke exit 0" in the driver log was the exit status of its `tee`, not of the
script.  CPU tests now assert smoke's own exit status for a failed France probe
and a non-repeating capsule, and exercise the stock-comparison branch.

### KL probe fix after the smoke rerun (a2, 2026-11-25 03:43)

The rerun passed the France probe (both servers answered Paris after a reasoning
preamble, first divergence at char 65) and the capsule (identical repeat), then the
KL cell crashed: vLLM returns `None` in `logprobs.top_logprobs[0]` (the first
prompt token has no context) and the probe called `.items()` on it.  The probe now
scores positions 1..L-1, treats any missing top-k entry as skipped, prints
`skipped S/considered` per window and fails a window above `--max-skipped-share`
(default 5 %).  It requests `temperature=1` (raw post-softcap logprobs, top-k
truncated, no renormalisation); its docstring and README state the comparison
honestly: the served fp16-path top-100 number approximates the quantizer's float32
full-logit 0.0996 nats / top-1 93.0 %, within `--max-kl`, not for equality.  CPU
tests cover the `None` at position 0 (the old parser raised) and an omitted
interior position, and the JSON carries `skipped_share`, the returned top-k size
(`topk_entries`, printed for the first window) and a truncation note.  The fixed
probe then ran on a2 at 04:10 and **passed**: 32 mixed windows x 2048, mean KL
0.09340, top-1 0.93090, 0 skipped (see the exploratory table below).

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

The contract compares the csrc sources of the base (`41ef6ba1e0`, the merged
`rdna_extras` commit; the branch was cut at `1fdcb3a66e` and merged up to the new
base in `bae1b21db0`) with the head.  The plan was re-run after the merge on
**clean `git archive` trees** of both commits
(`~/w4a8_runs/inkling-hi2-runtime/archdiff2/{base41,headbae}`, base label
`41ef6ba1e0`, head label `bae1b21db0`):

```bash
$V tools/rdna/archdiff_report.py plan \
  --base-tree ../archdiff2/base41 --head-tree ../archdiff2/headbae \
  --base-label 41ef6ba1e0 --head-label bae1b21db0 \
  --plan-out ../archdiff2/plan/plan.json --diff-out ../archdiff2/plan/source.diff
# "empty"; rc=0; source.diff 0 bytes
# plan.json: mode auto, changed_files [], units [], source_diff {}, tus_missing []
```

No translation unit changed, so `archdiff.sh` has nothing to compile or disassemble
for any arch (`bash tools/rdna/archdiff.sh` would print nothing to diff) and needs
no GPU.  Clean archives are used because a *configured* box tree also contains the
generated paged-attention units (`csrc/rocm/attention_paged_attn_*.cu` +
`.paged_attn_generated`, written at 02:59 by the 03:08 build): CMake takes its
source list from that stamp, not from a glob, so they are not strays in the build,
but a raw directory-to-directory plan would report them as head-only files.  Any
future change to `csrc/` on this branch must run `archdiff.sh` with all six arches.

### pr_receipt scan and dry run

The scan against the cut base lists five added `tolist()` lines, all in `amd/hi2.py`
at the tensor-parallel slice and reference-decode helpers (CPU paths:
`_select_cols`, `dequantize_stored_rows`, `reference_moe`).  None is in the per-step
serving path; the HI2 kernels receive precomputed index tensors.  The `DRY_RUN=1`
receipt with the **stock** recipe (base `../base-rdna_extras` vs a clean head
archive; both need a built `_rocm_C`, hence the two warnings below) exits 0 and
prints the production plan:

```text
pr_receipt: warning: BASE_TREE has no built vllm/_rocm_C*.so
pr_receipt: warning: HEAD_TREE has no built vllm/_rocm_C*.so
boots: XA:base:exact XB:head:exact A1:base:speed B1:head:speed B2:head:speed A2:base:speed
speed cells: 1024:512:1 1024:512:4 1024:512:8 16384:1024:1 16384:1024:4 16384:1024:8 32768:2048:1 32768:2048:4
exactness cells: 1024:128:1:4 16384:128:1:2
quality: 1 (windows .../quality/windows.json, limit 64)
stall probe: 1 (decoders 7, prefill 4096,16384, repeats 3)
diff scan: 5 added lines to review (0 in the per-step path): tolist x 5
```

Six boots (2 exactness + 4 speed including the quality floor), the contract's full
8-cell matrix, the two exactness cells, the quality collects (64 windows per domain)
and the stall probe on every speed boot.

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

### GPU facts already measured on a2 (exploratory; not the contract receipt)

a2 = HIP 0,1 = PCI `d1,d4` (two V620).  Short windows freed up on 2026-11-25; these
are single runs, **not** the contract gates, and the bench/smoke boots used the
kit's config (`MAXLEN=8192 MAXBAT=256 SEQS=1 GMEM=0.45`, `DSPARK=0`), not the
production recipes.  All of them were measured on the **pre-merge** branch (tree
tip `aae486796f`/`43a19980d2`, csrc cut `1fdcb3a66e`); after the merge
(`bae1b21db0`, csrc `41ef6ba1e0`) the HI2 kernels are unchanged (`archdiff units
[]`), but no number here has been re-measured on the merged tree.

| Fact | Artifact | Result |
| --- | --- | --- |
| Kernel vs fp64 reference, a2 03:19 | `~/w4a8_runs/inkling-hi2-runtime/kernel_test_a2.out` | **5/5 passed**: M=1 5.450e-4, M=8 7.402e-4, M=64 9.428e-4, TP=2 vs TP=1 5.044e-4, HIP path 5.435e-4 and alloc delta 0 B (4.94 s, after the `208230562b` device fix; the 03:12 run had failed in the float64 oracle) |
| Smoke + KL, a2 04:03-04:10 | `.../serve_smoke_a2.out` | France probe: both servers answer Paris, first divergence at char 65 -> PASS; capsule repeat identical -> PASS; **KL PASS** on the 32 mixed windows x 2048 tokens: mean 0.09340, SEM 0.01716, top-1 0.93090, 0 skipped, 100/100 top entries, `RESULT measured window PASS` (max junction 64.0 C, KFD evictions +0) |
| Bench, a2 04:11-04:14 | `.../bench_stock.json`, `.../bench_hi2.json` | stock load 9.03 GiB, KV 3.90 GiB/101,664 tok; HI2 load 6.87 GiB, KV 6.04 GiB/167,620 tok; prefill 2293.7 vs 1260.4 tok/s (0.549x); decode b=1 12.83 vs 23.52 ms/step (77.95 vs 42.52 tok/s), b=4 51.14 vs 93.99 ms/step (78.21 vs 42.56 tok/s; ratios 1.833x/1.838x); both windows valid |
| Long serving run, a2 07:43-08:02 | `.../logs/eval-serve.log` | HI2 dir via the kit's serve.sh, `MAXLEN=32768 MAXBAT=2048 SEQS=4 GMEM=0.45`, workspace 0.60 GiB, four concurrent requests for ~19 min (60-72 tok/s generation), no engine error, clean SIGTERM shutdown (the `EngineDeadError` at 08:02:43 is the output handler after the deliberate stop) |

The quantizer's float32 reference on the same 32 mixed windows
(`stage4/eval/bpw3.5_mixed.json`, `student`): KL 0.099626 +/- 0.017838, top-1
0.930191, PPL ratio 1.00951, expected acceptance 0.93365.  The served top-100 KL
(0.09340) and top-1 (0.93090) are consistent with it; the served number is a top-k
truncated fp16-path measurement and is compared within `--max-kl`, never for
equality.  The 03:43 crash (`top_logprobs[0]` is `None`) was fixed in
`aae486796f`; the 04:10 run is the fixed probe.

### Lease queue and stub-server fixes (2026-11-25, CPU only)

The kit verifier's last items, fixed with regressions:

- `lease.sh` never overwrites another owner's `a2.request`: the shared request
  file (lane README) is written only when absent or when it is ours and stale;
  a foreign request is queued behind, up to `LEASE_WAIT_SECONDS`.  The release
  trap removes the request only while it is still ours.
- An own holder is reused only while its recorded pid is alive **and** the
  >= 28 GiB free-memory check still passes; a dead-pid holder is dropped with a
  message and the normal wait applies.  (Before: an own holder returned success
  before the memory check, and a stale holder blocked a fresh acquire.)
- `test_hi2_slots.py` no longer leaks stub servers: the module overrides
  `tmp_path` with a per-test scratch dir under `$HI2_TEST_BASE` (default
  `~/.cache/inkling2-hi2-tests`, never /tmp), and both `DryKit.stop()` and a
  session fixture kill any `stub_http.py` this kit wrote.  The 15 stubs leaked
  by the Nov-24 runs under `/tmp/pytest-of-chenco_adm/...` were killed; a run
  now leaves 0.
- `mutants_launchers.py` grows to 16 (three lease mutants; the VRAM mutant
  follows the new expression): all 16 killed, `mutants.py` 21/21.
- README steps 2/3 and `tools/rdna/recipes/inkling2-hi2-tp2.env` state that the
  two-server layout needs `GMEM=0.45` per boot (0.92 is the single-server
  default); without the caps the second boot stays under 28 GiB free.
- The three new tests fail on the pre-fix lease (`3 failed, 37 deselected,
  121 s`, `subprocess.TimeoutExpired` per case); the slot module is now 40
  passed and the five-file gate 74 passed / 5 skipped (119.5 s).

## PR contract: change class, gates and the a2 run plan (CPU-only prep, 2026-11-25)

### Base and head, and what "before" can serve

- **Head**: `bae1b21db0`, the merge of `41ef6ba1e0` into the branch tip
  `43a19980d2` (26 commits on top of the base).  Against the base
  (`41ef6ba1e0..HEAD`) the change set is 23 files added + 7 modified —
  `vllm/models/inkling2_loop/amd/{glue,loader,model,moe}.py`, `tools/`, `docs/`,
  `tests/` — and **no csrc/CMake/setup change**; the csrc in the branch history
  comes from the base through the merge.
- **Base**: `41ef6ba1e0` (`origin/rdna_extras` as of 2026-11-25, PR #210 LoopCD +
  TP1 decode perf).  The branch was **merged** (not rebased) up to it in
  `bae1b21db0`; the PR base is now `41ef6ba1e0` and this contract's base with it.
  The three merge conflicts were resolution-only: the glue workspace constructor
  keeps both the HI2 flag and the LoopCD tap dtype, `_prepare_serving` passes both
  keywords, and the test-tethering allowlist keeps both test sets.  The merged
  tree's delta against `41ef6ba1e0` in `amd/{glue,loader,model,moe}.py` is still
  only the HI2 additions (1148+/7-, no deletion of the merged code).
  `41ef6ba1e0` is the first `rdna_extras` commit that changes `csrc` in this
  branch's history (5 TUs: `inkling2_attn/dspark/glue`, `moe_lowp_dot2`,
  `torch_bindings.cpp`), so the box `_rocm_C` was rebuilt from the merged tree
  (~10 min CPU under `build.lock`) and copied over the tree's; the a2 numbers in
  this receipt were measured on the **pre-merge** tree and stay exploratory.
- The **stock Inkling-2-Mini path is in the base**: `vllm/models/inkling2_loop/`,
  the `Inkling2LoopForConditionalGeneration` registry entry and the
  `inkling2-mini-tp{1,2,4}.env` recipes all exist at `41ef6ba1e0` (as at the cut
  `1fdcb3a66e`; introduced by
  `7eeb0da81b` and merged into `rdna_extras` before the cut; the draft PR #195 is
  not needed to serve it).  The base tree has `~/hfcache/inkling2-mini` with
  `inkling2-mini-tp2.env` (the stock-path boot is the receipt's XA boot and has not
  been run on the base yet); that recipe differs from head only in a comment.  So
  **base-vs-head gates on the stock model are definable and owed**.
- The **HI2 path is not in the base**: base cannot load an `rdna_hint_flex` (HI2)
  directory at all.  Therefore the base-vs-head token-exact, quality-floor and ABBA
  gates are **not definable for the HI2 model**; the definable substitutes are below
  and a reviewer must accept them.

**Callers and platforms reached.** The HI2 branch is taken only when the model's
`quantization_config.quant_method` is `rdna_hint_flex`:
`loader.read_expert_formats()` then returns `"hi2"` per layer, the MoE is
constructed with that format, and `moe.forward_hip` / the eager reference call
`_forward_*_hi2`, which begin with `_require_gfx10x()`.  `glue.py` adds the HI2
workspace fields only for that model and `model.py` wires the loader on the same
config.  So: gfx1030 (run below), every other ROCm arch (the path refuses with a
clear error; not run), CUDA (stock path not changed in csrc; not run), CPU (the
pure-torch HI2 reference runs in the CPU tests).  The stock NVFP4 route through the
same four files keeps its operations; the stock base-vs-head token-exact gate is
what measures that it did not change numerically.

### Change class (rows of `.github/CONTRIBUTING.md`)

| Row | Evidence here |
| --- | --- |
| New model | a new checkpoint format/path (HI2 3.5 bpw) inside the existing Inkling-2-Mini model; full set owed **on that model** |
| Shared Python code | `amd/{glue,loader,model,moe}.py` are reached by every platform that serves Inkling2Loop (CUDA, ROCm, CPU reference); CPU unit tests plus a gfx1030 receipt, callers/platforms listed |
| Code gated to one arch | the HI2 expert dispatch calls `_require_gfx10x()` and raises on any other arch (both the eager and serving paths), so no non-gfx10x device reaches the gfx1030 kernels; the stock path never enters that branch; no csrc change against the merged base (`41ef6ba1e0`), so `archdiff` is empty (below) |
| Lossy feature (INT2/INT4 experts) | opt-in by model directory + `inkling2-hi2-tp2.env`; needs the quality gate + task-eval A/B, and **maintainer sign-off** for the recipe |
| New/changed kernel | no new kernel, but a new dispatch of the existing grouped GEMMs: the fp64-reference kernel test is owed and ran on a2 (5/5) |

### Gate table: exact command, a2 time, comparison

Durations are estimates ([INFERENCE]) anchored on the measurements above (boot
~1-3 min, 32 KL windows 5.5 min, the contract's own note that its longest speed cell
is ~10 min; the script's plan is the 6-boot matrix printed above).

| Gate | Command | a2 time | Compares |
| --- | --- | --- | --- |
| pre-commit + CPU tests | `pre-commit run --files $(git diff --name-only 1fdcb3a66e..HEAD)`; the 5-file pytest gate | 0 | done: all hooks pass; 71 passed, 5 skipped |
| kernel vs reference | `HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 PYTHONPATH=$TREE HI2_TINY_DIR=$RT/tiny/hi2 $PY -m pytest tests/models/inkling2_loop/test_hi2_kernel_gpu.py -q -s -rA` (the box driver `kernel_test_a2.sh` adds the lease, PCI and 28-GiB gates) | ~1 min | HIP dispatch vs the fp64/dequant reference, M=1/8/64, TP=2 sums, zero allocation; ran 5/5 on a2 |
| archdiff | `archdiff_report.py plan` on clean `git archive` trees (command in the archdiff section) | 0 GPU | csrc sources: `changed_files []`, `units []`, `src.diff` 0 bytes; `archdiff.sh` has no TU to build |
| token-exact, stock model | `MODE=abba QUALITY=1 pr_receipt.sh` with `RECIPE=inkling2-mini-tp2 MODEL=$STOCK`; boots XA/XB | ~10 min | base vs head greedy hashes (1024:128 x4 prompts, 16384:128 x2) under `VLLM_RDNA2_MOE_DETERMINISTIC=1`, plus concurrent==solo (0 FAIL); INCONCLUSIVE if base does not repeat itself |
| quality, stock model | same run, `QUALITY=1` (boots A1/A2/B1/B2) | ~20-40 min | head KL/top-1 vs the base-vs-base bootstrap floor; PPL rise <= 2 %; greedy probes; concurrency 0 FAIL |
| speed ABBA, stock model | same run, `MODE=abba` | ~1.5-3 h | the 8 contract cells (1024:512 x c1/4/8, 16384:1024 x c1/4/8, 32768:2048 x c1/4), per request and aggregate; verdict by the ABBA rule, reported not gated |
| overlap + stall | rides every speed boot (`STALL=1`) | +5-15 min | c=4/8 overlap rule and the hard stall gate (no starved decoder, tokens during every injected prefill) |
| HI2 model full set | `BASE_TREE=$TREE HEAD_TREE=$TREE RECIPE=inkling2-hi2-tp2 MODEL=$HI2 MODE=abba QUALITY=1` (A/A) | ~2-4 h | the HI2 path against itself: token-exact A/A determinism, HI2 KL noise floor, HI2 speed (direction only, not a code claim), overlap/stall hard gates; **not** base-vs-head |
| HI2 quality vs stock | `kl_probe.py` on mixed/wiki/pydocs + `quality_ab.py` collect on both servers + `compare --max-ppl-rel 0.02` | ~10-30 min | HI2 vs stock: KL/top-1 (measured 0.09340 / 0.93090 on mixed) and PPL ratio (reference 1.00951) |
| task-eval A/B | `agentic_bench.py --hi2 $HOME/w4a8_runs/inkling-hi2/tree run --tasks gsm8k,toolcall --limit 120 --workers 8`, then `tfkld score --ref-results <stock.json>`; both servers up | ~30-90 min | accuracy and task-level logprob gap, stock vs HI2 (the harness the contract's row asks for) |

Total for the whole PR: **~5-9 h of a2**, i.e. two windows, not one: the stock
contract receipt (~2-4 h) and the HI2-model set (~2-4 h plus the task eval).

### Gates that cannot be satisfied as written

1. **Base-vs-head token-exact / quality-floor / ABBA for the HI2 model** — the base
   cannot load the directory.  Substitutes: the fp64 kernel test, the HI2 A/A
   receipt, the HI2-vs-stock KL/PPL gates and the task-eval A/B.  A reviewer has to
   accept the substitution; the PR must label these cells as such.
2. **Task-eval A/B and maintainer sign-off** — the harness exists (`agentic_bench.py`
   over the HI2 checkout's `bench/` suite) but has not been run; the sign-off is a
   human decision and cannot be produced by any command.
3. **The other arches (gfx90a/gfx942/gfx950/gfx1100/gfx1200, CUDA, CPU serving)** —
   no such hardware here.  The proof is indirect: no csrc change against the
   merged base (`41ef6ba1e0..HEAD`; archdiff `units []`) and the HI2 branch behind
   `on_gfx10x()`; the stock path is exercised only on gfx1030.  This is "not run",
   not a pass.
4. **Kernel shape coverage** — the test covers M=1/8/64 and TP=1/2; the dispatcher
   can send any per-step M up to `MAXBAT` (the dominant grouped-GEMM shapes are
   covered, but "every shape the dispatcher sends" is not enumerated).  Cheap to
   extend on one GPU: add 128/256/320 to the parametrization and rerun.
5. **The exploratory bench is not the speed receipt** — it ran at `MAXLEN=8192
   MAXBAT=256 SEQS=1 GMEM=0.45 DSPARK=0` with both servers resident; the contract
   requires the production recipe (stock: DSpark on, `MAXLEN=720000`, `MAXBAT=1024`,
   `SEQS=8`, `GMEM=0.92`; HI2: `MAXLEN=1048576`, `MAXBAT=320`, `SEQS=8`,
   `GMEM=0.92`, MTP off).  Do not quote the 1.833x/0.549x numbers as the receipt.

### Exact GPU sequence (lease -> gates -> release)

The two-slot protocol applies (`AGENTS.md` 2414-2416): per-slot quiet dirs
`~/w4a8_runs/lanes/quiet-a2`, a single `~/w4a8_runs/boot.lock` for boots, holders in
`~/w4a8_runs/lanes/lease`, `MEASURE_LOCK=` empty.  `pr_receipt.sh` waits out other
agents' quiet windows under `QUIET_DIR` and runs its measured cells inside its own
window (`QUIET_WIN_MAX=240`, renewed per cell); it never takes the lease itself.

```bash
# 0) CPU prep: one build for both trees, markers (no GPU)
TREE=$HOME/w4a8_runs/inkling-hi2-runtime/tree
BASE=$HOME/w4a8_runs/inkling-hi2-runtime/base-rdna_extras   # 41ef6ba1e0
RT=$HOME/w4a8_runs/inkling-hi2-runtime
V=$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340
PY=$V/bin/python
STOCK=$HOME/hfcache/inkling2-mini
HI2=$RT/serve-bpw3.5
mkdir -p "$RT/pr-receipt-stock" "$RT/pr-receipt-hi2" "$RT/logs"
printf '%s\n' bae1b21db05f550733e600a10ff14b03bfc7dafe > "$TREE/.sha"
printf '%s\n' 41ef6ba1e066571a3346c6f7d1f6c38419fb9c29 > "$BASE/.sha"
# csrc is byte-identical to the merged base `41ef6ba1e0` (the branch's delta
# against it is Python-only); the only directory differences are the generated
# paged-attn units + their stamp, which CMake consumes from the stamp.  The
# attached `_rocm_C` is the one rebuilt from the merged tree's csrc.
diff -qr "$BASE/csrc" "$TREE/csrc"
bash "$TREE/tools/rdna/port_v031/attach_so.sh" "$TREE" "$BASE"

# 1) stock-model contract gate (base vs head), one driver owns the a2 lease
cd "$TREE"
export OWNER=HI2InklingRuntime SLOT=a2 EXPECT_PCI_BUSES=d1,d4 \
  LEASE_DIR=$HOME/w4a8_runs/lanes/lease \
  QUIET_DIR=$HOME/w4a8_runs/lanes/quiet-a2 QUIET_OWN=$OWNER QUIET_SKIP=0 \
  BOOT_LOCK=$HOME/w4a8_runs/boot.lock \
  BASE_TREE=$BASE HEAD_TREE=$TREE BASE_VENV=$V HEAD_VENV=$V \
  RECIPE=inkling2-mini-tp2 MODEL=$STOCK GPUS=0,1 MODE=abba QUALITY=1 STALL=1 \
  CHANGE_CLASS="new model path (HI2 quantized Inkling-2-Mini) on shared Python; no csrc" \
  REACHED="every platform that serves Inkling2Loop; the HI2 dispatch is on_gfx10x-gated; the stock NVFP4 path is unchanged" \
  NOT_RUN="gfx90a, gfx942, gfx950, gfx1100, gfx1200; CUDA/CPU serving" \
  OUT=$RT/pr-receipt-stock
bash -c '
  set -u
  source tools/inkling_hi2/lease.sh
  lease_acquire || exit 2              # no foreign holder, >=28 GiB free per GPU
  trap lease_release EXIT TERM INT
  bash tools/rdna/pr_receipt.sh        # 6 boots; exit 0 = token-exact+quality+overlap+stall pass
'

# 2) HI2-model full set, A/A on the head tree (base cannot load HI2)
BASE_TREE=$TREE HEAD_TREE=$TREE RECIPE=inkling2-hi2-tp2 MODEL=$HI2 \
  OUT=$RT/pr-receipt-hi2 \
  bash -c 'source tools/inkling_hi2/lease.sh; lease_acquire || exit 2; trap lease_release EXIT TERM INT; bash tools/rdna/pr_receipt.sh'

export RT STOCK HI2 PY
# 3) HI2 vs stock: KL (wiki, pydocs), PPL and the task eval; the driver owns the lease
bash -c '
  set -u
  source tools/inkling_hi2/lease.sh
  lease_acquire || exit 2
  trap lease_release EXIT TERM INT
  MAXLEN=8192 MAXBAT=256 SEQS=1 GMEM=0.45 MODEL=$STOCK \
    SERVER_PID_FILE=$RT/server-stock.pid PORT=18231 \
    bash tools/inkling_hi2/serve.sh > $RT/logs/stock.log 2>&1 &
  for i in $(seq 1 160); do curl -sf http://127.0.0.1:18231/health >/dev/null && break; sleep 15; done
  STOCK_URL=http://127.0.0.1:18231 RUN_KL=1 KEEP=1 MODEL=$HI2 PORT=18230 \
    SERVER_PID_FILE=$RT/server-hi2.pid \
    MAXLEN=8192 MAXBAT=256 SEQS=1 GMEM=0.45 bash tools/inkling_hi2/smoke.sh
  # the two extra held-out sets (17 windows each); keep the window and the lock fresh per probe
  for set in wiki pydocs; do
    touch $HOME/w4a8_runs/lanes/quiet-a2/inkling-hi2-kl
    flock -n $HOME/w4a8_runs/boot.lock true || { echo "a boot started; stop and retry"; exit 1; }
    $PY tools/inkling_hi2/kl_probe.py --hi2-url http://127.0.0.1:18230 \
      --stock-url http://127.0.0.1:18231 --windows 17 --topk 100 --max-kl 0.15 \
      --ids $HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_${set}_ids.npy \
      --out $RT/kl_${set}.json
    rm -f $HOME/w4a8_runs/lanes/quiet-a2/inkling-hi2-kl
  done
  $PY tools/rdna/port_v031/quality_ab.py collect --url http://127.0.0.1:18231 \
    --model inkling2-mini --out $RT/ppl_stock.json
  $PY tools/rdna/port_v031/quality_ab.py collect --url http://127.0.0.1:18230 \
    --model inkling2-hi2 --out $RT/ppl_hi2.json
  $PY tools/rdna/port_v031/quality_ab.py compare $RT/ppl_stock.json $RT/ppl_hi2.json --max-ppl-rel 0.02
  AG=tools/rdna/port_v031/agentic_bench.py
  $PY $AG --hi2 $HOME/w4a8_runs/inkling-hi2/tree run --url http://127.0.0.1:18231 \
    --model inkling2-mini --tasks gsm8k,toolcall --limit 120 --workers 8 --out $RT/task_stock.json
  $PY $AG --hi2 $HOME/w4a8_runs/inkling-hi2/tree run --url http://127.0.0.1:18230 \
    --model inkling2-hi2 --tasks gsm8k,toolcall --limit 120 --workers 8 --out $RT/task_hi2.json
  $PY $AG --hi2 $HOME/w4a8_runs/inkling-hi2/tree tfkld score --url http://127.0.0.1:18230 \
    --model inkling2-hi2 --ref-results $RT/task_stock.json --out $RT/task_lp.json
  kill $(cat $RT/server-hi2.pid $RT/server-stock.pid)
'

# 4) release: the driver traps remove the holder; verify no HI2InklingRuntime.a2 remains
ls $HOME/w4a8_runs/lanes/lease
```

Every measured step keeps its own window: `smoke.sh`/`kl_probe.py` use
`lanes/quiet-a2/inkling-hi2-smoke` (re-touched every 5 min, <=15 min) and re-check
`flock -n boot.lock` before each cell; `pr_receipt.sh` uses
`lanes/quiet-a2/$OWNER` and its own `QUIET_WIN_MAX=240` renewal.  A KFD
`evicted_ms` jump or a thermal sample >= 95 C invalidates the window (both record
in the JSONs/markdown and fail the run).

## What still needs a GPU (ordered; commands in the PR-contract section)

Already run on a2 (exploratory, table above): kernel test 5/5, the HI2 boot with
FULL_AND_PIECEWISE, the greedy/capsule smoke, the 32-window KL (0.09340 / 0.93090),
the bench cells and a 19-minute c=4 serving run.  Remaining, in order:

1. CPU prep: base tree (`41ef6ba1e0`, the merged base) with the attached `_rocm_C`
   rebuilt from the merge (csrc is byte-identical to the base) and the `.sha`
   markers (`attach_so.sh`; only the generated paged-attn units + stamp differ).
2. **Stock contract receipt** (base vs head, `inkling2-mini-tp2`, production recipe,
   `MODE=abba QUALITY=1`): token-exact + quality + speed ABBA + overlap/stall.
   Exit 0 is the gate; the speed verdicts are reported.  ~2-4 h.
3. **HI2 model A/A receipt** (`inkling2-hi2-tp2`, production recipe, `MODE=abba
   QUALITY=1`, the same tree twice): determinism, HI2 noise floor, HI2 speed and
   overlap/stall.  ~2-4 h.
4. **HI2 vs stock quality** on the remaining held-out sets
   (`heldout_wiki_ids.npy`, `heldout_pydocs_ids.npy`; gates KL <= 0.15, top-1 >=
   0.90) and `quality_ab.py compare --max-ppl-rel 0.02` (reference PPL ratio
   1.00951); record PPL and the skipped share here.  ~10-20 min.
5. **Task-eval A/B** (gsm8k, toolcall; `agentic_bench.py` + `tfkld score`) against
   the stock server, plus the maintainer sign-off for `inkling2-hi2-tp2.env`.
   ~30-90 min and a human decision.
6. Kernel shape extension (M=128/256/320) if review wants the full dispatcher
   range, then rerun the kernel test (~5 min).
7. Vision and MTP/DSpark are outside the language-only recipe (`MTP=0`,
   `--language-model-only`); a separate recipe/PR would need its own gates.

## Known limits of this receipt

- The a2 measurements are single exploratory runs at kit settings (`MAXLEN=8192
  MAXBAT=256 SEQS=1 GMEM=0.45 DSPARK=0`, both servers resident), not the contract's
  production-recipe receipt and not a speed claim; the KL is a top-k truncated
  fp16-path measurement against a float32 full-logit reference.
- They predate the merge to `41ef6ba1e0` (`bae1b21db0`): the HI2 kernels' sources are
  unchanged (`archdiff units []`) and the branch's csrc equals the base's, but the
  merged tree has not been booted.
- The 5 GPU tests are skipped in the CPU gate; a green CPU run says nothing about
  the HIP kernels.
- The HI2 model has no base-vs-head token-exact/quality/speed gate (the base cannot
  load it); the substitutes are listed above and need reviewer acceptance.
- Kernel coverage is M=1/8/64 with TP=1/2, not every per-step M the dispatcher can
  send.
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

## 2026-11-25 addendum: DSpark, vision, MTP wiring (CPU only, no GPU run)

Trigger: the final card needs DSpark on and the vision tower served; the kit
forced `DSPARK=0` and `{"image":0}`.  Findings and the resulting change:

- Both blockers were in the kit, not the model.  The draft reads the *draft's*
  quantization config (`vllm/model_executor/models/utils.py:1149` via
  `vllm/v1/worker/gpu/spec_decode/dspark/utils.py:117`), so an HI2 target leaves
  it unquantized; it loads the stock bf16 `mtp.safetensors` (54 BF16 tensors,
  `amd/dspark.py:526/530/652`, dtype enforced on load) from the same directory
  (`allow_patterns_overrides`).  `prep_serve_dir.sh` already links that file.
  The vision tower is the registered multimodal class
  (`amd/multimodal.py:13/145-161`) with native HIP ops
  (`csrc/rocm/torch_bindings.cpp:1059-1063`, `CMakeLists.txt:1828`); the head
  tree's built `_rocm_C.abi3.so` registers 44 rocm ops including
  `inkling2_vit_gather_patches` (checked on the box; `~/w4a8_runs/
  inkling-hi2-runtime/tree`, not the older `tree-cp` control copy, which lacks
  them).  The 422 `model.visual.*` tensors are bf16 in the quantizer output.
- Changes: `serve.sh` gained `DSPARK` (spec block via the mini recipe, 720000/
  1024 defaults), `VISION` (drops `--language-model-only`, allows one image) and
  a `RECIPE` override; the HI2 recipe owns the same branches; `validate_dir.py`
  checks the vision census and (with `--require-mtp`) the MTP file, including
  bf16; `check_real_dir.py` gained `--vision` (bit-exact tower compare) and
  `--mtp`; `mtp_probe.py` and `vision_probe.py` are the GPU gates
  (README steps 3b and 8); `test_hi2_vision_mtp.py` covers all of it on CPU.
- The recipe now defaults to `DSPARK=1`; the first boot stays
  `DSPARK=0 VISION=0` (README step 3).  Nothing here changes the quantizer: the
  MTP file is the stock one and the vision tower is carried bit-exact.

Commands (CPU; `$TREE` = this worktree, `$V` = the box venv):

```bash
# the new tests with the rest of the CPU gate (counts below are from the run)
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
  PYTHONPATH=$H:$TREE:$HOME/w4a8_runs/pydeps/tfm-main $V -m pytest \
    $TREE/tests/models/inkling2_loop/test_hi2_loader.py \
    $TREE/tests/models/inkling2_loop/test_hi2_slots.py \
    $TREE/tests/models/inkling2_loop/test_hi2_vision_mtp.py \
    $TREE/tests/models/inkling2_loop/test_loader.py \
    $TREE/tests/models/inkling2_loop/test_hi2_kernel_gpu.py \
    $TREE/tests/models/inkling2_loop/test_hi2_reference.py -q
# the serve directory (prep_serve_dir.sh output; the raw quantizer directory has
# no MTP file), now with the vision and MTP comparisons
REQUIRE_MTP=1 bash $TREE/tools/inkling_hi2/prep_serve_dir.sh \
  ~/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5 \
  ~/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
  PYTHONPATH=$TREE $V $TREE/tools/inkling_hi2/check_real_dir.py \
    --model ~/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 \
    --stock ~/hfcache/inkling2-mini --vision --mtp
# the prepared serve directory (MTP required for the DSpark boots)
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
  PYTHONPATH=$TREE $V $TREE/tools/inkling_hi2/validate_dir.py \
    --model ~/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 --require-mtp
# the recipe's two branches, no GPU (assemble only)
PRINT=1 DSPARK=1 bash $TREE/tools/rdna/serve_rdna.sh RECIPE=inkling2-hi2-tp2 \
  MODEL=<serve dir> VENV=$V HIP_VISIBLE_DEVICES=0,1   # --speculative-config ...; max-model-len 720000
PRINT=1 DSPARK=0 bash $TREE/tools/rdna/serve_rdna.sh RECIPE=inkling2-hi2-tp2 \
  MODEL=<serve dir> VENV=$V HIP_VISIBLE_DEVICES=0,1   # max-model-len 1048576, no spec args
```

Run of 2026-11-25 on the box (CPU only, head tree `~/w4a8_runs/inkling-hi2-runtime/
tree`): the six-file CPU gate **90 passed, 5 skipped in 419 s**;
`check_real_dir.py --model <serve dir> --stock <stock> --vision --mtp` loaded
1125 text tensors with 0 missing parameters and 480 expert buffers, the lm_head's
first 200058 rows equal the stock unembed, 643 dense text tensors, **422/422
vision tensors** and **54/54 MTP tensors** bit-equal, then `PASS`;
`validate_dir.py --model <serve dir> --require-mtp` printed the vision census
(422 BF16 in `model-vision.safetensors`), the MTP census and `VALID`;
`mutants_launchers.py` killed **16/16** mutants (including the boot-lock mutant,
whose pattern now follows the `VISION` branch).  Two findings were real and
fixed: `compare_mtp` only checked the dtype of the served file (now both files),
and the `validate_dir` census used a stale `vision_bytes` name after the
refactor.  GPU numbers remain unmeasured (all GPUs leased).

## 2026-11-25 addendum: one-shot final-card driver (CPU proof)

`tools/inkling_hi2/validate_final_a2.sh` runs the ordered card as one detached,
resumable, stop-on-first-failure process: one lease, one quiet/boot.lock
discipline, every boot through `serve.sh`, per-step `STATUS.log`/JSON and a
`summary.md` table.  Steps: `kernel` (own lease window) -> `serve_dir` (CPU) ->
`smoke` -> `kl` (mixed gated at KL 0.15; long 8k/16k recorded with `--min-top1 0`)
-> `bench` -> `eval` (main/agentic-probe/vision on the a2 eval tree) -> `vision`
-> `dspark` (spec pair, 16384, GMEM 0.45 each) -> `boot720` (single HI2, 720000,
GMEM 0.92; `/v1/models.max_model_len` must be 720000).  Vision runs inside the
pair boot instead of after DSpark so a 720k KV failure cannot discard the vision
evidence; that is the only order change.  `DRY_RUN=1` assembles and dry-runs
every GPU argv on CPU without taking the lease.

```bash
TREE=$HOME/w4a8_runs/inkling-hi2-runtime/tree
MODEL=$HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5
bash $TREE/tools/inkling_hi2/validate_final_a2.sh
# prints the pid; then: tail -f $RUN/STATUS.log ; cat $RUN/summary.md
```

CPU gates on the box (no GPU in the environment):

```bash
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$HOME/w4a8_runs/pydeps/tfm-main \
$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340/bin/python -m pytest \
  tests/models/inkling2_loop/test_hi2_kernel_gpu.py \
  tests/models/inkling2_loop/test_hi2_reference.py \
  tests/models/inkling2_loop/test_hi2_loader.py \
  tests/models/inkling2_loop/test_hi2_slots.py \
  tests/models/inkling2_loop/test_hi2_vision_mtp.py \
  tests/models/inkling2_loop/test_hi2_final_driver.py -q
# 88 passed, 10 skipped (the GPU tests), 15 warnings in 212.21s (head 5d36945e5c)
```

`test_hi2_final_driver.py` is hermetic (stub tools; no GPU, no /tmp): dry-run
takes no lease, the disk guard stops before any dry-run, a PASSed step is skipped
on resume, and the first failure writes `| serve_dir | FAIL |` and stops.
A real `DRY_RUN=1` on the box against the 3.5 bpw directory passed in 30 s
(and again as `dryrun-2` after the bench/boot changes, RC=0)
(RC=0, no lease): the kernel-test collection, all five serve argvs, the four
probes' --help and the eval CLI --help ran on CPU; results under
`final/dryrun-1/`.  `tools/inkling_hi2/mutants_final.py` killed **5/5** mutants (disk guard off,
failed step not fatal, resume ignored, dry-run skips the tool help, wrong solo
bench flags); the bench test drives the pair and solo arms against fake health
servers, a fake bench and a fake serve and asserts the three calls with
`--decode-tokens 128`.  Every GPU
number and the 720k boot remain unmeasured because all GPUs were leased.

### 2026-11-25 fix: nested lease deadlock (a2 card run failed at smoke)

The first real run of the driver failed at `smoke` after 31 min: the driver's
kernel window and then every boot under it acquired the a2 lease; a boot that
had written its own `a2.request` and then had to wait on the VRAM gate queued
behind that same request (`our own a2.request is live (pid ...); queueing`)
until the 1800 s timeout.  The fix keeps one lease for the whole run: after the
kernel step the driver calls `lease_acquire` once, and every boot it starts gets
`LEASE_INHERIT=1`.  Inherit mode requires the slot holder to be ours, its pid
alive, no request pending, and the free gate the owner recorded at acquisition
(`<owner>.<slot>.gate`) to be >= 28 GiB; it writes no request/holder and its
release is a no-op, so a child cannot drop the run's lease.  A foreign or stale
holder falls back to the normal refuse/queue path.  A request written by the
same process no longer queues behind itself.  CPU proof: four lease tests (own holder,
foreign refusal, release no-op, self-request) plus the five driver tests on the
fake sysfs; box gate 88 passed, 10 skipped (212.21 s) and mutants 8/8 killed
(incl. unconditional inherit, release-deletes-holder and self-queue).

### 2026-11-25 fix: pair health gate (card1 smoke teardown killed the pair)

The card1 smoke PASSed, then `kl` failed in 5 s on a closed port: `smoke.sh`
stops its own HI2 (18230) when it exits, the driver's `ensure_pair` saw both
ports still answering during the teardown window, returned without booting, and
the KL client hit the dead server (`URLError` traceback).  Two changes:

- `smoke` is self-contained: the driver boots its own `smoke_stock` on 18231,
  `smoke.sh` boots/stops its own `smoke-hi2` on 18230, and both are stopped
  (health-waited) before the step returns.  Nothing the next steps use is booted
  by smoke.
- `kl`, `eval`, `vision` and the `bench` pair arm start with `require_pair`: both
  ports healthy, or one restart of a fresh pair (`restart_pair`: stop our
  stock/hi2/smoke leftovers, boot stock 18231 + HI2 18230 with the driver's own
  pidfiles, inherit mode, health-waited).  If the pair still is not healthy the
  step fails as `pair-unhealthy` before any Python client runs, so the step log
  has no traceback.  Every leftover of ours is stopped at teardown (`stop_all`
  covers `smoke_stock`/`smoke-hi2`).

CPU proof (box, CPU-only): `test_hi2_final_driver.py` **11 passed** (189 s),
including `test_pair_restarted_once_after_smoke_tears_it_down` (dead pair ->
one restart -> the probe runs against both health-checked URLs) and
`test_pair_unhealthy_fails_cleanly_without_traceback` (restart fails -> exit 1,
`pair-unhealthy`, no `Traceback`, the probe is never called).
`tools/inkling_hi2/mutants_final.py`: baseline 11 passed, **11/11 killed**
(the three new: gate ignored, restart twice, pair-unhealthy ignored), 393 s.
`ruff check` and `ruff format --check` clean on the two Python files.  The real
GPU steps (smoke, kl, eval, vision, bench, dspark, boot720) remain unmeasured:
a2/b4 are still leased by other jobs; the ordered checklist is unchanged in
`tools/inkling_hi2/README.md`.
