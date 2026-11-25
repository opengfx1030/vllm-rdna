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
together; `serve.sh` and `smoke.sh` clear `ROCR_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` (a stale filter must not hide or add cards), then
`tools/inkling_hi2/slot.py` enumerates the GPUs with exactly the server's
environment (`HIP_VISIBLE_DEVICES` only) and refuses a moved bus, a missing or extra
device, a non-V620 or a W7800 before the boot and before every measured cell. The
HTTP clients (`kl_probe.py`, `bench.py`) cannot see the GPUs and check the declared
env pair against the same table, recording the slot in their JSON.

As of 2026-11-24 the user approved both a2 cards (HIP 0,1) for this work; the
AGENTS.md table above is authoritative (a2 = HIP 0,1 / PCI `d1 d4`, b4 = HIP
2,3,4,5 / TP4). An earlier note in this file that read those numbers the other way
is superseded. This kit's holder is `HI2InklingRuntime.a2`; an older
`HI2Inkling.a2` is read as a foreign holder.

Every GPU command carries `HIP_VISIBLE_DEVICES=<leased HIP list>` on its own command
line; CPU-only commands use
`env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= <cmd>`.
Leases live in `~/w4a8_runs/lanes/lease/README.md` (holder files `<owner>.a2` /
`<owner>.b4`, requests, release trap; kill the holder's `sleep` on release). This
work's holder is `HI2InklingRuntime.a2` (`OWNER`, default `HI2InklingRuntime`); the
older `HI2Inkling` name is read only as a foreign holder. `serve.sh` writes the typed
request first, then the holder, and waits for 28 GiB free; with a foreign holder it
writes a request and exits (never deletes the file). `smoke.sh` still requires this
kit's holder. Both look at the fair queue (an older `.request`) before the boot and
at the slot's quiet file before every measured cell. Nested boots started by the
one-shot driver pass `LEASE_INHERIT=1`: they reuse the driver's live holder only
when it is ours, alive and the free gate it recorded at acquisition is >= 28 GiB;
they write no request and never release the holder (the driver does, once). A
foreign holder (or a missing/stale one) still refuses/queues as above, and a
request written by the same process no longer queues behind itself.

Protocol the scripts enforce (and the CPU tests check, see section 1):

* **Lease**: request -> holder -> VRAM threshold (28 GiB) -> trap release; a foreign
  holder, a requester ahead in the queue, or a live holder is refused. The lease
  README in `~/w4a8_runs/lanes/` is authoritative; this session's approval is
  recorded in `~/w4a8_runs/inkling-hi2-runtime/STATUS.md` with the exact commands
  (a2 = HIP 0,1 = PCI `d1 d4`).
* **Quiet**: `~/w4a8_runs/lanes/quiet-a2/inkling-hi2-smoke`, touched every 5 min for
  up to 15 min; the boot waits until no file in the slot is newer than 6 min, then
  re-checks *under* the lock. A >15 min window is split or restarted; it is never
  silently shortened.
* **boot.lock**: `flock -w 900`; held from just before torch/GPU until `/health`
  (`READY_SECONDS=1800`), then released while the server keeps serving. `smoke.sh`
  re-checks it (non-blocking) before each measured cell.
* **Litmus**: the device check runs three times -- before the quiet wait, before
  the launch and after `/health` -- and once per cell in `smoke.sh`; the child
  server sees `HIP_VISIBLE_DEVICES` only (ROCR/CUDA filters cleared), exactly the
  leased cards (V620, expected buses). The HTTP clients refuse `b4` and record the
  slot in their JSON.
* **Valid window**: junction temps sampled by `gpu_temps.py` (a 95 C sample aborts;
  a 90 C mean fails once the mean window is armed); a window with no pre-window KFD
  evictions or with any eviction during it (`kfd_evictions.py` against
  `SERVER_PID_FILE`) prints `INVALID` and exits non-zero. `bench.py` requires
  `--server-pid`.
* No measured window while a `docker buildx build` runs.

## 1. Preconditions (CPU only, no lease)

```bash
TREE=/path/to/vllm-inkling2-hi2-runtime
V=$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340/bin/python

# the new Python path imports against the tree's built _rocm_C
cd $TREE && env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
    PYTHONPATH=$TREE $V -c "import vllm._custom_ops, vllm.models.inkling2_loop.amd.moe"

# CPU gates: loader (23), the launcher/lease/window protocol (31), quantizer
# reference on the tiny HI2 directory (4; the fixture is built by the quantizer's
# own pipeline, ~3.7 min), legacy loader (6), the vision/MTP/recipe checks
# (test_hi2_vision_mtp.py, 19), plus the GPU kernel tests collected and skipped
# on CPU (5). H is the quantizer tree; it provides the `hi2` package the
# reference test imports.
H=$HOME/w4a8_runs/inkling-hi2/tree
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$H:$TREE:$HOME/w4a8_runs/pydeps/tfm-main $V -m pytest \
  tests/models/inkling2_loop/test_hi2_loader.py \
  tests/models/inkling2_loop/test_hi2_slots.py \
  tests/models/inkling2_loop/test_hi2_vision_mtp.py \
  tests/models/inkling2_loop/test_loader.py \
  tests/models/inkling2_loop/test_hi2_kernel_gpu.py \
  tests/models/inkling2_loop/test_hi2_reference.py -q
# 90 passed, 5 skipped (419 s, 2026-11-25); INKLING2_TINY_OUT=<dir>
# reuses the <dir>/stock + <dir>/hi2 pair built below instead of rebuilding it.
# The launcher test (37) needs no torch: it fakes the visible-device table (a fake
# torch under a fake venv), the stub HTTP server and the scripts, and covers the
# lease (request, holder, VRAM wait, trap release, foreign/live/queued refusals),
# quiet + boot.lock order, the three device checks, the b4 refusal, both thermal
# and both KFD window outcomes, the greedy/capsule probes and their failures
# (including smoke's non-zero exit), the KL parser (None at position 0 and omitted
# positions skipped), and the cells (capture + bench).

# mutant battery: the loader / name map / TP slice / reference tests must kill
# every mutant (21; exit 0). Use a scratch copy of the tree, not the checkout.
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$H:$HOME/w4a8_runs/pydeps/tfm-main INKLING2_TINY_OUT=$HOME/w4a8_runs/inkling-hi2-runtime/tiny \
$V $TREE/tools/inkling_hi2/mutants.py --tree <scratch copy>

# launcher mutant battery: must kill every mutant (16 as of 2026-11-25; exit 0,
# ~5 min). It copies tools/ and
# the launcher test into its own scratch dir; no PYTHONPATH or fixture needed.
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
$V $TREE/tools/inkling_hi2/mutants_launchers.py

# the serve directory (symlinks; the quantizer output is only read) with the
# stock bf16 MTP file required: DSpark loads it and the quantizer does not write it
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
REQUIRE_MTP=1 PYTHONPATH=$TREE $V $TREE/tools/inkling_hi2/prep_serve_dir.sh \
  $HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5 \
  $HOME/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5
# if the serve dir already exists add FORCE=1; MTP_STOCK overrides the stock file
# (default ~/hfcache/inkling2-mini/mtp.safetensors)

# the real quantizer output, loaded on CPU (the first-boot proxy): no GPU
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
PYTHONPATH=$TREE $V $TREE/tools/inkling_hi2/check_real_dir.py \
  --model $HOME/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 \
  --stock $HOME/hfcache/inkling2-mini --vision --mtp
# expected: 1125 text tensors, 0 missing parameters, 480 expert buffers,
# lm_head equals the stock unembed's first 200058 rows, 643 dense tensors
# bit-equal, vision tensors bit-equal to stock: 422/422, mtp.safetensors:
# 54 tensors bit-equal to stock: True.  The vision shard in the output is
# model-vision.safetensors; prep_serve_dir.sh links it and the stock MTP file,
# so the serve directory has both for VISION=1 and DSPARK=1.
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

### One-shot final-card run (no operator loop)

`tools/inkling_hi2/validate_final_a2.sh` is the whole checklist below in one
detached process: one lease, one quiet/boot.lock discipline, stop on the first
failure, resumable.  It writes `$RUN/STATUS.log` (every step), `$RUN/steps/*.json`
(rc, seconds, parsed numbers, log path) and `$RUN/summary.md` (the table to paste
into the receipt); a PASSed step is skipped on a rerun with the same `TAG`.

```bash
TREE=$HOME/w4a8_runs/inkling-hi2-runtime/tree   # this fork, synced
MODEL=$HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5
bash $TREE/tools/inkling_hi2/validate_final_a2.sh   # prints the pid and the tail command
```

Step order (all requested checks; grouped to keep the pair boot alive):
`kernel` (own lease window + tiny fixture) -> `serve_dir` (prep + validate +
check_real on CPU) -> `smoke` (self-contained: its own stock + its own HI2, both
stopped before it returns) -> `kl` (the first step on the run pair; mixed gated
at 0.15, long 8k/16k recorded) -> `eval` (two labels on the a2 tree) ->
`vision` -> `bench` -> `dspark` (spec pair: 16384, GMEM 0.45 each) -> `boot720`
(single HI2, 720000, GMEM 0.92; `/v1/models` must report 720000).  `bench` runs
last inside the pair boot and has TWO arms: the pair arm (both servers resident
at GMEM 0.45, the conservative number) and then a SOLO arm: stock is stopped and
HI2 is booted alone at GMEM 0.92 with the recipe's MAXBAT/MAXLEN.  Both arms use
`--decode-tokens 128` (less prefill contamination of a decode step) and both are
recorded in `steps/bench.json` with their flags; the card quotes the solo arm
(production runs one server) and footnotes the pair ratios.  `vision` before
`dspark`/`boot720` is deliberate: a 720k KV failure must not discard the vision
evidence.  Everything that shares the stock+HI2 pair runs under that one boot
(VISION=1, 32768, GMEM 0.45 each); all boots go through `serve.sh` (28 GiB lease
gate, quiet windows, boot.lock, HIP->PCI litmus).  After smoke the driver boots
one fresh stock+HI2 pair with its own pidfiles; every step that talks to the
pair starts with a health gate (`require_pair`): a dead port restarts the pair
once, and if it still does not come up the step fails as `pair-unhealthy`
instead of dying on a closed port mid-probe.  `vision` records the probe result
in `numbers.vision_probe` and, when the probe defects before a verdict, falls
back to the eval vision compare (`numbers.vision_eval`; comparable, >= 50 paired
items, non-inferior within its `mdd`); a probe verdict that failed a check stays
fatal.  `steps/kl.json` carries the three KL numbers (and a resumed run backfills
them from `steps/kl-*.json` if the first attempt left them empty).  ~4 h
(5 h budget).
Useful knobs: `DRY_RUN=1` (assemble and dry-run every GPU argv on CPU, no
lease), `ONLY=smoke,kl`, `RESUME=1` (default), `BOOT_WAIT`, `CTX`,
`GMEM_PAIR`, `KL_MIXED_WINDOWS`, `TAG`.  `DETACH=0` for a foreground/tmux run.
CPU proof: `tests/models/inkling2_loop/test_hi2_final_driver.py` (stubs, no GPU,
no /tmp; includes the pair health gate: dead pair -> one restart -> pass, and
restart-fails -> clean pair-unhealthy + no traceback) and
`tools/inkling_hi2/mutants_final.py` (kills the disk-guard-off,
failed-step-not-fatal, resume-ignored, dry-run-skips-tools, pair-gate-skipped,
pair-retried-twice and pair-unhealthy-ignored mutants).

## 2. Ordered GPU checklist

One lease of two V620s: the a2 slot (HIP 0,1; PCI `d1 d4`) from Main. b4
(HIP 2,3,4,5; `d9 dc df e2`) is the TP4 lease and is not used by this TP=2 kit.
Each command repeats `HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4` on its own
line; `serve.sh`/`smoke.sh` resolve the pair to the live PCI buses before the boot
and before the measured cells and refuse a moved or foreign device. Every step
prints a `RESULT ... PASS` or a numeric record; stop at the first failure and keep
the log.

0. **Take the a2 lease.** `serve.sh` queues on `a2.request` (trap armed first;
   another owner's request is never overwritten — it waits for that file to go,
   up to `LEASE_WAIT_SECONDS=1800`), waits until no other `<owner>.a2` holder
   exists and both cards have 28 GiB free, and reuses its own holder only while
   the holder's recorded pid is alive and the free-memory check still passes (a
   stale holder is dropped), then writes `HI2InklingRuntime.a2`. It exits non-zero
   on a busy `boot.lock` (retry after the other boot) or a foreign holder (inspect
   `ls ~/w4a8_runs/lanes/lease/`; never delete another agent's file). Check the box
   with `$V tools/rdna/port_v031/kfd_evictions.py` and
   `$V tools/rdna/mono/w4a8/gpu_temps.py --count 1` before starting; a boot must not
   run while another slot boots or a `docker buildx build` runs.

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
   The gate needs the tree's `_rocm_C` built with the four Inkling/HI2 ops **and**
   the venv's `_moe_C` (`moe_align_block_size`); without either the file skips
   (a skip is not a pass). The build of 2026-11-25 (5e2c740de3) has both.

2. **Stock reference server** (same lease, TP=2, language-only, 8192 context) so the
   probes have a baseline:

   ```bash
   MODEL=$HOME/hfcache/inkling2-mini SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   SERVER_PID_FILE=$HOME/w4a8_runs/inkling-hi2-runtime/server-stock.pid \
   VENV=$(dirname $(dirname $V)) PORT=18231 MAXLEN=8192 MAXBAT=256 SEQS=1 \
   GMEM=0.45 \
   bash $TREE/tools/inkling_hi2/serve.sh > ~/w4a8_runs/inkling-hi2-runtime/logs/stock.log 2>&1 &
   ```

   Verifies: the recipe boots an NVFP4 stock directory unchanged (no regression from
   the loader changes). Wait for `/health`; record "Model loading took" and the
   `max_model_len=8192`.

   **Two-server layout:** steps 2-5 keep the stock and the HI2 server resident at
   the same time, so each must cap its KV cache with `GMEM=0.45` (~13.5 GiB of
   the 30 GiB per card; `0.92` is the kit's single-server default). Without the
   caps the second boot never sees 28 GiB free per card, and its lease wait times
   out after `LEASE_WAIT_SECONDS`.

3. **HI2 server**: same command with `MODEL=$HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5`,
   `PORT=18230`, `SERVER_PID_FILE=$HOME/w4a8_runs/inkling-hi2-runtime/server-hi2.pid`
   and `GMEM=0.45` (both servers stay resident; see the step-2 note), log
   `logs/serve.log`; the env pair stays
   `SLOT=a2 HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4`. Verifies: all 24 layers
   find their `gate_up`/`down` stacks (`HI2 expert groups incomplete` and
   `unmapped ... tensor` must NOT appear); the engine reports the model memory
   (expect ~7.1-8.6 GiB/rank for the weights, versus 18.68 GiB for the stock NVFP4
   at TP=1); FULL_AND_PIECEWISE graph capture succeeds; `VLLM_INKLING2_PATH` is
   unset (native HIP path) and the log shows "Inkling-2 HIP serving path: workspace
   ...". If capture fails, rerun with `EAGER=1` and record the difference (FULL
   graphs are a production requirement; an eager-only boot is a finding).

3b. **DSpark / MTP probe** (the production card's speculative path). The draft is
   the stock bf16 `mtp.safetensors` that `prep_serve_dir.sh` links into the serve
   directory; its 3 layers and Markov head are not quantized and read the target's
   hidden-state taps, which `_forward_hip_hi2` writes like the stock path. Boot
   **two HI2 servers of the same directory** (spec and plain) with a small context
   so both fit without the stock server:

   ```bash
   MODEL=$HOME/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   SERVER_PID_FILE=$HOME/w4a8_runs/inkling-hi2-runtime/server-hi2-plain.pid \
   VENV=$(dirname $(dirname $V)) PORT=18231 MAXLEN=16384 MAXBAT=1024 SEQS=8 \
   GMEM=0.45 DSPARK=0 \
   bash $TREE/tools/inkling_hi2/serve.sh > ~/w4a8_runs/inkling-hi2-runtime/logs/hi2-plain.log 2>&1 &

   MODEL=$HOME/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   SERVER_PID_FILE=$HOME/w4a8_runs/inkling-hi2-runtime/server-hi2-dspark.pid \
   VENV=$(dirname $(dirname $V)) PORT=18230 MAXLEN=16384 MAXBAT=1024 SEQS=8 \
   GMEM=0.45 DSPARK=1 \
   bash $TREE/tools/inkling_hi2/serve.sh > ~/w4a8_runs/inkling-hi2-runtime/logs/hi2-dspark.log 2>&1 &

   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/mtp_probe.py \
     --spec-url http://127.0.0.1:18230 --plain-url http://127.0.0.1:18231 \
     --log ~/w4a8_runs/inkling-hi2-runtime/logs/hi2-dspark.log \
     --drop-batch 3 --json ~/w4a8_runs/inkling-hi2-runtime/mtp_probe.json
   ```

   The spec boot must log `Inkling-2 DSpark weights: bf16 -> fp16, ... 54 tensors`
   (no `mtp.safetensors` parse error, no `get_draft_quant_config` failure: the
   draft's config has no quantization and the HI2 quant config stays on the
   target) and `SpecDecoding metrics:` lines. Verifies, all hard checks in the
   probe: greedy outputs token-identical with and without DSpark on the 8
   code/chat/agentic prompts, deterministic on repeat, clean non-empty text; a
   mean acceptance length and per-position acceptance rates from the log (record
   them; the stock card's numbers are the reference, HI2's are unmeasured); decode
   tokens/s at c=1 (spec vs plain) and c=2; at `--drop-batch 3`
   `vllm:spec_decode_num_drafts_total` does not move and every request succeeds
   (`[[1,2,2],[3,8,0]]`). Exit 0 = all checks passed; keep `mtp_probe.json`.
   Stop both servers afterwards (send SIGTERM to the pid files; the lease trap
   releases the holder).

1. **Greedy smoke** with the HI2 server (it boots its own) and the stock server as
   the reference:

   ```bash
   MODEL=$HOME/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5 SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   VENV=$(dirname $(dirname $V)) STOCK_URL=http://127.0.0.1:18231 \
   bash $TREE/tools/inkling_hi2/smoke.sh
   ```

   Verifies: the smoke takes its own lease route through `serve.sh` (request,
   holder `HI2InklingRuntime.a2`, 28 GiB wait, quiet wait, boot lock until
   `/health`; stop the step-3 server first, the default port is the same); the
   pre-launch device check and a per-cell check see only the leased V620s;
   `/v1/models` name and context; the France probe runs `GREEDY_TOKENS=160`
   tokens (the model's reasoning prefix counts, so the city is accepted anywhere
   in the reasoning + answer text); with `STOCK_URL` set the same request runs on
   the stock server and the log prints both outputs and the first-divergence
   character position (set `STOCK_KL=0` to keep only that comparison and skip the
   held-out KL cell); the greedy capsule sends the same request twice and
   requires a non-empty deterministic repeat (it does not judge wording). A
   failed Python probe aborts `smoke.sh` with a non-zero exit before any
   measured-window PASS (the CPU tests assert this). The smoke re-touches
   `lanes/quiet-a2/inkling-hi2-smoke` every 5 min, re-checks the boot lock before
   each probe, and ends with `RESULT measured window PASS` (or `INVALID` plus
   exit 1) from the junction temperatures and the KFD eviction counters;
   `$RUN_DIR/logs/temps-*.csv` and the serve log keep the evidence. It refuses b4
   and a foreign holder.

2. **KL vs stock on the quantizer's held-out windows** (mixed set, first 32 of 96,
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
   prompt-token counts equal between the servers.  Positions are scored from 1
   (position 0 has no context and vLLM returns `None`); any position whose top-k
   entry a server omitted is skipped, printed as `skipped S/considered`, and a
   window with more than `--max-skipped-share 0.05` of them fails instead of
   silently scoring fewer positions.  The request uses `temperature=1` and raw
   post-softcap logprobs truncated to the top-k (no renormalisation); the first
   window prints the number of top entries each server returned (`... (requested
   100)`) and the JSON records them as `topk_entries` — fewer than requested means
   the served KL is truncated further, so the served number is an fp16-path top-100
   approximation of the quantizer's float32 full-logit eval: compare it within
   `--max-kl`, not for equality.  Repeat with
   `heldout_wiki_ids.npy` (17 windows, reference 0.0468) and
   `heldout_pydocs_ids.npy` (0.0305) — those are stricter relative checks than the
   mixed set. Both servers must have `--max-logprobs 128` (serve.sh passes it).
   A measured window is <= 15 min: split a long set into `--windows` chunks and
   re-touch the quiet file (and re-check the boot lock) between chunks; `smoke.sh`
   caps each measured probe at `CELL_TIMEOUT=900` s and dies at the cap (rerun with
   a smaller `--windows`), never silently shortening a probe.

3. **Bench cells**, same lease, stock then HI2:

   ```bash
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/bench.py --url http://127.0.0.1:18231 \
     --server-pid $(cat ~/w4a8_runs/inkling-hi2-runtime/server-stock.pid) \
     --label stock --log ~/w4a8_runs/inkling-hi2-runtime/logs/stock.log \
     --out ~/w4a8_runs/inkling-hi2-runtime/bench_stock.json
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/bench.py --url http://127.0.0.1:18230 \
     --server-pid $(cat ~/w4a8_runs/inkling-hi2-runtime/server-hi2.pid) \
     --label hi2 --log ~/w4a8_runs/inkling-hi2-runtime/logs/serve.log \
     --baseline ~/w4a8_runs/inkling-hi2-runtime/bench_stock.json \
     --out ~/w4a8_runs/inkling-hi2-runtime/bench_hi2.json
   ```

   Verifies: prefill prompt-tokens/s, decode wall per step at batch 1 and 4, and the
   engine's model-load/KV/workspace lines, plus the window record (max junction
   temperature, KFD evictions). `--server-pid` is required for a valid window: it
   locates the serve process's KFD context (evictions before/after) and lets the
   temperature watcher kill the server on a hot sample. Without it the window is
   unsampled and the run exits 3. An `INVALID` window (exit 3) is rerun, never
   recorded. No pass/fail on the numbers: record the ratios. Known
   expectation [inference]: the block-64 HI2 GEMM pads an 8-pair decode to 64 rows,
   so batch-1 decode can be slower than the stock NVFP4 kernel; the memory cell must
   show the HI2 weights smaller. A decode step slower than ~10x stock is a finding.

4. **Graph-mode A/B** (only if step 3 needed it): rerun step 4/5 with `EAGER=1`; the
   numbers from the eager boot are not the serving record, they bound the graph
   contribution.

5. **Production-context boot and the vision session** (single-server, `GMEM=0.92`):

   (a) The recipe default (DSpark on, 720000) — this is the context/KV check:

   ```bash
   MODEL=$HOME/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   SERVER_PID_FILE=$HOME/w4a8_runs/inkling-hi2-runtime/server-hi2-720k.pid \
   VENV=$(dirname $(dirname $V)) PORT=18230 EAGER=0 \
   bash $TREE/tools/inkling_hi2/serve.sh > ~/w4a8_runs/inkling-hi2-runtime/logs/hi2-720k.log 2>&1 &
   ```

   serve.sh with `DSPARK=1` defaults `MAXLEN=720000 MAXBAT=1024`; the boot must
   reach `/health`, log `max_model_len=720000`, the model-memory line and the KV
   blocks, and then `smoke.sh` (step 4) must pass against it. If the KV does not
   fit, record the largest `MAXLEN` that boots (the KV is 32 KiB/token/rank) —
   that number is the card's context, not 720000.

   (b) Vision (`VISION=1` loads the multimodal class; the tower's 422 `model.visual.*`
   tensors come from the directory's vision shard). For the stock comparison both
   servers are resident:

   ```bash
   # stock reference (as in step 2) + HI2 with images
   MODEL=$HOME/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 SLOT=a2 \
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   SERVER_PID_FILE=$HOME/w4a8_runs/inkling-hi2-runtime/server-hi2-vision.pid \
   VENV=$(dirname $(dirname $V)) PORT=18230 MAXLEN=8192 MAXBAT=256 SEQS=4 \
   GMEM=0.45 VISION=1 DSPARK=0 \
   bash $TREE/tools/inkling_hi2/serve.sh > ~/w4a8_runs/inkling-hi2-runtime/logs/hi2-vision.log 2>&1 &

   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/vision_probe.py \
     --make-images ~/w4a8_runs/inkling-hi2-runtime/images
   HIP_VISIBLE_DEVICES=0,1 EXPECT_PCI_BUSES=d1,d4 \
   python3 $TREE/tools/inkling_hi2/vision_probe.py \
     --hi2-url http://127.0.0.1:18230 --stock-url http://127.0.0.1:18231 \
     --images ~/w4a8_runs/inkling-hi2-runtime/images \
     --json ~/w4a8_runs/inkling-hi2-runtime/vision_probe.json
   ```

   The probe takes server **bases** (`http://host:port`, no `/v1` or other path)
   and resolves each server's model id from `/v1/models`; it prints both the base
   and the id and posts to `<base>/v1/chat/completions`.  This matters: the recipe
   serves `inkling2-hi2`, and vLLM answers **404** for any other model name
   (`"model": "default"` was the card1 vision failure).  In the driver a probe
   defect (404/transport, no verdict JSON) does not discard the eval vision
   evidence: the step passes when `eval/cmp-vision.json` is comparable, paired on
   >= 50 items and non-inferior within its `mdd`, recording `vision_probe=FAIL`
   and `vision_eval=<summary>`; a probe that *reached a verdict* and failed a
   check (empty/garbled image answer, stock disagreement) stays fatal.

   `VISION=1` allows `VISION_IMAGES=1` image per prompt by default; a multi-image
   eval set needs `VISION_IMAGES=<n>` on the boot line.  The boot must show the
   vision tower as loaded (no `unexpected vision tensor`, no missing-parameter
   line) and `/v1/models` must report image support (the stock recipe serves
   images the same way). The probe requires, per image: a
   clean one-sentence answer (HTTP 200, `finish_reason` stop/length, no U+FFFD),
   word overlap with the stock answer >= `--min-overlap 0.3`, and a mixed batch of
   images and text in flight at `--concurrency 4` (it defaults to the built-in 4
   PNGs; pass the eval harness's image directory instead if it exists). Keep
   `vision_probe.json`; the outputs' texts are the evidence. The tower's bit-exact
   equality is already the CPU `--vision` gate (422/422), so a coherent answer here
   gates the served path, not the weights.

## 3. Deliberate limits and unrun items

* As of 2026-11-25 every GPU is leased: everything after step 1 is prepared and
  CPU-checked but unrun. DSPARK=1 and VISION=1 are wired in `serve.sh` and the HI2
  recipe (defaults: DSPARK on, vision off) and tested on CPU; the first boot should
  still be `DSPARK=0 VISION=0`.
* DSpark: the draft is the stock bf16 `mtp.safetensors` (54 tensors), no conversion
  and no quantizer change; the HI2 serve directory must carry it
  (`validate_dir.py --require-mtp`, `prep_serve_dir.sh` links it). Identity,
  acceptance length and speed are unmeasured on HI2; acceptance is expected to be
  lower than stock (the experts are quantized) and the probe records it without a
  gate.
* Vision: the tower is bf16 bit-exact (CPU check 422/422), the head build registers
  the `inkling2_vit_*` ops (44 rocm ops in `torch.ops._rocm_C`), and the stock
  recipe serves images; the HI2 image path is unmeasured. `vision_probe.py` is the
  gate; the tower is loaded only with `VISION=1` (no `--language-model-only`).
* Long context: 720000 with DSpark is the stock card's number. The HI2 KV is the
  same 32 KiB/token/rank; step 8a must show the blocks that fit or the honest
  lower `MAXLEN`.
* TP=4 and CUDA: the kernel test is gfx1030-only, as the fork requires.
