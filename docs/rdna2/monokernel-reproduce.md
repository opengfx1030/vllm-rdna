# Reproduce the MonoKernel (#94) A/B numbers

How to rerun the same-session `--monokernel` off/on A/B that produced the
results table in [#94](https://github.com/opengfx1030/vllm-rdna/pull/94)
(RDNA2 MonoKernel decode for Qwen3.8-Flash-Next, merged into `rdna_extras` as
`1c1426d585`).

Everything here comes from the PR description and the raw run in the
infrastructure repo, `bench_results/2026-10-16_mono-layout/` (`ab.sh`,
`ab.out`, `ab2.out`, `ab_table.md`, `build.sh`, `pytest_mono.sh`,
`pytest*.log`, `serve-*/`). Items that the sources do not record are marked
**not recorded**.

## What it reproduces

Four cells per recipe: 1k/512 and 16k/1k, at c=1 and c=8. Recipes
`flashnext-mtp0` and `flashnext-mtp2`. Same build, same session, `--monokernel`
off and on.

`ab_table.md` (output of `tools/rdna/mono/ab_table.py`). `n` is the number of
serves per arm, not the request count:

| recipe | cell | arm | n | prefill tok/s/req | decode tok/s/req | TTFT s | ITL med ms | ITL mean ms |
|---|---|---|---|---|---|---|---|---|
| mtp0 | 1024x512-c1 | off | 1 | 2033.5 | 44.97 | 0.50 | 22.2 | 22.2 |
| mtp0 | 1024x512-c1 | on | 1 | 2204.6 | 50.74 | 0.46 | 19.7 | 19.7 |
| mtp0 | 1024x512-c1 | **on vs off** | | +8.4% | +12.8% | -7.8% | -11.4% | -11.4% |
| mtp0 | 1024x512-c8 | off | 1 | 524.0 | 25.84 | 1.95 | 35.1 | 38.7 |
| mtp0 | 1024x512-c8 | on | 1 | 574.0 | 28.56 | 1.78 | 31.7 | 35.0 |
| mtp0 | 1024x512-c8 | **on vs off** | | +9.5% | +10.5% | -8.7% | -9.5% | -9.5% |
| mtp0 | 16384x1024-c1 | off | 1 | 2096.1 | 44.38 | 7.82 | 22.5 | 22.5 |
| mtp0 | 16384x1024-c1 | on | 1 | 2112.9 | 50.03 | 7.75 | 20.0 | 20.0 |
| mtp0 | 16384x1024-c1 | **on vs off** | | +0.8% | +12.7% | -0.8% | -11.2% | -11.3% |
| mtp0 | 16384x1024-c8 | off | 1 | 1070.8 | 13.34 | 15.30 | 35.6 | 75.0 |
| mtp0 | 16384x1024-c8 | on | 1 | 1083.2 | 13.97 | 15.13 | 32.6 | 71.6 |
| mtp0 | 16384x1024-c8 | **on vs off** | | +1.2% | +4.7% | -1.1% | -8.5% | -4.5% |
| mtp2 | 1024x512-c1 | off | 1 | 2052.8 | 58.88 | 0.50 | 28.6 | 28.7 |
| mtp2 | 1024x512-c1 | on | 1 | 2284.4 | 77.66 | 0.45 | 24.8 | 24.8 |
| mtp2 | 1024x512-c1 | **on vs off** | | +11.3% | +31.9% | -10.1% | -13.4% | -13.5% |
| mtp2 | 1024x512-c8 | off | 1 | 905.2 | 29.47 | 1.13 | 61.0 | 72.0 |
| mtp2 | 1024x512-c8 | on | 1 | 1003.2 | 32.54 | 1.02 | 55.9 | 65.7 |
| mtp2 | 1024x512-c8 | **on vs off** | | +10.8% | +10.4% | -9.8% | -8.3% | -8.8% |
| mtp2 | 16384x1024-c1 | off | 1 | 2022.2 | 68.37 | 8.10 | 29.1 | 29.2 |
| mtp2 | 16384x1024-c1 | on | 1 | 2039.6 | 104.58 | 8.03 | 25.5 | 25.5 |
| mtp2 | 16384x1024-c1 | **on vs off** | | +0.9% | +53.0% | -0.9% | -12.4% | -12.4% |
| mtp2 | 16384x1024-c8 | off | 1 | 1396.8 | 13.92 | 11.73 | 62.3 | 166.2 |
| mtp2 | 16384x1024-c8 | on | 1 | 1459.6 | 14.35 | 11.23 | 58.3 | 160.5 |
| mtp2 | 16384x1024-c8 | **on vs off** | | +4.5% | +3.1% | -4.3% | -6.4% | -3.4% |

The PR table has the same values. The PR rounds prefill to whole tok/s
(2033.5 -> 2034, ...) and shows only the median-ITL change. There is no other
difference.

Which serves the rows come from:

- MTP0 rows: `serve-ab-mtp0-off`, `serve-ab-mtp0-on` (driver output `ab.out`).
- MTP2 rows: `serve-ab2-mtp2-off`, `serve-ab2-mtp2-on` (driver output
  `ab2.out`). The first MTP2 pair in `ab.out` is not in the table: its
  `mtp2-on` boot failed (see [Gotchas](#gotchas)) and the pair was rerun as
  `ab2`.
- The env-alias boot (`serve-envboot-mtp0-env`, 1k/512 c=1 only):
  2209.9 / 50.84 / 0.46 s / 19.7 ms, against 2204.6 / 50.74 / 0.46 s /
  19.7 ms for `--monokernel` (`ab.out`).

The exact `ab_table.py` command line is **not recorded**. The `n=1` MTP2 rows
match `ab2` alone, so it was run on these four directories:

```bash
python3 tools/rdna/mono/ab_table.py <run-dir>/serve-ab-mtp0-off <run-dir>/serve-ab-mtp0-on \
    <run-dir>/serve-ab2-mtp2-off <run-dir>/serve-ab2-mtp2-on
```

## Software state

| Item | Value | Source |
|---|---|---|
| Fork tree | `1c1426d585` for the MTP2 rows. See the next paragraph for MTP0 | inferred, see below |
| vLLM version string | `0.31.0+rdna` (`SETUPTOOLS_SCM_PRETEND_VERSION`; the tree had no `.git`, so the logs print `Failed to read commit hash`) | `build.sh`, `serve.log` |
| Venv | `venv-7.14.0_0.31.0-final`, Python 3.12 | `ab.sh`, `pytest*.log` paths |
| torch / ROCm in that venv | **not recorded**. The loaded `librocblas.so.5` matched the `rocm7.14-rocblas5.5` profile, which `tunableop/profiles.json` registers for torch `2.12.0+rocm7.14.0`, ROCm `7.14.60850`, rocBLAS `5.5.0.cd957402`. The 0.31 venv lineage in `tools/rdna/port_v031/build.sh` is the same torch | `serve.log` line 1, `profiles.json` |
| ROCm SDK (build) | `/opt/rocm/core-7.14` (hipcc, headers) | `build.sh` |
| TunableOp | profile `rocm7.14-rocblas5.5`, auto-selected for rocBLAS build `f30bb442e9b5`; lookup on, tuning off | first two lines of every `serve.log` |
| Native extensions | `setup.py build_ext --inplace` in the tree, `PYTORCH_ROCM_ARCH=gfx1030`, `RelWithDebInfo`, `MAX_JOBS` default 16; 390 s | `build.sh`, `build.out` |
| Model | `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`, snapshot `0939125b929543a783ce700c90e36dd1a575c00c` | `summary.txt` boot line |
| PLE table | `PLE=int4`, `PLE_QUANT_DIR` = `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` snapshot `4f861b63f69e61bfc2e22130ec91ec67f03ec43e`, subdirectory `ples_int4` | `summary.txt` boot line |

**Which commit.** The tree on the bench host was an rsync copy without
`.git`, so no serve log names a SHA. The evidence:

- `33892a9af4` ("accept the Qwen4Exp MTP drafter's config under
  `--monokernel`") was committed at 10:44 local time. That is the end of the
  failed `ab-mtp2-on` boot (10:24) and the start of `pytest4.log` (10:44,
  142 passed against 141 before). The `ab2` MTP2 serves started at 10:47.
- In the MTP2 `ab2` logs the per-rank part line comes from
  `mono_decode_rdna2.py:198`. That is its line number at `1c1426d585`. In the
  MTP0 and env-alias logs it comes from line 196, so those serves ran on the
  parent state, before `33892a9af4`.
- `33892a9af4`, `ae4159cb6e` (the head of #94) and `1c1426d585` have the
  same tree (`5a3e418bae`).
- `33892a9af4` changes only `mono_decode_rdna2.py` (drafter config check),
  one test and `ab_table.py`. With MTP=0 there is no drafter config, so the
  MTP0 arms run the same code at `1c1426d585`. The extensions were built once
  at 08:54 (`build.out`), before that commit. It has no `csrc` change, so no
  rebuild was needed.

Use `1c1426d585`. Later `rdna_extras` commits change the launcher, the recipes
and `csrc/rocm` (for example #100, #104, #105), so do not use a newer tip for
this reproduction.

**Two vLLM trees.** The server ran from the A/B tree (`serve_rdna.sh` puts it
first on `PYTHONPATH`). The probes and the `vllm bench serve` client ran from
the venv's own editable install. That was a different tree, as the
`Failed to read commit hash` warning path in the bench logs shows. The commit
of that client tree is **not recorded**.

## Hardware and environment

- 4x Radeon PRO V620 (gfx1030), TP=4 with expert parallel. The original run
  used four GPUs behind one PCIe switch (PLX). Set
  `GPUS=<4 GPUs on one PCIe switch>`; it becomes `HIP_VISIBLE_DEVICES`.
- Both recipes set this environment (`tools/rdna/recipes/flashnext-mtp{0,2}.env`
  at `1c1426d585`). It is the "production recipe environment" in the PR:
  `HSA_FORCE_FINE_GRAIN_PCIE=1`, `GPU_MAX_HW_QUEUES=4` (#107),
  `HSA_ENABLE_SDMA=0`, `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_RDNA_AR=1` with
  `VLLM_RDNA_AR_MAX_KB=64` / `VLLM_RDNA_AR_ONESHOT_KB=64`,
  `VLLM_RDNA_FUSED_SE=1`, `TUNED_CONFIG=1`, `OMP_NUM_THREADS=4`.
- The launcher (`rdna_launcher_common.sh`) and `serve_validate.sh` set
  `ROCPROFILER_REGISTER_ENABLED=0` (#108), `VLLM_ROCM_USE_AITER=0`,
  `VLLM_RDNA_FORCE_FP16=1`, `TORCH_BLAS_PREFER_HIPBLASLT=0`,
  `HF_HUB_OFFLINE=1`.
- Serve shape (`serve.log` "non-default args"): `--dtype float16`,
  `--max-model-len 262144`, `--block-size 1024`, `--max-num-seqs 8`,
  `--max-num-batched-tokens 2048`, `--kv-cache-memory-bytes 4026531840`,
  `--attention-backend RDNA_ATTN`, prefix caching, `--mamba-cache-mode align`,
  `FULL_AND_PIECEWISE` with `compile_ranges_endpoints: []`. The capture sizes
  are `[1,2,4,8]` for MTP0 and `[3,6,12,24]` for MTP2. MTP2 adds
  `--speculative-config '{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true}'`.
- The drivers apply no clock, power-cap or fan settings. `clocks.log` in each
  serve directory logs sclk, mclk, power and use at 1 Hz.

To see the full environment and command for an arm without launching:

```bash
PRINT=1 EXTRA_ARGS="--monokernel" bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 \
    MODEL=<model-dir> VENV=<venv> PLE=int4 PLE_QUANT_DIR=<ple-int4-dir> HIP_VISIBLE_DEVICES=<gpus>
```

## Steps

### 1. Tree and build

```bash
git clone https://github.com/opengfx1030/vllm-rdna.git <tree>
cd <tree> && git checkout 1c1426d585
VENV=<venv> bash tools/rdna/mono/build_tree.sh   # setup.py build_ext --inplace, gfx1030
```

`build_tree.sh` matches the recorded `build.sh` except that it defaults to
`MAX_JOBS=32` (the recorded `build.sh` defaults to 16). Job count does not
change the binaries. It builds in place without pip; the launcher serves the tree through
`PYTHONPATH`. Check that `vllm/_rocm_C.abi3.so` exists (the run produced a
173 MB file, `build.out`).

### 2. Unit tests (the recorded gate)

`pytest_mono.sh` runs on one GPU with the venv's ROCm SDK libraries on
`LD_LIBRARY_PATH`, `PYTHONPATH=<tree>`, `VLLM_ROCM_USE_AITER=0`,
`VLLM_RDNA_FORCE_FP16=1`, `ROCPROFILER_REGISTER_ENABLED=0` and
`HF_HUB_OFFLINE=1`:

```bash
python -m pytest -p no:cacheprovider -q -rs \
    tests/models/qwen4_exp/test_amd_mono_rdna2.py \
    tests/models/qwen4_exp/test_amd_mono_rdna2_kernels.py \
    tests/models/qwen4_exp/test_qsa_amd.py
```

Expected at `1c1426d585`: **142 passed** (`pytest4.log`, 66.6 s). The
`pytest3.log` run (141 passed) is from before `33892a9af4` added the
drafter test. `pytest1.log` (1 failed: `test_off_by_default` needed the Hub
offline) and `pytest2.log` (24 failed: K2 shape mismatch) were earlier
iterations in the same session. Their failures match what
`67c3bd6ef5` ("K2 result order; no model resolution in the switch test")
fixed.

### 3. The A/B serves

The run used `ab.sh` (infrastructure repo). It calls
`tools/rdna/mono/run_specs.sh`, which runs `tools/rdna/mono/serve_2a.sh`, which
runs `tools/rdna/port_v031/serve_validate.sh` -> `tools/rdna/serve_rdna.sh`.
Host-neutral equivalent:

```bash
cd <tree>
export RUN=<run-dir> GPUS=<4 GPU HIP indices> PORT=18790 VENV=<venv>
export MODEL=<model-dir> PLE_QUANT_DIR=<ple-int4-dir>
export SMI_DEVS="<rocm-smi ids of the same 4 GPUs>"   # idle check; default "0 1 2 3"
export LOCK=0                                         # the HIP 6-9 lock is a shared-host convention
export CELLS="1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"
SPECS='mtp0|off|ab|;mtp0|on|ab|' bash tools/rdna/mono/run_specs.sh
SPECS='mtp2|off|ab2|;mtp2|on|ab2|' bash tools/rdna/mono/run_specs.sh
# env-alias boot (1k/512 c=1 only)
CELLS="1024:512:1" CACHE_TAG=ab SPECS='mtp0|env|envboot|' bash tools/rdna/mono/run_specs.sh
```

What each arm sets (`serve_2a.sh`):

| Arm | Switch | Compile cache |
|---|---|---|
| `off` | nothing | `<run-dir>/cache-2a-<tag>-off` |
| `on` | `EXTRA_ARGS=--monokernel` (no `--additional-config`, so all parts use their default width gates) | `<run-dir>/cache-2a-<tag>-on` |
| `env` | `VLLM_ROCM_MONO_DECODE=1`, no flag | `<run-dir>/cache-2a-ab-env` |

Every arm also passes `PLE=int4 PLE_QUANT_DIR=...` to the recipe, plus
`BENCH_EXTRA="--temperature 0"` and `LOAD_HOOK=tools/rdna/mono/parity_hook.sh`.
`run_specs.sh` waits for the GPUs to be idle before each serve. The original
lock also kept a 90 s gap between serves (`LOCK_YIELD_S`); with `LOCK=0`, wait
between serves yourself if you want the same spacing.

One arm by hand, without the drivers:

```bash
EXTRA_ARGS="--monokernel" bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp0 \
    MODEL=<model-dir> VENV=<venv> PLE=int4 PLE_QUANT_DIR=<ple-int4-dir> \
    HIP_VISIBLE_DEVICES=<gpus> PORT=18790
```

### 4. What `serve_validate.sh` does per serve

In order, each step logged to `summary.txt`:

1. Boot. Ready means `/v1/models` answers. A cold compile cache took 811-892 s;
   `ab-mtp2-off` took 321 s because it reused `cache-2a-ab-off` from
   `ab-mtp0-off`.
2. Greedy probe (`tools/probe_greedy_correctness.py`, 3 prompts).
3. Perplexity probe (`ppl_probe.py`, 261 tokens).
4. Prefix probe (`prefix_probe.py`, cold vs warm TTFT).
5. Capture-ladder warm-up (`ladder_warmup.py`, default sizes, decode batches
   1..8; 21-33 s). It keeps first-use JIT and tuning out of the bench cells.
6. Concurrency probe (`concurrency_probe.py`, 48 greedy checks at c=N and under
   slot churn against solo outputs).
7. Bench cells, in the order of `CELLS`:

   ```bash
   python -m vllm.entrypoints.cli.main bench serve \
       --backend openai --endpoint /v1/completions \
       --base-url http://127.0.0.1:<port> --model flash-next --tokenizer <model-dir> \
       --dataset-name random --random-input-len <in> --random-output-len <out> \
       --num-prompts <n> --max-concurrency <c> --ignore-eos \
       --request-rate inf --seed 12345 --save-result --result-dir <serve-dir> \
       --result-filename bench-<in>x<out>-c<c>.json --temperature 0
   ```

   `n` is 4 at c=1 and 32 at c=8 (`max(4c, 8)`).
8. Load hook: `logprob_parity.py dump` writes greedy top-5 logprobs for 8
   prompts to `parity.json` and counts the `mono ...` engagement lines.
9. Probe after load, marker summary, fault-marker count in `serve.log`, and
   new amdgpu lines in `dmesg` (needs `sudo -n`, else `n/a`).

### 5. The four metrics (`tools/rdna/port_v031/bench_metrics.py`)

From each `bench-*.json`:

- **prefill tok/s/req** = mean input length / mean TTFT
  (`total_input_tokens / completed / (mean_ttft_ms / 1000)`).
- **decode tok/s/req** = `1000 / mean_tpot_ms`.
- **TTFT s** = `mean_ttft_ms / 1000`.
- **ITL ms** = `median_itl_ms` (the PR's ITL column) and `mean_itl_ms`.

`summary.txt` also prints aggregate prefill
(`min(c, completed) * input length / p99 TTFT`) and aggregate output tok/s.
These are not in the PR table. `ab_table.py` computes the same per-request
metrics and averages them over serves per arm.

## Validation

A run is valid when all of these hold. Every serve behind the table met
them (`ab.out`, `ab2.out`, `summary.txt`):

- `PROBE PASS` (3/3), `PPL PROBE PASS` with PPL 4.75-4.85, `PREFIX PASS`,
  `CONCURRENCY PASS` with `0 failing`, `LOAD HOOK rc=0 PASS`,
  `PROBE-AFTER-LOAD PASS`.
- Every bench cell ends `ok 4` (c=1) or `ok 32` (c=8).
- `fault markers in serve.log: 0`.
- `serve.log` starts with `TunableOp profile rocm7.14-rocblas5.5 auto-selected
  for rocBLAS build f30bb442e9b5`. A different hash means a different rocBLAS
  build, and the shipped rows do not apply.
- On-arms: each of the 4 ranks logs `mono rdna2: --monokernel on TP rank N:
  k1 (<= 24 rows), k2 (<= 8 rows, full,piecewise,eager), gdn (<= 64 rows),
  qsa (HIP sparse decode >= 3 rows)`. `load_hook.txt` shows k1, k2 and gdn
  taking decode widths (`mono k1_rdna2: <w> rows (full) -> K1 kernel`,
  `mono gdn_rdna2: decode <w> rows -> fused kernel`, and `verify <w> rows`
  for MTP2). The QSA part logs `mono qsa_rdna2: ... -> mono` in `serve.log`
  only. Off-arms log no `mono` lines.
- Markers: `Using CompressedTensorsWNA16RDNA2MoEMethod`, `rdna_ar: one-shot
  all-reduce active` on every rank, PLE `Qwen4ExpPLEInt4EmbeddingMethod`. KV
  cache: 297,756 tokens (MTP0) or 266,945 (MTP2). Model load: 18.89 GiB
  (MTP0) or 20.63 GiB (MTP2) per rank.

The recorded result does not include an off-vs-on logprob comparison. Both
arms saved `parity.json`, but no `logprob_parity.py compare` output is
recorded. You can run it as an extra check.

## Gotchas

- **MTP2 decode tok/s/req follows drafter acceptance.** The PR says to read
  ITL for the MTP2 cells. Recorded acceptance lengths (`bench-*.json`,
  `spec_decode_acceptance_length`): 16k/1k c=1 off 1.99 vs on 2.67, 1k/512
  c=1 1.69 vs 1.93. That explains most of the +53 % and +32 % decode rows. The
  same off config served twice (`ab-mtp2-off`, then `ab2-mtp2-off`) gave
  16k/1k c=1 decode 83.29 vs 68.37 tok/s/req (acceptance 2.44 vs 1.99), with
  ITL 29.3 vs 29.1 ms.
- **Run-to-run noise at c=8.** Between those two off serves, prefill at c=8
  moved 3-5 % (1k/512: 951.7 vs 905.2; 16k/1k: 1440.2 vs 1396.8). That is the
  same size as the on/off prefill delta in the MTP2 16k/1k c=8 cell (+4.5 %).
  Each arm in the table is one serve.
- **`GPU_MAX_HW_QUEUES=4` is required.** With 2, HIP queue sharing makes one
  to three ranks slower next to the persistent kernel
  (`docs/rdna/monokernel/fg1-skew.md`). The recipes set 4. If the launcher
  sets less, the MonoKernel raises it to 4 for its workers (knob
  `hw_queues`).
- **MTP2 + `--monokernel` before `33892a9af4` refuses to boot**:
  `ValueError: --monokernel on gfx10x supports Qwen4Exp only, not
  ['Qwen4ExpMTP']` (`serve-ab-mtp2-on/serve.log`). `1c1426d585` has the fix.
- **Cold start.** A fresh compile cache adds 13-15 min to boot. The warm-up
  step must run (`WARMUP=1`, default) or first-use JIT lands in a bench
  cell. The A/B used a separate cache per arm, so neither arm reused the
  other's compiled graphs.
- **Benign log noise.** Every arm, passing ones included, logs `Failed to
  import Triton kernels ... is_hip_gfx1250` and `Failed to read commit hash`.
- **dmesg.** `serve-ab-mtp0-on` recorded `new amdgpu dmesg faults: 17` with
  `fault markers in serve.log: 0` and all gates passing. The matching dmesg
  lines are **not recorded**. All other serves recorded 0.
- **Host-specific guards in the drivers.** `serve_validate.sh` refuses
  `GPUS` that contain HIP index 0 or 1 (a bench-host rule). On another host,
  remove that `case` block locally or use other indices. `serve_2a.sh`,
  `parity_hook.sh` and `run_specs.sh` default `MODEL`, `PLE_QUANT_DIR`,
  `VENV`, `RUN` and `LOCKDIR` to bench-host paths. Override them as in step 3.
- **QSA threshold wording.** The PR description says the HIP sparse decode
  starts at 8 rows. The code at `1c1426d585` (`mono/rdna2/qsa.py`
  `MIN_ROWS = 3`) and the startup log say 3 rows. The measured runs used
  the code value.

## Context

Earlier stages (infrastructure repo `bench_results/`), not part of the table:
`2026-10-14_monokernel-p0/` (P0 probe and Model Runner V2 serve),
`2026-10-14_monokernel-2a/` (FFN kernel, Stage E), `2026-10-15_fg-pcie/`
(`HSA_FORCE_FINE_GRAIN_PCIE` 1 vs 0, kernel off), `2026-10-16_rocprof-off/`
(`ROCPROFILER_REGISTER_ENABLED` unset vs 0, kernel off). Records in the fork:
[`docs/rdna/monokernel/`](../rdna/monokernel/README.md).

## Files

At `1c1426d585`:

- [`tools/rdna/mono/run_specs.sh`](../../tools/rdna/mono/run_specs.sh):
  queue of serves (spec `recipe|arm|tag|knobs`), idle wait, clock log.
- [`tools/rdna/mono/serve_2a.sh`](../../tools/rdna/mono/serve_2a.sh): one
  serve, maps `ARM` to `--monokernel` / `VLLM_ROCM_MONO_DECODE` /
  `--additional-config`.
- [`tools/rdna/mono/parity_hook.sh`](../../tools/rdna/mono/parity_hook.sh),
  [`logprob_parity.py`](../../tools/rdna/mono/logprob_parity.py): load hook.
- [`tools/rdna/mono/ab_table.py`](../../tools/rdna/mono/ab_table.py): the
  A/B table.
- [`tools/rdna/mono/build_tree.sh`](../../tools/rdna/mono/build_tree.sh):
  in-place native build.
- [`tools/rdna/port_v031/serve_validate.sh`](../../tools/rdna/port_v031/serve_validate.sh):
  boot, probes, warm-up, bench cells, markers.
- [`tools/rdna/port_v031/bench_metrics.py`](../../tools/rdna/port_v031/bench_metrics.py):
  the four metrics.
- [`tools/rdna/serve_rdna.sh`](../../tools/rdna/serve_rdna.sh),
  [`tools/rdna/rdna_launcher_common.sh`](../../tools/rdna/rdna_launcher_common.sh),
  [`tools/rdna/recipes/flashnext-mtp0.env`](../../tools/rdna/recipes/flashnext-mtp0.env),
  [`flashnext-mtp2.env`](../../tools/rdna/recipes/flashnext-mtp2.env):
  launcher and recipes.
- [`vllm/models/qwen4_exp/amd/mono/rdna2/README.md`](../../vllm/models/qwen4_exp/amd/mono/rdna2/README.md):
  parts, knobs, refusals.
- [`tunableop/rocm7.14-rocblas5.5/`](../../tunableop/rocm7.14-rocblas5.5/):
  the TunableOp rows used.
