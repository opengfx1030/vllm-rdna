# LeapQuant PR 170: GPU validation, 27B AWQ at TP2 and Flash-Next at TP4 (2026-11-15)

Status: validation record for PR #170
([config flag, validator, numerics tests, page packing](leapquant-config-guards-2026-11-14.md)).
Branch `feat/gdn-leapquant-pack-layers` at `791392e04b`. The three earlier notes
of this PR say "no GPU ran this change". This note is the GPU run.

Sections 1 to 6 ran on **two V620 GPUs (gfx1030, HIP 0,1 = PCI buses d1 and d4
of the box par1-cs25, lease `a2`) with the Qwen3.8-27B AWQ model at TP2**.
Sections 7 to 11 are the Flash-Next TP4 matrix (HIP 2-5, lease `b4`, run
`matrix-flashnext-20261115-195724`, five arms). Section 11 says what is not
validated.

All runs are correctness and function runs. No timed window was taken, no quiet
file was written, and this note makes no throughput claim.

## Summary

| Item | Result |
|---|---|
| Build | `_rocm_C` builds from the branch head, no warning in the edited file, no op schema changed |
| Kernel tests | 26 + 46 pass on HIP 0 and on HIP 1, also under an external matmul load and at the Flash-Next per-rank heads (4, 12). Nothing to triage. At the 27B per-rank heads (8, 24): 24 + 46 pass, and two 512 KiB-page cases cannot be built there (a limit of the test, section 2) |
| Control | A `_rocm_C` built from the kernel source before the fold guard fails 2 of the 46 safety tests: the tests see the defect the guard fixes |
| CPU tests | 250 passed, 1 skipped (pre-existing), 0 failed |
| Rejection at boot | 8 setups are refused in the API server before any worker starts, with the exact messages of section 5 |
| Boot with the flag (27B, TP2, MTP2, `FULL_AND_PIECEWISE`) | boots and serves; greedy, PPL, prefix-cache (cold text equals warm text) and concurrency probes pass; no fault, no eviction |
| KV capacity at block 1024 | flag off 328,658 tokens, flag on 325,019 (-1.1%): no gain at the recipe's block, as the planner said |
| Quality, 27B flag on against flag off | long-context PPL -0.014%; prompt-level KL 1.1e-4 nats/token, top-1 0.9977; decode KL 2.1e-4 and 1.0e-4 nats/position, no confident flip. No floor on the 27B (`A2` collects were cut off) |
| Page packing on the 27B | kernels: 4 packed-page tests pass. Planner: applies at block 1024 (3 groups of 16 become 2 of 24, 331,129 tokens, +0.75% against flag off) and is a no-op at block 512. The 27B packing boots are not run (dropped) |
| Flash-Next TP4: boots and capacity (section 7) | `A`/`A2` 266,945 tokens, groups (1024 x4, 8, 1024); `B` and `BNM` 273,456, (512 x4, 8, 512), +2.44% against `A`; `P` (packing, block 640) 273,220, (640, 640, 640, 8, 640), -0.09% against `B`. All five arms: probes pass, 0 KFD evictions, 0 faults |
| Flash-Next TP4: quality against the floor (section 8) | prompt-level KL 0.01194 against floor 0.01189 nats/token, dNLL -0.00098 [-0.00213, +0.00016] against floor +0.00067, top-1 0.9673 against 0.9671; decode KL (512 tokens) 6.7e-3 against floor 9.0e-3, long decode 1.02e-2 against 9.8e-3; long-context PPL +0.173% against floor -0.336%. Packed against padded: KL 0.01142, PPL -0.342% |
| Flash-Next TP4: MonoKernel off, `BNM` (section 9) | boots with the same capacity, probes pass, 0 evictions, MonoKernel lines absent; quality at the floor except one open sign: `A` to `BNM` long decode KL 1.9e-2 against 9.8e-3 (175 positions) |
| Bugs found | none in the GPU tests of the 27B run. The Flash-Next matrix found one cosmetic defect: `VLLM_RDNA_GDN_LEAPQUANT_PACK` was not registered in `vllm/envs.py` (unknown-variable warning); fixed in this PR |
| Not validated | 27B floor and packing boots (dropped), any timed throughput, MTP drafter layers outside the recipes, V1 runner (section 11) |

**Status of the arms.** `A` and `B` are complete. `A2` (the boot-to-boot
floor) booted at 18:12 and passed the probes (section 3), but its quality
collects were cut off in the prompt-level collect (window 22 of 64): at about
18:17 the chain and the `A2` server were terminated from outside. The cause was
not identified. The server log shows a graceful shutdown, and there is no fault,
eviction or OOM. The hand-off script then released the lease `a2` and removed
the scratch (everything except the logs and `tree-pr170`), and another job took
`a2` at 18:18. So `P`, `C512` and `C512P` never booted, and this note has no
floor and no packed engine boot. `compare.txt` in `quality-20261115-162845/`
was written by the hand-off script and fails on the missing `A2` files: ignore
its floor parts. Sections 3, 4 and 6 mark what is missing, and "Still pending
on the 27B" has the commands.

## Setup

| | |
|---|---|
| Box, GPUs | par1-cs25, HIP 0,1 = V620 buses d1, d4 (checked before every run), 30 GiB each |
| Software | torch 2.12.0+rocm7.14.0, HIP 7.14.60850, AMD clang 23.0.0git, venv `venv-7.14.0_0.32.0` |
| Tree | `git ls-files` of the branch at `791392e04b` (`GIT_SHA` in the tree) with a `_rocm_C.abi3.so` built from that source. The other extension modules (`_C`, `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `cumem_allocator`, `fs_io_C`, `spinloop`) come unchanged from an earlier tree: the branch changes no csrc outside `csrc/rocm/` |
| Model | `cyankiwi/Qwen3.8-27B-AWQ-INT4` (snapshot `63768c10`), 64 layers: 48 GDN (16 key heads, 48 value heads, head dims 128/128, conv kernel 4) and 16 full-attention layers (4 KV heads of 256) plus 1 MTP draft layer. TP2: 24 value heads and 2 KV heads per rank |
| Launcher | `tools/rdna/serve_rdna.sh RECIPE=27b-awq TP=2 MTP=2` through `serve_validate.sh`: MTP2, `FULL_AND_PIECEWISE`, V2 model runner, prefix caching with `--mamba-cache-mode align`, block 1024, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`, `--kv-cache-memory-bytes 12000000000` for every arm |
| `REFUSED_GPUS` | `serve_validate.sh` refuses HIP 0 and 1 by default (its comment still says "0-1 = W7800"). HIP 0,1 are V620s here (buses d1, d4, the leased slot `a2`), so every run clears the list with the documented env knob `REFUSED_GPUS=` and checks the PCI buses itself |
| Scripts, logs | `~/w4a8_runs/leapquant/pr170-gpu/` on the box (see [Commands](#commands)) |

## 1. Build and op registration

`_rocm_C` only (`cmake --build --target=_rocm_C`, `MAX_JOBS=8`, ccache, one
configure plus one build, 93 s and 817 s). The build log has no warning for
`gdn_leapquant_rdna2.hip` (the edited file) or `mono_gdn_rdna2.hip`.

| | |
|---|---|
| `_rocm_C.abi3.so` | 92,467,472 bytes, md5 `8a5d43759e91bdaf6906e3854f379917` |
| `torch.ops._rocm_C` schemas | 177 (161 RDNA ops) |
| LeapQuant ops present | `gdn_lq_store_rdna2`, `gdn_lq_normalize_rdna2`, `gdn_lq_decode_rdna2`, `gdn_lq_materialize_rdna2`, `mono_gdn_lq_rdna2`, `precopy_mamba_align_fused_rdna2` |
| Schema diff against the `_rocm_C.abi3.so` of the older `tree-pack` build (2026-11-14 03:00, before the fold-guard commit of 21:59) | 0 lines: no op signature changed |

The binary was built from the branch head `791392e04b`. The commits that add
this note change no code except one comment: the "NOT BUILT, NOT RUN" lines in
the header of `gdn_leapquant_rdna2.cu` are deleted. The binary stays valid for
the branch head.

## 2. Kernel tests on gfx1030 (HIP 0)

`pytest -v -rs` on the rebuilt `_rocm_C`, `HIP_VISIBLE_DEVICES=0`.

| File | Result | Time |
|---|---|---|
| `tests/kernels/mamba/test_gdn_leapquant_rdna2.py` | **26 passed**, 0 failed, 0 skipped | 13.4 s |
| `tests/kernels/mamba/test_gdn_leapquant_safety_rdna2.py` | **46 passed**, 0 failed, 0 skipped | 146.6 s |
| the same two files again, with an external FP16 matmul loop on HIP 0 (twice) | 26 + 46 passed, 26 + 46 passed | 16 s + 116 s, 16 s + 115 s |
| the same two files on HIP 1 (bus d4, the second GPU of the TP2 pair) | 26 + 46 passed | 11.6 s + 51.1 s |
| the same two files at the per-rank head geometry of Flash-Next at TP4, key and value heads (4, 12) instead of (2, 6), HIP 0 | 26 + 46 passed | 12.2 s + 76.5 s |
| the same two files at the per-rank geometry of the 27B at TP2, (8, 24), HIP 0 | 24 passed + 2 not buildable (below) + 46 passed | 12.0 s + 105.1 s |

The kernel file has 26 tests, not the 22 of the earlier notes: the four
packed-page tests of the packing change were added to it.

The geometry rows come from `geometry_pytest.py`. Both test files keep their
geometry in module constants, and a collection hook swaps them for the heads of
one rank of a real model. They show that the kernels (blob stride, grid, the
MonoKernel scratch) work at the Flash-Next shapes. They are not a Flash-Next
run: no model, no engine. At (8, 24) the two 512 KiB-page cases of
`test_grouped_normalize_matches_per_layer` cannot be built: three pages of
512 KiB do not hold 24 blobs of 28,768 or 37,024 bytes plus the 25,600-byte
conv state, and the test's `as_strided` view fails with `setStorage ... out of
bounds` before any kernel runs. The 27B page is 939,776 bytes, so that case does
not exist for it. At (4, 12) the page is the Flash-Next page of 469,888 bytes
and fits.

The 46 safety tests by item of the numerics note
([isolation note](leapquant-gdn-numerics-and-isolation-2026-11-14.md)):

| Item | Tests | Result |
|---|---:|---|
| 1. Fold guard: degenerate states finite, accurate and orthonormal; neighbour heads unchanged; exactly rank-one trajectory | 3 | 3 passed |
| 2. Determinism: `store`/`normalize`/decode and the MonoKernel path, bitwise, 200 repeats each, second stream running matmuls | 2 | 2 passed (64 s and 42 s) |
| 3. Stale state of recycled blocks: header range check (10), records past the cursor (3), new tenant on the leftovers of a recycled block (8), first decode on a fresh block (1), MonoKernel on a fresh block (1) and after a store (4), align copy overwrite (4) | 31 | 31 passed |
| 4. Flush race: grouped normalize equals per-layer launches (8, 25 repeats, windows 16 and 32, with and without load), batched decode equals singleton launches (2) | 10 | 10 passed |

No failure, so nothing to triage as fold guard, test bug or pre-existing. The
HIP fold guard (`lq_orthonormalize`, the warm start and the smoothing sum) was
**built and run here for the first time** and passes every test written for it.
The "NOT BUILT, NOT RUN" note in the header of `gdn_leapquant_rdna2.cu` is
removed in the same change as this note.

**Control: the tests see the defect the guard fixes.** The two files also ran
against a `_rocm_C` built from the kernel source before the fold-guard commit
(`24a431484a~1`, `mutation_build.sh`, md5 `cc99b9b2de7e6625c8520725e7a38035`,
same tree otherwise). The kernel file gives 26 passed: its tests use ordinary
states. The safety file gives **44 passed and 2 failed**, and the two are the
fold-guard tests with an exactly rank-deficient state:

| Test | Pre-guard kernel | Guarded kernel |
|---|---|---|
| `test_fold_of_degenerate_states_is_finite_accurate_and_orthonormal` | fails: three compensator columns are the same vector up to sign, Gram off-diagonal 0.9998 against a tolerance of 0.002 | passes |
| `test_flush_of_an_exactly_rank_one_trajectory` | fails: worst output error 2.18e-3 against the bound 2e-3 | passes |

Neighbouring GPU tests on the rebuilt `_rocm_C`: `test_precopy_mamba_align.py`
85 passed, `test_gdn_decode_rdna2.py` 37 passed, `test_gdn_zero_draft_spec_rows.py`
4 passed.

CPU tests of the PR on the box, no GPU visible (`HIP_VISIBLE_DEVICES=`):

| File | Result |
|---|---|
| `tests/kernels/mamba/test_gdn_leapquant_oracle.py` | 122 passed (999 s) |
| `tests/benchmarks/test_replayssm_e2e_decode_speedup.py` | 18 passed |
| `tests/config/test_leapquant_config.py` | 38 passed |
| `tests/v1/kv_connector/unit/test_leapquant_state_refused.py` | 6 passed |
| `tests/v1/core/test_leapquant_scheduling.py` | 25 passed, 1 skipped (`requires two concurrently scheduled batches`) |
| `tests/v1/core/test_leapquant_page_packing.py` | 31 passed |
| `tests/tools/rdna/test_serve_replayssm_state_dtype.py` | 7 passed |
| `tests/tools/rdna/test_decode_kl.py` | 3 passed |
| total | 250 passed, 1 skipped, 0 failed |

The PR body listed `test_cli_flag_parses_and_restricts_its_values` as a failure
(HF download on an offline box). It passes on this run.

## 3. Boot, probes and KV capacity

The arms boot one after the other through `serve_validate.sh` on HIP 0,1: `A`
flag off, `B` with `--replayssm-state-dtype leapquant`, `A2` flag off again (the
boot-to-boot floor of section 4), then the packing and block-size arms of
section 6. Each boot runs the launcher probes (greedy, PPL, prefix cache,
concurrency), then the quality collects of section 4 against the same server,
then the greedy probe again. `A` and `B` passed all of them, with no KFD
eviction after READY (`evicted_ms` +0 on both ranks) and no amdgpu dmesg fault.
`A2` passed the probes; its collects were cut off (status above). `P`, `C512`
and `C512P` never booted.

**Server log with the flag on (arm B).**

- The validator accepts the setup. The API server builds `VllmConfig`, the
  engine starts, and the non-default args hold `replayssm_state_dtype:
  'leapquant'` next to `tensor_parallel_size: 2`, `block_size: 1024`,
  `mamba_cache_mode: 'align'` and `kv_cache_memory_bytes: 12000000000`.
- Each worker builds its 48 GDN layers through the layer-time
  `verify_leapquant_gdn` call. `info_once` logs for the first local rank only:
  `GDN recurrent state: LeapQuant INT8 (window 32, r 4, 37024 B/head)`, later
  `LeapQuant GDN: one normalize launch per KV-cache group (16 layers)`. The
  scheduler prints `LeapQuant scheduling: bounded output admission=True,
  decoder-first FCFS`.
- The platform pads the GDN page to the attention page: `Padding mamba page
  size by 123.15%` (arm A, FP32 state: `29.13%`).
- No line names `VLLM_RDNA_GDN_LEAPQUANT`. No WARNING names LeapQuant or
  ReplaySSM. The "Unknown vLLM environment variable" warnings are the RDNA
  variables of the launcher (`VLLM_RDNA_AR_*`, `VLLM_CAUSAL_CONV1D_RDNA2_*`,
  and others), the same in every arm.
- The server prints `state_format` nowhere. The string is covered by the CPU
  tests, and its effects are visible in the lines above (page padding, group
  sizes). The metric `vllm:cache_config_info` carries
  `replayssm_state_dtype="auto"` for `A` and `"leapquant"` for the flag-on arms
  of section 6 (the capture was added to the hook after `B` had run).

**KV capacity: the engine's own number.** `GPU KV cache size` at
`max_model_len` 262,144 and `kv_cache_memory_bytes` 12,000,000,000 per rank.
The last column is the number the CPU planner what-if gave before the boot
(section 6 shows how it was built).

| Arm | State | Block | GPU KV cache size | Concurrency | Group sizes | GDN page padded by | Planner |
|---|---|---:|---:|---:|---|---|---:|
| `A` | FP32 | 1024 | **328,658** | 1.25x | (1024, 1024, 1024, 1024) | 29.13% | 328,658 |
| `A2` | FP32 | 1024 | **328,658** | 1.25x | (1024, 1024, 1024, 1024) | 29.13% | 328,658 |
| `B` | LeapQuant | 1024 | **325,019** | 1.24x | (1024, 1024, 1024, 1024) | 123.15% | 325,019 |

The arms `P`, `C512` and `C512P` never booted (section 6 has the planner values
for them). The second flag-off boot `A2` gives the same capacity and groups as
`A`.

At the recipe's block 1024 **LeapQuant adds no KV tokens on the 27B**: 325,019
against 328,658 (-1.1%). The attention page (2 MiB per layer at block 1024) is
larger than both state pages, so the planner pads the GDN page up to it either
way (29.13% for FP32, 123.15% for LeapQuant) and the smaller LeapQuant page
saves nothing. With the flag the planner also reserves one state block more
per request in each GDN group, 5 instead of 4
(`MambaSpec.max_memory_usage_bytes` counts `1 + max_concurrent_batches +
num_speculative_blocks` = 1 + 2 + 2: "LeapQuant admission accounts for retained
inputs of unresolved batches"). The same holds on Flash-Next at block 1024
(#168: 266,945 tokens both ways).

**Probes.**

| Arm | greedy | PPL probe | prefix cache | concurrency (48 checks) |
|---|---|---|---|---|
| `A` | 3/3 | 4.90 (NLL 1.590, 261 tokens) | cold TTFT 3.82 s, warm 1.14 s, hits +2048, cold text = warm text | 0 failing, 5 near-tie, 43 identical |
| `A2` | 3/3 | 4.90 (NLL 1.590, 261 tokens) | cold TTFT 3.81 s, warm 1.14 s, hits +2048, cold text = warm text | 0 failing, 6 near-tie, 42 identical |
| `B` | 3/3 | 4.90 (NLL 1.590, 261 tokens) | cold TTFT 3.83 s, warm 1.14 s, hits +2048, cold text = warm text | 0 failing, 4 near-tie, 44 identical |

The prefix probe is the check of the V2 align copy marker: a warm request
resumes from LeapQuant state blocks that `get_leapquant_copy_spec` copied, and
its text equals the cold text. The TTFTs are the probe's own output, not a
measurement window.

**Memory outside the KV cache.** `Model loading took` is 11.45 GiB per rank for
`A` and 10.70 GiB for `B`. [INFERENCE] The 0.75 GiB difference is the permanent
GDN state arenas that the flag-off path allocates for `FULL_AND_PIECEWISE`
capture: 25 slots (`max_num_seqs` 8 x 3, plus the null slot) for each of the 48
layers, 1.76 GiB per rank as FP32 and 0.99 GiB as blobs (a difference of 0.77
GiB). This run fixes `kv_cache_memory_bytes`, so the saving does not show in the
capacity above. With utilization-based sizing it would.

## 4. Quality: flag off (`A`) against flag on (`B`)

The collects run against the live server of each arm with the same inputs. They
are the checks of #168, run again with the fold guard, on the 27B instead of
Flash-Next:

| Check | Inputs | `A` (FP32 state) to `B` (LeapQuant) |
|---|---|---|
| Prompt level (`topk_kl.py`, prefill path, top-20) | 64 windows of 2048 tokens of this tree's `docs/**/*.md` | PPL 3.1955 to 3.1956. dNLL +0.00003 [-0.00008, +0.00013]. **KL 0.00011** nats/token [0.00008, 0.00015], p99 0.0016. **Top-1 0.9977** [0.9973, 0.9980]. abs dlogp p50 / p99 / max 0.0000 / 0.0716 / 0.862, identical tokens 50.6% |
| Long-context PPL (`quality_ab.py`) | 6 passages of 12,122 to 18,569 tokens (91,495 tokens), prefilled in chunks of 2048, which crosses many LeapQuant stores; 256 greedy tokens each | **3.4140 to 3.4135 (-0.014%)**, gate +-2%: PASS. Per passage -0.07% to +0.02%, mean abs dlogp 0.0082. Greedy exact 3 of 6 (first divergence at token 5, 129 and 160 of 256) |
| Decode path (`decode_kl.py`) | 32 passages of 1,356 to 2,928 tokens, 512 greedy tokens each, 8 concurrent | **KL mean 2.1e-4**, p50 4.3e-6, p99 2.8e-3, max 0.18 over 9,170 positions (coverage 0.980). 10 of 32 identical, divergence median at token 337, 0 confident flips |
| Decode path, long | 8 passages of 14,715 to 21,892 tokens, 2048 greedy tokens each, 8 concurrent | **KL mean 1.0e-4**, p50 2.4e-6, p99 1.7e-3, max 0.020 over 6,629 positions (coverage 0.993). 1 of 8 identical, divergence median at token 528, 0 confident flips |

[INFERENCE] Each 2048-token window of the prompt-level check is one prefill
chunk, so the stored state is not read back inside a window. That check bounds
what the flag does to the prefill path. The long-context and decode checks read
the state back.

**No verdict against a floor yet.** The floor is the same comparison between two
flag-off boots (`A` against `A2`): it says how much of a difference above is the
boot and how much is the flag. `A2` booted, but its collects were cut off, so
only the gate of `quality_ab.py` applies (a PPL change of at most 2%), and it
passes with 0.014%.
For scale only: #168 measured on Flash-Next at TP4 a prompt-level KL of 0.0120
against a floor of 0.0117 and a long-context PPL change of +0.15% against a
floor of +0.10%. That is another model, another TP and a MoE with its own noise.
The 27B numbers above are smaller.

## 5. Rejection at boot (real launcher, 27B recipe, HIP 0,1)

Each case runs `tools/rdna/serve_rdna.sh RECIPE=27b-awq TP=2 MTP=2` with the
flag (`REPLAYSSM_STATE_DTYPE=leapquant`) and one setup the validator refuses.
All of them stop in the API server while it builds `VllmConfig`. The server
logs of the cases hold no `EngineCore` or `Worker` line, so no worker started
and no weight was loaded. A refusal takes about 40 s, the import time of the
API server.

| Case (launcher argument) | Exit | Message (exact) |
|---|---:|---|
| `EXTRA_ARGS='--kv-transfer-config {"kv_connector":"SharedStorageConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"..."}}'` | 1 | `ValidationError: 1 validation error for VllmConfig` / `Value error, --replayssm-state-dtype leapquant cannot be used with a KV connector or KV offloading (--kv-transfer-config, --kv-offloading-size). A connector would move LeapQuant state blocks between instances or to the host, and no connector knows the format. Remove the connector or the offloading option.` |
| `EXTRA_ARGS='--kv-offloading-size 4'` | 1 | the same message |
| `EXTRA_ARGS='--use-replayssm'` | 1 | `Value error, --replayssm-state-dtype leapquant cannot be used with --use-replayssm. --use-replayssm selects the Mamba2 and KDA ReplaySSM kernels and removes the speculative state blocks. LeapQuant keeps its own record ring in each state block. Remove --use-replayssm.` |
| `VLLM_USE_V2_MODEL_RUNNER=0` (align mode is on) | 1 | `Value error, --replayssm-state-dtype leapquant with mamba align prefix caching (--mamba-cache-mode align) needs the V2 model runner. Its fused align copies commit the LeapQuant records. Set VLLM_USE_V2_MODEL_RUNNER=1 or turn prefix caching off.` |
| `EXTRA_ARGS='--enable-mamba-cache-stochastic-rounding --mamba-ssm-cache-dtype float16'` | 1 | `Value error, --replayssm-state-dtype leapquant cannot be used with --enable-mamba-cache-stochastic-rounding. Stochastic rounding is for FP16 SSM caches. A LeapQuant state is INT8 with FP16 records. Remove --enable-mamba-cache-stochastic-rounding.` |
| a model without GDN (Qwen2.5-0.5B-Instruct, `MTP=0`) | 1 | `Value error, --replayssm-state-dtype leapquant stores the recurrent state of Gated DeltaNet layers. This model config has no linear_key_head_dim or linear_value_head_dim. Remove the flag.` |
| `REPLAYSSM_STATE_DTYPE=bogus` | 2 | `main.py serve: error: argument --replayssm-state-dtype: invalid choice: 'bogus' (choose from auto, leapquant)` |
| stale `VLLM_RDNA_GDN_LEAPQUANT=1`, in the environment or as a launcher argument | 1 | `ERROR: VLLM_RDNA_GDN_LEAPQUANT=1 was replaced by REPLAYSSM_STATE_DTYPE=leapquant (--replayssm-state-dtype leapquant). Unset it: the engine ignores it and would serve FP32 state.` The launcher exits before it starts anything |

Not run: non-gfx10x (the box has only gfx1030), AITER GDN, FlashInfer GDN, the
fused CUDA speculative decode and head dimensions other than 128. The CPU tests
of the config-guards note cover them. The connector-creation guard in
`KVConnectorFactory` sits behind the config check and is reachable only if the
config check is bypassed: it is covered by `test_leapquant_state_refused.py`
(6 passed).

## 6. Page packing on the 27B (`VLLM_RDNA_GDN_LEAPQUANT_PACK=1`)

**Kernels.** The four packed-page tests of the kernel file pass on gfx1030
(`test_packed_pages_match_isolated_blobs` and
`test_grouped_normalize_on_packed_pages`, two parametrizations each). Store,
normalize, decode and grouped normalize on layers at `k x page` offsets inside
one block equal the isolated blobs, and the bytes of the neighbouring layers and
of the slack stay unchanged. `mono_gdn_lq_rdna2` and the materialize kernel
have no packed-page test.

**Planner what-if on the 27B (CPU, before any boot).** The real KV-cache
planner and platform code, driven by the harness of the packing note with the
27B geometry (48 GDN layers, 16 attention layers and the MTP layer, 24 value
heads and 2 KV heads per rank), 12,000,000,000 bytes of KV per rank and
`max_model_len` 262,144. It gave the engine's number to the digit for `A` and
`B`.

| Config | Groups (GDN) | GPU KV cache size | Against flag off |
|---|---|---:|---:|
| FP32, block 1024 (`A`) | 3 x 16 | 328,658 | |
| LeapQuant, block 1024 (`B`) | 3 x 16 | 325,019 | -1.1% |
| LeapQuant + packing, block 1024 (`P`) | 2 x 24 | 331,129 | +0.75% |
| LeapQuant, block 512 (`C512`) | 3 x 16 | 334,768 | +1.9% |
| LeapQuant + packing, block 512 (`C512P`) | 3 x 16 | 334,768 | +1.9%, identical to `C512`: a no-op |
| LeapQuant + packing, block 576, 608, 640 | 3 x 16 | 332,828, 331,931, 331,843 | |
| LeapQuant + packing, block 768 | 2 x 24 | 333,637 | |

Packing applies on the 27B from a block of about 656 tokens: at the recipe's
block 1024 it turns 3 GDN groups of 16 into 2 of 24 (22,554,624 bytes per GDN
block) and gains 1.9% over the padded LeapQuant boot. At block 512 it changes
nothing. The LeapQuant GDN page is 939,776 bytes (888,576 of state, 51,200 of
conv), the FP32 page 1,624,064, and the attention page 2 MiB at block 1024 and
1 MiB at block 512.

**Engine boots: not run.** `P`, `C512` and `C512P` never booted (the chain was
terminated, see the status above). They must print, for `P`: `kv cache group sizes (1024,
1024, 1024)`, `GPU KV cache size: 331,129 tokens`, and `LeapQuant page packing:
48 Mamba layer(s) of 939776 bytes use 2 KV-cache group(s); one block holds 24
of them.` For `C512`: 334,768 tokens. For `C512P`: the same 334,768 and the
same group sizes as `C512`. [INFERENCE until booted] The prefix probe of `P`
checks the align-mode copy on unpadded pages, and its quality collects against
`B` check that packed output equals padded output.

## Still pending on the 27B (lease `a2`, HIP 0,1)

`A2` (the floor), `P`, `C512` and `C512P`. `A2` has probes but no collects, and
the other three never booted. The hand-off script released `a2` at 18:18 and
another job holds it now (`HI2.a2`, until 2026-11-16T04:18), so take `a2`
first (ask Main). Then run the arms into the same run directory, with the
partial `A2` directory kept aside:

```bash
cd ~/w4a8_runs/leapquant/pr170-gpu
R=$(cat quality.path); mv "$R/A2" "$R/A2.partial"
PROFILE=27b QUIET_OWN=<agent> ARMS="A2 P C512 C512P" RUN=$R bash run_matrix_pr170.sh
bash compare_pr170.sh
bash extract_boot_pr170.sh      # boot lines, capacities and probes of every arm
```

Then fill sections 3, 4 and 6 and tick the two 27B items of the PR body. The
Triton and Inductor caches of this run were deleted, so the first boot compiles
again.

## 7. Flash-Next at TP4: boots and KV capacity

Run on HIP 2-5 (lease `b4`) of the same box, between 19:57 and 20:58 on
2026-11-15, by `run_matrix_pr170.sh` (`PROFILE=flashnext`). Five arms booted one
after the other through `serve_validate.sh`, each with the probes, the quality
collects of section 8 and the greedy probe again. Run directory on the box:
`~/w4a8_runs/leapquant/pr170-gpu/matrix-flashnext-20261115-195724/` (one
directory per arm with `serve.log`, `summary.txt`, the probe files and the
collects; `compare.txt`, `topk_compare.md|json`, `*.compare.json` at the top).
The small result files are copied to
[`bench_results/2026-11-15_pr170-tp4/`](../../bench_results/2026-11-15_pr170-tp4/)
(`compare.txt`, `topk_compare.md`, `topk_compare.json`, the three
`decode_kl_*.compare.json`, and `bnm_compare.txt` for the arm that
`compare_pr170.sh` does not know). The files are unchanged copies, except that
`topk_compare.md` has one heading line added for the markdown lint.

| | |
|---|---|
| Tree | `tree-pr170` on the box: `GIT_SHA` `791392e04b` with the `_rocm_C` of section 1. The commits after it change only docs and one comment, plus (in the commit that adds this section) the registration of `VLLM_RDNA_GDN_LEAPQUANT_PACK` in `vllm/envs.py`, which this run did not have |
| Model | `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` (snapshot `0939125b`), served as `flash-next` |
| Engine | TP4 with EP4, MTP2 (`num_spec_tokens` 2), `FULL_AND_PIECEWISE`, V2 model runner, prefix caching with `--mamba-cache-mode align`, `--max-num-seqs 8`, `--max-num-batched-tokens 2048` (2560 for `P`), `--kv-cache-memory-bytes 4026531840` per rank, `max_model_len` 262,144 |
| Arms | `A`, `A2`: recipe `flashnext-mtp2` with `--monokernel`, flag off (`A2` is the boot-to-boot floor). `B`: recipe `flashnext-leapquant-mtp2` (flag on, MonoKernel on, block 512). `BNM`: `B` with `EXTRA_ARGS=` (MonoKernel off). `P`: `B` with `VLLM_RDNA_GDN_LEAPQUANT_PACK=1 BLOCK_SIZE=640 MAXBAT=2560` |

**Capacity: the engine's own numbers.**

| Arm | State | MonoKernel | Block | GPU KV cache size | Concurrency | Group sizes | GDN page padded by | GPU blocks | Model loading (GiB per rank) |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| `A` | FP32 | on | 1024 | **266,945** | 1.02x | (1024, 1024, 1024, 1024, 8, 1024) | 29.13% | 278 | 20.56 |
| `A2` | FP32 | on | 1024 | **266,945** | 1.02x | (1024, 1024, 1024, 1024, 8, 1024) | 29.13% | 278 | 20.56 |
| `B` | LeapQuant | on | 512 | **273,456** | 1.04x | (512, 512, 512, 512, 8, 512) | 11.58% | 556 | 20.28 |
| `BNM` | LeapQuant | off | 512 | **273,456** | 1.04x | (512, 512, 512, 512, 8, 512) | 11.58% | 556 | 20.28 |
| `P` | LeapQuant, packed | on | 640 | **273,220** | 1.04x | (640, 640, 640, 8, 640) | none (no padding line) | 444 | 20.37 |

`A` and `A2` give the same capacity and the same groups. `B` and `BNM` give the
same capacity that PR 168 measured (273,456 at block 512). `A` and `B` differ in
two things, the flag and the block of the recipe (1024 against 512), exactly as
in PR 168. `P` prints the same capacity the CPU planner computed before the
boot (273,220 at block 640, see the capacity analysis linked below): the planner
and the engine agree to the digit.

**Boot-level lines** (from `compare.txt` and `serve.log` of each arm).

| Line | `A`, `A2` | `B` | `BNM` | `P` |
|---|---|---|---|---|
| `GDN recurrent state: LeapQuant INT8 (window 32, r 4, 37024 B/head)` | absent | present | present | present |
| `LeapQuant GDN: one normalize launch per KV-cache group (N layers)` | absent | 12 layers | 12 layers | 18 layers |
| `LeapQuant scheduling: bounded output admission=True, decoder-first FCFS` | absent | present | present | present |
| `LeapQuant page packing: 36 Mamba layer(s) of 469888 bytes use 2 KV-cache group(s); one block holds 19 of them.` | absent | absent | absent | present |
| `vllm:cache_config_info` `replayssm_state_dtype` | `"auto"` | `"leapquant"` | `"leapquant"` | `"leapquant"` |
| `mono rdna2: --monokernel on TP rank N` lines | 4 | 4 | **0** | 4 |
| lines naming the retired `VLLM_RDNA_GDN_LEAPQUANT` | 0 | 0 | 0 | 1 (see below) |
| WARNING naming leapquant or replayssm | 0 | 0 | 0 | 1 (see below) |
| fault markers in `serve.log`, new amdgpu dmesg faults | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| KFD `evicted_ms` since READY (TP0, TP1, TP2, TP3) | +0 | +0 | +0 | +0 |
| READY | 281 s | 281 s | 311 s | 280 s |

The one line in `P` is the "Unknown vLLM environment variable detected:
VLLM_RDNA_GDN_LEAPQUANT_PACK" warning. The `compare.txt` counters match it as a
substring of the retired switch name `VLLM_RDNA_GDN_LEAPQUANT`; the engine did
not read the retired switch. The cause is that the packing switch was read from
`os.environ` and not registered in `vllm/envs.py`. The commit that adds this
section registers it (`vllm/envs.py`, with a test in `tests/test_envs.py`; the
switch is read through `envs` and is not a compile-cache factor). That change is
Python only and was tested on CPU, not booted: no arm of this run has it. The
other unknown-variable warnings are the RDNA variables of the launcher, the same
set in every arm (`VLLM_RDNA_AR_*`, `VLLM_GDN_HIP_PREFILL`, and others); the
three flag-on arms `B`, `BNM` and `P` also name `VLLM_RDNA_DECODE_SHARE`,
`VLLM_RDNA_DECODE_STALL_MS` and `VLLM_DECODE_STALL_MAX_OVERHEAD`, which only the
`flashnext-leapquant-mtp2` recipe sets.

**Probes** (launcher probes after READY, same inputs in every arm).

| Arm | greedy | PPL probe | prefix cache | concurrency (48 checks) |
|---|---|---|---|---|
| `A` | 3/3 | 4.80 (NLL 1.570) | cold 1.02 s, warm 0.34 s, 3.0x, hits +2048 of 5550 queries, warm hit rate 73.8% | 0 failing, 13 near-tie, 2 soft, 33 identical |
| `B` | 3/3 | 4.80 (NLL 1.569) | cold 1.16 s, warm 0.21 s, 5.6x, hits +2560, 92.3% | 0 failing, 10 near-tie, 5 soft, 33 identical |
| `A2` | 3/3 | 4.87 (NLL 1.582) | cold 1.02 s, warm 0.33 s, 3.1x, hits +2048, 73.8% | 0 failing, 14 near-tie, 1 soft, 33 identical |
| `P` | 3/3 | 4.86 (NLL 1.581) | cold 1.10 s, warm 0.21 s, 5.3x, hits +2560, 92.3% | 0 failing, 14 near-tie, 2 soft, 32 identical |
| `BNM` | 3/3 | 4.82 (NLL 1.572) | cold 1.09 s, warm 0.20 s, 5.5x, hits +2560, 92.3% | 0 failing, 15 near-tie, 0 soft, 33 identical |

All probes pass in all five arms. The prefix probe confirms that a warm request
resumes from LeapQuant state blocks (V2 align copy) with cold text equal to warm
text, with and without MonoKernel and on packed pages. The warm hit rate differs
between the flag-off arms (73.8%) and the flag-on arms (92.3%) because the hit
grain is the block: the cache serves whole blocks of the shared prefix, 2 x 1024
= 2048 tokens at block 1024 (`A`, `A2`), 5 x 512 = 2560 at block 512 (`B`,
`BNM`) and 4 x 640 = 2560 at block 640 (`P`). The TTFTs are the probe's own
output, not a measurement window.

## 8. Flash-Next at TP4: quality against the boot-to-boot floor

The collects of section 4 (same tools, same inputs) ran against each live
server. The floor is `A` against `A2`: two boots of the same flag-off setup. The
head is `A` against `B`. Confidence intervals are the 95% intervals of
`topk_kl.py` over 64 windows of 2048 tokens.

**Prompt level** (`topk_kl.py`, prefill path, top-20, docs of this tree):

| Pair | PPL ref to cand | dNLL [95% CI] | KL nats/token [95% CI] | KL p99 | Top-1 [95% CI] |
|---|---|---|---|---:|---|
| floor: `A` to `A2` | 2.7474 to 2.7493 | +0.00067 [-0.00060, +0.00201] | 0.01189 [0.00984, 0.01418] | 0.1878 | 0.9671 [0.9637, 0.9704] |
| **head: `A` to `B`** | 2.7474 to 2.7447 | **-0.00098** [-0.00213, +0.00016] | **0.01194** [0.00994, 0.01420] | 0.1835 | **0.9673** [0.9639, 0.9706] |
| pack: `B` to `P` | 2.7447 to 2.7464 | +0.00060 [-0.00051, +0.00173] | 0.01142 [0.00954, 0.01348] | 0.1742 | 0.9678 [0.9645, 0.9709] |
| MonoKernel: `B` to `BNM` | 2.7447 to 2.7489 | +0.00152 [+0.00031, +0.00285] | 0.01102 [0.00922, 0.01302] | 0.1733 | 0.9682 [0.9650, 0.9715] |
| `A` to `BNM` | 2.7474 to 2.7489 | +0.00054 [-0.00061, +0.00168] | 0.01187 [0.00990, 0.01408] | 0.1830 | 0.9665 [0.9631, 0.9699] |

The head KL (0.01194) equals the floor (0.01189): the flag adds nothing the
boot-to-boot noise does not already have. Its dNLL interval contains zero and
lies on the negative side of the floor's. Top-1 is 0.9673 against 0.9671. The
pack row (packed pages against padded pages at the same format) is also at the
floor. The `B` to `BNM` dNLL interval excludes zero by 0.0003: switching the
MonoKernel changes the kernel path of the GDN decode, and its shift of +0.0015
nats/token is of the size of the floor's interval half-width (0.0013), while its
KL and top-1 are at the floor. [INFERENCE] That is kernel-path noise, not the
LeapQuant state: `A` to `BNM` has dNLL +0.00054 [-0.00061, +0.00168], inside the
floor.

**Decode path** (`decode_kl.py`; KL per position up to the first token where
the two greedy runs disagree, so each pair compares only a few positions per
passage; the divergence median is 5 to 15 tokens and no pair has an identical
passage, floor included):

| Pair | 512 tokens, 32 passages: KL mean, positions, confident flips | Long, 2048 tokens, 8 passages of 15k-22k: KL mean, positions, confident flips |
|---|---|---|
| floor: `A` to `A2` | 9.0e-3, 1832, 3 | 9.8e-3, 331, 4 |
| **head: `A` to `B`** | **6.7e-3**, 2093, 3 | **1.02e-2**, 202, 2 |
| pack: `B` to `P` | 1.05e-2, 1524, 4 | not collected for `P` |
| MonoKernel: `B` to `BNM` | 9.4e-3, 1615, 4 | not compared |
| `A` to `BNM` | 1.00e-2, 1730, 3 | 1.9e-2, 175, 2 (p50 6.9e-4 against 2.6e-4 of the floor) |

The head is at or below the floor on the 512-token set (6.7e-3 against 9.0e-3)
and at the floor on the long set (1.02e-2 against 9.8e-3), with no more
confident flips than the floor. The one value above the floor is `A` to `BNM`
on the long set (1.9e-2 against 9.8e-3, from 175 positions of 8 passages, where
one passage can move the mean); the same pair on the 512-token set is 1.0e-2
against 9.0e-3. The tool gives no verdict for decode KL, so I do not call this a
failure, and I do not call it noise either: it is the one open sign in this
matrix, and it belongs to the MonoKernel-off path, not to the flag.

**Long-context PPL** (`quality_ab.py`, 6 passages, 256 greedy tokens each; the
gate of the tool is a PPL change of at most 2%):

| Pair | PPL | Change | Greedy exact | Mean first divergence | Gate |
|---|---|---|---:|---:|---|
| floor: `A` to `A2` | 3.3319 to 3.3207 | -0.336% | 2 of 6 | 72.3 | PASS |
| **head: `A` to `B`** | 3.3319 to 3.3377 | **+0.173%** | 3 of 6 | 92.7 | PASS |
| pack: `B` to `P` | 3.3377 to 3.3262 | -0.342% | 3 of 6 | 92.7 | PASS |
| MonoKernel: `B` to `BNM` | 3.3377 to 3.3239 | -0.412% | 1 of 6 | 50.3 | PASS |
| `A` to `BNM` | 3.3319 to 3.3239 | -0.240% | 1 of 6 | 38.7 | PASS |

The floor itself moves by 0.34%, larger than the head's 0.17%: the flag is
inside the boot-to-boot spread. The MonoKernel-off arm diverges earlier in
greedy decoding (exact 1 of 6; against `A`, first divergence at token 0 for two
passages). With 6 passages and near-tie tokens that is not enough to call it more
than a different kernel path, and its PPL is inside the gate and the floor.

## 9. Flash-Next at TP4: MonoKernel off (`BNM`)

`BNM` is `B` with `EXTRA_ARGS=` empty: the recipe runs without `--monokernel`.
It is not in `compare.txt` (`compare_pr170.sh` knows `A`, `B`, `A2`, `P`); the
boot, probe and eviction facts are in the tables of section 7 and the quality
comparisons in `bench_results/2026-11-15_pr170-tp4/bnm_compare.txt`, computed
afterwards on the box on CPU with the same tools from the collects of the run.

- The MonoKernel is really off: the four `mono rdna2: --monokernel on TP rank N`
  lines and the `mono gdn_rdna2: fused GDN decode core enabled` line are absent,
  the non-default args do not name `monokernel`; `A`, `B` and `P` have all of
  them.
- The setup is accepted by the validator and boots: LeapQuant INT8 line,
  `GPU KV cache size: 273,456 tokens`, groups `(512, 512, 512, 512, 8, 512)`, 12
  layers per normalize launch: the same as `B`. The capacity does not depend on
  the MonoKernel.
- Probes pass (greedy 3/3, PPL 4.82, prefix 5.5x with cold text equal to warm
  text, concurrency 0 failing), 0 KFD evictions, 0 faults.
- Quality (section 8): prompt-level and long-context numbers are within the
  floor; the long decode set is the one open sign.

## 10. What #170 adds to capacity, honestly

- The capacity gain that ships is #168's: 273,456 tokens at block 512 against
  266,945 at block 1024 flag off, **+2.44%** (`B` against `A`, measured again
  here). It comes from the flag and the recipe's smaller block together.
- **Packing at block 640 gives 273,220 tokens: -0.09% against `B` at block 512**
  (-236 tokens), +2.35% against `A`. At the block `B` already uses (512) packing
  changes nothing: it is the same three groups of 12 layers. Packing pays only
  where the block is large (on the 27B at block 1024 the planner gives +0.75%,
  against -1.1% padded; that boot did not run). It also lets Flash-Next boot at
  block 640 with five groups instead of six and no page padding, which is the
  only effect `P` shows.
- What #170 adds: the config flag `--replayssm-state-dtype` with its validator
  (8 refused setups, section 5), the fold guard of the HIP kernel (tests and
  control of section 2), and the opt-in packing. It does not add KV capacity on
  the shipped Flash-Next recipe.
- Why the capacity cannot be much larger: the attention KV is 92% of the pool and
  the state a request needs is 2.1% of it, so even deleting all state gains
  +6.64%. The options table is in the
  [capacity analysis](https://github.com/opengfx1030/vllm-rdna/blob/feat/leapquant-capacity-analysis/docs/rdna2/leapquant-capacity-analysis-2026-11-15.md)
  (branch `feat/leapquant-capacity-analysis` at `1baf9a7f37`, CPU only, built on
  this PR's head `57c2c1c6f0`). It predicted 273,220 for block 640: the boot of
  `P` agrees.

## 11. What is not validated

- **27B floor and packing boots.** `A2`, `P`, `C512` and `C512P` on the 27B at
  TP2 never ran (section 3 and "Still pending on the 27B"). They are dropped: the
  27B is not the subject of this PR, and the only 27B result about the flag is
  `A` against `B` without a floor. The 27B packing numbers are planner values.
- **No timed throughput window.** No tok/s, TTFT window or quiet file exists for
  any arm. The probe TTFTs and the 5.5x prefix speedups are single probe
  requests, not benchmarks. This PR makes no speed claim and none should be
  taken from this run. The `Model loading` difference (20.28 against 20.56 GiB
  per rank) is not a measurement of memory saved with utilization-based sizing:
  every arm fixed `--kv-cache-memory-bytes`.
- **MTP drafter layers** are exercised only through the `flashnext-mtp2` and
  `flashnext-leapquant-mtp2` recipes: the probes and the quality collects ran
  with speculative decoding on. No acceptance length or drafter-only check was
  taken.
- **V1 model runner** is out of scope by the decision of the user. The
  validator refuses LeapQuant with align prefix caching on the V1 runner
  (section 5); only the V2 runner was booted.
- One boot per arm and one floor pair (`A`, `A2`); one packing block (640); TP4
  with EP4 only; `max_num_seqs` 8; quality inputs are the repo's docs, not a
  task benchmark.
- The `vllm/envs.py` registration of `VLLM_RDNA_GDN_LEAPQUANT_PACK` was not
  booted (CPU tests only).
- #168's quality checks were re-run only through this matrix (arms `A`, `A2`,
  `B`); there is no separate re-run.

Commands of this run (the scripts are on the box in
`~/w4a8_runs/leapquant/pr170-gpu/`; the lease helper is sourced in the shell
that starts the matrix; boots take the box boot lock):

```bash
export LEASE=b4 OWNER=<agent>
source ~/w4a8_runs/leapquant/lease.sh
lease_take "PR170 Flash-Next TP4 validation" 300 && lease_check_buses
cd ~/w4a8_runs/leapquant/pr170-gpu
PROFILE=flashnext QUIET_OWN=<agent> ARMS="B A A2" bash run_matrix_pr170.sh
PROFILE=flashnext QUIET_OWN=<agent> ARMS="P" RUN=$(cat matrix-flashnext.path) bash run_matrix_pr170.sh
PROFILE=flashnext QUIET_OWN=<agent> ARMS="BNM" RUN=$(cat matrix-flashnext.path) bash run_matrix_pr170.sh
bash compare_pr170.sh "$(cat matrix-flashnext.path)"
```

`DRY=1` prints the launches without booting.

## Commands

Run on par1-cs25 as `<user>`. The scripts are on the box, not in the
repository: `~/w4a8_runs/leapquant/pr170-gpu/` holds them and every log of this
note (`build_pr170.log`, `kernel-tests-*`, `cpu-tests-*`, `negative-*`,
`quality-*`, `compare.txt` inside the quality run). Every GPU command sets
`HIP_VISIBLE_DEVICES` and checks the PCI buses first.

```bash
# lease a2 (HIP 0,1 = buses d1, d4); a separate holder script keeps it for the session
export LEASE=a2 OWNER=LeapQuant; source ~/w4a8_runs/leapquant/lease.sh
lease_take "PR170 GPU validation" 300 && lease_check_buses
cd ~/w4a8_runs/leapquant/pr170-gpu
bash build_pr170.sh                       # CPU: flock build.lock, MAX_JOBS=8, only the _rocm_C target
TAG=main bash run_kernel_tests.sh         # HIP 0: op schemas, then the two kernel files
TAG=load1 LOAD=1 bash run_kernel_tests.sh # the same with an FP16 matmul loop on HIP 0 (also load2)
DEV=1 BUS=d4 TAG=hip1 bash run_kernel_tests.sh   # the same on HIP 1 (bus d4)
GEO=4,12 TAG=geo-fn bash run_kernel_geometry.sh   # the same at the Flash-Next per-rank heads (also GEO=8,24 TAG=geo-27b)
bash mutation_build.sh                    # CPU: _rocm_C from the kernel source before the fold guard, in tree-pr170-pre
T=$HOME/w4a8_runs/leapquant/tree-pr170-pre TAG=mutation bash run_kernel_tests.sh   # HIP 0: the control of section 2
bash run_cpu_tests.sh                     # no GPU visible: the eight CPU files of the PR
bash run_negative_boots.sh                # launcher refusals of section 5
ARMS="B A A2 P C512 C512P" bash run_quality_pr170.sh   # boots, probes and the quality collects
bash compare_pr170.sh                     # KL, PPL and decode-KL against the floor
```

`run_quality_pr170.sh` calls `serve_validate.sh` with `RECIPE=27b-awq TP=2 MTP=2
KV=12000000000` and `REPLAYSSM_STATE_DTYPE=leapquant` for the flag-on arms
(`VLLM_RDNA_GDN_LEAPQUANT_PACK=1` for `P`, `BLOCK_SIZE=512` for `C512`).
Its hook runs, per arm:

| Collect | Command | What it measures |
|---|---|---|
| prompt level | `topk_kl.py collect --top 20 --limit 64` on 64 windows of 2048 tokens of this tree's `docs/**/*.md` | KL, dNLL, top-1 of the prefill path |
| long-context PPL | `quality_ab.py collect --passages 6 --passage-chars 48000 --gen-tokens 256` | teacher-forced PPL over 6 passages of 12k-19k tokens; chunked prefill crosses many LeapQuant stores |
| decode path | `decode_kl.py collect --passages 32 --passage-chars 6000 --gen-tokens 512 --workers 8` | KL of greedy decode, 8 concurrent requests |
| decode path, long | `decode_kl.py collect --passages 8 --passage-chars 60000 --gen-tokens 2048 --workers 8` | the same at 15k-22k context |

`compare_pr170.sh` runs `topk_kl.py compare --pair floor=A,A2 --pair head=A,B`,
`decode_kl.py compare A B --floor A2` and `quality_ab.py compare` (A to B, A to A2).
`run_matrix_pr170.sh` is the same flow for either model; it produced sections 7 to 10.
`mutation_build.sh` builds the control `.so` of section 2.
