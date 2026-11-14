# LeapQuant GDN: one flag, one validator (2026-11-14)

LeapQuant stores the Gated DeltaNet (GDN) recurrent state as an INT8 blob. See
[LeapQuant INT8 GDN recurrent state](leapquant-gdn-2026-11-13.md) for the
format, the kernels, and the measurements. This document covers four changes:

1. The flag `--replayssm-state-dtype leapquant` replaces the environment
   variable `VLLM_RDNA_GDN_LEAPQUANT`.
2. One validator rejects every unsupported setup at start-up.
3. Each LeapQuant state carries a format identity in its KV cache spec.
4. KV connectors and KV offloading are rejected when LeapQuant state is on.

The LeapQuant RFC ([vllm#64725](https://github.com/vllm-project/vllm/issues/64725))
lists these checks. This document says which ones this fork takes, which ones it
changes, and why.

**Status.** CPU unit tests pass (see [Tests](#tests)). No part of this change
ran on a GPU. [Pending GPU validation](#pending-gpu-validation) lists the checks
that need a gfx103 host.

## Credits

| Source | What this change takes from it |
|---|---|
| RFC [vllm#64725](https://github.com/vllm-project/vllm/issues/64725), lihaoyang-amd | The list of checks: the rejected setups, the format identifier, and the flag design (sections 3 and 5). No code. |
| Draft PR [vllm#52962](https://github.com/vllm-project/vllm/pull/52962), askliar | The flag name `--replayssm-state-dtype` and its place next to `--replayssm-buffer-len`. No code, no tests. |
| ReplaySSM [vllm#48018](https://github.com/vllm-project/vllm/pull/48018) and GDN decode [vllm#62004](https://github.com/vllm-project/vllm/pull/62004), Johnny-Liou | Background for the `--use-replayssm` decision. #62004 is not in this tree. No code. |
| LeapQuant, [arXiv 2609.38166](https://arxiv.org/abs/2609.38166) | The state format. This change does not alter it. |

The validator, the format identity, the spec and connector changes, the
launcher changes, and the tests are original work for this fork. The files that
use the RFC list name it in their header: `leapquant_gdn.py` and the two new
test files.

## The flag

| Interface | Name | Values | Default |
|---|---|---|---|
| CLI and API server | `--replayssm-state-dtype` | `auto`, `leapquant` | `auto` |
| Python | `CacheConfig.replayssm_state_dtype` | `"auto"`, `"leapquant"` | `"auto"` |
| Launcher (`tools/rdna/serve_rdna.sh`) | `REPLAYSSM_STATE_DTYPE` | empty, `auto`, `leapquant` | empty: no flag |
| Removed | `VLLM_RDNA_GDN_LEAPQUANT` | | |

- `auto`: the model keeps its own GDN state. Nothing changes.
- `leapquant`: each GDN state is an INT8 residual with FP16 compensators and a
  ring of FP16 records. This change alters model outputs. It runs only on RDNA2
  (gfx10x) with the HIP kernels.

The launcher puts the flag after `--block-size` in the server command.

The engine does not read `VLLM_RDNA_GDN_LEAPQUANT` any more. There is no alias.
The launcher stops with an error when `VLLM_RDNA_GDN_LEAPQUANT=1` is set, in the
environment or in a recipe. Without this stop, the server would start with FP32
state and the user would not know. A server started without the launcher still
gets the engine warning `Unknown vLLM environment variable detected`. The
option `--fail-on-environ-validation` turns that warning into an error.

### Fixed parameters

The flag has no other parameter. These values are constants in
`vllm/model_executor/layers/mamba/ops/leapquant_gdn.py`.

| Parameter | Value | Reason |
|---|---|---|
| Compensator rank r | 4 | The RFC fixes r at 4. |
| Window p | 32 | See [Why the window is 32](#why-the-window-is-32). |
| Record dtype | FP16 (the decay stays FP32) | The qualified layout. |
| Head dimensions | K = V = 128 | The kernels are written for 128. |
| Layout version | 1 | Raise it with every layout change. |

## Why `--use-replayssm` is rejected

The RFC says `--replayssm-state-dtype leapquant` requires `--use-replayssm`.
This fork rejects the combination. Four facts support this:

1. Qwen GDN models do not declare ReplaySSM support. In this tree only
   `nemotron_h` and `kimi_k3/nvidia` declare `SupportsReplaySSM`.
   `validate_mamba_cached_kernel` raises `--use-replayssm is not supported for
   architecture ...` for every other model. A rule that requires the flag
   rejects every Qwen GDN model.
2. `--use-replayssm` sets `num_speculative_blocks` to 0 for Mamba specs. The
   MTP2 layout, the scheduler policy, and the MTP2 measurements in the first
   document use the speculative state blocks. [INFERENCE: with zero
   speculative blocks the block tables that the LeapQuant kernels read would
   change. No test covers this.]
3. This tree has no GDN ReplaySSM decode kernel. vllm#62004 is open upstream.
   LeapQuant keeps its own record ring in each state blob.
4. If the validator accepted `--use-replayssm` and ignored it, the user would
   think that ReplaySSM is active. It would not be. That is a silent fallback.
   The validator raises instead.

Revisit this rule when #62004 lands and Qwen GDN models declare
`SupportsReplaySSM`.

## Why the window is 32

The RFC reuses `--replayssm-buffer-len` as the window. This fork fixes the
window at 32 and does not read `--replayssm-buffer-len`.

The table shows the cost of other windows for Flash-Next at TP4 and MTP2: 12
value heads per rank, a 25 KiB convolution state, and a 512 KiB page. The blob
sizes come from the layout oracle (`LeapQuantLayout.nbytes`). The oracle ran on
a CPU.

| Window | Bytes per head | 12 heads + convolution state | Fits a 512 KiB page |
|---:|---:|---:|---|
| 16 | 28,768 | 362.125 KiB | yes |
| 32 | 37,024 | 458.875 KiB | yes |
| 40 | 41,152 | 507.250 KiB | yes |
| 41 | 41,664 | 513.250 KiB | no |
| 48 | 45,280 | 555.625 KiB | no |

Reasons:

1. `--replayssm-buffer-len` defaults to 16. If the window followed it, the flag
   alone would select window 16, not the qualified window 32.
2. The window sets the blob size. Windows 16 and 32 fit the same 512 KiB page,
   so the padded page size and the KV capacity (273,456 tokens, group sizes
   `(512, 512, 512, 512, 8, 512)`) in the first document are the same for both.
   No window above 40 fits this page at TP4 and MTP2.
3. Window 32 has the accepted serving controls (A1, C1, C2, A2) in the first
   document. The GPU kernel tests cover windows 16 and 32.
4. The format identity contains `p32`. A later change of the window cannot mix
   states.
5. Windows 33 to 40 fit the page at this TP and MTP size but have no
   qualification. A free window needs a page-fit rule in the validator and new
   GPU runs. This change does not add them.

## Rejected setups

`verify_leapquant_gdn` in `vllm/model_executor/layers/mamba/ops/leapquant_gdn.py`
is the only place that decides. It raises `ValueError`. Every message starts
with `--replayssm-state-dtype leapquant` and names the cause and the fix. There
is no fallback to FP32 state.

The check runs at three points:

| Point | When | What it knows |
|---|---|---|
| Config | `VllmConfig.__post_init__`, after the model-specific config update | The whole config and the model's text config. |
| Layer | Each LeapQuant GDN layer, in `_init_leapquant` | The layer's head dimensions, the selected decode kernel, the selected prefill backend. |
| Connector | `KVConnectorFactory.create_connector` | The KV cache config. |

At the config point, pydantic wraps the message in a `ValidationError`, which is
a `ValueError`. The model-specific config update sets `mamba_cache_mode`, so the
validator must run after it. A test covers this order.

| Setup | Checked at | Origin |
|---|---|---|
| Host is not gfx10x ROCm with `VLLM_RDNA_ALL_HIP=1` | config, layer | First PR |
| `--use-replayssm` | config | This change. The RFC requires the flag; this fork rejects it. |
| Stochastic rounding (`--enable-mamba-cache-stochastic-rounding`) | config | RFC |
| KV connector (`--kv-transfer-config`) | config, connector | RFC |
| KV offloading (`--kv-offloading-size`) | config, connector | RFC |
| Mamba align prefix caching on the V1 model runner | config, layer | First PR |
| AITER GDN kernels enabled | config, layer | First PR |
| FlashInfer GDN prefill backend | layer | First PR |
| Fused CUDA GDN spec-decode kernel | layer | First PR |
| Head dimensions other than 128, from the layer | layer, state shape | First PR |
| Head dimensions other than 128, from the model config | config | This change |
| Model config without GDN head dimensions | config | RFC (non-GDN and non-KDA layers) |
| GDN layer class without LeapQuant support | layer | First PR |

Notes:

- `MambaConfig` already rejects stochastic rounding on every platform except
  CUDA when it is built. The LeapQuant rule is a second check. It fires only
  for a config that changes after it is built.
- The first PR (#168) already rejected the rows marked "First PR", in several
  places. This change moves them into the validator without changing their
  meaning. The AITER row uses the same two predicates as before.
- A model config has no GDN head dimensions when its text config lacks
  `linear_key_head_dim` or `linear_value_head_dim`. Kimi-Linear and GLM-5.3-Flash
  (KDA) have no such attribute in this tree. [INFERENCE: no KDA checkpoint
  config was loaded.]
- Olmo-Hybrid, MiniCPM-V 4.6, and Alice-AI use GDN layers that do not read
  LeapQuant blobs. Those layers raise `--replayssm-state-dtype leapquant is not
  implemented for <layer>` when they are built.
- A drafter can get its own `VllmConfig` that holds the draft model config
  (the Qwen4Exp MTP model builds one with `replace`). The validator then skips
  the model-config check, because the target model decides. The layer check
  still runs for any GDN layer of the drafter. The real Flash-Next MTP draft
  config (`Qwen4ExpMTP`) has the head dimensions 128 and 128.

### What stays supported

| RFC | This fork | Why |
|---|---|---|
| Rejects speculative decoding | MTP stays supported (MTP2 is the qualified width) | The layout carries `num_spec`: capacity = window + `num_spec`. The scheduler policy and the measurements use MTP2. |
| Rejects `--monokernel` | Supported | `mono_gdn_lq_rdna2` reads LeapQuant blobs. |
| Requires `--use-replayssm` | Rejects it | See [above](#why---use-replayssm-is-rejected). |
| Window = `--replayssm-buffer-len` | Window 32, fixed | See [Why the window is 32](#why-the-window-is-32). |

Mamba align prefix caching on the V2 model runner and FULL_AND_PIECEWISE also
stay supported. The align copy marker now comes from the spec (see below). That
path has no GPU run yet. [INFERENCE: it must behave as before.]

## State format identity

The RFC asks for a format identifier (format, r, p, record dtype) so that FP32
state and LeapQuant state are never mixed. This change puts it at the spec
level.

`LeapQuantLayout.state_format` builds the string. For the MTP2 serving layout:

```text
leapquant-int8/v1/k128v128/r4/p32/cap34/rec-float16
```

| Part | Meaning |
|---|---|
| `leapquant-int8` | The format. |
| `v1` | The layout version (`LEAPQUANT_LAYOUT_VERSION`). |
| `k128v128` | Key and value head dimensions. |
| `r4` | Compensator rank. |
| `p32` | Window. |
| `cap34` | Record ring capacity: window + `num_spec`. |
| `rec-float16` | Record dtype. |

The identity does not contain `power_iters`. That value changes how a fold fits
the compensators. It does not change what a blob holds.

How the identity is used:

- `MambaSpec.state_format` holds it. `None` means the model's own state.
- It is part of the `MambaSpec` key and of `is_uniform_with_collection`. Layers
  with different formats never share a KV cache group. Two layouts of equal
  size but different versions are different specs.
- `KVCacheConfig.mamba_state_formats` lists the formats of a KV cache config.
  The connector factory reads it.
- `select_state_copy_funcs` in `vllm/v1/worker/mamba_utils.py` reads it. A
  LeapQuant spec gets the copy marker `get_leapquant_copy_spec`. The model's own
  `get_mamba_state_copy_func()` cannot see the config, so the spec carries the
  decision. Before this change the environment variable carried it.

A test pins the byte offsets of the serving layout to the layout version. If a
change moves a section, the test fails until the version and the table change.

**No marker inside the blob.** The blob header stays `[cursor, pending, 0, 0]`.
A zero blob stays the valid zero state, and a fresh block needs no
initialization. The kernels would need a rebuild to read or write a marker.
Kernel support for an in-blob marker is not built.

**No consumer yet outside the process.** This tree has no KV compatibility hash
and no connector that accepts LeapQuant state. The connector rejection is the
safeguard. When a connector gains support, the format identity must go into its
compatibility key.

## KV connectors and offloading

A connector moves state blocks and does not know the LeapQuant blob format. The
change rejects all connectors, also the ones that support hybrid models.

| Point | Rule |
|---|---|
| Config | `--kv-transfer-config` with a connector, or `--kv-offloading-size`, with `--replayssm-state-dtype leapquant` raises at config time. `--kv-offloading-size` becomes a connector late in config construction, so the validator checks the option itself. |
| Connector creation | `KVConnectorFactory.create_connector` raises when the KV cache config has a quantized Mamba state format. The scheduler and the worker setup both call it. |

The connector point reads the spec identity, not the flag. It also stops a
future path that builds a LeapQuant spec without the flag.

## Flag-off behavior

With `--replayssm-state-dtype auto` the server does the same as before. These
differences remain:

- `--help` lists the new option.
- The `vllm:cache_config_info` metric has a new label, `replayssm_state_dtype`,
  because the metric lists all `CacheConfig` fields. `CacheConfig.metrics_info()`
  returns 37 keys before and 38 after, and the new value is `auto`.
- `MambaSpec` has a new field, `state_format`, with the default `None`.
- The state calculators have a new optional argument.
  `gated_delta_net_state_copy_func()` no longer reads any switch.
- `leapquant_gdn_enabled()` and the environment variable are gone.

`CacheConfig.compute_hash` ignores the new field, so the compile cache key does
not change. The next table lists what a CPU run observed on par1-cs25 (no GPU
visible). The base tree is the first PR head, `7b4aff7fe3`. The new tree is this
branch. Flag-off values are equal in both trees:

| Value | Base | New |
|---|---|---|
| `CacheConfig().compute_hash()` | `d9403301…` | `d9403301…` |
| `CacheConfig(block_size=1024, mamba_cache_mode="align")` hash | `39f6b341…` | `39f6b341…` |
| `VllmConfig` default hash | `5a3fcb0883` | `5a3fcb0883` |
| Real Flash-Next config, block 512, prefix caching, TP1: `VllmConfig` hash | `7c5848228a` | `7c5848228a` |
| Same with MTP2 | `c916d43bb5` | `c916d43bb5` |
| GDN state dtypes, shapes, copy function names (3 dtype cases, 4 shape cases) | | equal |
| `MambaSpec` page size, `max_memory_usage_bytes` (TP4, MTP2 shapes) | 418,816; 1,675,264 | 418,816; 1,675,264 |

The Flash-Next rows load the real model config on the CPU with
`on_rdna2_hip` and the device capability stubbed. They are config probes, not
server starts.

### Dry run of the recipe

`PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=flashnext-leapquant-mtp2` prints
the server command without starting it. The run used the same tree path, the
same model path, and the same `VENV` before and after the change. The GPU
variables were unset. No GPU was used.

Result: the command is the same except for the flag spelling.

- The command has 44 tokens before and 46 after.
- The command after the change equals the command before it, with
  `--replayssm-state-dtype leapquant` inserted after `--block-size 512`.
- The environment block loses `export VLLM_RDNA_GDN_LEAPQUANT='1'` and gains
  nothing.
- Every other plan line is equal.

Other launcher checks, all in print mode:

| Check | Result |
|---|---|
| `VLLM_RDNA_GDN_LEAPQUANT=1` in the environment | Exit 1. The message names `REPLAYSSM_STATE_DTYPE=leapquant`. |
| `VLLM_RDNA_GDN_LEAPQUANT=1` as a launcher argument | Exit 1. Same message. |
| `VLLM_RDNA_GDN_LEAPQUANT=0` | The plan prints (exit 0). |
| `REPLAYSSM_STATE_DTYPE=auto` | The command has `--block-size 512 --replayssm-state-dtype auto`. |
| `RECIPE=flashnext-mtp2 REPLAYSSM_STATE_DTYPE=leapquant` | The command has `--block-size 1024 --replayssm-state-dtype leapquant`. |
| `RECIPE=flashnext-mtp2` | The command has no `--replayssm-state-dtype`. |
| All 14 recipes, flag off, against the launcher and recipes of the first PR head | 13 plans are identical. The 14th is `flashnext-leapquant-mtp2`, which differs as above. The tree name was normalized. |

The plans and the output of every check in this document are in
`bench_results/2026-11-14_leapquant-config-guards/`: `dryrun-before.txt`,
`dryrun-after.txt`, and `cpu-receipts.txt`.

## Migration

| Before | After |
|---|---|
| `VLLM_RDNA_GDN_LEAPQUANT=1` in the server environment | `--replayssm-state-dtype leapquant` |
| `VLLM_RDNA_GDN_LEAPQUANT=1` on the launcher command line or in a recipe | `REPLAYSSM_STATE_DTYPE=leapquant` |
| `RDSE_ENV_KEYS` exported the variable | The launcher passes the CLI flag |

```bash
bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 REPLAYSSM_STATE_DTYPE=leapquant
bash tools/rdna/serve_rdna.sh RECIPE=flashnext-leapquant-mtp2
```

The worker log prints `GDN recurrent state: LeapQuant INT8 (...)`.

## Tests

All tests in this section ran on a CPU, on par1-cs25, with no GPU visible
(`HIP_VISIBLE_DEVICES=`). Run them from the repository root:

```bash
HIP_VISIBLE_DEVICES= pytest -q -rs \
  tests/config/test_leapquant_config.py \
  tests/v1/kv_connector/unit/test_leapquant_state_refused.py \
  tests/v1/core/test_leapquant_scheduling.py \
  tests/tools/rdna/test_serve_replayssm_state_dtype.py
```

| File | What it checks | Result |
|---|---|---|
| `tests/config/test_leapquant_config.py` (new) | The flag and the CLI. Each config-time refusal. Each layer-time refusal. The state calculators, the layout offsets, the state format, the spec key, the copy functions, and `mamba_state_formats`. | 38 passed |
| `tests/v1/kv_connector/unit/test_leapquant_state_refused.py` (new) | The connector factory refusal. | 6 passed |
| `tests/v1/core/test_leapquant_scheduling.py` (changed) | The growth gate follows the flag. | 25 passed, 1 skipped. The skip pre-dates this change: it needs two concurrent batches. |
| `tests/tools/rdna/test_serve_replayssm_state_dtype.py` (new) | The launcher key and the retired switch, in print mode. | 7 passed. Against the launcher of the first PR head, 6 of the 7 fail. |
| `tests/kernels/mamba/test_gdn_leapquant_rdna2.py` (not changed) | The GPU kernels. | 22 skipped: it needs a gfx103 GPU. Pending GPU validation. |

The three CPU test files for the engine together: 69 passed, 1 skipped. The
launcher test file and `tests/tools/rdna/test_serve_pci_buses.py` together:
11 passed.

The new tests use real config objects: `CacheConfig`, `VllmConfig`, `EngineArgs`,
a saved tiny Qwen3-Next config, and `KVCacheConfig`. They patch host facts
only: `on_rdna2_hip`, the AITER predicates, and the runner and removed
environment variables. One test builds a GDN layer without its constructor
and replaces `MambaBase.get_kv_cache_spec`, so that it checks the spec stamp
alone. The launcher test runs the real launcher and the real recipes with a
stub Python, so no GPU is touched. The Buildkite pipelines do not run it. Like
its neighbours in `tests/tools/rdna`, it is in the tethering allowlist.

### Regression runs

The same test files ran in a tree of the first PR head and in this tree.
The failing tests are the same tests in both trees.

| Set | First PR head | This branch |
|---|---|---|
| 11 files: connector, offloading, KV cache utils and planning, contiguous packing, mamba align chunk split, single-type manager (`tests/v1/kv_connector/unit/*`, `tests/v1/core/*`) | 8 failed, 442 passed, 1 skipped | 8 failed, 442 passed, 1 skipped |
| 13 files: engine arguments, KV cache dtype, config utils, GDN and mamba kernels, GDN metadata builder, mamba align block table, KDA metadata, Alice-AI, Qwen4Exp PLE, GLM-5 KDA state dtype | 173 failed, 246 passed, 204 skipped | 173 failed, 246 passed, 204 skipped |

The box has no GPU. Two causes were read from the output: a failed model
download (`test_multi_example_connector_consistency`) and an assertion on a
chunk boundary (`test_flash_next_tp4_materializes_replay_state`). The other
failures were not investigated one by one. [INFERENCE: most need a GPU.] These
tests do not give evidence for this change. Run them again on a GPU host.

## Pending GPU validation

Nothing in this change ran on a GPU. These checks must run on a gfx103 host
before the change is accepted as equal to the environment-variable path:

1. **Boot and serve.** Start `RECIPE=flashnext-leapquant-mtp2` with the flag.
   Compare with the first document: the KV capacity (273,456 tokens), the
   group sizes `(512, 512, 512, 512, 8, 512)`, the worker log line, and the
   greedy, prefix-cache, and concurrency probes.
2. **Layer check in a worker.** The layer-time validator call runs in each GDN
   layer of each rank, including the MTP drafter layers and the MonoKernel
   path.
3. **Align copy marker.** On the V2 runner, prefix-cache hits must use
   `get_leapquant_copy_spec` through `select_state_copy_funcs`. Before this
   change the environment variable set it.
4. **Graphs and speculation.** FULL_AND_PIECEWISE capture and replay, MTP2,
   and `--monokernel` must behave as before with the flag.
5. **Kernel tests.** Run `tests/kernels/mamba/test_gdn_leapquant_rdna2.py` on
   gfx103. The layer constructor path changed.
6. **Rejection at boot.** Start with `--kv-transfer-config` or
   `--kv-offloading-size` and the flag. The server must stop before it loads
   weights. CPU tests cover the config point only.
7. **Dynamic speculation.** The error text changed. The behavior must not.
8. **Tests that failed in both trees.** The 8 and 173 tests under
   [Regression runs](#regression-runs) fail on a CPU host with and without
   this change. Run them on a GPU host.

## Not done

- No marker inside the blob (kernel support not built).
- No window or rank option, and no KV compatibility key. See above.
- RFC section 4, items 11 to 13 (kernel, builder, and end-to-end tests;
  accuracy against a per-window 16-bit baseline; performance on other GPUs):
  this change adds none of them.
- `qwen4_exp/nvidia`, Olmo-Hybrid, MiniCPM-V 4.6, and Alice-AI do not pass the
  flag to their state calculators. LeapQuant cannot run on NVIDIA. The other
  three reject the flag when their layers are built.
