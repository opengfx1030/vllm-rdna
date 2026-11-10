# Hetero MoE hub: slot-pool quant method, runtime, launcher (2026-11-07)

Branch `feat/hetero-hub`, merged from `origin/feat/hetero-moe` (spec), `origin/feat/rdna3-flashnext`
(`30b6a47eca`, R3 port) and `origin/feat/expert-stats` (solver and validator). This is the hub side of
slices S2, S4-H1 and S5-H1 of `docs/rdna2/hetero-moe-design-2026-11-07.md` (sections 6, 9, 10, 11 and
Appendix C): the process that owns the engine, the plan on the hub, `HeteroMoEMethod`, the per-step
protocol, the launcher and the two recipes. The `hx` transport, the expert server process and the HIP
kernels belong to `feat/hetero-hx` (`docs/rdna2/hetero-hx-2026-11-07.md`).

**Status.** Everything below that says "tested" was run on a CPU only: the Mac for the plan, tables,
runtime, launcher and harness tests, the box (`par1-cs25`, CPU-only environment) for the quant method
tests, which need the vLLM stack. GPU: the V620 stand-in (interim layout: hub HIP 1, servers HIP 2-5; no
W7800) boots in eager and in production compiled mode (section 10.1, numbers in section 13): self-test
2.6e-03 and 2.8e-03, PPL 4.85, c1 ITL 43.9 ms, c8 ITL 50.2 ms; the prefix probe fails on its 2 GiB KV pool
and the concurrency probe flips in about one compiled run of five (open, section 11 item 9). Every W7800
step (section 10.2) is blocked until a W7800 is installed. What is still `[INFERENCE]` is marked.

## 1. What the hub side does, in one paragraph

With `VLLM_HETERO_MOE=1` and a placement plan (`hetero-moe-placement/1`) the engine's Flash-Next target
MoE layers keep only the plan's hub-resident experts (130 slots per layer in H1, 512 experts exist).
The layer's `expert_map` is the device `slot_map[E]` (slot or -1), the weights are loaded and repacked
by the platform's own W4A16 MoE method, and `apply` becomes dispatch, shared expert, hub fused MoE on
the masked ids, combine. Dispatch (the `hx` kernel) applies the spill rule and sends the pairs that are
not on the hub to the expert servers (V620 processes, one device each); the servers compute their
partials and push them back; combine adds them to the hub partial. The model runner announces every
forward to the servers (`post_step`) before its kernels are launched. The MTP draft layer is hub-only
and keeps the plain method. H1 is the degenerate H2 case: one island, `island_mode: replicate`, no
all-reduce, `apply()` returns the island's whole sum.

## 2. Files

| File | Role |
|---|---|
| `vllm/distributed/hetero_moe/plan.py` | loads and validates a plan with the same checker as the solver, answers `hub_resident`, `server_experts`, `cold_copies`, owner and island questions, numpy only |
| `vllm/distributed/hetero_moe/tables.py` | `slot_map`, `n_cold`, `cold_srv`, `owned` per layer, spill K table (`default_spill_k`), `split_pairs_loop` (the oracle of the dispatch rule) |
| `vllm/distributed/hetero_moe/ref_hx.py`, `ref_moe.py` | in-process reference of the `hx` ops (vectorised `split_pairs`, combine order of the wire) and a CPU W4A16 reference MoE, for tests |
| `vllm/distributed/hetero_moe/runtime.py` | `HubRuntime`: per-step protocol, one MoE layer, boot self-test, counters (with the raw `hx_status` words), the optional per-step log; process singleton and the model-runner hooks |
| `vllm/distributed/hetero_moe/cli.py` | plan command line for the launcher (`validate`, `hubs`, `servers`, `buckets`, `retarget`, `stripe`), numpy and regex only, no vLLM import |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_hetero.py` | `HeteroMoEMethod` mixin and `maybe_wrap_hetero_method` |
| `vllm/model_executor/layers/fused_moe/expert_map_manager.py` | explicit local expert list (the slot pool) for `determine_expert_map`, hetero hook in `ExpertMapManager` |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py` | `_hetero_wrap` at both WNA16 return points of `get_moe_method` |
| `vllm/v1/worker/gpu/model_runner.py` | `runner_pre_forward` / `runner_post_forward` around every target forward (V2 runner) |
| `vllm/envs.py` | `VLLM_HETERO_MOE*` (section 8) |
| `tools/rdna/rdna_hetero_launch.sh`, `tools/rdna/serve_rdna.sh`, `tools/rdna/rdna_launcher_common.sh` | launcher stanza: servers first, supervised hub, teardown (section 7) |
| `tools/rdna/recipes/flashnext-hetero-h1.env`, `flashnext-hetero-h1-v620.env` | the H1 recipe (hub on the W7800; buses of the layout before the 2026-11-08 case move) and the V620-only stand-in (interim layout) |
| `tools/rdna/tp8pp2/driver.sh`, `slots.sh`, `tempwatch.py` | arm harness: `CFG=h1` (slot `HA`, hub W7800 plus slot A; old layout) and `CFG=h1v` (slot `HV`, the interim layout's V620 stand-in: hub HIP 1, servers HIP 2-5), `TEMPWATCH_GRACE` (section 10) |
| `tools/rdna/port_v031/kfd_evictions.py` | opt-in `--hetero` / `KFD_EVICT_HETERO=1` (exported by the driver for `h1` and `h1v`): the `evicted_ms` windows also cover the hub's engine core (TP1 has no `VLLM::Worker`) and the expert servers, labels `EngineCore` and `hx_<device id>` |
| `bench_results/2026-11-10_hetero-hub-standin/` | `H1v.W1.plan.json` (the stand-in's plan) and `make_plan.sh` (the exact commands) |
| `tests/quantization/test_hetero_moe_{plan,runtime,method}.py`, `hetero_moe_support.py`, `tests/tools/rdna/test_hetero_recipes.py`, `test_tp8pp2_hetero_cfg.py` | CPU tests (section 9) |

## 3. The plan on the hub

`plan.py` accepts exactly what `tools/rdna/hetero_moe/solve.py` writes (`hetero-moe-placement/1`) and
re-runs the solver's validator on load, so a hand edited or stale plan is refused with a message, never
half used. The questions the hub asks:

- `hub_resident(hub, layer)`: the experts of the layer's slot pool in slot order (this order is
  `slot_map`);
- `server_experts(server, layer)`: ascending expert ids, the server's local slot order (HxServer builds
  its expert map from the same call);
- `cold_copies(layer, island)`: per expert the island servers that hold a copy, primary first.

`tables.py` turns that into four per-layer arrays the dispatch kernel and the reference both read:
`slot_map` int32 `[E]`, `n_cold` int32 `[E]`, `cold_srv` int32 `[E, 8]` and `owned` uint8 `[E]`
(`owned` is 1 for every expert in `replicate` mode; in `partition` mode an expert owned by the other
island is dropped, not an error). The spill K table is `K(T) = spill_k[smallest width >= T]` and the
plan's `prefill_k` (no spill) above the largest width. Defaults per HeteroDesign's correction of the
spec: H1 `3:10, 6:21, 12:39, 24:65, 48:110`, H2 per hub `3:7, 6:13, 12:23, 24:39, 48:63`; other server
counts scale linearly with servers / 4 (minimum 1) [INFERENCE until S5 tuning]. Override with
`VLLM_HETERO_MOE_SPILL_K="3:2,6:4,..."` or the plan's `runtime.spill_k`.

A pair is valid when its expert id is in `[0, E)` and owned by the island. An owned expert that is
neither kept on the hub nor has a server copy is a plan error: the reference raises, and the `hx`
dispatch kernel writes an abort record of phase 7 (`HX_PH_PLAN`, HxServer's `hx_proto.py`).

`cli.py` is run by path so the launcher needs no vLLM import:

```bash
python vllm/distributed/hetero_moe/cli.py stripe --out plan.json          # H1: hub HIP 0, experts 1-4
python vllm/distributed/hetero_moe/cli.py stripe --out plan.json \
    --hub-hips 1 --expert-hips 2,3,4,5 --stripe 16 --hub-arch gfx1030    # a V620 hub, uniform stripe
python vllm/distributed/hetero_moe/cli.py validate plan.json             # one line, exit 0 or 1
python vllm/distributed/hetero_moe/cli.py servers plan.json              # island id hip switch arch
python vllm/distributed/hetero_moe/cli.py buckets plan.json              # 3,6,12,24,48
```

`stripe` runs the ExpertStats solver without statistics (the uniform stripe) with the H1 command of
spec 6.11 (`--islands 1 --hubs-per-island 1 --island-mode replicate --hub-active-gib 9.4 --max-hub-per-layer 130
--mtp-policy hub --mtp-expert-bytes 9830400` and the section 7.2 cost constants). With the defaults the
hub keeps 128 experts per layer (`--stripe 4`); the 130-slot figure of the spec is the cap the solver may
use with statistics.

**Plans solved on real routing statistics** (ExpertStats, T10, `bench_results/2026-11-07_expert-stats-t10/`):
`H1.W1.plan.json` and `H1.W2.plan.json` (one island, replicate, 130 slots per target layer, 6752 hub experts,
`cold_complete`: every expert also has a server copy, so a missed or spilled pair always finds an owner),
`H2.W1`/`H2.W2` (two islands, partition, 115 slots per layer). All four load and build tables
(`test_plans_solved_on_real_routing_statistics_load`). The MTP draft is a `hub_only` layer numbered
`num_target_layers + N` (96, the stats collector's rule): the hx tables, the 64-layer protocol limit and the
servers see the 48 target layers only; the draft keeps its experts on the hub through the standard path.
Device indices in a plan are positions inside the island (hub 0, `exp0..exp3` = HIP 1-4 in listed order).

**The stand-in plan** (`bench_results/2026-11-10_hetero-hub-standin/H1v.W1.plan.json`, plan
`4f5011cdc2d89dcd`; `make_plan.sh` has the commands and reproduces the file byte for byte): the same W1
statistics solved for a V620 hub (HIP 1) and four V620 servers (HIP 2-5): 40 slots per target layer and the
MTP draft's 512 experts on the hub (2432 hub experts, 5.74 GiB), 26 GiB of experts per server,
`cold_complete`, hub arch `gfx1030` (`cli.py retarget`), V620 costs for the hub (6.7 us per pair, 3.1 us
fixed) and the measured 13.9 GB/s links. The T10 plan `H1.W1` does not fit a V620 hub: its 130 slots per
layer are 14.7 GiB of target experts on top of about 15.4 GiB of other weights (measured on the first
stand-in boot: 19.03 GiB of weights with 32 slots per layer, 0.113 GiB per slot and layer, MTP head
included) against 29.69 GiB free. The solver's estimate for the stand-in plan is a decode hub hit of 29 %
(`H1.W1`: 59 %): it checks the mechanism, not the speed.

## 4. `HeteroMoEMethod`

`get_moe_method` returns the platform's WNA16 method (`CompressedTensorsWNA16RDNA2MoEMethod` on gfx1030,
the gfx1100 `CompressedTensorsWNA16MoEMethod` backend on the W7800). `maybe_wrap_hetero_method` swaps its
class for `Hetero<Class>` (a cached dynamic subclass with `HeteroMoEMethod` in front) when the plan lists
the layer and sends part of its experts to the servers. Weight loading, repack and the fused kernel stay
the inner method's. What the mixin adds:

- **Slot pool.** `ExpertMapManager` asks `hub_slot_experts(layer_name, E, rank)` and sizes the layer to
  the resident experts: `local_num_experts == n_slots`, `layer.expert_map` is the int32 `slot_map[E]`.
  The weight loader's `map_global_to_local` therefore places each checkpoint expert into its slot and
  skips the others (a GPU `.item()` per call, about 220k calls: acceptable [INFERENCE], measure at the
  first load). `create_weights` refuses a slot-count mismatch; `process_weights_after_loading` refuses a
  `slot_map` that is not contiguous int32 and registers the layer's hub-only MoE for the boot self-test.
- **`apply`** (fp16 activations only, TP1 or EP only, V2 model runner only; all three are checked at
  build time):

```text
rt.apply_layer(layer, x, topk_weights, topk_ids)
  first hx layer of the forward:   begin_step(rows)           # servers' epoch, K(T)
  hx.dispatch(hub, x, ids, w, hub_ids, layer)                 # spill rule, pushes to the servers
  shared_experts(input, MK_INTERNAL_OVERLAPPED)               # hub works while the servers do
  y = inner.apply(layer, x, w, hub_ids, None, None)           # resident experts, -1 skipped
  hx.combine(hub, y, y, layer)                                # + the servers' fp16 partials
```

- **Contracts.** x fp16 `[T, 2560]` contiguous and 16 byte aligned, `topk_ids` int32 `[T, 10]`,
  `topk_weights` fp32, `hub_ids` int32 `[T, 10]` from one persistent buffer shared by all layers (graph
  captures bake its address). Rows above `max_num_batched_tokens` are refused. `mk_can_overlap_shared_experts`
  is true: the shared expert runs between dispatch and the hub MoE on the same stream, so the runner must
  not fork it.
- **Hub-only and unlisted layers** (the MTP draft, layer 48 of the plan) keep the plain method; they hold
  all 512 experts (the checkpoint's bf16 drafter in the first run, 4.69 GiB per spec 5.2a) and never call
  `hx`.
- **Kernel facts relied on** (read from code): the RDNA2 and RDNA3 MoE paths skip ids below 0 and ids whose
  `expert_map` entry is -1 (`get_local_expert_id` in `csrc/moe/moe_align_sum_kernels.cu` returns -1 for
  ids outside `[0, num_experts)` before the map is looked up, and the RDNA2 path passes
  `ignore_invalid_experts=True` whenever an `expert_map` exists); the gfx1100 Q32 deterministic mode needs
  fp16 and returns NaN at 65520 or more.

## 5. The per-step protocol

Servers must be told about a forward before its kernels are launched, exactly once, and never for a graph
capture (a capture executes nothing). Two places announce:

1. **The model runner hook** (`vllm/v1/worker/gpu/model_runner.py`): `runner_pre_forward(num_tokens, replay)`
   after the cudagraph mode is chosen, `runner_post_forward()` after the forward (before the PLE output
   release). `pre_forward` posts `post_step(rows, bucket)` with `bucket = rows` for a graph replay (the
   padded width, which must be one of the servers' captured widths) and `0` for an eager forward, and arms
   the runtime. `post_forward` disarms and calls `hub.check()`, which raises `HxAbort` when the device or
   the host saw the transport fail. `hx_status` is a 64 byte copy on a side stream: it does not wait for
   the compute stream.
2. **`begin_step`** in the first hx layer of a forward consumes the arm. When the runner did not announce
   (eager launches outside `execute_model`: the profile run, capture warm-ups, dummy runs) it posts the
   step itself. During a capture it posts nothing, bakes the width into the graph and refuses widths the
   servers have no graph for, with a message that names `cudagraph_capture_sizes`, the spill table and the
   server `--buckets`.

The runtime connects lazily at the first MoE `apply` (the profile run) and refuses to connect inside a
capture. The first `get_runtime` call also asserts that `gcnArchName` of the process' device equals the
plan's `arch` for the hub (spec 6.2) before it creates the `HxHub`.

## 6. Boot self-test

The server runs the whole 48-layer chain for every announced step, so a single-layer check would leave
it waiting. The hub therefore runs one eager three-row step through every hx layer right after
`HxHub.create`: pairs are drawn from experts that are resident on the hub, owned, and have a server copy;
each layer's output is computed twice, hub-only (the layer's registered `apply` on its own slot pool) and
through dispatch, servers and combine (the spill rule sends some of the same experts to the servers), and
the two must agree within `rtol 3 %` of the largest value plus `1e-3`. NaN (the abort output) or a wrong
expert id to server slot mapping fails the engine start with the layer number. It checks the mapping,
the wire, the combine and the kernels across the two architectures in about one step of work. The
tolerance held on the first V620 stand-in boot (largest relative difference 3.22e-03, 48 layers, section 13);
it is [INFERENCE] on the W7800 hub (the V620 decode kernels use fp16 activations for three rows; a W4A8
int8-activation path would change the error scale).

## 7. Launcher and recipes

`VLLM_HETERO_MOE=1` in a recipe makes `serve_rdna.sh` a supervisor (no `exec`): it validates the plan,
starts one expert server per plan expert device, waits for every ready file, runs the hub engine as a
child, and tears the group down together. The numbers of hubs, islands and servers come from
`cli.py hubs|servers`, never from a constant.

```bash
python vllm/distributed/hetero_moe/cli.py stripe --out $RUN/plan.json
bash tools/rdna/serve_rdna.sh RECIPE=flashnext-hetero-h1 MODEL=$CKPT VENV=$VENV \
    VLLM_HETERO_MOE_PLAN=$RUN/plan.json                 # PRINT=1 prints stanza and commands only
```

- **Servers** start first, each in its own session (`setsid`) with `HIP_VISIBLE_DEVICES` set to its own
  device, the bus given by `VLLM_HETERO_MOE_SERVERS` (islands separated by `;`, buses by `,`, mapped in
  plan order onto the island's expert devices and resolved to HIP indices at launch; unset falls back to
  the plan's `hip` column with a warning, because HIP indices move), `--expect-bus <hex bus>` so a moved
  index can never put a server on the wrong card, and the explicit kernel environment
  (`VLLM_RDNA2_W4A8_SDOT4=1`, `VLLM_RDNA2_MOE_DETERMINISTIC=1`, `GPU_MAX_HW_QUEUES=4`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, TunableOp off). Their command line is
  `python -m vllm.distributed.hetero_moe.server --plan P --device-id expN --model M --ready-file F
  --buckets <cli.py buckets> --max-rows 2048 --wait-ms 2000 --expect-bus BB`; logs and ready files are in
  `$HETERO_LOG_DIR` (default `<run dir>/hetero`).
- **Supervisor.** Ready wait 900 s by default (`HETERO_READY_TIMEOUT`); a server that dies before ready
  stops the launch (exit 1, log tail shown); a server that dies under a running hub stops the hub (exit 3);
  the hub's exit code is passed through. SIGTERM or the hub's end tears down the hub first, then the
  servers, then KILL after `HETERO_TERM_TIMEOUT` (30 s) per stage. `rdna_kill_stale` also matches the
  server module. `serve_validate.sh` and the arm harness SIGTERM the supervisor's process group and give
  it 180 s (`STOP_GRACE=180`): the hub and the servers are in their own sessions, so a shorter grace would
  orphan them.
- **Venv.** For anything that touches the W7800 use `VENV=~/Apps/vllm/venv-7.14.0` (the only venv with the
  gfx1100 torch and rocm-sdk device packages; the `_0.31` / `_0.32` venvs die with `hipErrorInvalidImage`)
  and `PYTHONPATH=<exact-SHA tree>`; the servers get the same venv.
- **Recipes.** `flashnext-hetero-h1.env`: hub TP1 on bus `cb`, servers `ce d3 d8 db` (the buses of the layout
  before the 2026-11-08 case move; no W7800 is installed now), port 18099, MTP2,
  `COMPILE_MODE=3`, `FULL_AND_PIECEWISE`, `MAXBAT=2048`, `VLLM_USE_V2_MODEL_RUNNER=1`,
  `VLLM_RDNA3_MOE_DETERMINISTIC=1` and `VLLM_RDNA2_MOE_DETERMINISTIC=1` (spec 6.9; the servers get the latter
  from the launcher), `VLLM_HX_FLUSH=256` (the hx digest check on the hub and on every server until the W7800
  path is proven; a caller's `VLLM_HX_FLUSH=0` turns it off), no gfx1030 HIP family on the hub, no `rdna_ar`
  (TP1), `KV=10 GiB`
  (raise toward the 12.1 GiB of spec 5.2a after the first boot measures the free memory),
  `FEATURES="prefix-caching mamba-align vision-cap"` (no `expert-parallel`). `flashnext-hetero-h1-v620.env`
  (interim layout): hub on bus `d4` (HIP 1), four servers `d9 dc df e2` (HIP 2-5), port 18098, `ATTN=fa`,
  `MAXLEN=16384`, `KV=2 GiB`, the production TP4 V620 kernel environment plus `VLLM_RDNA2_MOE_DETERMINISTIC=1`
  (the hub is a gfx1030 engine) and `VLLM_HX_FLUSH=256` as above, and no `VLLM_HETERO_MOE_SPILL_K`
  override (the plan's K table, else the defaults of section 3). It also sets `RDNA_ROCMINFO_STUB=1`: aiter
  (imported through flash_attn by the hub and by every server) runs `rocminfo` and takes the first gfx name,
  which on a box with only V620s is `gfx1030`, and aiter rejects it (`KeyError: Unknown GPU architecture:
  gfx1030`). With the knob the launcher puts `tools/rdna/rdna3/cpu_stubs` (a `rocminfo` stub) first on the
  PATH of the hub and of every server; a recipe with a gfx1100 device in the box leaves it off.
- **Capture sizes.** `rdna_mtp_args 2` captures `[3,6,12,24]` (decode width 3 times 1, 2, 4, 8). The
  servers' buckets (`3,6,12,24,48`) are a superset. A larger `cudagraph_capture_sizes` entry is refused at
  capture with a message; set `CG_SIZES` and the spill table together.

## 8. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_HETERO_MOE` | 0 | master switch; every other `VLLM_HETERO_MOE_*` is ignored while 0 |
| `VLLM_HETERO_MOE_PLAN` | empty | `hetero-moe-placement/1` JSON (required when the switch is 1) |
| `VLLM_HETERO_MOE_ROLE` | empty | `hub` for the engine, `expert` for a server (set by the launcher) |
| `VLLM_HETERO_MOE_DEVICE_ID` | empty | plan device id(s); hub ids in TP/EP rank order, comma separated |
| `VLLM_HETERO_MOE_SERVERS` | empty | server PCI buses, `;` between islands, `,` between buses |
| `VLLM_HETERO_MOE_SPILL_K` | empty | spill K override, `"3:10,6:21,..."` (empty = plan, then defaults) |
| `VLLM_HETERO_MOE_WAIT_MS` | 2000 | device-side poll limit before a step aborts with a record |
| `VLLM_HETERO_MOE_STATS` | 0 | log the `hx_status` counters every 1000 forwards and the per-layer hub / cold / spilled pairs at close |
| `VLLM_HX_DEV_LIB` | unset | HxServer's development build of the hx ops (`hx_dev.so`) while `_rocm_C` lacks them |
| `VLLM_HETERO_MOE_STEP_LOG` | unset | file prefix of the hub's per-step log, `<prefix>.<pid>`: one JSON line per announced step (wall clock ns, hx epoch, who announced it, rows, bucket, running requests as `[req_id, prompt_len, num_computed, num_scheduled]`; `num_computed` is the V2 runner's CPU upper bound). For diagnosing a wrong row; the launcher exports it when it is an override (`EXTRA_OV`) |
| `VLLM_HX_FLUSH` | 0 | HxServer's ordering knob of the hx transport (bit 1 control-page read-back, 2 payload read-back, 4 atomic flag, 8 consumer settle; read modes 16, 32, 48). It is read from the **process environment** by the hub (`hx.py`) and by every server (`server.py`, `--flush` default, decimal): set it with `SERVE_ENV="VLLM_HX_FLUSH=8"` of the harness, never with `EXTRA_OV` (that only sets launcher variables); the servers inherit it from the supervisor |

## 9. CPU tests

| File | Tests | What it proves |
|---|---:|---|
| `tests/quantization/test_hetero_moe_plan.py` | 33 | plan load/validation, tables, spill K defaults and scaling, `owned` and lost pairs, hub-only layer ids beyond the protocol, the four ExpertStats T10 real-statistics plans and the committed stand-in plan (40 slots per layer, V620 hub, four servers, budgets), the CLI (`validate`, `hubs`, `servers`, `buckets`, `retarget`, `stripe`) |
| `tests/quantization/test_hetero_moe_runtime.py` | 38 | partial-sum identity (hub residents plus remote partials equal the full MoE for H1 and H2 shapes, with and without spill), the call sequence, the armed step protocol, capture rules, the boot self-test (passes; catches a wrongly mapped server, NaN, an unregistered layer), counters (totals, per-layer, the `hx_status` line, the digest line of a digest hub, a hub without `layer_stats`), the step log (one line per announced step, queued as tuples and written every `STEP_LOG_EVERY` forwards and at close, off by default, a broken batch turns it off and never fails a forward) |
| `tests/quantization/test_hetero_moe_method.py` | 19 | the wrapped method on the real vLLM classes: slot-map checks, call order with and without a shared expert, hub-only layers untouched, runner hooks, `get_runtime` (arch check, connect, self-test, `layer_stats=True` only with `VLLM_HETERO_MOE_STATS=1`) |
| `tests/tools/rdna/test_hetero_recipes.py` | 30 | the launcher: PRINT stanza of both recipes with a stub venv (the stand-in with its committed plan, the interim buses and `RDNA_ROCMINFO_STUB`), bus resolution, disjointness, `--expect-bus`, the `VLLM_HETERO_MOE_STEP_LOG` override, a full `serve_rdna.sh` run with stub servers and hub for both recipes (the stub `rocminfo` first on the stand-in's PATH, in the hub and in every server), and the supervisor with stub servers and hubs (ready wait, a dead server, exit codes, SIGTERM, a hub that ignores TERM) |
| `tests/tools/rdna/test_tp8pp2_hetero_cfg.py` | 42 | the arm harness: slot `HA` (W7800 first, then slot A) refuses a V620 in the hub position and a W7800 in a server position, slot `HV` (hub HIP 1 bus `d4`, servers HIP 2-5) refuses the old layout and misplaced GPUs, slots A, B and AB unchanged, `driver.sh` start-up report of `h1` / `h1v` (default plans), `need_leases` (`a2` + `b4`, foreign holders, requests ignored) and `check_plan` with `driver.sh` sourced; the `evicted_ms` capture of the hub's engine core and the expert servers (labels, the `timeout` wrapper skipped, the `KFD_EVICT_HETERO` default, the CLI flag, the driver exporting it for `h1` / `h1v` only); the driver's hetero concurrency gate (run, a FAIL reported but not an arm failure, `CONC_GATE=probe` and other configs skip it) |
| `tests/tools/rdna/test_hetero_conc_gate.py` | 14 | the stand-in concurrency gate (`tools/rdna/hetero_moe/conc_gate.py`) on hand-built arm directories: PASS with token flips only reported, `solo` bit-identical across runs and `solo2` equal to `solo` (a logprob alone breaks it), the digest counters of the hub and every server (bad, unhealed, a missing line, no server; the last counters line wins), the exit codes |

Commands and results:

```bash
# Mac: plan, tables, runtime (71 passed with the method file excluded)
uv run --quiet --no-project --with numpy --with regex --with pytest --with torch python -m pytest \
    --noconftest -q -p no:cacheprovider tests/quantization/test_hetero_moe_plan.py \
    tests/quantization/test_hetero_moe_runtime.py
# Mac: launcher, harness and gate (86 passed: recipes 30, harness 42, gate 14; 157 with the two lines above
# in one run)
uv run --quiet --no-project --with numpy --with regex --with pytest python -m pytest --noconftest -q \
    -p no:cacheprovider tests/tools/rdna/test_hetero_recipes.py tests/tools/rdna/test_tp8pp2_hetero_cfg.py \
    tests/tools/rdna/test_hetero_conc_gate.py
# box, exact-SHA tree, CPU only (910 passed on 97634f2553: method 19, plan 33, runtime 38, gate 14 and HxServer's
# tests/distributed/hetero_moe 806; section 13)
TREE=$PWD nice -n 10 bash tools/rdna/rdna3/cpu_pytest.sh -q -p no:cacheprovider \
    tests/quantization/test_hetero_moe_method.py tests/quantization/test_hetero_moe_plan.py \
    tests/quantization/test_hetero_moe_runtime.py tests/tools/rdna/test_hetero_conc_gate.py \
    tests/distributed/hetero_moe
```

`cpu_pytest.sh` of the port branch (a stub `rocminfo`: `aiter` runs `rocminfo` at import and fails without a GPU)
needs no wrapper since the merge of `origin/feat/rdna3-flashnext` `05c6ec578e`; before it, `PYTHON=` pointed at
`py_noaiter`, a wrapper that blocks the `aiter` and `flash_attn` imports (these tests need neither). Before the
case move the `PRINT=1` stanza of both recipes was also checked
with the real venv (`~/Apps/vllm/venv-7.14.0`): `PCI_BUSES='cb'` resolved to HIP 0, the servers `ce d3 d8 db` to
HIP 1-4 (stand-in: hub `ce` = HIP 1, servers `d3 d8 db` = HIP 2-4), plan ids `371b3bae4cfc3193` (H1 stripe) and
`bae5599036440b3d` (stand-in stripe). The interim buses of the stand-in (`d4` = HIP 1, `d9 dc df e2` = HIP 2-5)
are checked the same way in section 13.

## 10. GPU runs

The V620 stand-in booted in eager mode before the 2026-11-08 case move (self-test passed, PPL 4.87, section 13).
It now runs on the interim layout (six V620s, no W7800; box rules): its eager and its production compiled
boots, probes, PPL and concurrency repeats are in sections 10.1 and 13. Every W7800 step is blocked: no W7800
is installed. HxServer's stack is on
`origin/feat/hetero-hx` and was validated by HxServer on V620s and on the W7800 before the move (T1, the T5
fault tests, a real-weight T5).

All runs use the arm harness (`tools/rdna/tp8pp2/driver.sh`, `CFG=h1` and `CFG=h1v` added in this branch):
leases, `boot.lock` and quiet windows, bus verification of the whole slot, junction temperatures (abort at 5 s
mean 90 C or 95 C; `TEMPWATCH_GRACE=120` for the hetero configs, because the supervisor stops hub and servers
in their own sessions), memory and `evicted_ms` capture, probes, PPL, concurrency probe, optional measured
hook. It needs `ulimit -l unlimited` (a fresh ssh login), the lease holder files, and an exact-SHA tree.
The `evicted_ms` windows (`serve_validate.sh`, `hook.sh`) cover the hub's engine core and the expert servers
(`KFD_EVICT_HETERO=1`, exported by the driver): with TP1 there is no `VLLM::Worker` process, so the eager run
before the case move (`evicted_ms: no vLLM workers with a KFD context found`) had validated nothing.

| CFG | Slot | Hub engine sees | Servers | Leases | Refused |
|---|---|---|---|---|---|
| `h1` (old layout, blocked) | `HA` | HIP 0 (W7800, bus `cb`) | `ce d3 d8 db` (HIP 1-4) | `<owner>.slotA` and `<owner>.w7800cb` | HIP 5 |
| `h1v` (interim layout) | `HV` | HIP 1 (V620, bus `d4`, leaf A) | `d9 dc df e2` (HIP 2-5, leaf B) | `<owner>.a2` and `<owner>.b4` (`STANDIN_LEASES`) | none; HIP 0 (`d1`) is not used |

Slots A, B, AB and HA keep the buses of the layout before the move (`verify_slot` refuses them on today's box);
`HV` is the interim layout. The harness defaults the plan of `h1` to the T10 `H1.W1` and of `h1v` to `H1v.W1`
(both committed in the tree; `PLAN=<json>` overrides) and validates it (`cli.py validate`, CPU only) before it
waits for memory.

Tree (exact SHA of this branch, merged with `feat/hetero-hx`; `SHA` is the 40 hex commit). The fat
`gfx1030;gfx1100` `_rocm_C` carries the hx ops (CMake builds them in); use the ABSOLUTE script path (a relative
one breaks the script's own load check):

```bash
T=$HOME/vllm-tp8pp2-${SHA:0:10}
git archive $SHA | ssh $BOX "mkdir -p $T && tar -x -C $T && echo $SHA > $T/.sha"
ssh $BOX "bash $T/tools/rdna/rdna3/build_rocm_c.sh $T 'gfx1030;gfx1100' 8"   # build lock, installs $T/vllm/_rocm_C.abi3.so
# the other extension modules and triton_kernels from the R3 fat tree (their csrc is unchanged)
ssh $BOX "cp ~/vllm-rdna3-flashnext/vllm/{_C,_C_stable_libtorch,_moe_C_stable_libtorch,cumem_allocator,fs_io_C,spinloop}.abi3.so $T/vllm/ \
    && mkdir -p $T/vllm/third_party && cp -rp ~/vllm-rdna3-flashnext/vllm/third_party/triton_kernels $T/vllm/third_party/"
```

A full build takes about 20 minutes. To avoid it for every SHA, keep one build workspace tree (its path keys the
build directory, so ninja stays incremental), sync `csrc/` and `CMakeLists.txt` of the new tree into it, rebuild,
load-check the `.so` (`tools/rdna/rdna3/rocm_c_load_check.py check`) and copy it into the exact tree; that is
what `~/w4a8_runs/hetero_hub/finish_build.sh <sha10> <previous sha10>` does on the box.

### 10.1 V620 stand-in (interim layout: hub HIP 1, servers HIP 2-5; no W7800)

Hub = V620 `d4` (HIP 1, leaf A, gfx1030), TP1, slot pool of 40 experts per layer (plan `4f5011cdc2d89dcd`,
section 3); servers = `d9 dc df e2` (HIP 2-5, leaf B). It exercises the whole hub path (plan, slot pool,
`expert_map`, dispatch, shared expert, combine, runner hooks, self-test, supervisor) except the W7800 kernels.
The hub link crosses leaves and every GPU link is Gen4 x8 (about 13.9 GB/s): its numbers differ from the
same-switch layout the design assumed; they are recorded as measured, not tuned. Measured memory (section 13):
the hub peaks at 26,826 MiB (eager) and 29,698 MiB (compiled) of about 30 GiB, each server at 27.5 GiB (11,015
expert-layer pairs). If the hub or a server does not fit, `make_plan.sh` takes the budgets
(`--max-hub-per-layer`, `--expert-budget-gib`).

```bash
# leases HubMethod.a2 AND HubMethod.b4 first (box rules: queue HxServer, ExpertStats, HubMethod; every GPU of
# both leaves with >= 28 GiB free), a fresh ssh login (ulimit -l unlimited). Eager first, then compiled
# (one change per boot). Recipe keys are overridden with KEY=value in EXTRA_OV (a variable in the caller's
# environment would be overwritten by the recipe file). Never reuse an ARM name: the driver overwrites it.
A=$HOME/w4a8_runs/tp8pp2/arms
OWNER=HubMethod SHA=$SHA CFG=h1v ARM=h1v-int-eager PORT=18098 CELLS=none \
    EXTRA_OV="EAGER=1 VLLM_HETERO_MOE_STATS=1 HETERO_READY_TIMEOUT=1800" \
    setsid nohup bash tools/rdna/tp8pp2/driver.sh > $A/h1v-int-eager.driver.log 2>&1 < /dev/null &
# production compiled mode (COMPILE_MODE=3, FULL_AND_PIECEWISE): probes, PPL, concurrency probe, prefix probe,
# c1 and c8 cells (1,024-token prompts: of the 8 requests only two are resident in the 2 GiB pool; the 16k
# cells do not fit)
OWNER=HubMethod SHA=$SHA CFG=h1v ARM=h1v-int-compiled PORT=18098 HOOK=1 QUALITY=0 PREFIX_PROBE=1 \
    CELLS_SPEC="1024:512:1:4:12301 1024:512:8:32:12303" \
    EXTRA_OV="VLLM_HETERO_MOE_STATS=1 HETERO_READY_TIMEOUT=1800" \
    setsid nohup bash tools/rdna/tp8pp2/driver.sh > $A/h1v-int-compiled.driver.log 2>&1 < /dev/null &
# compiled, no cells, CONC_REPEAT=8 more concurrency probes after the boot, the per-step log, and a knob for the
# hub AND the servers: SERVE_ENV is exported to the whole process tree (VLLM_HX_FLUSH is parsed in decimal)
OWNER=HubMethod SHA=$SHA CFG=h1v ARM=h1v-int-conc-flush8 PORT=18098 HOOK=1 QUALITY=0 PREFIX_PROBE=1 \
    CELLS_SPEC=none CONC_REPEAT=8 SERVE_ENV="VLLM_HX_FLUSH=8" \
    EXTRA_OV="VLLM_HETERO_MOE_STATS=1 HETERO_READY_TIMEOUT=1800 VLLM_HETERO_MOE_STEP_LOG=$A/h1v-int-conc-flush8/hetero/steps" \
    setsid nohup bash tools/rdna/tp8pp2/driver.sh > $A/h1v-int-conc-flush8.driver.log 2>&1 < /dev/null &
```

The box keeps these as helpers (not in the tree): `~/w4a8_runs/hetero_hub/run_arm.sh`
`[OWNER=..] [SX=..] [CR=..] [STEPLOG=0|1] bash run_arm.sh eager|compiled|conc|verify [suffix] [sha10]` (`verify`:
compiled, `CONC_REPEAT` probes, then the c1 and c8 cells; `SX` is `SERVE_ENV`), `auto_verify.sh` (waits for the
keeper's leases, runs the verify arm, releases after a grace period) and `check_env.sh`, which writes
`arms/<arm>/env_check.txt` with the `VLLM_HX_FLUSH` and step-log values the EngineCore and every `hx_exp` process
see (verify it for every arm that sets a knob).

Gates for the stand-in (since `97634f2553`): the "boot self-test passed" line and the counters in the hub log,
probes 3/3, PPL 4.75 to 4.9 (`ppl.txt`; measured 4.82 to 4.87), no abort record, and the hetero concurrency gate
(`conc_gate.txt`; `tools/rdna/hetero_moe/conc_gate.py`, run by `driver.sh` after the arm, `CONC_GATE=probe` skips
it). The gate PASSes when the hx digest is clean on the hub and on every server (last counters line of
`serve.log` and `hetero/<id>.log`: 0 bad at the first read, 0 unhealed; this is the transport test) and `solo` is
bit-identical in tokens and logprobs in every run with `solo2` equal to `solo` (the hub MoE is deterministic).
Token flips of `together`, `churn` and `long_together` against the T=3 `solo` reference are reported, never
failed (HxServer, hx doc 8.9: the batch composition decides how a row's pairs are split over devices, so a
token comparison is not a transport test); the probe's own PASS or FAIL stays in `summary.txt` and `runs.tsv` as
information. Variants of the transport go through `SERVE_ENV` (`VLLM_HX_FLUSH=<decimal>` for the hub AND the
servers, never `EXTRA_OV`: the launcher exports only the keys it knows, and the servers read their own
environment); say so in the report. If memory makes the stand-in impossible, record the largest valid subset
and the number that blocked it.

### 10.2 W7800: T6-H1 eager, then T7-H1 compiled and T14 (BLOCKED: no W7800 is installed)

Blocked since the 2026-11-08 case move (box rules, INTERIM), and the user's approval for W7800 use is still
open. The commands below are for the layout before the move (hub bus `cb`, servers `ce d3 d8 db`, leases
`slotA` and `w7800cb`) and stay as they were: when a W7800 is back, its bus, the servers' buses (`PCI_BUSES` and
`VLLM_HETERO_MOE_SERVERS` of `flashnext-hetero-h1.env`), slot `HA` (`HUB_HIP`, `HUB_BUSES`, `SLOT_A_*`) and the
lease names are re-mapped first; the plan names devices by position only. They also wait for Rdna3Port's
blocks A and C and HxServer's go. The W7800 arms run the real-statistics plan of ExpertStats (T10), which is
the harness default of `h1`; the stand-in's is the solved `H1v.W1`.

```bash
T=$HOME/vllm-tp8pp2-${SHA:0:10}
REAL=$T/bench_results/2026-11-07_expert-stats-t10/H1.W1.plan.json   # 130 slots per layer, plan a6fecb0c623d1c78
# T6-H1, eager: boot, probes 3/3, PPL, concurrency probe, self-test line, counters
OWNER=HubMethod SHA=$SHA CFG=h1 ARM=h1-eager PORT=18099 CELLS=none PLAN=$REAL \
    EXTRA_OV="EAGER=1 VLLM_HETERO_MOE_STATS=1" setsid nohup bash tools/rdna/tp8pp2/driver.sh \
    > $HOME/w4a8_runs/tp8pp2/arms/h1-eager.driver.log 2>&1 < /dev/null &
# TP4 slot A reference on the same tree and text (PPL within 0.02, MTP acceptance 1.84 / 2.01):
OWNER=HubMethod SHA=$SHA CFG=tp4a ARM=tp4a-ref PORT=18700 CELLS=none bash tools/rdna/tp8pp2/driver.sh
# T6-H1 soak, 1 h of c8 (ten cells of 96 requests, about 6 min each [INFERENCE]):
OWNER=HubMethod SHA=$SHA CFG=h1 ARM=h1-soak PORT=18099 HOOK=1 PLAN=$REAL \
    CELLS_SPEC="$(for i in 1 2 3 4 5 6 7 8 9 10; do printf '1024:512:8:96:%d ' $((12310 + i)); done)" \
    EXTRA_OV="EAGER=1 VLLM_HETERO_MOE_STATS=1" setsid nohup bash tools/rdna/tp8pp2/driver.sh \
    > $HOME/w4a8_runs/tp8pp2/arms/h1-soak.driver.log 2>&1 < /dev/null &
# T7-H1 = T14, production compiled mode, fresh boot per arm, ABBA: tp4a, h1, h1, tp4a
for arm in "tp4a tp4a-a" "h1 h1-a" "h1 h1-b" "tp4a tp4a-b"; do
    set -- $arm; ov=""; plan=""; [[ $1 == h1 ]] && ov="VLLM_HETERO_MOE_STATS=1" && plan=$REAL
    OWNER=HubMethod SHA=$SHA CFG=$1 ARM=$2 HOOK=1 PLAN=$plan EXTRA_OV="$ov" bash tools/rdna/tp8pp2/driver.sh
done
```

T14 reads per arm: probes, PPL, MTP acceptance at c1 and c8, `Available KV cache memory` / `GPU KV cache size`
in `serve.log` (at least 470k tokens), the hub counters (`VLLM_HETERO_MOE_STATS=1`), `temps1s.log`,
`evicted_ms`. First-build speed gates (spec 5.4a): green c1 at most 36.9 ms and c8 at most 60.8 ms, red above
40.2 or 72.3 ms (stop and trace), 2048-token chunk at most 1.5 s with the scalar MoE kernel.
`KV=10737418240` in the H1 recipe is raised toward 12.1 GiB after the first boot reports the free memory.

## 11. Risks

1. **The W7800 hub path has never run on a GPU** (no W7800 is installed). The V620 stand-in ran eager and
   compiled (section 13): the self-test tolerance holds (largest relative difference 2.63e-03 to 3.22e-03
   against the 3 % limit), the TP1 shapes (QSA group 12, GDN 16 / 48 heads) run on the V620 kernels and PPL is
   4.82 to 4.87. A W7800 result is provisional while the wrong-row event of item 9 is open.
2. **Padded rows.** The router's padded rows may carry valid expert ids rather than -1; the runtime treats
   them as real rows (they are dispatched and combined, wasting a few pairs per graph replay). [INFERENCE:
   not yet observed on a GPU.]
3. **Capture sizes against server buckets.** A capture or replay width the servers have no graph for is a
   hard error by design; `cudagraph_capture_sizes`, `VLLM_HETERO_MOE_SPILL_K` and `--buckets` must change
   together.
4. **`slot_map` lives in two places** (`layer.expert_map` and the transport's tables). A later live
   placement flip (S8) must update both.
5. **The supervisor's timing is observed on one box only**: stubs on macOS, and the real vLLM process tree of
   the V620 stand-in (clean teardown, `evicted_ms` +0, no orphan); a W7800 hub is untried.
6. **The leases are shared**: the stand-in takes both of the interim layout (`a2` leaf A, `b4` leaf B) and
   waits in the queue behind HxServer's soak and ExpertStats' arms; every boot follows the lease and
   quiet-window rules.
7. **Harness gaps.** `memwatch.py` follows only the supervisor's process group (hub and servers run in
   their own sessions); its per-bus VRAM rows are complete. `tempwatch.py` waits `TEMPWATCH_GRACE` seconds
   (120 for `h1` and `h1v`, 15 otherwise) between SIGTERM and SIGKILL of the group; a teardown longer than
   that would orphan the hub and servers, and the next launch's `rdna_kill_stale` cleans them up.
   `kfd_evictions.py --hetero` walks the supervisor's ppid tree; on the box its windows list the EngineCore
   and `hx_exp0` to `hx_exp3` (`evicted_ms` +0 for all five in every stand-in arm).
8. **Not verified:** the W7800's torch device name (`verify_slot` expects "W7800" in it).
9. **Wrong rows in compiled mode: numerics, not transport (hx doc 8.9).** With the hub's MoE not deterministic
   the concurrency probe failed in 4 of 19 compiled runs (`together` prompt 0 token 28, `churn (11, 1)` prompt 1
   token 29: sequence position 35, margins 1.06 to 3.38 nats) and a heavy per-step log made the flips vanish (0
   of 9). HxServer's digest arms (`VLLM_HX_FLUSH=256`, every exchange verified end to end) logged 0 bad on the
   hub and on the four servers over about 2.56 M exchanges, and with `VLLM_RDNA2_MOE_DETERMINISTIC=1` on the
   hub `solo` and `solo2` are bit-identical in every run: the flips were the hub's arrival-order fp16 epilogues
   amplified at fragile positions, plus the dependence of a row's result on how its pairs are split over
   devices (batch composition); the one FAIL left is `together 0` token 28 at margin 1.031 against the 1.0
   threshold. The gate is therefore the digest plus `solo`/`solo2` identity (section 10.1), and the digest
   stays on by default until the W7800 path is proven. The compiled arm with the new defaults is in section 13.
10. **Prefix probe FAIL on the stand-in** (hits +0 of 3,854 queries in every compiled arm). The 2 GiB pool is
    12 pages and a request of 1k to 3.8k tokens takes about 4.5 of them, so the probe's cached prefix is
    [INFERENCE] evicted before the warm request. Re-test on the W7800, whose pool is larger.
11. **Hub memory margin.** The compiled stand-in hub peaks at 29,698 MiB (minimum free 0.94 GiB), so neither
    the KV pool (two resident requests of about 1k tokens) nor the slot pool can grow, and c8 is a batch of
    two. A leaner plan (6 to 8 slots per layer) would trade hub hits for KV [INFERENCE].
12. **aiter and `rocminfo`.** aiter takes the first gfx name `rocminfo` lists and rejects gfx1030; the stand-in
    recipe stubs `rocminfo` (`RDNA_ROCMINFO_STUB`, section 7). On a box with a W7800 whichever agent
    `rocminfo` lists first decides [INFERENCE]; set the knob if a V620 comes first.
13. **The spill path is barely exercised**: 3,155 of 42.8 M not-resident pairs in 6.5k steps (the K defaults
    of section 3 hold at the stand-in's load; the W7800's 130-slot plan has another profile).

## 12. Deviations from the spec and agreements

- `HxHub.create(plan, hub_device_id, *, tables, max_rows, top_k, hidden, wait_ms, ...)` (HxServer's
  signature) rather than the shorter Appendix C form; `post_step(T, bucket)` with bucket 0 for an eager
  forward; `hx.begin_step/dispatch/combine/status` module functions with the signatures of `ref_hx`.
- The `owned` table and the lost-pair rule (agreed with HeteroDesign and HxServer).
- Two-place step announcement (agreed, Appendix C.6): runner hook plus `begin_step`; never during capture.
- The recipe is `flashnext-hetero-h1.env`; its V620-only twin is `flashnext-hetero-h1-v620.env`.
- `VLLM_HETERO_MOE_SPILL_K` and the spill defaults per the corrected 6.2 table.
- Boot self-test runs hub-only versus remote on the same experts instead of a synthetic known answer,
  because the server's chain is 48 layers deep.
- The arm harness (`tools/rdna/tp8pp2`) gained `CFG=h1` (slot `HA`: W7800 hub plus slot A) and `CFG=h1v`
  (slot `HV`: V620 hub HIP 1, servers HIP 2-5 of the interim layout) for T14; the hub engine sees only its
  own GPU and the launcher starts the servers. `driver.sh` takes the plans from the tree (T10 `H1.W1` for
  `h1`, `H1v.W1` for `h1v`) instead of generating a stripe plan.
- The stand-in runs a stats-solved plan of its own (`H1v.W1`, 40 slots per layer), not the T10 `H1.W1`: that
  plan's 130 slots per layer do not fit a V620 hub (section 3).
- Per-layer counters use HxServer's `HxHub.create(..., layer_stats=True)` and `hub.layer_stats()` (agreed
  by message; they land with `feat/hetero-hx`).
- `cli.py stripe` takes the H1 command of spec 6.11 as its defaults (the solver of this tree accepts the
  cost flags of that command).
- Additions of this branch that are not in the spec: the launcher knob `RDNA_ROCMINFO_STUB` (stand-in
  recipe), `KFD_EVICT_HETERO` and `kfd_evictions.py --hetero` (the hub has no `VLLM::Worker`),
  `VLLM_HETERO_MOE_STEP_LOG` (per-step log of the announced steps, off by default) and the second stats line
  `hetero MoE hx status` (the eight `hx_status` words next to the counters).
- The recipes default to `VLLM_RDNA2_MOE_DETERMINISTIC=1` for the hub and `VLLM_HX_FLUSH=256` (Main's decision
  after HxServer's hx doc 8.8 and 8.9), and the stand-in's concurrency gate is the digest plus `solo`/`solo2`
  bit-identity with the token flips against the T=3 solo reported as information (`conc_gate.py`, section 10.1):
  the probe's own verdict no longer decides a hetero arm.

## 13. Progress and resume point (2026-11-10)

**Digest tree and gate (2026-11-10, evening; Main's RESUME after HxServer's digest arms).** `decd1dc383`
merges `origin/feat/hetero-hx` `fc64ca9554` (the digest kernels: `hx_core.cuh` and `hx_ops.cu` changed, so
`_rocm_C` is rebuilt, `finish_build.sh` 35 s, 199 ops registered, one more than before: `hx_dig_stats`; the merge
was committed with `--no-verify`, its files are HxServer's) and `97634f2553` adds what section 12 lists last: the
recipes' `VLLM_RDNA2_MOE_DETERMINISTIC=1` and `VLLM_HX_FLUSH=256`, the `hx digest` stats line, `conc_gate.py` and
its driver hook. The gate was run on HxServer's arms of the same day: `h1v-int-concdet` (deterministic hub MoE)
gives `solo` 8/8 items bit-identical in 17 runs and `solo2` equal to `solo` in 136/136 items, `h1v-int-concdig`
(non-deterministic hub MoE) 0/8 and 0/136, which is HxServer's finding (8.9); the hub's digest counters parse
(850,032 and 864,864 ok, 0 bad). Those arms ran on a tree whose servers logged no final digest line, so their
server lines read "no counters line" (the gate FAILs on a missing line by design). CPU tests on the box, exact
tree `97634f2553`: **910 passed in 380 s** (method 19, plan 33, runtime 38, gate 14 and HxServer's
`tests/distributed/hetero_moe`, 806).

**Stand-in on the interim layout, GPU results (2026-11-10).** Hub HIP 1 (bus `d4`), servers HIP 2-5, plan
`H1v.W1` (40 slots per layer), TP1, MTP2, `KV=2 GiB`; both leases held 12:45 to 14:12 box time. Exact trees
`~/vllm-tp8pp2-<sha10>` of `23e1edb1c7` (launcher stub), `5db4a10410` (stats line, per-step log) and
`f7238d70a4` (buffered per-step log); arm directories `~/w4a8_runs/tp8pp2/arms/h1v-int-*`.

- The first boot of the day failed in aiter (`KeyError: Unknown GPU architecture: gfx1030`, `rocminfo` lists
  only V620s): `RDNA_ROCMINFO_STUB=1` in the stand-in recipe (`23e1edb1c7`, section 7).
- Eager (`h1v-int-eager`): READY 281 s (model load 19.98 GiB in 166.5 s), KV 24,903 tokens, **self-test
  passed, largest relative difference 2.63e-03**, probes 3/3, **PPL 4.85** (261 tokens), **CONCURRENCY PASS**
  (0 failing, 12 near-tie, 1 soft, 35 identical of 48; 92.7 s), probe after load PASS, `evicted_ms` +0 for the
  EngineCore and `hx_exp0` to `hx_exp3`, 0 dmesg faults. Counters: hub 735.1 / servers 5142.9 pairs per step
  at 1001 steps (87.5 % remote), 85.3 % at 1506 steps. Hub VRAM peak 26,826 MiB (allocator `max_reserved`
  25.79 GiB), servers 27,532 to 27,554 MiB.
- Compiled (`h1v-int-compiled`, COMPILE_MODE=3, FULL_AND_PIECEWISE): READY 351 s (backbone compile 55.8 s,
  eagle head 4 s, graph capture 3 s and 0.60 GiB), **self-test passed, 2.84e-03**, probes 3/3, **PPL 4.85**.
  c1: decode 44.00 tok/s/req, **ITL median 43.9 ms**, TTFT 0.89 s, 4 of 4. c8: 41.54 tok/s/req (75.13
  aggregate), **ITL median 50.2 ms** (mean 53.2), TTFT 35.6 s, 32 of 32; with prompts of about 1k tokens the
  pool (12 pages) holds two requests (KV usage 75.7 %, up to 6 waiting), so c8 is a batch of two. Counters at
  6,509 steps: hub 885.4, servers 6,576.4 pairs per step (88.1 % remote), 42.8 M pairs not resident, 3,155 by
  the spill rule. Hub VRAM peak 29,698 MiB (`max_reserved` 28.59 GiB, minimum free 0.94 GiB), servers 27.5
  GiB, junction temperature hub 77 C at most, servers 54 to 57 C, 0 faults, `evicted_ms` +0.
- Prefix probe: FAIL in every compiled arm (queries +3,854, hits +0; cold 1.57 s, warm 1.58 s); prefix hits
  do occur later (concurrency probe, 'long' phase, 5 to 16 %). The TP4 production arms get 53.1 % with 266,945
  tokens of KV; the 2 GiB pool is the suspect [INFERENCE].

Concurrency probe in compiled mode (the harness's probe plus `CONC_REPEAT=8`, one boot per arm):

| Arm (tree) | Per-step log | Failing runs |
|---|---|---|
| `h1v-int-compiled` (`23e1edb1c7`) | off | 1 of 1 |
| `h1v-int-conc` (`23e1edb1c7`) | off | 2 of 9 (runs 3 and 6) |
| `h1v-int-concstepbase` (`5db4a10410`) | on, a JSON line per step | 0 of 9 |
| `h1v-int-concoff` (`5db4a10410`) | off | 1 of 9 (run 5) |

The failures: `together` prompt 0, token 28 of 64, three times (margins 1.41, 1.06 and 1.72 nats) with the
identical wrong continuation (`... The capital of Portugal is\n\n<think>\nThe user is listing capitals ...`
where the solo run says `Lisbon`), and `churn (11, 1)` prompt 1, token 29 (margin 3.38; the text stops at
`Tides return again.\n` where the solo run continues). Both are at absolute sequence position 35 (prompt length
7 plus 28, 6 plus 29); [INFERENCE] a special token is emitted there (the probe runs `ignore_eos`). No hx abort,
no record, no counter anomaly in any run (`hx status` at the close of `concstepbase`: record 0x0, epoch = steps
= 9,741, dispatches = combines = 467,568 = 48 per step). The per-step log of `concstepbase` (9,741 steps):
bucket 3 6,560 steps, bucket 6 2,730, 12 213, 24 55, plus eager prefill chunks; in every episode at most two
requests run at once (the `together` phase of 8 requests runs as four sequential pairs, decode in the bucket-6
graph, prefill joining as T12 or T24 steps). The step that produced a wrong token was never captured: the
flips vanished while that heavy log was on. Not run: eager repeats, the flush arms, HxServer's digest arm.

**First GPU contact: the V620 stand-in (`CFG=h1v`, eager; layout before the 2026-11-08 case move: slot A, hub HIP 1, servers HIP 2-4).**
Tree `~/vllm-tp8pp2-7f9edff0ba` plus an overlay of the wrapper fix of `af277e6cbb` (two files copied by `scp`),
plan `bae5599036440b3d` (stripe, 32 slots per layer), arm directory `~/w4a8_runs/tp8pp2/arms/h1v-eager`.

- The first boot failed in the model load: `TypeError: __class__ assignment: 'Hetero...' object layout differs
  from 'CompressedTensorsWNA16RDNA2MoEMethod'`. The real methods derive from ABC, a plain mixin cannot take
  over their layout. Fixed in `af277e6cbb` (the wrapper builds a new instance; the test stub is an ABC).
- The second boot: the three real-weight expert servers came up through the launcher (ready files), the hub
  loaded, **`hetero MoE boot self-test passed: 48 layers, largest relative difference 3.22e-03`** (limit 3 %),
  `GPU KV cache size: 24,903 tokens` (stand-in memory), probes served, **`PPL 4.87 (261 tokens) PASS`**,
  **`CONCURRENCY PASS`** (`0 failing, 14 near-tie, 5 soft, 0 solo-noise, 29 identical of 48 checks`, 91.4 s: the
  near-tie and soft rows are low-margin tokens, no failure), probe after load PASS, a clean `done`. The box was
  drained for the hardware move afterwards (SIGTERM, clean teardown, no KFD process left, no new dmesg faults).
- Memory at 32 slots per layer (`torchmem`, `memwatch.tsv`): hub weights 19.03 GiB, allocator `max_reserved`
  24.75 GiB of 29.98, final allocation about 21.57 GiB, hub VRAM peak 25.8 GB, servers about 20.6 GB each.
- Counters of that run (`serve.log`; 32-slot stripe plan, prefill-heavy probes): at 1001 steps `pairs per step:
  hub 343.7, servers 5556.4 (94.2 % remote)`; at close (`counters (final)`) 1491 steps, hub 926,696 pairs,
  servers (not resident) 15,728,288, servers (spill rule) 16,856, then 48 lines `layer N: hub H, cold C, spilled
  S`. The stripe plan puts about 6 % of the pairs on the hub; the stats-solved `H1v.W1` is the first plan whose
  counters mean something.
- At the drain the EngineCore did not exit within the 15 s cleanup grace and was force-killed, then the supervisor
  SIGKILLed the hub's process group; the stock TP arms of 2026-11-07 (`pcie-A`, `-B`, `-0`, `-F`) log the same
  two lines, so it is this vLLM build's shutdown, not the hetero path.
- Not obtained: the compiled boot (`h1v-compiled`) and everything on the W7800 (T6-H1, soak, T14).
- Harmless log noise of this venv: `Failed to import from amdsmi` (warning; the amdsmi fix is in the tree) and
  `Failed to import Triton kernels ... is_hip_gfx1250` (mxfp4 path, not used).

**After the case move (2026-11-10), CPU side:** the stand-in is re-mapped to the interim layout: recipe
`flashnext-hetero-h1-v620.env` (hub `d4`, servers `d9 dc df e2`), slot `HV` (hub HIP 1, servers HIP 2-5, leases
`a2` and `b4`), the solved plan `H1v.W1` (the T10 `H1.W1` does not fit a V620 hub), 135 CPU tests on the Mac.
`flashnext-hetero-h1.env`, slots A, B, AB and HA and the `h1` commands keep the buses of the layout before the
move: they are for the W7800 and stay blocked.

Head `f6c32e5f2c` is the merge of `origin/feat/rdna3-flashnext` `05c6ec578e` (the arch of a pinned process
comes from KFD sysfs, not amdsmi; a host-side row check of the shared-expert tile ops in
`hc_prefill_rdna2.cu`). Its exact tree is `~/vllm-tp8pp2-f6c32e5f2c` (`git archive`, then
`finish_build.sh f6c32e5f2c 7f9edff0ba`: build rc=0, `_rocm_C` loads with 198 ops registered, 0 undefined; only
`hc_prefill_rdna2.cu` differed from the previous build workspace). Checks on the box, no GPU used:

- CPU tests on that tree: **86 passed in 159 s** (method 19, plan 33, runtime 34) with
  `TREE=$PWD bash tools/rdna/rdna3/cpu_pytest.sh -q -p no:cacheprovider tests/quantization/test_hetero_moe_{method,plan,runtime}.py`
  (stub `rocminfo`, the real venv, aiter imports).
- `PRINT=1` with the real venv, committed plan, no `HIP_VISIBLE_DEVICES` on the line:
  `PCI_BUSES='d4' -> HIP_VISIBLE_DEVICES=1`; servers `d9 dc df e2` resolve to HIP 2 3 4 5, matching the plan;
  plan `4f5011cdc2d89dcd` (1 hub, 4 expert devices, 48 target layers plus the MTP layer). A
  `HIP_VISIBLE_DEVICES=0,1,2,3,4,5` on that line makes the check fail ("HIP 2 is both the hub engine's device
  and the device of server exp0"): do not set it there.
- Sysfs: buses `d1 d4 d9 dc df e2` all sit at `0000:BB:00.0` with `temp2_input` (junction) and
  `mem_info_vram_used`, so `tempwatch.py` and `memwatch.py` work unchanged.

**Resume (2026-11-10; HubMethod stopped at its request budget, leases released 14:12:38):**

1. Done: the exact trees of `23e1edb1c7`, `5db4a10410` and `f7238d70a4` are shipped and built; GPU arms above.
2. The wrong-row hunt, one fresh boot per arm, each with 9 probes. The box helper
   `~/w4a8_runs/hetero_hub/run_arm.sh` takes `OWNER` (the lease holder), `SX` (SERVE_ENV: reaches the hub and
   the servers), `CR` and `STEPLOG`: `OWNER=<agent> SX="VLLM_HX_FLUSH=8" CR=8 STEPLOG=0 bash run_arm.sh conc
   flush8 f7238d70a4` from a fresh ssh login with `<agent>.a2` and `<agent>.b4` held; then `=3` and `=32`; an
   eager repeat (`EAGER=1 HOOK=1 CONC_REPEAT=8 CELLS=none` in `EXTRA_OV`; not scripted); then HxServer's digest
   arm (`=256`: it needs the digest code of `feat/hetero-hx` merged into a new exact tree, `finish_build.sh`,
   and `describe_digest()` in `log_stats`; `bad=0` over a failing run would exonerate the transport). With
   `STEPLOG=1` (cheap since `f7238d70a4`) a failing run yields the step (epoch, bucket, composition) that
   produced the token at position 35.
3. Optional: a leaner stand-in plan (6 to 8 slots per layer) for a larger KV pool and more than two resident
   requests (`bench_results/2026-11-10_hetero-hub-standin/make_plan.sh` takes the budgets).
4. The W7800 steps of section 10.2 wait for an installed and approved W7800: re-map buses, slot `HA` and the
   lease names first.
