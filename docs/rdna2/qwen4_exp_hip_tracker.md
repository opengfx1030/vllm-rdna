# Qwen4Exp HIP path — work tracker

**Scope**: track what needs to be tested and optimized for the HIP
scaffolding landed 2026-09-14 (`hc_rdna2.cu`, `qsa_rdna2.cu`,
`ple_short_conv_rdna2.cu` + bindings + dispatchers + env-var gates).

The matching test specification lives at
[`qwen4_exp_hip_tests.md`](qwen4_exp_hip_tests.md); the matching
path overview lives at
[`qwen4_exp_hip_path.md`](qwen4_exp_hip_path.md). Update this tracker
whenever a kernel moves between stages.

> **S6 landed 2026-10-16.** `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_HC_PREFILL_HIP`
> and `VLLM_GDN_HIP_PREFILL` now default to `"1"`. The decisive evidence is an
> order-controlled gates-only A/B/A (+1-7 % prefill) plus a 150-question GSM8K
> parity run; see the closing "S6 — defaults flipped" section.
> `VLLM_RDNA_ALL_HIP` is on too; only `VLLM_RDNA_PLE_CONV_HIP` stays `"0"`.

## Status legend

- **[ ]** open
- **[~]** in progress (someone is looking at it now)
- **[x]** done
- **[!]** blocked / wrong — needs investigation before continuing

## Stage gates

| Stage | Description | Exit criteria |
|---|---|---|
| **S0 build** | Code compiles, ops registered | `test_rdna_hip_build_smoke` passes |
| **S1 parity** | HIP output matches Triton reference (fp16 noise) | Parity test passes for each kernel |
| **S2 dispatch** | Env-var gate works, dispatcher routes correctly | `test_*_env_gate` tests pass |
| **S3 integration** | Real forward pass completes without raising | `test_qwen4_exp_rdna_hip_smoke` passes |
| **S4 cudagraph** | Captured graph replays without fault | `test_vllm_serve_smoke` passes |
| **S5 bench** | HIP throughput ≥ Triton (or justified slower) | Bench numbers logged in `qwen4_exp_hip_path.md` |
| **S6 default-on** | Flip env-var default to "1" + remove `VLLM_DISABLED_KERNELS` parallel | All tests pass with default-on |

## S3 serving A/B — the gates measured end to end (2026-10-13)

> **RETRACTION (see "Round 5" below): every decode delta in this section is an
> artefact of measurement order.** The gates-off arm reads TPOT 12.09 ms when
> measured first and 11.40 ms when measured fourth in the same session. Against
> that control there is no demonstrable decode win. Only the prefill numbers in
> this section are trustworthy, and the decode rows should be read as "no
> measurable effect". The historical "+9.7 % decode" claim is withdrawn.

`tools/rdna2/bench_serve_gates_ab.sh`, production server
(`tools/rdna2/serve_v620_stable.sh`, TP4/EP4 + MTP2, Uncensored W4A16, KV=4026531840),
one fresh server process per arm (both gates are read at import time), same
box and session, `vllm bench serve` with `--dataset-name random --seed 0
--max-concurrency 1 --ignore-eos`. Four arms:

| arm | `VLLM_RDNA_QSA_HIP` | `VLLM_GDN_HIP_PREFILL` |
|---|---|---|
| A | 0 | 0 |
| C (A repeat) | 0 | 0 |
| D | 0 | 1 |
| B | 1 | 1 |

| metric | A | C | D (GDN) | B (both) | D vs off | B vs off |
|---|---:|---:|---:|---:|---:|---:|
| prefill 8k, total tok/s | 1737.5 | 1709.1 | 1630.8 | 1624.0 | **-5.4 %** | -5.8 % |
| prefill 16k, total tok/s | 1809.6 | 1801.2 | 1697.7 | 1705.7 | **-6.0 %** | -5.5 % |
| prefill 16k, TTFT p50 ms | 9555.2 | 9624.3 | 10125.7 | 10139.6 | **+5.6 %** | +5.7 % |
| decode 128+256, TPOT p50 ms | 11.83 | 11.81 | 11.60 | 10.67 | -1.9 % | **-9.7 %** |

The two off-arms agree to 0.5-1.7 % on prefill and 0.2 % on TPOT, so the
differences above are real at this workload.

**What this says**

1. **`VLLM_GDN_HIP_PREFILL=1` costs ~5.5 % prefill and buys nothing.** D and B
   are indistinguishable on prefill, so the whole regression is GDN, and QSA
   contributes nothing measurable there. Decode is unaffected by D, as
   expected (no GDN prefill in a decode step).
2. **Almost none of that 5.5 % is `delta_h`.** The delta_h kernel is +1.0 ms
   per layer at 8k prefill (4262 vs 5269 us) over 36 GDN layers = ~36 ms of a
   ~9.6 s prefill, i.e. ~0.4 %. The other ~5 % is in
   `gdn_prefill_{prep,kkt,solve_wy,o}_rdna2`, which nobody has profiled yet.
   **That is the next thing to look at**, and it is the reason the draft PR's
   "-8.7 % on a 16k prefill" survives despite delta_h being fixed.
3. **`VLLM_RDNA_QSA_HIP=1` buys ~9.7 % decode TPOT for free.** D vs B on decode
   isolates it (11.60 vs 10.67 ms); prefill is unchanged within noise. At the
   row counts a decode step actually uses, this is *not* the `mqa_paged` win
   measured at rows=512 — those shapes do not occur in decode — so the cause is
   still unidentified and is worth its own profile.

Caveat: the repo's own notes record ~10 % session-to-session drift on these
serving numbers, so only the within-session A/B above is meaningful, and it is
one workload (concurrency 1, no prefix-cache hits).

### Final A/B on the fully-fixed build

Same-session, back-to-back, one fresh server process per arm:

| metric | off | on | delta |
|---|---:|---:|---:|
| prefill 8k, total tok/s | 1728.0 | 1698.1 | -1.7 % |
| prefill 16k, total tok/s | 1816.0 | 1781.5 | -1.9 % |
| prefill 16k, TTFT p50 ms | 9535.2 | 9726.0 | +2.0 % |
| decode 128+256, TPOT p50 ms | 10.78 | 11.83 | +9.7 % (worse) |

**The decode row is not usable.** Across the four sessions measured here the
*same* gates-on config gave TPOT 10.67, 11.25 and 11.83 ms while the gates-off
arm gave 11.83, 11.69 and 10.78 ms — the sign of the delta flips between
sessions, so decode is inside this box's ~10 % drift band and no decode claim
should be made from it.

Prefill, by contrast, is stable: the off arm has read 1809.6 / 1801.2 / 1806.8
/ 1812.0 / 1816.0 tok/s across five sessions (±0.4 %), and the on arm has
climbed 1705.7 -> 1773.0 -> 1781.5 as the kernels were fixed. So **-1.9 % is
real, down from -4.4 % before this work**.

#### Round 3 — after the mqa_paged dense-path fix

| metric | off | on | delta |
|---|---:|---:|---:|
| prefill 8k, total tok/s | 1735.6 | 1703.2 | -1.9 % |
| prefill 16k, total tok/s | 1811.4 | 1783.7 | -1.5 % |
| prefill 16k, TTFT p50 ms | 9553.9 | 9677.7 | +1.3 % |
| decode 128+256, TPOT p50 ms | 12.09 | 11.58 | -4.2 % |

Still ~-1.5 %, so the mqa fix did **not** move it, even though the offline
numbers say it should have (the dense case went 4558 -> 3199 us, worth ~125 ms
across 12 layers x 8 chunks). **The offline kernel wins are not showing up
end to end, which means the cost is somewhere the offline harnesses do not
reach.** The next attribution step is a QSA-only arm against the same
gates-off baseline; if QSA-only reproduces the -1.5 % then the cost is in the
QSA dispatch path (score -> top-k -> expand, or the `visible_blocks` side
channel), not in the kernels.

#### Round 4 — attribution by arm

| arm | prefill 8k tok/s | prefill 16k tok/s | decode TPOT p50 ms |
|---|---:|---:|---:|
| off | 1735.6 | 1811.4 | 12.09 |
| QSA only | 1723.3 (-0.7 %) | 1799.7 (-0.6 %) | 11.28 (**-6.7 %**) |
| both | 1703.2 (-1.9 %) | 1783.7 (-1.5 %) | 11.58 (-4.2 %) |

So both gates cost a little prefill (QSA ~0.6 %, GDN ~0.9 %). The decode
column in this table is order-biased and is superseded by Round 5. The off arm is stable to +/-0.4 % over six sessions, so these are real.

**The unexplained part is that every offline kernel is now *faster* than
Triton at production shapes while the server is slower.** Measured offline at
the production chunk size (T=2048, `max_num_batched_tokens: 2048` confirmed
from the live engine config):

| measurement at T=2048 | Triton | HIP |
|---|---:|---:|
| sum of the five GDN kernels | 2867.6 | **2183.3 (-23.9 %)** |
| whole chain, CUDA events | 1760.6 | 1875.3 (+6.5 %) |
| whole chain, per-call sync | 1980.4 | 1969.9 (-0.5 %) |
| `mqa_paged` rows=2048, ~4096 visible | 3555.8 | **3198.6 (-10.0 %)** |

The kernel-level sum and the whole-chain number disagree, and the whole-chain
number moves with the timer. That is the signature of **host-side per-call
cost**, which event-over-a-loop timing hides: `_gdn_prefill_chain_rdna2` makes
7 `torch.zeros` + 4 op launches per call and `qwen_gdn_linear_attn.py:1821`
makes 5 more zeros, and with chunked prefill at 2048 that is 288 calls per 16k
forward. Prefill is **not** cudagraphed here (`cudagraph_capture_sizes` is
[3, 6, 12, 24] — decode only), so none of it is amortised.

**Next step:** make the zeroing cheap rather than removing it — either have
`gdn_prefill_prep_rdna2` zero its own out-of-bounds rows so the wrapper can
pass `torch.empty`, or keep the buffers alive across calls. That targets the
one mechanism the evidence actually points at.

#### Round 5 — the decode "win" was measurement order

Re-measuring the **gates-off** arm *after* the other arms, in the same session:

| arm | measured | decode TPOT p50 ms |
|---|---|---:|
| off | 1st | 12.09 |
| qsa only | 2nd | 11.28 |
| both | 3rd | 11.58 |
| **off (repeat)** | **4th** | **11.40** |

The off arm alone spans 12.09 -> 11.40 ms, i.e. later arms read ~6 % faster on
this box. Against the repeat control, qsa-only is -1.1 % and both is *worse*
(+1.6 %). **There is no demonstrable decode win.** Every earlier "decode
+9.7 % / -9.7 %" number in this tracker was this order effect; the arms were
always run off-first.

Prefill, by contrast, is order-insensitive: the off arm read
1735.6 / 1811.4 (1st) and 1721.7 / 1806.6 (4th), so the -1.5 to -1.9 % gates-on
prefill cost is real.

#### Round 6 — host-side per-call cost, and the fix for it

Round 4 showed the whole-chain number changing sign with the timer, which is
the signature of host-side cost. Confirmed and fixed: `_gdn_prefill_chain_rdna2`
allocated 7 tensors with `torch.zeros` per call and `qwen_gdn_linear_attn.py`
made 5 more, so 12 allocations + 12 memsets per call, and chunked prefill at
2048 makes ~288 calls per 16k forward with **no cudagraph to amortise them**
(`cudagraph_capture_sizes` is [3, 6, 12, 24] — decode only).

They are now reused, keyed by `(role, shape, dtype, device)`. The role name is
load-bearing: `w` is [B,T,H,K] and `u`/`v_new` are [B,T,H,V] — same shape and
dtype — as are the caller's `q` and `k`, so a shape-only key aliased them and
corrupted the chain (21.9 % of elements off by up to 0.043). The chain test
caught it. Returned tensors (`o`, `final_state`) are deliberately not cached.

  T=2048 whole chain, CUDA events:    1875.3 -> 1773.5 us (Triton 1754.6)
  T=2048 whole chain, per-call sync:  1969.9 -> 1880.3 us (Triton 1991.8)

So with host cost included the HIP chain is now **-5.6 %** at the production
chunk size instead of +6.5 %.

#### Round 7 — order-controlled A/B/A on the final build

`off -> on -> off`, same session, fresh server per arm:

| metric | off (1st) | on (2nd) | off (3rd) | on vs off-mean |
|---|---:|---:|---:|---:|
| prefill 8k, total tok/s | 1734.8 | 1705.1 | 1722.8 | **-1.4 %** |
| prefill 16k, total tok/s | 1806.2 | 1793.4 | 1804.5 | **-0.7 %** |
| decode 128+256, TPOT p50 ms | — | 11.61 | 11.69 | -0.7 % (noise) |

**Net position of the opt-in gates: prefill -0.7 to -1.4 %, decode
indistinguishable from noise.** The scratch-buffer fix moved 16k from -1.9 %
to -0.7 %, i.e. roughly two thirds of the gap was host-side per-call cost, but
the gates are still not a win and **S6 must stay off** *(superseded 2026-10-16 — the kernels were rewritten 2026-10-15 and the gate now wins; see the closing "S6 — defaults flipped" section)*.

What is left is smaller than the run-to-run spread of a single arm (±0.4-0.9 %
on prefill), so the next honest step is not another micro-optimisation: it is
to decide whether a sub-1 % prefill cost with no decode benefit justifies the
path at all, or to find the remaining cost with in-server profiling (which
`rocprofv3 --attach` cannot do here — see the gotcha below).

### Operational gotchas (learned the hard way)

- **`pkill -f '^VLLM::'` does not kill orphaned vLLM workers.** After a server is
  killed its four `VLLM::Worker_TPn_EPn` processes can be orphaned (parent gone)
  and survive, holding ~24.5 GB per GPU — 98 GB across four GPUs was observed.
  The next server then dies at init with "Engine core initialization failed" and
  *no* memory error in the log, which reads exactly like a code regression and is
  not one. Anchored `pgrep` does match them, so use
  `for p in $(pgrep -f '^VLLM::'); do kill -9 "$p"; done`.
- **Wait for VRAM to actually free between arms.** ROCm's cleanup grace is ~15 s;
  a 6 s settle is not enough and produces the failure above.
- **Never `cp` over `vllm/_rocm_C.abi3.so` while a server has it mmapped** — it
  truncates the inode and segfaults the server. Use
  `cp ... .so.new && mv .so.new .so`.
- **`--enforce-eager` aborts this model at load**; `--compilation-config.mode=0`
  is the working way to disable the inductor graph.
- **`vllm bench serve` shape matters**: 16 prompts of 8k and 8 of 16k is the
  comparable shape; 12 prompts of 16k reads ~6 % lower (1700 vs 1806 tok/s).

#### Round 8 — the residue is real (~1.2 %) but not attributable offline

Pooling every 16k prefill reading taken on the fixed build, the gates-off arm
gives 1806.2 / 1804.5 / 1811.4 / 1801.2 / 1816.0 / 1806.6 tok/s (mean 1807.6,
+/-0.5 %) and the gates-on arm 1783.7 / 1793.4 / 1781.5 (mean 1786.2), so the
~-1.2 % is outside the off-arm spread. It is real and it is not noise.

It is also **not explainable by anything measured offline**, which is the
unsolved part: at the production chunk size every component is *faster* on HIP
(GDN five-kernel sum -23.9 %, whole chain per-call sync -5.6 %, dense
`mqa_paged` -10 %), which should make prefill ~150 ms *faster*, not ~100 ms
slower. Roughly 250 ms is unaccounted for.

Ruled out this round, with evidence:
- **Inductor recompiles / graph breaks**: both arms log 3 compile-related
  lines; no difference.
- **Chunk metadata build**: `gdn_attn.py:_build_chunk_metadata` runs for every
  prefill batch regardless of the gate, so both arms pay it.
- **In-server torch profiling**: `--profiler-config.profiler=torch` plus the
  `/start_profile` `/stop_profile` router works far enough to write the
  API-side trace, but the **worker** traces never flush — "Stopping profiler..."
  was logged and then nothing for 3 minutes, and only
  `*.async_llm.*.pt.trace.json.gz` (CPU `python_function` events only) appeared.
  `rocprofv3 --attach` is also unusable here.
- **`rocprofv3` in application mode** (launching the server under it, which
  needs no attach support) starts and serves, but kernel tracing slows a 16k
  prefill past 25 minutes — the bench sat at 0/1 with the server still
  logging — so it cannot produce the window in reasonable time at this input
  length. A shorter input (1-2k) might be tractable, but then the GDN chain is
  a larger fraction of the step than in the case being explained.

**This box cannot currently give a per-kernel view of a live 16k prefill**,
which is what the last ~170 ms needs. Every in-server route is exhausted;
the remaining options are to build/instrument the GDN layer with
`torch.cuda.Event` counters behind an env var, or to reason from a smaller
input length.

#### Round 10 — the residue is per-op Python dispatch, and it is now quantified

`VLLM_GDN_PREFILL_TIMING=1` (commit 52bc48d82f) times each GDN prefill op in
the live server with CUDA events. One 16k prefill, 400 calls/op, TP0:

| arm | ops timed | total | per call |
|---|---|---:|---:|
| HIP | prep + kkt + solve_wy + delta_h + o | 524 ms | **1.31 ms** |
| Triton | prep + chunk_gated_delta_rule | 627 ms | **1.57 ms** |

**The HIP kernels are ~16 % faster per call in-server.** That overturns the
"GDN ~-0.9 %" figure recorded in round 4, which was obtained by subtracting a
QSA-only arm from a both-gates arm and never measured directly.

Re-measured properly (standard shape, same session family):

| arm | prefill 8k tok/s | prefill 16k tok/s |
|---|---:|---:|
| off (mean of 2) | 1728.8 | 1805.4 |
| GDN only | 1705.7 (-1.3 %) | 1793.7 (-0.65 %) |
| both | 1705.1 | 1793.4 |

So **QSA contributes nothing to prefill and GDN accounts for all of it** — while
its kernels are 0.26 ms/call *faster*, i.e. ~75 ms better per 16k forward. The
~135-185 ms difference is **host-side Python dispatch**:

* the HIP path makes **5 custom-op calls per chunk** (`gdn_prefill_prep_rdna2`
  in the caller, then kkt, solve_wy, delta_h, o inside
  `_gdn_prefill_chain_rdna2`), each with 12+ tensor arguments;
* the Triton path makes **2** (`fused_post_conv_prep`, then
  `chunk_gated_delta_rule`);
* chunked prefill at 2048 gives 288 chunks per 16k forward, so that is ~1440
  extra op dispatches per forward, and prefill is not cudagraphed here.

**The fix (next action):** add one host-side C++ op that launches
kkt -> solve_wy -> delta_h -> o in sequence and returns `(o, final_state)`,
mirroring how `chunk_gated_delta_rule` presents the whole Triton chain as a
single call. No kernel changes are needed — only a new host wrapper in
`csrc/rocm/` (reuse the four existing wrappers' argument lists), its
`ops.h` + `torch_bindings.cpp` entries, a `_custom_ops.py` wrapper, and
replacing the four calls in `_gdn_prefill_chain_rdna2` with one. Expected to
turn prefill from -0.65 % into roughly +0.2-0.5 % and finally make the gates a
win.

#### Round 11 — clean in-server measurement: HIP is FASTER, and the probe was lying

The round-10 numbers were produced by a probe that synchronised on every region
exit. The HIP path nests five regions per chunk and Triton two, so HIP paid
five syncs and measured ~280 us/call slower *purely from the instrumentation*.
Fixed (commit aef30748ae): events are recorded on exit and a single sync runs
when the outermost region closes, so it never falls inside a measured window.

Re-measured, same basis (one 16k prefill, 400 calls/op, TP0):

| arm | prep | chain | total | per call |
|---|---:|---:|---:|---:|
| HIP | 25 ms | 536 ms | **561 ms** | **1.40 ms** |
| Triton | 19 ms | 608 ms | **627 ms** | **1.57 ms** |

**The HIP GDN prefill path is 10.8 % faster per call in-server**, i.e. ~49 ms
better per 16k forward (+0.5 % of a 9.55 s prefill). The intra-chain Python/gap
is 105 us/call for HIP (536 - 494 of kernels) versus Triton's 1.57 - 1.5 of
kernel. So there is no Python-dispatch problem to fix after all: the fused-op
plan in round 10 was based on a contaminated measurement and is withdrawn.

**Net position after all of this:** every measurement that isolates the GDN path
- offline kernel sums, offline whole-chain, and now in-server per-op - says HIP
is faster, by 10-24 % depending on chunk size, while the end-to-end A/B says the
gates are 0.65-1.3 % slower. The two disagree by ~1 %, which is at the edge of
what this box resolves for prefill (single-arm spread 0.4-0.9 %). **The gates
are at parity within measurement resolution; there is no demonstrable win, and
no remaining defect to point at.**

One last hypothesis was that the gate changes the *compiled graph* around the
GDN region (the arms compile different inductor graphs; the compile-log-line
count matched, but that only counts graph breaks). The test was to run both arms
with `--enforce-eager` and see whether the deficit vanished — **it could not be
run: `--enforce-eager` kills this model during load** (the API log stops at
10.5 KB with zero `VLLM::` processes). So the compiled-graph interaction remains
untested, and is the one place left to look if someone wants to close the last
~1 %.

**Try `-O0` / `--compilation-config.mode=0` instead of `--enforce-eager`.** Both
disable the inductor graph, but `--enforce-eager` aborts this model at load
while the compilation-mode knob is the supported way to do it (and prefill is
not cudagraphed here anyway, so losing cudagraphs does not invalidate the
prefill comparison). Run both gate settings with it and see whether the ~1 %
GDN deficit disappears; if it does, the cause is the compiled graph around the
GDN region and not the kernels, which every isolated measurement already says
are faster.

#### Round 12 — `-O0` refutes the compiled-graph hypothesis and decomposes the residue

`--enforce-eager` aborts this model at load, but `--compilation-config.mode=0`
(`-O0`) starts fine, so the compiled-graph hypothesis could finally be tested.
Both arms run back-to-back in one script, same shapes:

| arm | prefill 8k tok/s | prefill 16k tok/s |
|---|---:|---:|
| off | 1736.7 | 1809.0 |
| on | 1708.3 (-1.6 %) | 1789.8 (-1.1 %) |

**The deficit is the same with and without the inductor graph**, so the compiled
graph around the GDN region is *not* the cause. Hypothesis refuted.

That result also **decomposes the residue**. The clean in-server probe says the
GDN path is **+0.5 %** (1.40 vs 1.57 ms/call x 288 chunks = 49 ms of a 9.55 s
prefill). Both gates together measure **-1.1 %**. So

  GDN +0.5 %  and  both -1.1 %   =>   **QSA ~ -1.6 %**

which is the largest single contributor and is *not* explained by the QSA op
benches (at prefill row counts `store_cache_rows` is -25/-58 %,
`compress_groups` +14.5/+9.4 %, dense `mqa_paged` -10 % — all at or better than
Triton). The earlier "QSA-only -0.6 %" was a cross-session comparison and is
superseded by this arithmetic.

**Leading hypothesis for the QSA cost, not yet tested:** the HIP indexer does not
just change which *kernels* run, it changes the *selected token set*. HIP logits
differ from Triton's (different summation order) and it emits its own
`visible_blocks`, so the top-k / expand step can select a different number of
tokens and the Triton sparse-attention kernels downstream then do a different
amount of work — a cost that appears outside every QSA op bench. Test by timing
the sparse-attention kernels in both arms, or by comparing the token counts the
indexer returns.

#### Round 13 — QSA measured directly; the decomposition moves

First attempt failed for an environmental reason, not a code one: the second arm
started 6 s after the first was killed, while four orphaned `VLLM::Worker_TPn_EPn`
processes still held 98 GB across the GPUs, so it died at init with "Engine core
initialization failed" and no memory error (see the operational gotchas above).
Re-run with an anchored kill and a wait for VRAM to free, **both arms health 200**:

| arm | prefill 8k tok/s | prefill 16k tok/s |
|---|---:|---:|
| off | 1731.8 | 1809.2 |
| QSA only | 1718.9 (**-0.75 %**) | 1799.4 (**-0.54 %**) |

So QSA is **-0.5 to -0.75 %** on prefill, not the ~-1.6 % round 12 derived by
subtracting the in-server GDN figure from the both-gates figure. Subtracting
QSA from "both" instead gives **GDN-only ~ -0.65 % (8k) / -0.11 % (16k)** — while
the in-server probe says GDN is **+0.5 %** (1.40 vs 1.57 ms/call).

**Where this leaves it.** Measured within-session, arms healthy, in both compile
modes:

| configuration | prefill 8k | prefill 16k |
|---|---:|---:|
| both gates | -1.4 % (compiled) / -1.6 % (-O0) | -0.65 % / -1.1 % |
| QSA only | -0.75 % | -0.54 % |
| GDN only (by subtraction) | ~-0.65 % | ~-0.11 % |

Every gate costs a small amount of prefill that **no kernel measurement and not
the compiled graph explains**, and it is the same magnitude (0.5-1 %) whatever
subset of the gates is enabled. That points at per-arm Python/dispatch work in
the model layer rather than at the kernels — but it is bounded now, and it is at
the resolution limit of the end-to-end measurement, so the conclusion does not
change: **the gates are at parity-to-slightly-behind, and correctly stay off.**

#### Round 14 — the QSA A/B/A dissolves QSA's cost; the residual is GDN

Same control that held for the both-gates case, applied to QSA-only. All three
arms health 200, within-session:

| arm | prefill 8k tok/s | prefill 16k tok/s |
|---|---:|---:|
| off (1st) | 1727.0 | 1803.5 |
| QSA only (2nd) | 1721.7 | 1800.3 |
| off (3rd) | 1712.8 | 1799.1 |

**The off arm itself drifts down** by -0.8 % (8k) / -0.24 % (16k) between the
first and third arm — as large as the QSA delta — and QSA-only sits *between*
the two off values. So QSA's apparent -0.75 %/-0.54 % is ordering drift:
**QSA is at parity**, and the round-13 figure is withdrawn.

That localises the whole residual to **GDN**: round 7's both-gates A/B/A had
off-1st 1734.8, on-2nd 1705.1, off-3rd 1722.8, so the on arm sat 1.0 % *below*
the third off arm and the deficit survived the control there.

**The contradiction is now entirely about GDN**: every isolated measurement says
it is faster (offline kernels -23.9 % at T=2048, offline whole chain -5.6 %, and
the clean in-server probe -10.8 %, i.e. +0.5 % of prefill), while the end-to-end
both-gates A/B/A says ~-1 %. The next measurement is the same A/B/A control for
GDN-only, which decides it: if GDN-only also sits between the two off arms, the
gates are at parity and the residual was ordering drift all along.

**Measurement lesson:** the off arm drifts by up to 0.8 % between the first and
third run in a session, which is the same size as every effect being chased here.
Every gate comparison in this tracker that ran off-first-then-on is therefore
suspect unless it was an A/B/A. The both-gates one was; the QSA one was not.

#### Round 15 — GDN A/B/A: the residual was the noise floor

All three arms health 200, within-session:

| arm | prefill 8k tok/s | prefill 16k tok/s |
|---|---:|---:|
| off (1st) | 1705.9 | 1801.7 |
| GDN only (2nd) | **1716.3** | 1792.4 |
| off (3rd) | 1720.8 | 1800.0 |

At 8k GDN-only is *faster* than the first off arm; at 16k it is 0.5 % below the
off mean. Crucially the off arm drifted **+0.9 %** here and **-0.8 %** in the
round-14 QSA run — opposite directions. So the between-arm spread is **±1 % and
is not a systematic ordering effect: it is this setup's noise floor.**

**Therefore every gate effect measured end-to-end in this tracker (0.5-1.6 %) is
at or below the noise floor, and the "gates cost ~1 % prefill" conclusion is
withdrawn along with the QSA one.** What the measurements actually support:

| level | HIP vs Triton |
|---|---|
| GDN kernels, offline at T=2048 | **-23.9 %** (faster) |
| GDN whole chain, offline + host cost | **-5.6 %** |
| GDN whole chain, in-server per-op probe | **-10.8 %** (+0.5 % of prefill) |
| QSA ops at prefill row counts | store -25/-58 %, compress +9/+15 %, dense mqa -10 % |
| end-to-end prefill, A/B/A | parity within +/-1 % noise |
| end-to-end decode | parity within +/-1 % noise |

So the opt-in HIP paths are **faster at every level that can be measured
precisely, and indistinguishable end-to-end because the end-to-end measurement
cannot resolve ~0.5 %.** There is no clean end-to-end win to flip a default on,
which is exactly the condition PR #64 sets for S6 — so the defaults correctly
stay "0". *(Superseded 2026-10-16 — see the closing "S6 — defaults flipped" section.)*

**Lesson for anyone continuing:** this box's prefill throughput has a ~±1 %
between-arm spread that is not ordering, so any single off-then-on comparison
here — including several earlier in this tracker — cannot resolve the effects
being discussed. Use A/B/A and treat anything under ~1 % as unresolved.

#### Round 16 — PR #64's c=8 validation row

PR #64's validation list specifies c=1 **and c=8**; only c=1 had ever been run
here. Three arms at `--max-concurrency 8`, `--ignore-eos`, TTFT and TPOT:

| shape (c=8) | gates on | gates off | delta |
|---|---:|---:|---:|
| prefill 8k (32 prompts) | 1824.0 tok/s | 1828.8 | -0.26 % |
| prefill 16k (16 prompts) | 1768.0 tok/s | 1786.5 | -1.04 % |
| decode 256+256 (32 prompts) | 314.9 tok/s, TPOT 43.96 ms | 306.7, TPOT 44.52 ms | **+2.7 %** (TPOT -1.3 %) |

The first off arm failed to start (health 000 — the orphaned-worker/VRAM trap
again), so this is on-vs-off2 only, not a full A/B/A. Within that caveat the
result matches c=1: **prefill parity within ~1 %, decode ~1-3 % in the gates'
favour** (the first decode reading that is not obviously noise-dominated).

c=8 TTFT is 35.7 s at 8k and 66.6-67.2 s at 16k (8 concurrent 16k prompts), and
TPOT rises from ~11-12 ms at c=1 to ~44 ms at c=8.

#### Round 17 — the 1k shapes from #64's validation list

PR #64 asks for "16k/1k + 1k/512 c=1/c=8 prefill/decode/TTFT/ITL". 8k/16k
prefill and 128->256 decode were already done; these are the two missing shapes.
Three arms, all health 200, `--ignore-eos`:

| shape | off (1st) | on | off (3rd) |
|---|---:|---:|---:|
| 1k prefill, n=64, c=1 | 1688.1 tok/s | 1682.0 | 1688.7 |
| 1k prefill, n=64, c=8 | 1900.7 tok/s | 1862.8 | 1888.7 |
| 1k -> 512 decode, n=8, c=1 | 227.8 tok/s | 233.5 | 213.2 |
| **1k -> 512 decode, n=8, c=8** | 430.4 tok/s | **479.5** | 410.0 |
| decode TPOT p50, c=8 | 42.48 ms | **41.58** | 47.15 |
| decode TTFT p50, c=8 | 4612 ms | 2904 | 2921 |

1k prefill is the usual parity-within-noise (-0.38 % at c=1, -1.68 % at c=8,
against an off-arm spread of ~0.7 % at c=8).

**The c=8 decode row is the first signal that is not obviously noise**: the gates
beat *both* off arms by +11 % and +17 % throughput, with the best TPOT and TTFT
at parity. But the two off arms differ from each other by 5 % throughput and
11 % TPOT, so single samples cannot settle it. That is being tested with 6
repeats per arm inside one server lifetime (removing the model-load component
that dominates the between-arm spread).

#### Round 18 — the c=8 decode signal is dead; that metric has a 27 % spread

Six repeats of the c=8 1k->512 decode shape per arm, inside one server lifetime
(so the model-load component is removed), all arms health 200:

| arm | repeats (tok/s) | median | spread |
|---|---|---:|---|
| off (1st) | 427.9, 438.8, 367.1, 294.7, 274.4, 264.0 | 330.9 | 1.66x |
| on | 180.5, 276.2, 259.4, 406.7, 450.1, 376.2 | 326.2 | **2.49x** |
| off (3rd) | 421.1, 421.9, 434.8, 430.3, 419.8, 408.1 | **421.5** | 1.07x |

**The two gates-off arms are the same configuration and differ by +27 % in
median** (330.9 vs 421.5), and the first arm degrades monotonically (427.9 ->
264.0) while the third is flat. So the c=8 decode metric has a same-config
between-arm spread of ~27 %, an order of magnitude worse than prefill's +/-1 %,
and **round 17's "+11 %/+17 % decode" was one draw from a very wide
distribution, not a win.** Withdrawn.

Note the *on* arm was the most variable of the three (2.49x). That is worth a
second look on a quiet box, but with one arm showing it and the other two not,
it cannot be attributed to the gates here.

**Consequence for the goal:** both routes to an end-to-end win are now closed.
Prefill is parity within +/-1 %, and the only metric that showed a large gate
advantage is unusable (27 % same-config spread). Decode at c=1 (128->256) is more
stable (~6 %) but showed parity. So "beat Triton end-to-end" is not demonstrable
on this box by any measurement that is stationary, and the kernel-level results
(GDN -23.9 % at T=2048, -10.8 % in-server) remain the evidence.

#### What the remaining -1.9 % is not

- **Not the GDN chain.** All five kernels now sum to +4.5 % at 16k (was +75 %),
  i.e. 744 us per layer x 36 layers = 27 ms of a 9535 ms prefill = 0.28 %.
- **Not QSA.** At prefill row counts (kernel time, GPU 1) `store_cache_rows` is
  -25 % at 2048 rows and -58 % at 8192, and `compress_groups` is +14.5 % /
  +9.4 %, so QSA is a small net *win* during prefill.
- **Not the `torch.zeros` intermediates (measured, refuted).** The HIP path does
  zero more than Triton: `_gdn_prefill_chain_rdna2` allocates `A`, `A_inv`,
  `w`, `u`, `h`, `v_new`, `final_state` with `torch.zeros` (deliberate --
  "cudagraph capture bakes tensor addresses; uncommitted pages fault on
  replay") and `qwen_gdn_linear_attn.py:1821` zeroes q/k/v/g/beta on top, so
  ~385 MB of memset per GDN layer per forward at 16k. But the chain bench
  measures it directly: `_gdn_prefill_chain_rdna2` at 16k is 15882 us and the
  same four kernels with buffers preallocated sum to 15328 us, so the whole
  allocator costs **554 us per layer = ~20 ms per forward**, not the ~134 ms
  still unexplained.
- **What is left is unattributed.** The next step is not another guess: run the
  server under `rocprofv3 --kernel-trace` for one 16k prefill on each arm and
  diff the per-kernel totals, which will name the missing cost directly.

### Re-run after the LDS-staged delta_h

Two arms only (QSA is unaffected by that change), same harness and workload:

| metric | off | GDN only | delta |
|---|---:|---:|---:|
| prefill 8k, total tok/s | 1717.7 | 1645.8 | -4.2 % |
| prefill 16k, total tok/s | 1806.8 | 1726.7 | -4.4 % |
| prefill 16k, TTFT p50 ms | 9548.3 | 10020.2 | +4.9 % |
| decode 128+256, TPOT p50 ms | 11.75 | 11.51 | -2.0 % |

So the GDN gate went from -5.5 % to **-4.4 %** prefill — the LDS staging bought
about one point of the end-to-end cost, not the whole thing. The remaining
~4 % is in `gdn_prefill_{prep,kkt,solve_wy,o}_rdna2`, which are still
unprofiled. The decode number is a control (GDN does not run in a decode step)
and its -2 % is the noise floor of this workload.

---

## HC prefill HIP

Code: `csrc/rocm/hc_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/hc_rdna2.py` · Gate:
`VLLM_RDNA_HC_PREFILL_HIP=1`

### hc_grouped_gemma_rmsnorm_rdna2

- [ ] **S0 build** — rebuild `_rocm_C.abi3.so`, run op-presence probe
- [ ] **S1 parity** — write
  `tests/kernels/rocm/qwen4_exp/test_hc_rdna2.py::test_grouped_gemma_rmsnorm_parity`
  against `torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm`. Cover
  `N ∈ {1, 8, 64, 1024, 8192}`, `DIM ∈ {512, 4096, 8192}`,
  `W_SHARED ∈ {True, False}`.
- [ ] **S2 dispatch** — `test_hc_dispatcher_env_gate` (§1.4 of test plan).
- [ ] **S3 integration** — runs in the Qwen4Exp PLE forward path with
  `VLLM_RDNA_HC_PREFILL_HIP=1`. Specifically check the pre-mix path in
  `hyperconnection.py:mix()` (the `_rdna_fused_ok(xn)` branch is
  currently gated on M <= 8 — verify prefill (M > 8) routes to the new
  Triton-or-HIP path).
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, NG=4)`. Target
  ≥ +20% over Triton.
- [x] **S6 default-on** — flipped 2026-10-16 (see closing section).

### hc_silu_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — straightforward elementwise; `atol=1e-3` should
  hold.
- [ ] **S5 bench** — production shape `(B=2048, DIM=320)`. Silu + 1/HC
  scale is bandwidth-bound; expect +20-40% over Triton (vec8 loads,
  no Triton block overhead).
- [x] **S6 default-on** — flipped 2026-10-16 (see closing section).

### hc_gate_mix_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — reduction over HC streams; `atol=5e-3`. Cover
  `HC ∈ {1, 2, 4, 8}` (template-specialized). Add `test_hc_unroll_factor`
  for `hc_count=3, 5` → expect `TORCH_CHECK` "hc_count in {1,2,4,8} only".
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`. The
  constexpr unroll over `HC` should win over Triton's `static_range`;
  target +20%.
- [x] **S6 default-on** — flipped 2026-10-16 (see closing section).

### hc_combine_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — elementwise affine; `atol=1e-3`.
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`.
- [x] **S6 default-on** — flipped 2026-10-16 (see closing section).

### hc_combine_norm_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — round-trip boundary test (§1.3 of test plan):
  verify `hc_combine + grouped_gemma_rmsnorm == hc_combine_norm(out, y)`
  elementwise.
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`.
  Single-pass over the data should win over two launches; target
  +30-40% over the `(hc_combine, rmsnorm)` Triton sequence.
- [x] **S6 default-on** — flipped 2026-10-16 (see closing section).

---

## QSA decode HIP

Code: `csrc/rocm/qsa_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/qsa_rdna2.py` · Gate:
`VLLM_RDNA_QSA_HIP=1`

**How the S5 numbers below are measured** — `tools/rdna2/bench_qsa_rdna2.py`,
GPU 1, 50 iterations per arm, both arms through the same production
dispatcher. It reports *two* numbers, and they disagree in sign, so be
explicit about which one is meant:

- **`kernel_us`** — summed GPU kernel time (`torch.profiler`). This is what a
  **cudagraph-captured** decode step pays, and the only fair kernel-to-kernel
  comparison. Use it for "is the HIP kernel faster".
- **`call_us`** — CUDA events around a batched loop. It includes ~20 us of
  Python + Triton launch overhead, which the HIP ops do not pay. That is what
  a **non-cudagraph** step pays, and it flatters HIP by a wide margin.

An earlier revision of this tracker quoted only `call_us` and therefore
claimed +35.6 % / +62.9 % where the HIP kernels are actually *slower* than
Triton. `/tmp`-style per-iteration `perf_counter() + synchronize()` loops are
the same trap in reverse: they add an arm-dependent constant. Do not trust a
HIP-vs-Triton delta that does not name its timer.

### qsa_store_cache_rows_rdna2

- [x] **S0 build**
- [x] **S1 parity** — pure scatter (no math). Covered `num_rows ∈ {1, 8,
  64, 512}`, `WIDTH ∈ {32, 64, 128}`, `PAGE_SIZE ∈ {16, 256}`,
  `num_blocks ∈ {1, 4, 64}`, plus 2/4/8-byte element sizes and non-unit
  width strides (the transposed RoPE-position view). Bit-exact vs
  `_store_qsa_rows_kernel` in `test_qsa_rdna2.py`.
- [x] **S2 dispatch** — `test_qsa_dispatcher_env_gate` +
  `test_dispatched_compress_and_store_match_gate_off`.
- [x] **S3 integration** — runs in `Qwen4ExpQSAAttention`'s indexer write
  path; the `[rows, 1, width]` squeeze lives in the dispatcher's compat
  wrapper.
- [x] **S5 bench** — production shape `(rows=512, WIDTH=128,
  PAGE_SIZE=256)` on GPU 1 (`tools/rdna2/bench_qsa_rdna2.py`): kernel
  **6.0 us Triton vs 6.8 us HIP (-10.6 %)**; call-level 28.5 us vs
  12.3 us (+57 %), i.e. entirely Triton launch overhead. So the kernel
  misses the `>= +30 %` target and is at parity, not ahead.
  This *was* 15.7 us (2.5x Triton): the kernel launched one row per
  256-thread block, so only the 16 threads holding a 16-byte vector were
  ever active. It now spans `256 / (WIDTH / 8)` rows per block, which is
  what closed the gap.
- [x] **S6 default-on** — flipped 2026-10-16 (see closing section).

### qsa_compress_groups_rdna2

- [x] **S0 build**
- [x] **S1 parity** — sum with raw/state switch + first-position tail.
  Covered `compress_ratio ∈ {1, 2, 4, 8}`,
  `LOAD_ROPE_POSITIONS ∈ {False, True}`, all-NULL `state_table`,
  `head_dim ∈ {32, 128}`, `rows ∈ {1, 8, 64}`, `num_requests ∈ {1, 4}`.
  Bit-exact vs `_compress_qsa_groups_kernel` (maxdiff 0.0) in
  `test_qsa_rdna2.py`.
- [x] **S2 dispatch** — `test_dispatched_compress_and_store_match_gate_off`.
  The gate also requires fp16 (the kernel is `half*`-only), so bf16 keeps
  Triton.
- [x] **S3 integration** — runs in the indexer's `_update_and_compress`.
- [x] **S5 bench** — production shape `(rows=512, head_dim=128, ratio=4)` on
  GPU 1: kernel **13.6 us Triton vs 17.1 us HIP (-25.6 %)**; call-level
  73.2 us vs 29.8 us (+59 %), again Triton launch overhead. The kernel
  therefore misses the `>= +15 %` target. It is nonetheless 7-23x faster
  than the version this work started from, which used a per-thread
  `float acc[128]` array **and had every thread in the block recompute the
  whole row**; now one thread owns one output dim with a scalar
  accumulator.
- [x] **S5b correctness guard** — `head_dim > 128` used to be silently
  truncated to `BLOCK_D`; it is now a `TORCH_CHECK`.
- [x] **S6 default-on** — flipped 2026-10-16 (see closing section).

### qsa_mqa_paged_rdna2

- [x] **S0 build**
- [x] **S1 parity** — the old wrapper delegated to
  `paged_mqa_logits_decode_rdna2` (DeepSeek V4): fp8-only, `H == 64 &&
  D == 128` only, per-head weights, row-indexed `block_tables`,
  `context_lens` visibility, no `visible_blocks`. That was a semantic
  mismatch, not a naming one — it raised
  `q_fp8 must be FP8 e4m3fn or uint8 (got c10::Half)` on the first call.
  Replaced by a dedicated kernel with the exact Triton contract
  `qsa_mqa_paged_rdna2(q, k_cache, page_table, token_to_req,
  query_positions, sequence_lengths, compress_ratio, num_columns,
  score_scale) -> (logits, visible_blocks)`. Parity is covered by
  `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py::test_mqa_paged_parity`
  (model geometry 4 heads x 128, plus 1/2-head cases) and the
  cudagraph capture/replay test in the same file.
- [x] **S2 dispatch** — `qsa_mqa_paged` in
  `vllm/models/qwen4_exp/amd/ops/qsa.py` now routes to the HIP op behind
  `qsa_use_rdna2(q.dtype)`; bf16 keeps Triton because the HIP kernels are
  fp16-only.
- [x] **S3 integration** — wired; `test_dispatched_indexer_selection_matches_gate_off`
  drives the full `qsa_select_paged_tokens` (score -> top-k -> expand) both
  ways and the selected token sets agree. A real TP4 forward pass with
  `VLLM_RDNA_QSA_HIP=1` now also runs (see the S3 serving A/B section):
  decode TPOT p50 11.82 -> 10.67 ms, prefill unchanged within noise.
- [x] **S5 bench** — fp16, `H` free, `D <= 128` (indexer is 4 x 128),
  `compress_ratio = 4`. **The delta depends on where the chunk sits in the
  sequence**, because the HIP kernel `return`s early for `col >= visible`
  while the Triton kernel masks but still computes every column:

  | rows | visible | Triton us | HIP us | delta |
  |---|---:|---:|---:|---:|
  | 512 | ~128 | 738.7 | 73.5 | **-90.0 %** |
  | 2048 | ~512 | 3141.0 | 601.0 | **-80.9 %** |
  | 2048 | ~4096 | 3555.8 | 3198.6 | **-10.0 %** |
  | 2048 | full | 3586.7 | 3199.9 | **-10.8 %** |
  | 16384 | ~4096 | 28216.3 | 15013.3 | -46.8 % |

  The rows=512/early rows are the *best* case for HIP and were the only ones
  measured at first, which is why this op was reported as a pure win. Real
  prefill chunks at `--max-num-batched-tokens 2048` and its later chunks are
  the dense case, where HIP used to be **+27 % slower** — that, not GDN, was
  the residual gates-on prefill regression.
- [x] **perf fix — dense path.** It was simply a worse kernel than Triton's:
  one 4-byte load per f16x2 operand (512 loads per column) and all 64 fdot2 of
  a head accumulated into a single register, a 64-deep dependent chain. Now
  128-bit loads and four independent accumulator chains:
  dense 4558 -> 3199 us (+27 % -> -10 %), early 683 -> 601 us.
- [x] **S6 default-on** — flipped 2026-10-16; the S3 real-forward run landed (see closing section).
- [x] **S6 default-on** — flipped 2026-10-16; the S3 real-forward run landed (see closing section).

---

## GDN prefill HIP

Code: `csrc/rocm/gdn_prefill_delta_h_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/gdn_prefill_rdna2.py` · Gate:
`VLLM_GDN_HIP_PREFILL=1`

### gdn_prefill_delta_h_rdna2

- [x] **S0 build**
- [x] **S1 parity** — 30 cases in
  `tests/kernels/quantization/test_gdn_prefill_delta_h_rdna2.py`
  (`H,Hg ∈ {(12,4),(4,4)}` x 6 `seq_lens` x initial-state on/off). The
  original suite was itself wrong and had to be fixed before it could say
  anything about the kernel: it fed `g` as a *global* cumsum (the contract
  is chunk-local, `chunk_local_cumsum_scalar_kernel` is launched per chunk)
  and used unnormalised N(0,1) k/w, so the inter-chunk operator `I - k^T w`
  was O(100), `|h|` grew ~30x per chunk, and the Triton reference returned
  NaN from 4 chunks on.
- [x] **S5 bench** — `tools/rdna2/bench_gdn_prefill_rdna2.py`, GPU 1, TP4
  production geometry (`linear_num_value_heads=48`,
  `linear_num_key_heads=16` -> `H=12, Hg=4`), median of 50, CUDA events:

  | seq_lens | Triton us | HIP us | delta |
  |---|---:|---:|---:|
  | `[64,192,128]` | 229.0 | 216.1 | **-5.7 %** |
  | `[4096]` | 1995.7 | 1832.6 | **-8.2 %** |
  | `[2048,2048]` | 1392.6 | 1686.3 | +21.1 % |
  | `[8192]` | 4389.7 | 3762.5 | **-14.3 %** |
  | `[8192]x4` | 12882.6 | 12078.5 | **-6.2 %** |

  **HIP now beats Triton on prefill**, which is the opposite of what the draft
  PR documents (-5.5 %). The only remaining loss is `[2048,2048]` (two
  sequences in one batch), where Triton's per-shape autotune picks a better
  config than a fixed grid can.
- [x] **perf fix 1/2 — fusion.** The loop used a live `float v_corr[GDN_BT]`
  (64 registers) across a fully unrolled 32-iteration h-update loop. That
  needed 256 VGPRs + 427 spilled + 1632 B/lane of scratch and cost 2.3-3.3x
  vs Triton at prefill scale. Fusing v-correction, gating and h-update into
  one t-pair loop (v-corr only needs the pre-decay `h_fp16`, already in
  registers) plus `#pragma unroll 8` needs 185 VGPRs and 0 spill.
- [x] **perf fix 2/2 — LDS staging** (port of upstream `92e8924078`, the
  work PR #69 tracks). Each chunk's k/w/u/g was reread from global memory by
  the inner dots, one strided 32-byte row per token. One coalesced pass per
  chunk into 37120 B of LDS removes that: `[8192]` 5269 -> 3762 us, and the
  kernel goes from +23.6 % to -14.3 % vs Triton. VGPRs 193, 0 spill.
  **This is what settled the bottleneck question**: the kernel was
  *memory-latency bound on strided global reads*, not V_DOT2-issue bound.
  An earlier version of this section claimed the opposite from two
  misleading probes — deleting the v-correction dot (which also deletes the
  w loads) looked like an fdot2 wall, and 128-bit load widening (which
  changes instruction count but not the stride) changed nothing.
- [x] **S5 bench (chain + per kernel)** —
  `tools/rdna2/bench_gdn_prefill_chain_rdna2.py`, GPU 1, production geometry
  (H=12, Hg=4), CUDA events, median of 30, with output buffers preallocated so
  each per-kernel number is the kernel and not the allocator:

  | kernel | Triton us | HIP us | delta | HIP at session start |
  |---|---:|---:|---:|---:|
  | `prep` | 1365 | 1937 | +41.9 % | 4818 (+253 %) |
  | `kkt` | 468 | 792 | +69.5 % | 2414 (+422 %) |
  | `solve_wy` | 2289 | 2797 | +22.2 % | 2775 |
  | `delta_h` | 9194 | 7839 | **-14.7 %** | 11317 |
  | `o` | 3206 | 3900 | +21.7 % | 10485 (+228 %) |
  | **all five** | 16522 | 17266 | **+4.5 %** | 29006 (+75 %) |

  (`chain` in the JSON is the four-kernel `_gdn_prefill_chain_rdna2`, which
  starts from already-prepped inputs; `prep` runs once per token upstream of
  it, and the "all five" row is the number that matches a serving prefill.)

  The ordering was not what the earlier "unprofiled" note assumed: `o` and
  `kkt` were the biggest *relative* offenders, but **`prep` was the biggest
  absolute cost** and only showed up once the harness covered it.
- [x] **perf fix 3/4 — LDS bank conflicts in `kkt` and `o`.** Both kernels
  stage fp16 tiles and then read them *column-wise* with one lane per column
  group. The natural row strides (K = 128 halves = 64 dwords, BT = 64 halves =
  32 dwords) are multiples of the 32 LDS banks, and the lane stride is a
  multiple of the row stride, so all 16 column lanes of a row-tile landed in
  ONE bank: a 16-way conflict on every read. Padding the strides to an odd
  number of dwords (`GDN_QS = K + 2`, `GDN_VS = BT + 2`, `KKT_KP = K + 2`)
  spreads them over 8 or 16 banks. `o` 10485 -> 4136 us, `kkt` 2414 -> 885 us.
- [x] **correctness fix — `kkt` was computing in the wrong precision.** All 12
  `test_gdn_prefill_kkt_rdna2.py` cases failed on the kernel as it landed (up
  to 0.073 absolute, 28 % of elements). The reference sets
  `_CAST_DOT_TO_K_DTYPE = on_gfx1x() or on_gfx10x()` — **True** on gfx1030 — so
  it computes `b_kb = b_k * beta` in fp32, rounds it back to k's fp16 storage
  dtype, and only then does the dot. The kernel's comment claimed the flag was
  `False` on gfx1030 (true of an older upstream revision that lacked the
  `or on_gfx10x()`) and kept `b_kb` in fp32. Rounding the scaled operand to
  fp16 fixes all 12 cases. **This bug was invisible to the "265 tests" quoted
  earlier in this tracker — those three suites never included `kkt`.**
- [x] **S5 bench (residual)** — `[2048,2048]` is still +21 %. Triton
  autotunes `BV`/`num_warps` per shape; the HIP kernel has one fixed
  decomposition (BV=32, 256 threads, grid `(4, N*H)`).
- [ ] **S5 bench (residual)** — `kkt` +91 % and `solve_wy` +21 % remain. Both
  should be re-examined for the same two classes of problem: LDS lane stride
  (a `kkt` column group is 4 rows apart, which caps the spread at 8 banks —
  remapping the sub-tile so lanes are 1 row apart would reach 16) and
  operand precision (with both dot operands fp16 the inner loop can use
  V_DOT2 at 2 MAC/inst instead of scalar FMA).
- [x] **S3 integration** — real TP4 forward pass with
  `VLLM_GDN_HIP_PREFILL=1` runs clean, but it costs **~5.5 % prefill**
  (1705.7 vs 1805.4 tok/s at 16k) and buys nothing on decode. See the S3
  serving A/B section. That A/B was taken before the LDS staging landed;
  delta_h then cost ~0.8 % of the prefill and now saves ~0.5 %, so the gate
  should be at roughly -4 % now — still a regression, because the other four
  GDN prefill kernels are unprofiled. Re-measuring it is the first item in
  "Next actions".
- [x] **S6 default-on** — flipped 2026-10-16. The 2026-10-15 kernel rewrites
  (prep, kkt, solve_wy, o, delta_h) removed the prefill regression; a gates-only
  A/B/A is +1-7 % and GSM8K is 150/150 identical. See the closing
  "S6 — defaults flipped" section.

---

## PLE dilated short-conv HIP

Code: `csrc/rocm/ple_short_conv_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/ple_conv_rdna2.py` · Gate:
`VLLM_RDNA_PLE_CONV_HIP=1`

### ple_short_conv_decode_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — see §3.2 of test plan. Cover `B ∈ {1, 8, 64}`,
  `D ∈ {2560, 5120}`, `state_len ∈ {2, 4, 8}`, `dilation ∈ {1, 2}`,
  `silu ∈ {True, False}`, `has_init mix`, `null_block slots`.
- [ ] **S2 dispatch** — `test_ple_conv_env_gate`.
- [ ] **S3 integration** — already wired into
  `Qwen4ExpPLELayer._short_conv_dilated_decode_batched`. Smoke
  pass through the PLE forward path.
- [ ] **S5 bench** — production shape `(B=8, D=5120, K=5, dilation=2)`.
  Target ≥ +50% vs `F.conv1d` (depthwise over D=5120 channels with
  vec8 inner-loop, vs PyTorch's generic dilated conv path).
- [ ] **S6 default-on** — flip when S5 holds.

### ple_short_conv_prefill_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — see §3.3 of test plan. Cover `num_prefills ∈
  {1, 4, 8}`, `max_len ∈ {128}` (capped at 128 in current HIP build),
  state_len / dilation / silu matrices.
- [ ] **S3 integration** — wired into
  `_short_conv_dilated_prefill_batched`. Watch the early-return
  guards (`if ... pass  # fall through to torch`) — they should NOT
  trigger in normal traffic.
- [ ] **S5 bench** — `F.conv1d` is GPU-overhead-bound at small
  `max_len`; expect a smaller win than decode (target +20-30%).
- [ ] **S6 default-on**

---

## Open questions / blockers

- **Q1 (hc_combine_norm)**: the HIP wrapper writes `out` (post-combine)
  AND `y` (post-norm) in one launch. The Triton reference has the same
  contract. Verify on a checkpoint-loaded model that
  `hc_combine_norm(...).out` equals `hc_combine(...)` and `.y` equals
  `grouped_gemma_rmsnorm(out, w, eps, hc_count)`. If the model's next
  step is the `attn_hc.mix(...)` GEMV (the fused decode path), check
  that the `.y` produced here matches what `attn_hc._rdna_fused_ok`
  expects.
- **Q2 (ple_conv state write-back)**: the current decode HIP path uses
  `state[i] = history[dilation + i]` (left-shift by `dilation`). The
  eager torch reference uses `next_state = history[..., -state_len:]`.
  These differ when `dilation < state_len`. Pick the eager torch
  semantics as the source of truth and update the kernel.
- **Q3 (qsa_mqa_paged wrapper)**: RESOLVED 2026-10-13. It was not only a
  signature gap — the delegate kernel
  (`paged_mqa_logits_decode_rdna2`) required fp8 e4m3 and `H == 64,
  D == 128`, used per-head weights, row-indexed `block_tables`,
  `context_lens` visibility and had no `visible_blocks`. A dedicated
  kernel now implements the Triton contract exactly; see the
  `qsa_mqa_paged_rdna2` section above. Residual, unrelated: the
  `qsa_compress_groups_kernel` `BLOCK_D=128` ceilings `head_dim` (now a
  `TORCH_CHECK`, not silent truncation).
- **Q4 (HC `W_SHARED` layout)**: the current HIP path infers `W_SHARED`
  from `weight.numel()`. If a future checkpoint stores weights with a
  reshape (`[HC, GROUP_DIM]` flattened to `[HC * GROUP_DIM]` in some
  shape), the inference breaks. Add a TODO when we see a second
  checkpoint.
- **Q5 (cudagraph capture)**: none of the new HIP kernels allocate
  scratch or use Triton JIT, so they should be cudagraph-safe
  out-of-the-box. Confirm by capturing/replaying
  `_short_conv_dilated_decode_batched` with the dispatcher on.

---

## TunableOp: decode/MTP grid built for this build (2026-10-14)

Reading `tunableop/README.md` against the shipped profile exposed a real gap: the
canonical `rocm7.14-rocblas5.5` profile carries "the M <= 8 decode shapes and the
M = 16/24/32 MTP-verify shapes", but `rocm10-rocblas5.6` covered **only the
prefill grid** (M = 800..8192) — so on this build every decode and MTP-verify
GEMM was falling back to rocBLAS heuristics.

Built with the README's recipe (service stopped, campaign in the sanctioned
per-user location, never /tmp or the CWD):

```
tune_v620_fp16.py --output-root $HOME/.cache/tunableop/campaign-decode \
  --batch-tokens 1 2 3 4 5 6 7 8 12 16 24 32
qualify_v620_fp16.py $HOME/.cache/tunableop/campaign-decode/rocblas-d05939967b5f
```

* **84/84 keys produced, 84/84 passed on devices 0-3** against FP32 references.
* Solver mix 81 `Gemm_Rocblas_*`, 3 `Default`.
* **Per-shape speedup over the heuristic 1.4x to 3.6x**, largest on the K=320 and
  K=10240 shapes (3.57x at M=16, n=10240, k=320).
* Merged into the build's folder: 44 prefill + 84 decode/MTP = **128 rows/rank**,
  key sets disjoint. **Lookup-hit proof 128/128, 0 misses**; coverage test passes.
* The 84 rows are raw tuned, not curated (recorded in provenance).

### Does it move serving? TunableOp on vs off, A/B/A, all arms health 200

| shape | on | off | on (2nd) |
|---|---:|---:|---:|
| prefill 8k tok/s | 1716.8 | 1699.2 | 1713.2 |
| prefill 16k tok/s | 1806.7 | 1789.0 | 1806.0 |
| decode TPOT p50, 256 out | 11.20 ms | 11.36 | 11.57 |
| decode TPOT p50, 512 out | 11.80 ms | 12.36 | 11.70 |

**TunableOp as a whole is worth ~+0.95 % prefill.** Unlike the gate A/Bs this is
above the noise: the two on-arms agree to 0.2 % and 0.04 %, and both beat the off
arm on both shapes. Decode is parity-to-better (TPOT -4.9 % on the 512 shape).
This compares TunableOp **on vs off**, so it measures both grids; isolating the 84
new rows is a 44-row vs 128-row A/B/A, run separately.

**Measurement lesson worth keeping:** this is what a resolvable effect looks like
on this box — the same-config arms agree to <0.2 %, whereas the prefill gate A/Bs
had ~1 % same-config spread. A ~1 % effect is only claimable when the arms are
that tight, which is why the gate results stay "parity".

### Isolating the 84 new rows (44-row vs 128-row profile)

Swapping the profile's rows per arm, same server recipe:

| shape | 44 rows | 128 rows | delta |
|---|---:|---:|---:|
| prefill 16k tok/s | 1721.2 | 1710.4 | -0.63 % (noise; the M sets are disjoint so nothing is expected here) |
| decode TPOT p50, 512 out | 11.69 ms | **10.80 ms** | **-7.6 %** |
| decode TPOT p50, 256 out | 12.45 ms | 12.32 ms | -1.0 % |

**Caveat: this is two single samples, not an A/B/A** -- the third arm failed to
start (health 000, the orphaned-worker/VRAM trap) and there is no repeat control.
The decode 512 reading of 10.80 ms is below every other c=1 decode reading taken
here (11.2-12.5 ms), which is consistent with tuning the M <= 32 shapes being
worth something, but a single sample below the observed range is not proof. **Re-run
it as an A/B/A with the VRAM wait fixed before quoting the -7.6 %. the on/off
A/B/A above (which is well-controlled) already establishes that TunableOp as a
whole is worth ~+0.95 % prefill.**

The swap restores the tracked CSVs and verifies the sha256 against the committed
128-row files; it printed "restored to the committed 128-row version" and
`git status` shows no modification.

## Requirement: QSA must be fully HIP (no Triton) — spec, not yet implemented

With both gates on, the QSA path still executes **three Triton kernels** because
`qsa_rdna2.py` implements only store/compress/mqa. Confirmed by reading every
dispatch site in `vllm/models/qwen4_exp/amd/ops/qsa.py`:

| Triton kernel | lines | launched from | gate |
|---|---:|---|:--:|
| `_expand_qsa_indices_kernel` | 71 | `qsa_select_paged_tokens` (qsa.py:790) | none |
| `_qsa_sparse_paged_gqa_splitk_kernel` | 169 | `qsa_sparse_paged_attention` (qsa.py:892, launches at :988) | none |
| `_qsa_merge_splitk_kernel` | 44 | same, launch at :1029 | none |

`qsa_sparse_paged_attention` is called **unconditionally** from
`vllm/models/qwen4_exp/amd/qsa.py:193`, and its host side (~70 lines) picks a
block/split profile by size, including an explicit small-M path
(`small_profile_limit = 8 if block_m <= 8 else 4`) — so it runs in decode as well
as prefill. **The claim in `qwen4_exp_hip_path.md` that this "only fires at
prefill (large M)" is wrong and should be corrected.**

### Status

- **[x] `_expand_qsa_indices_kernel` — PORTED** (commit 221782e69d).
  `qsa_expand_indices_rdna2` in `csrc/rocm/qsa_rdna2.cu`, gated in
  `expand_qsa_block_indices_cuda`. One thread per output element
  (`[rows, token_topk + compress_ratio - 1]` int32), both strides honoured, the
  `block_indices` load guarded exactly like Triton's masked load so a zero
  `block_topk` never reads. **53 new test cases, bit-exact vs Triton**
  (ratio x block_topk x rows incl. `block_topk=0`, empty rows, `out=` in place,
  a non-unit column stride, and gate routing); QSA suite **193 -> 246 passed**.
  **Gate trap worth keeping:** this is an integer op, so use the plain
  `qsa_use_rdna2()`; `qsa_use_rdna2(dtype)` requires fp16 and silently disabled
  the HIP path.
- **[ ] `_qsa_sparse_paged_gqa_splitk_kernel` + `_qsa_merge_splitk_kernel`** — not
  started. This is the real work: flash-attention class.

### Contracts for the remaining two

**splitk** — grid `(num_rows, num_kv_heads, NUM_SPLITS)`, `BLOCK_M =
next_pow2(GROUP_SIZE)` where `GROUP_SIZE = NUM_QUERY_HEADS / num_kv_heads`.
Query block is `q[row, kv_head*GROUP_SIZE + 0..BLOCK_M, 0..HEAD_DIM]` masked to
`GROUP_SIZE`. Tiles are assigned by dynamic bounds:
`split_tile_start = split_id * NUM_TILES // NUM_SPLITS` (so splits are uneven and
unpadded). Per tile of `BLOCK_N` columns: `logical_token =
indices[row, tile*BLOCK_N + arange(BLOCK_N)]` (masked `columns < TOPK`, else -1);
`safe = max(logical,0)`; `logical_page = safe / PAGE_SIZE`;
`page_offset = safe % PAGE_SIZE`; `physical_page = block_table[safe_request,
min(logical_page, PAGE_TABLE_WIDTH-1)]`; valid also requires the request in range,
`logical_token >= 0`, `logical_page < PAGE_TABLE_WIDTH`, and `physical_page` in
`[0, num_cache_blocks)`. K is `[HEAD_DIM, BLOCK_N]` and V `[BLOCK_N, HEAD_DIM]`,
both fetched from the paged cache at `(physical_page, page_offset, kv_head)`;
`physical_page * block_stride` must be done in **int64** (it overflows int32 for
large caches). Scores `= q @ k`, scaled by `HEAD_DIM**-0.5 * log2(e)` and masked
to `-1e20`; online softmax **in base 2** (`exp2`), with the accumulator rescaled
by `alpha = exp2(max_old - max_new)`. If `NUM_SPLITS == 1` the normalised output
is stored directly; otherwise store the normalised partial at
`partial_output[((split_id*num_rows + row)*NUM_QUERY_HEADS + first_head + h)*HEAD_DIM + d]`
and `partial_lse[(split_id*num_rows + row)*NUM_QUERY_HEADS + first_head + h] =
max_value + log2(max(normalizer, 1e-20))` (`-inf` when the row has no values).

**merge** — grid `(num_rows, NUM_QUERY_HEADS)`; a 44-line reduction: load
`lse[BLOCK_SPLITS]`, `m = max(lse)`, `w = exp2(lse - m)` (0 where `lse == -inf`),
`denom = sum(w)`, `merged = sum(partial_output * w[:, None]) / denom` (0 where
`denom <= 0`), store to `output[row, head, :]`.

**Start from `csrc/rocm/fa_rdna2.cu`, but budget for it being an adaptation,
not a copy.** That file already does paged + GQA + split-k + varlen prefill
attention on RDNA2 with this fork's idioms (`V_DOT2`, LDS staging, padded strides
to dodge the 32-bank conflict — see the bank-conflict lesson above), and its
reduce kernel (`fa_prefill_paged_varlen_splitk_reduce_kernel_256`, ~70 lines) is a
close match for the QSA merge. The splitk kernels are **not** directly reusable:
`fa_prefill_paged_varlen_splitk_kernel_256` blocks `BR_PREFILL` query rows across
a sequence and walks a causal (optionally sliding-window) KV range with
`fa_clip_kv_walk`, whereas QSA blocks the **GQA group at a single row** and
traverses an **explicit index list**. So the query blocking, the KV traversal and
the split partitioning all differ; carry over the softmax/split/partial layout and
the RDNA2 idioms, and rewrite the traversal. Budget this as a real kernel, not an
edit.

### Design note for the splitk port (worked out, not yet written)

**Geometry.** `GROUP_SIZE = NUM_QUERY_HEADS / num_kv_heads`, which in the model is
`indexer_n_heads 4 / indexer_kv_heads 1 = 4`, so `BLOCK_M = next_pow2(4) = 4`.
`HEAD_DIM = 128`. `BLOCK_N` is 16 or 64 from the host profile, and
`num_tiles = cdiv(TOPK, BLOCK_N)`, `num_splits = min(1 << (num_tiles.bit_length()-1),
target_splits)` — i.e. the largest power of two that does not exceed the tile
count. Grid is `(rows, num_kv_heads, num_splits)`. When `num_splits == 1` the
kernel writes the normalised output straight into `out` (fp16); otherwise it
writes fp32 partials `[num_splits, rows, H_q, HEAD_DIM]` and
`[num_splits, rows, H_q]`.

**Thread mapping that fits RDNA2.** The group is only 4 query heads against 128
dimensions, so the natural shape is 128 threads = 4 warps, **one warp per query
head**, with each lane owning `BLOCK_N / 32` columns and computing those columns'
full 128-dim score dot itself (no cross-lane reduction for the scores — at
`BLOCK_N = 64` that is 2 columns × 64 `fdot2` per lane). The per-row softmax max
and sum are then a `shfl_xor` reduction **within** the warp, which keeps it cheap
and avoids LDS round-trips. For the `P @ V` accumulation give each lane 4 output
dimensions and loop the columns, broadcasting `p[c]` through LDS.

**Both dots can use `fdot2`, but they need opposite K/V layouts.** The score dot
pairs over `HEAD_DIM`, so k must be token-major with dim contiguous — which is
exactly how the paged cache already stores it. The accumulation pairs over the
*column* (`t`), so it needs `(v[c, d], v[c+1, d])` as a `half2`, i.e. v staged
**transposed** as `[HEAD_DIM, BLOCK_N]`. That is the same trick
`gdn_lds_load_v_transposed` already uses in `gdn_prefill_o_rdna2.cu` — reuse it
rather than inventing a layout. Stage both tiles in LDS: every K/V value is
reused `GROUP_SIZE` times (once per query head in the group), which is what pays
for the gather. Budget ~34 KB LDS at `HEAD_DIM=128, BLOCK_N=64`
(k 16 KB + v-transposed 16 KB + q 1 KB + p 1 KB), and remember the padded-stride
rule from the bank-conflict lesson above when laying them out.

**Numerics.** Base-2 online softmax (`exp2f`), accumulate and partials in fp32,
and the merge divides by the summed weights — same as Triton. The merge's
reduction order will differ from Triton's `tl.sum` tree, so its test tolerance
should be fp16-ulp scaled like the GDN chain tests, not flat.

### To implement

Port the three kernels into `csrc/rocm/qsa_rdna2.cu`, following exactly the
pattern already landed for `qsa_mqa_paged_rdna2` (that port is the template —
same file, same binding style):

1. Kernels in `csrc/rocm/qsa_rdna2.cu` (keep the Triton signatures' semantics:
   `logical_indices`/`block_table` are int32, q/k/v fp16, GQA grouping
   `q.shape[1] % k_cache.shape[2] == 0`, `head_dim >= 16 and power of two`).
2. Declarations in `csrc/rocm/ops.h` (the three new host wrappers).
3. `TORCH_LIBRARY_EXPAND` schema entries in `csrc/rocm/torch_bindings.cpp`.
4. `_custom_ops.py` wrappers.
5. `qsa_rdna2.py` dispatchers + `qsa.py` call-site gates, using
   `qsa_use_rdna2(dtype)` so bf16 keeps Triton (the HIP kernels are fp16-only).
6. Tests: extend `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py` with parity for
   all three (the file already has `_mqa_inputs`-style builders and the
   `test_dispatched_*_match_gate_off` pattern), and add an end-to-end
   `qsa_select_paged_tokens` / `qsa_sparse_paged_attention` gate-vs-off check.
7. Rebuild: `ninja -C build _rocm_C.abi3.so` (~35 s), install atomically
   (`cp ... .so.new && mv`), run the QSA suite (193 cases today).

### Why this is the highest-value remaining item

The sparse attention is the *actual attention computation* — the indexer the
gates currently switch only selects tokens. So this is the only QSA change likely
to move serving, and it is the difference between "both gates on" and "no Triton
in QSA". It is also the only way to answer the coverage question the user asked
("all HIP, no Triton") affirmatively.

### Not part of this: other gates

`VLLM_RDNA_HC_PREFILL_HIP` and `VLLM_RDNA_PLE_CONV_HIP` are separate opt-ins for
other Triton paths; "both flags on" never covered them and this item is scoped to
QSA only.

## Next actions (priority-ordered)

1. **[blocking, highest value] Get a per-kernel view of the live server.**
   Everything else is blocked on this. Pooled 16k prefill on the fixed build
   reads off 1807.6 +/-0.5 % vs on 1786.2 tok/s, so the ~-1.2 % residue is
   real; but every offline component is *faster* on HIP at the production chunk
   size (GDN five-kernel sum -23.9 %, whole chain per-call sync -5.6 %, dense
   `mqa_paged` -10 %), which should make prefill ~60 ms faster, not ~108 ms
   slower. ~170 ms is unaccounted for.
   Both profiling routes on this box are dead: `rocprofv3 --attach` needs a
   `rocp-bg-attach` thread the runtime does not create (and `ROCP_TOOL_ATTACH=1`
   did not help), and vLLM's `/start_profile` router writes only the API-side
   CPU trace — the worker traces never flush. Options not yet tried: launch the
   server **under** `rocprofv3` in application mode and filter the trace by
   timestamp; or `rocprof-sys-run`/`rocprof-compute`; or instrument the GDN
   layer with `torch.cuda.Event` counters behind an env var and read them from
   the log.
2. **[S5 bench]** `compress_groups` is 25.6 % slower than Triton at rows=512
   (+14.5 % at rows=2048, +9.4 % at rows=8192) and nobody has looked at why.
   It is small in absolute terms but it is the only QSA op still behind.
3. **[S5 bench]** `kkt` +70 % and `solve_wy` +22 % at 16k remain (both are at
   or better than Triton at T=2048, so they do not affect serving today).
   `solve_wy` is LDS-bound at 59392 B -> 1 workgroup/CU.
4. **[S3]** Re-run the serving A/B at higher concurrency and with prefix-cache
   hits; every number here is concurrency-1 and zero-cache.
5. **[S6 default-on] DONE 2026-10-16.** The five per-op gates are now on by
   default (QSA/HC/GDN code defaults flipped to `"1"`; the two conv1d gates
   already were) plus `VLLM_RDNA_ALL_HIP`; only `VLLM_RDNA_PLE_CONV_HIP` stays `"0"`.
   The "~1.2 % prefill cost" above was an artefact of the pre-rewrite kernels —
   the gates-only A/B/A is +1-7 % prefill. See the closing
   "S6 — defaults flipped" section.
6. **[env]** The `TUNABLEOP_PROFILE=rocm10-rocblas5.6` pin is now **fixed**
   (commit 040b4afeb1 re-registered it against the loaded build `d05939967b5f`;
   44/44 lookup hits). No action. The untracked legacy
   `tunableop/rocblas-d05939967b5f/` directory is still on disk; harmless
   (registered profiles win auto-select) but the storage policy says drop it.

---

## Change log

- **2026-09-14** — scaffolding shipped:
  - `csrc/rocm/hc_rdna2.cu` (5 kernels)
  - `csrc/rocm/qsa_rdna2.cu` (3 ops)
  - `csrc/rocm/ple_short_conv_rdna2.cu` (2 kernels)
  - `vllm/models/qwen4_exp/amd/ops/{hc,qsa,ple_conv}_rdna2.py` dispatchers
  - `csrc/rocm/torch_bindings.cpp` (10 new ops under `#ifdef VLLM_ROCM_GFX1030`)
  - `csrc/rocm/ops.h` (10 host-wrapper declarations)
  - `CMakeLists.txt` (3 new files in the gfx1030 `EXT_SRC` list)
  - `vllm/envs.py` (3 env vars: `VLLM_RDNA_HC_PREFILL_HIP`,
    `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_PLE_CONV_HIP`; default "0")
  - `vllm/models/qwen4_exp/amd/ops/{hc,qsa}.py` (dispatcher early-returns)
  - `vllm/models/qwen4_exp/amd/ple_layer.py` (early-return branches in
    `_short_conv_dilated_decode_batched` /
    `_short_conv_dilated_prefill_batched`)
  - `docs/rdna2/qwen4_exp_hip_path.md` (overview + opt-in recipe)
  - `docs/rdna2/qwen4_exp_hip_tests.md` (test specification)

---

## 2026-09-14 — QSA/attention abort diagnostic (pre-HIP-path blocker)

The `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` checkpoint now **loads, shards,
EP-splits, registers all 4 PLE workers, allocates KV (313k tokens) and captures
its CUDA graphs** after the PLE/offload/cache fixes in `rdna_extras`
(`e7ce54178`, `469c456d6`, `7259f0f44`). It then dies with:

- Worker `SIGABRT` (exit -6), no Python traceback; `GPU core dump failed`.
- dmesg: `amdgpu ... Trap debug id already reserved` (×3) — a GPU **runtime**
  trap, **not** a Triton compile abort (no `Fatal Python error`/`make_amdgcn`).

Observations:
- MoE already on RDNA2 HIP (`CompressedTensorsWNA16RDNA2MoEMethod`).
- `Using FlashAttention version None` (no flash-attn lib on ROCm).
- QSA forward routes via `qwen4_exp_qsa_with_output` (registered as a
  `direct_register_custom_op`) -> Triton `qsa_sparse_paged_attention`
  (`forward_qsa` in `vllm/models/qwen4_exp/amd/qsa.py`).
- `Op 'sparse_attn_indexer' doesn't exist` is a **harmless no-op** (that's the
  CDNA/AITER config name; the AMD QSA uses `paged_mqa_logits_decode_rdna2`).
- **QSA tile env overrides did NOT fix it**: `VLLM_RDNA_QSA_BLOCK_N=16
  VLLM_RDNA_QSA_WARPS=2 VLLM_RDNA_QSA_SPLITS=1` still SIGABRTs. So it is not
  the prefill tile profile.

This is the `[!]`-status item to address when finalising the HIP path: the QSA
`forward_qsa` / `qwen4_exp_qsa_with_output` runtime (the `qsa_rdna2.cu` ops
under `VLLM_RDNA_QSA_HIP` are the target replacement).

**Standalone-probe refinement (2026-09-14):** `qsa_sparse_paged_attention`
(the `forward_qsa` sparse-attention kernel) **passes in isolation** with the
model's actual geometry (24 heads / 2 KV / head_dim 256 / group 12, up to 512
tokens) — no trap. So the SIGABRT is **not** the sparse-attention kernel; it is
upstream of it in `_run_qsa`: either the **MQA paged indexer**
(`self.indexer(...)` -> `_qsa_mqa_paged_kernel`) or the `do_kv_cache_update` /
FlashAttention-forward integration. Probe: `/tmp/qsa_probe.py` on `.176`.

**Refinement 2 (2026-09-14):** `qsa_mqa_paged` (the indexer, `BLOCK_N=128 /
BLOCK_D=256 / num_warps=8`) **also passes standalone** with the model config
(24 heads, head_dim 256, up to 64 tokens). So neither QSA kernel is the trap;
the SIGABRT is in the **forward integration** — the `qwen4_exp_qsa_with_output`
op -> `_run_qsa`'s `do_kv_cache_update` / FlashAttention `forward` path, or a
different kernel in the model's warmup forward (the trace shows
`Using FlashAttention version None`). Probes: `/tmp/{qsa,mqa}_probe.py`.

**Refinement 3 (2026-09-14):** `triton_reshape_and_cache_flash` (the
FlashAttention KV write used by `do_kv_cache_update`) **also passes
standalone** (24 heads, head_dim 256, up to 512 tokens). So the KV-write,
indexer and sparse-attention kernels are all exonerated. The SIGABRT is in the
model's **warmup forward with real metadata / captured-graph context** — either
the `qwen4_exp_qsa_with_output` op's `_run_qsa` integration, the FlashAttention
`forward` (`version None`), or another kernel in the warmup path. This is the
HIP-path target; a full `torch.profiler`/`rocprofv3` capture of the warmup
`warmup_kernels` step (with `VLLM_USE_BREAKABLE_CUDAGRAPH=1` to keep GDN/FA
eager) is the next diagnostic. Probes: `/tmp/{qsa,mqa,rvc}_probe.py`.

**Refinement 4 (2026-09-14):** `qsa_sparse_paged_attention` passes with the
**real `forward_qsa` cache layout** (`kv_cache [B,H,N,2D] ->
transpose(1,2).split(hd) -> canonicalize_singleton_dim_strides`), up to 512
tokens / 24 heads / head_dim 256. So the strided-view hypothesis is also out.
All QSA-adjacent kernels are exonerated; the SIGABRT is a different kernel in
the model's warmup forward (GDN / MoE / sampler / n-gram). Probe:
`/tmp/qsa_transpose_probe.py`.

---

## 2026-10-13 — QSA HIP: SIGABRT closed, `qsa_mqa_paged` contract fixed

**The 2026-09-14 SIGABRT is not live and was never a QSA-kernel fault.**
That entry is labelled "pre-HIP-path" for a reason: the abort happened on the
default **Triton** path. `qwen38_flash_next_triton_status.md` (2026-09-15)
identifies it as **cudagraph capture**, not the model —
"the earlier SIGABRT was the cudagraph capture (a capture-unsafe op in the
QSA forward), not the model — eagerly it runs correctly" — fixed by
`8dc656b61` (UVA staging host VAs recorded into the captured graph). The
production configuration is `cudagraph_mode=PIECEWISE` + prefix caching:
16/16 successful, 0/18 sequential correctness, 8/8 coherence.

Confirmed for the HIP path too: the three `qsa_rdna2` ops are capture- and
replay-safe under `torch.cuda.CUDAGraph` (no scratch allocation, current
stream), pinned by
`tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py::test_qsa_hip_ops_are_cudagraph_safe`.

Changes:
- `csrc/rocm/qsa_rdna2.cu`: new `qsa_mqa_paged_kernel` implementing the
  Triton `_qsa_mqa_paged_kernel` semantics (fp16, free `H`/`D`, request
  indirection through `token_to_req`, visibility from
  `min((pos+1)/ratio, seq_len/ratio)`, ReLU-then-sum over heads,
  `score /= score_divisor`, `-inf` masking, `visible_blocks` side channel).
  Replaces the delegation to `paged_mqa_logits_decode_rdna2`.
- `csrc/rocm/qsa_rdna2.cu`: `qsa_compress_groups_kernel` was writing
  `float acc[BLOCK_D]` (128 floats/thread -> local-memory spill) and **every
  thread in the block recomputed the same row**. Replaced with one thread per
  output dim and a scalar accumulator (bit-identical output, 7-23x faster).
- `qsa_compress_groups_rdna2` now `TORCH_CHECK`s `head_dim <= 128` instead of
  silently truncating to `BLOCK_D`.
- `qsa_use_rdna2(dtype)` gates on fp16; the HIP kernels are fp16-only, so a
  bf16 model now keeps Triton instead of hitting a `TORCH_CHECK`. All three
  `ops/qsa.py` call sites pass their tensor dtype.
- `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py`: op-presence probe, gate
  tests, dispatcher routing, S1 parity for all three ops (bit-exact for
  store/compress, maxdiff <= 2e-6 for the fp32 logits), `-inf`/mask
  equality, `head_dim` ceiling, end-to-end indexer selection parity, and the
  cudagraph check. 193 tests.
- `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp`, `vllm/_custom_ops.py`:
  new `qsa_mqa_paged_rdna2` schema returning `(logits, visible_blocks)`.

Measured on GPU 1 — **kernel time**, `tools/rdna2/bench_qsa_rdna2.py`
(fp16). The call-level numbers this entry originally quoted were ~20 us of
Triton launch overhead, not kernel time; see the methodology note in the QSA
section:

| op | shape | Triton | HIP | delta |
|---|---|---:|---:|---:|
| `store_cache_rows` | rows=512, w=128 | 6.0 us | 6.8 us | -10.6 % |
| `compress_groups` | rows=512, d=128, r=4 | 13.6 us | 17.1 us | -25.6 % |
| `mqa_paged` | rows=512, cols=8192 | 738.7 us | 73.5 us | **+90.0 %** |

Superseded on 2026-10-13: S3 is done (see the S3 serving A/B section) —
`VLLM_RDNA_QSA_HIP=1` is worth ~9.7 % decode TPOT, and
`VLLM_GDN_HIP_PREFILL=1` costs ~5.5 % prefill. Both defaults stay "0" until
the remaining GDN prefill kernels are profiled.

## QSA sparse attention ported: the QSA path is now fully HIP (2026-10-13)

`_qsa_sparse_paged_gqa_splitk_kernel` + `_qsa_merge_splitk_kernel` are now
implemented as `qsa_sparse_paged_attention_rdna2` (split-K, warp-per-GQA-head,
base-2 online softmax, fp32 partials merged by log-sum-exp). With
`VLLM_RDNA_QSA_HIP=1`, **no Triton kernel runs in the QSA path any more** — the
three leftovers (expand, splitk, merge) are all HIP.

Correctness is solid: **bit-exact against Triton on the `num_splits == 1` path**,
and 3–6e-5 abs / ~1e-3 rel (1–2 fp16 ulps) with splits, where the only difference
is the merge reordering the partial sums. 22 new cases in `test_qsa_rdna2.py`
(GQA 4/1 and 2/2, topk 8/64/129, rows 1/7/40, a fully masked row, `out=`), and the
tolerance is pinned at `rtol=5e-3, atol=1e-4` rather than a loose 1e-2 that would
hide a real bug. Full GDN+QSA suite: 408 passed.

**It is slower than Triton, so enabling that flag is now a regression.** Kernel
time, GPU 0, `torch.profiler`, 30 iters:

| shape | Triton | HIP | delta |
|---|---:|---:|---:|
| decode `rows=8, topk=512` | 56.2 us | 105.2 us | +87 % |
| prefill `rows=512, topk=2048` | 800.9 us | 2072.3 us | +159 % |
| GQA2 `rows=64, topk=1024` | 216.4 us | 585.3 us | +170 % |

This is a deliberately simple first version, written for correctness. A second
pass (2026-10-14) measured *why* it is slow instead of guessing, and the answer is
not the one the first write-up assumed.

### The gap is not MFMA, and not instruction count

**gfx1030 has no MFMA**, so Triton's `tl.dot` cannot be a different compute class:
compiling a `tl.dot` for gfx1030 and reading the ISA shows it lowers to
`v_dot2c_f32_f16`, the same instruction this kernel already uses. The gap is in the
memory path. `rocprofv3 --pmc` over one call (GPU 0, 20 iterations,
`rows=512, topk=2048`, `BLOCK_N=64`, 4 splits):

| counter, per call | HIP | Triton | ratio |
|---|---:|---:|---:|
| `SQ_INSTS_VALU` | 108.6 M | 35.1 M | 3.1x |
| `SQ_INSTS_SALU` | 421.7 M | 0.63 M | **669x** |
| `SQ_INSTS_LDS` | 6.2 M | 6.9 M | 0.9x |
| total | 536.5 M | 42.6 M | 12.6x |

The SALU excess is the two integer divisions by a runtime `page_size` per column,
per warp, twice per tile (8x what Triton does); Triton instead does one masked 2-D
tile load per operand, stages it and reads it back 128 bits at a time, while this
kernel issues 4-byte `half2` loads for `k` and four separate 2-byte
`global_load_ushort` for `v`, per column, per lane, each with its own 64-bit address
computation.

### Three cheap fixes were implemented and measured — all washes

| variant | decode `rows=8 topk=512` | prefill `rows=512 topk=2048` | verdict |
|---|---:|---:|---|
| baseline | 105.3 us | 2070.5 us | — |
| (2) index→page lookup hoisted to LDS | 99.9 / 99.3 us | 2116.3 / 2119.3 us | **wash**, reverted |
| (a) same, counters re-measured | — | — | 536.5 M → 330 M instructions (-38 %) |
| 4 independent `fdot2` chains for the score dot | 105.1 us | 2082.3 us | **wash**, reverted |

A/B/A in one session (`/tmp/bench_sparse.py`, GPU 0); the box drifts 5-10 % *between*
sessions, so only in-session pairs mean anything here.

The two results together are the useful finding: **cutting 38 % of the instructions
changed nothing, and 4x the ILP in the score dot changed nothing**, so this kernel is
latency-bound on the scattered per-column global loads, and neither instruction count
nor dot-chain depth is the lever. (2) and the chain split were therefore reverted
rather than landed as unmeasurable churn.

### What is left, and what it has to beat

Fixes (1) and (3) — the shared staged K/V — remain, and the measurements above say
they must land **as one rewrite**: K and V gathered once per tile into LDS with a
vectorised 8-wide gather shared across the group's warps, then read back as `b128`
for the score dot and in transposed form for `P @ V`. That is the only remaining
change that touches the measured bottleneck (redundant, scalar, scattered global
loads). (4) — hoisting `q` — still needs the kernel templated on `HEAD_DIM` first:
`head_dim` is runtime, so `q_reg[d2]` is dynamically indexed and the resource probe
shows 272 B/lane of scratch instead of registers. Do not retry (4) before the
template lands.

**Validate any rewrite with `rocprofv3` time, not with counters**: this pass is the
proof that counter reductions here do not imply speed.

Until the rewrite lands, `VLLM_RDNA_QSA_HIP=1` should stay off: previously it made
the QSA path a wash (HIP indexer + Triton attention), and it is now a net loss.
**SUPERSEDED — the rewrite landed on 2026-10-14; see the next section.**

## The rewrite landed: the HIP sparse attention now beats Triton at head_dim 256 (2026-10-14)

The "what is left" rewrite above (shared LDS-staged K/V, vectorised gather,
unrolled `P @ V`) landed, together with an LDS-staged `partial_lse` and a 4-chain
ILP merge kernel. **The section's verdict is superseded: `VLLM_RDNA_QSA_HIP=1` is
no longer a regression.**

It also turned out that every earlier measurement was taken at the wrong geometry.
The model's main attention is **`head_dim = 256`** (partial_rotary_factor 0.25 over
head_dim 256 -> rotary_dim 64, `mrope_section` [11,11,10]), not 128, so with the flag
on the server died in warmup: `qsa_sparse_paged_attention_rdna2: head_dim must be
<= 128`. Two defects in the rewrite:

* the staged K/V tiles used one 128+8 LDS row stride. The kernel is now templated on
  the stride and dispatches 136 (head_dim <= 128, ~20 KB LDS, 3 workgroups/CU) or 264
  (head_dim <= 256, ~39 KB, 1 workgroup/CU). `BLOCK_N = 64` does not fit at 264 and is
  not instantiated.
* `ACC_DIMS` was hardcoded 4, so 32 lanes covered only dims [0,128) and dims
  [128,256) were never written (48 % of the output). It is now derived from the LDS
  stride, with an `if constexpr` 16-byte `uint4` read in `P @ V` for the 8-dim case.

A third bug produced `NaN` on fully masked rows: clang contracts
`col_score * scale_log2 - next_max` into an FMA, so the exact product minus the
*rounded* max left a ~1e11 residue for the `-1e20` masked sentinel and `exp2` turned
it into `inf` (`inf * 0` V = NaN). The scores are now scaled before the max reduction
so the subtraction is exact, and the sentinel is `-INFINITY`.

Measured at the production geometry (group_size 6, kv_heads 1, **head_dim 256**,
topk 2048, page_size 1024), in-session vs Triton, `torch.profiler` kernel time:

| rows | Triton | HIP | delta |
|---:|---:|---:|---:|
| 3 | 169.3 us | 120.9 us | -28.6 % |
| 6 | 262.4 us | 200.4 us | -23.6 % |
| 12 | 433.6 us | 364.5 us | -15.9 % |
| 24 | 468.6 us | 436.7 us | -6.8 % |
| 64 | 548.7 us | 510.8 us | -6.9 % |
| 256 | 1772.9 us | 1200.5 us | -32.3 % |
| 512 | 3424.4 us | 2339.7 us | -31.7 % |
| 1024 | 4950.0 us | 4639.3 us | -6.3 % |
| 2048 | 9708.6 us | 9298.3 us | -4.2 % |

HIP is faster at all nine served shapes. `test_qsa_rdna2.py` parametrizes
head_dim {128, 256} and is at 290 passed.

### The rest of the gated path, measured the same way (2026-10-14)

`tools/rdna2/bench_qsa_rdna2.py` now sweeps rows 3..2048 (in-session HIP vs Triton,
`kernel_us`): `store_cache_rows`, `compress_groups` and `mqa_paged` are faster at
every shape. **`expand_indices` is slower at every shape** (e.g. 63.5 -> 129.8 us at
rows 1024, 76.4 -> 90.3 us at rows 2048) — it is a small integer compaction kernel and
is the one gated op that still loses. `tools/rdna2/bench_gdn_prefill_chain_rdna2.py`
adds the 1024/2048 served chunks: the GDN prefill chain is at parity, not faster
(-2.6 % at 8k, +1.5 % at 2k, +21 % at 16k x2), with `prep`/`kkt`/`solve_wy`/`o` slower
and `delta_h` faster.

### M-RoPE and the metadata builder are Triton-free now

Two more Triton sites were removed from the QSA path: `build_qsa_metadata` is routed
to the pure-torch builder under the gate, and the 2-D (multimodal) M-RoPE now uses the
existing gfx1030 HIP kernel (`mrope_forward_rdna2`) from both
`MRotaryEmbedding.forward_hip` (the HIP branch previously sat in the unreachable
`forward_cuda`) and `amd/indexer_qsa.py::apply_qsa_rope`.

### The runtime audit (both flags, text + multimodal)

With `VLLM_RDNA_QSA_HIP=1` **and** `VLLM_GDN_HIP_PREFILL=1`, the real server (every
worker instrumented via a `PYTHONPATH` `sitecustomize.py` that logs every
`JITFunction.run`) launches **zero Triton kernels whose innermost vLLM frame is in the
QSA path, the GDN prefill chain, or `rotary_embedding`**. The multimodal request does
reach the mm path (`_bilinear_pos_embed_kernel`, `_prepare_rope_positions_kernel`), so
this is not a vacuous pass.

The only Triton left in the GDN layer is `causal_conv1d` — and that is **recipe
configuration, not a fallback**: `tools/rdna/recipes/*.env` set
`VLLM_CAUSAL_CONV1D_RDNA2_{FWD,UPDATE}=0`. Those are a third pair of knobs, not the two
flags under test. Independently, the HIP conv kernels only supported
`conv_states.size(2) == width - 1`, which MTP=2 never satisfies; that is fixed (the
host wrappers now use the effective `state_len = width - 1`, exactly as Triton does,
and the guards accept extra capacity). Even with the knobs on, the MTP spec-decode
update (`num_accepted_tokens` is not None) has no HIP implementation and stays Triton.
`amd/ops/hc.py` (hyper-connection) is also Triton and is not gated by either flag.

## TunableOp: the whole grid re-measured, and the 256/512-token chunk family (2026-10-14)

`tools/rdna2/check_v620_tuning.py` reported **7 of the 14 shapes the live cache/batch
config needs as missing** — the live launcher resolves to `--block-size 1024
--max-num-batched-tokens 2048`, so the chunks that actually run are M = 1024 and 2048.
Two mechanisms had kept them missing, and both are deliberate:

1. `tools/rdna2_028/tunableop_rows_pipeline.sh` tunes **only the shapes present in
   `--shape-source`**, so the 26 prefill shapes the 2026-10-12 curation dropped were
   never re-tested.
2. `curate_tunableop_rows.py` folds `Default` rows into `dropped` by design ("an absent
   row and a `Default` row are behaviourally identical").

All 168 shapes (7 dense `(N,K)` x 12 prefill M + the 84 decode/MTP rows) were therefore
re-tuned and then measured heuristic / current / new **in one session** (the 3 % margins
are meaningless across sessions: the same rows drift 5-10 %).

**The 7 live-grid holes stay holes.** A fresh tune does not beat the heuristic by 3 % on
any of them (`10240x320`, `4096x2560`, `2560x1536`, `3584x2560` at M=1024; `10240x320`,
`512x2560`, `10240x2560` at M=2048). Filling them with an explicit `Default` row would
silence the warning without changing behaviour, and would pin those shapes against online
tuning — so the warning is left as-is. It is telling the truth.

**The real gap was one the checker cannot express.** The live launcher sets
`VLLM_RDNA_DYNAMIC_PREFILL=1`; its ladder is `RUNGS = (256, 512, 1024)` with
`LPT_SHARED = 512`, and `scheduler.py` sets `num_new_tokens = long_prefill_token_threshold`
whenever more than one request is eligible. At ≥4 eligible requests a **512-token chunk**
reaches the GEMMs, and 256 at 8 — and there was no row for M = 256/512 at all.
`check_v620_tuning.py` takes only `--block-size`/`--max-num-batched-tokens`, so it cannot
see this family.

| shape | M | heuristic | tuned row | delta |
|---|---:|---:|---:|---:|
| `10240x2560` | 512 | 4.33 ms | **1.22 ms** | −72 % |
| `336x10240` | 512 | 1.05 ms | **0.24 ms** | −77 % |
| `10240x2560` | 256 | 2.32 ms | **0.59 ms** | −75 % |
| `512x2560` | 512 | 0.229 ms | **0.102 ms** | −55 % |
| `2560x1536` | 512 | 0.243 ms | **0.207 ms** | −15 % |
| `3584x2560` | 800 | 0.851 ms | **0.728 ms** | −15 % |

The tuned solver reaches ~22 TFLOPS on the large-K shapes where the heuristic manages ~6,
so these are real, not noise. 11 rows added; 2 shipped prefill rows replaced by a
*different* faster solver (`10240x2560` M=1024, `2560x1536` M=3200).

**Two traps worth remembering.** (a) Two further "adopt" decisions carried the *same*
solver id as the shipped row — same-session noise, not a better kernel — and were
rejected; only adopt when the solver actually changes. (b) 18 decode/MTP rows (M=2..8)
measured 10-77 % slower than the heuristic, which looks damning, but that region is
launch-bound: identical shape families flip sign between M=1 (row 22-58 % *faster*) and
M=2..8, on ~14 MFLOP GEMMs timed at 80-240 µs. The decode grid is left byte-identical;
deleting those rows needs a small-M measurement (no per-call sync, larger reps) first.

Result: **139 rows/rank** (55 prefill + 84 decode/MTP), **139/139 matched independent FP32
on all four devices**, **139/139 lookup hits, 0 misses**. `tests/tools/test_v620_tuning_coverage.py`
passes; note that it asserts the *warning* path, so a green test is not a completeness proof.

Not yet measured: the end-to-end size of the 256/512 win. It only materialises when the
controller is in the shared cell, so a single-request prefill benchmark will not show it —
that needs a mixed-load run with ≥4 eligible requests plus a `[dyn-prefill] ADJUST` line
confirming a 512 cap.

---

## S6 — defaults flipped (2026-10-16)

`VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_HC_PREFILL_HIP`, `VLLM_GDN_HIP_PREFILL` and the
`VLLM_RDNA_ALL_HIP` umbrella are now `"1"` by default; only
`VLLM_RDNA_PLE_CONV_HIP` stays `"0"` (not in the measured set). The umbrella
gates 29 call sites beyond the five prefill gates (decode sampler, top-k/top-p,
QSA metadata, MTP conv1d + GDN spec-verify, unquantized MoE, vision pos-embed);
its prefill win is the gates' win, and its decode-side ports are exercised by the
`VLLM_RDNA_ALL_HIP=1` test suite.

### Decisive measurement — order-controlled gates-only A/B/A
`off -> on -> off`, three fresh servers, umbrella `0` in all three arms, prefix
caching off, `VLLM_RDNA_DYNAMIC_PREFILL=0`, 19 cases/arm, 0 failures
(`tools/rdna2/e2e_prefill_ab.sh gates`; logs
`.pi/sweep_{triton_gc_off1,gates_gc_on,triton_gc_off2}/`). `drift` = off2 vs off1
(identical config), `delta` = on vs mean(off1, off2). Values = total tok/s.

| scenario | conc | off1 | on | off2 | drift | delta |
|---|---:|---:|---:|---:|---:|---:|
| 1k | 4 | 1777.8 | 1879.5 | 1797.8 | +1.12 % | **+5.13 %** |
| 1k | 8 | 1835.0 | 1993.9 | 1890.7 | +3.03 % | **+7.04 %** |
| 1k | 32 | 1868.3 | 1983.0 | 1880.8 | +0.67 % | **+5.78 %** |
| 2k | 1 | 1799.4 | 1843.0 | 1791.0 | -0.46 % | **+2.66 %** |
| 2k | 4 | 1789.7 | 1850.3 | 1794.0 | +0.24 % | **+3.26 %** |
| 2k | 16 | 1784.4 | 1836.3 | 1781.9 | -0.14 % | **+2.98 %** |
| 8k | 8 | 1750.2 | 1776.6 | 1754.3 | +0.23 % | **+1.39 %** |
| 16k | 8 | 1729.3 | 1744.4 | 1734.8 | +0.32 % | **+0.71 %** |

Drift is ±0.5 % at 2k/8k/16k, so +3 % is 6x the noise; TTFT p50 tracks it
(-2.7..-3.2 % at 2k). The win decays with prompt length (1k ~+6 %, 2k ~+3 %,
8k ~+1.3 %, 16k ~+0.7 %), because QSA/GDN/HC are a fixed slice while 16k is
dominated by full-context attention identical in both arms. This supersedes
Round 7 (gates-only prefill was -1.4 % / -0.7 % on the pre-rewrite build) and
the umbrella+gates sweep, whose +2.5-5.4 % at 1k/2k showed the umbrella adds
nothing over the gates.

### Accuracy — GSM8K parity (a substitution, not the repo eval)
`.pi/gsm8k_g_{off,on}.json`: 150 GSM8K questions, 5-shot, greedy
(`temperature=0`, `seed=42`), chat endpoint, `max_tokens=1024`. **off 0.9800
(147/150), on 0.9800 (147/150), 0 discordant pairs, 150/150 identical answers.**
Not pinned at ceiling (3 wrong), so the test could have detected a difference.
`evalscope` is not installed, so the prescribed
`tests/evals/qwen4_exp/test_accuracy.py` (GSM8K + AIME25 thresholds) did not run.

### Owed, not done
* **A TP=4 PP / logit-fidelity run for the GDN gate.** The old "6.4 % lower PP"
  claim is pre-fix (2026-10-07, `e8d1701784`) and was never re-measured; GSM8K
  cannot rule out a logit-fidelity change. Counter-evidence: the kernels have
  fp16-ulp-scaled parity tests against an fp64 oracle, and for `delta_h` HIP is
  the *more* accurate side (6.8e-4 vs 2.5e-3).
* `tools/rdna/recipes/flashnext*.env` also had to change (they pinned
  `VLLM_GDN_HIP_PREFILL=0`, and `serve_rdna.sh` re-applies the caller env over
  the recipe, so the code default alone would not have reached the launcher).
