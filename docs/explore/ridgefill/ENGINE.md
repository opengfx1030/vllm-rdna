# Engine: RidgeFill on dest V1 (v0.33)

**Status: untested explore draft.** Flag default off. Do not merge this into
a recipe. The model math and the gfx1030 apply path are in `README.md` and
`SILICON.md`. This page is where the fill sits in the V1 runner, and what it
does to chunks, MTP, prefix cache, and the CPU KV tier.

Checked against this tree at `cb81247c13`. The inference note this page
started from named `qwen3_next.py` as the Flash-Next hook. That file is the
27B hybrid. Flash-Next on ROCm is `vllm/models/qwen4_exp/amd/model.py`.

## What is public

| Artifact | Public? | Use here |
| --- | --- | --- |
| Paper | OpenReview `ymLpDkJIGs`, Zenodo `10.5281/zenodo.23890247` | Method and quoted numbers |
| Technical note | Zenodo `10.5281/zenodo.23179168` | Earlier note. The PDF's own DOI. |
| Maps | Hugging Face, Apache-2.0 | Refit required for W4A16 / EXL3 / HI2 |
| ridgefill-data v1.8 (`f3b343fe`) | Code Apache-2.0, results CC BY 4.0 | Contract: `code/c5/`, `code/kva/`, `code/engine-fix/` |
| `tcclaviger/vllm:29.08.10` | Image only. Patches not public (paper Appendix F). | Not opened on this box (no Docker). Do not port it. RDNA4 / MXFP4. |
| llama.cpp-ridgefill `b0439f7bb` | MIT | Algorithm reference. Not copied. |
| radiance | Apache-2.0 on codeberg `main` | Separate engine. Left out. |

`code/c5/convert.py`, `code/c5/engine_check.py`, and `code/kva/kva.py` are
in the v1.8 tree. `convert.py` documents `layer.L` `[2560, 10241]` bf16 and
says the engine files `vllm/v1/ced.py` and
`vllm/models/qwen4_exp/amd/ced.py` live in the image. Those two files are
not in ridgefill-data. This implementation does not reconstruct them by
disassembly. It follows the paper and the public file layout.

## Mechanism the engine has to honor

- Prefill only. Decode is the full model (paper sections 1 and 2.1).
- Split `S=24`, tail `T=2048` for Flash-Next. Filled rows skip the late MoE.
  QSA still writes K, V, and indexer keys. GDN still updates its state.
- Their measured run: no MTP, chunks of 2048, prefix caching off, about two
  thirds of the routed experts in host memory (paper section 5). That PCIe
  term is not a V620 forecast.

## (a) Model hook, preferred, and what is actually wired

**Where a GPU port edits:** `Qwen4ExpDecoderLayer.forward` in
`vllm/models/qwen4_exp/amd/model.py`, after the HC combine that materializes
the four streams and before `mix` normalizes them (the `combine_and_mix` /
`mix` branch around the top of that `forward`). Filled rows at layers
`>= split`: apply the map, run projections and the cache write, skip
`self.mlp`. Exact rows, including the tail, call the layer as they do now.

**What this draft does:** `Qwen4ExpForCausalLM.forward` calls
`Session.begin_forward` only when `self.model._ridgefill` is not `None`. That
object is `None` unless `VLLM_RIDGEFILL=1`. The call records a row plan and
uploads two index tensors, outside the compiled region. The compiled model
then reaches the fill steps through opaque custom ops
(`torch.ops.vllm.ridgefill_*`), so a forward with no filled rows passes
`None` and traces the flag-off graph. `capture_split` checks that the carried
residual is 10240 wide. Neither reads nor writes the activation values. A
flag-on boot is still numerically exact. The `None` check is the whole
default-path difference in this file.

`qwen3_next.py` `Qwen3NextDecoderLayer.forward` always runs `self.mlp`
after attention (the post-attention norm and the MLP call). That is the
27B hook, later, and it is not edited. 27B has no four-stream residual;
both widths are 5120.

CUDA-graph mode in the flashnext recipe is `FULL_AND_PIECEWISE`. Do not put
the map GEMM on the decode capture. If the prefill graph will not capture
the filled-row branch, the first soak uses `--enforce-eager`.

## (b) Chunks

`SchedulerConfig.enable_chunked_prefill` defaults true
(`vllm/config/scheduler.py`). Recipes set `MAXBAT=2048`. The tail is the
last T tokens of the **full** prompt.

`policy.plan_prompt` marks `[0, filled_end)` filled. With v2 alignment,
`filled_end` rounds down to the hash block so the exact tail only grows and
no block mixes filled and exact tokens. v1 does not cache, so the unaligned
cut is enough: a chunk that crosses the cut is `mixed`
(`policy.chunk_class`) and the layer masks rows inside the chunk.

Do not build the 27B two-sweep (probe, then exact rows) in this draft.
Flash-Next plain mode has no exact prefix rows (paper section 5.2). First
soak can use one sequence if a mixed batch of fill-prefill plus unrelated
decode misbehaves. Their C5 run used `--max-num-seqs 1`. The recipe's
`SEQS=6` can stay for the flag-off path.

## (c) MTP, forced off in v1

| Option | When |
| --- | --- |
| **C1. Refuse speculative config while the flag is on** | This draft. `bind_flashnext` raises. Covers MTP and any other drafter, because filled rows have no final hidden state. |
| **C2. Fill the prefill, enable MTP only on the exact decode** | Later, after RULER and the dtype pre-check. Acceptance rate is meaningless until the MTP head dtype is known. |
| **C3. A `final` map for filled-row hidden states** | The release does not ship one (`convert.py`). Do not invent it. |

`flashnext-mtp2.env` is the production MTP=2 recipe. It is unchanged. The
example env sets `MTP=0`.

Community report, not a dest measurement: the Flash-Next HF checkpoint
ships its MTP head in bf16, and on the tcclaviger stack MTP accepts nothing
unless that head stays bf16 (`R9K_MTP_MLP=bf16`).
<https://x.com/tunapedia/status/2110968988847243678>. Before anyone reads an
acceptance rate, on the baseline MTP recipe or on a future C2 run, record
the MTP head dtype of **our** quant. A silent 0% is not a RidgeFill result.

## (d) Prefix cache and the CPU tier

**v1, this draft.** `enable_prefix_caching` must be false while the flag is
on, or `bind_flashnext` raises. The launcher does not lock APC off: the
flashnext `FEATURES` list includes `prefix-caching`, and
`ENABLE_PREFIX_CACHING` defaults to 1 (`tools/rdna/serve_rdna.sh`). The
example env sets it to 0.

**v2, gated by the multi-turn trace.** `VLLM_RIDGEFILL_APC=v2` plus
`VLLM_RIDGEFILL_MAP_SHA`. `generate_block_hash_extra_keys`
(`vllm/v1/core/kv_cache_utils.py`) appends
`("ridgefill", split, map_sha, mode)` for a block that ends at or before
the block-aligned filled cut. Exact blocks add nothing, so a chat-history
prefix that was computed exactly still matches. A filled block matches only
a later request that opted into the same map. It must not be treated as
exact. The flag-off path returns before the import, so those hashes do not
change.

**CPU KV.** `policy.cpu_tier_may_store` is false for a filled block while
the flag is on. The call site to wire, and not wired in this draft, is the
store path in `vllm/v1/kv_offload/cpu/manager.py` and the hash use in
`vllm/v1/simple_kv_offload/manager.py`. Wiring it inside the default store
path is the GPU follow-up; the helper is the contract. A filled block is
never reloaded as an exact prefix.

PLE offload is orthogonal (n-gram table). Do not port their
`code/c5/shim/sitecustomize.py`. Quality mode's scattered exact rows are
why the paper's protected-row speedup gave a lot of the gain back; that
mode is not implemented (`VLLM_RIDGEFILL_MODE=quality` raises).

## (e) TP=4

Replicate the full fp16 map on every rank (about 1.17 GiB each). Do not
column-shard it. The residual is already 10240 wide on every rank; the
README cites the lines. Their correction tensors are head-sharded for
TP=2. Quality mode is out, so this draft does not load them. Extend the
shard to TP=4 (`48 % 4 == 0`) only if that mode is built.

## (f) What the flag does on a boot

`VLLM_RIDGEFILL=1` with MTP or prefix caching: process raises before
serving. With both off and no map path: serves the exact model, and the
forward records the split width. With a map path: `validate_pins` against
`VLLM_RIDGEFILL_PINS`, and a mismatch raises. There is no approximate
forward in this commit.

Workers that do not see the environment stay on the exact path. That is
fail-safe. A follow-up can move the switch into `--additional-config` so
it cannot depend on the process environment. The fork's RDNA rules prefer
that. This draft keeps the env var the task asked for.

## Qualitative gain

Paper ceiling is about `1/r` of the work still done (lower layers, tail,
maps): 1.64× at 9K and 1.88× at 32K, measured 1.50× and 1.60× on two R9700s
with most experts in host memory (section 5.3). On 4× V620 with resident
experts, expect less. Do not publish a tok/s until `TESTPLAN.md` has a row.

## Unknowns

- Bytes inside `tcclaviger/vllm:29.08.10` (image not opened).
- Whether dest's QSA selection is already deterministic across two boots.
- Monokernel expert cache interaction. Recipes do not require it.
- 27B engine shape (no 10240-wide stream). Second target only.
