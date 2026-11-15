# LeapQuant capacity: what limits KV tokens on Flash-Next TP4, and what turns the INT8 state into tokens

Status: analysis, CPU only. No GPU boot, lease or running server was used. Every
token count comes from the real KV-cache planner and the real platform block
rule, run by `tools/rdna/leapquant_capacity.py` on branch
`feat/gdn-leapquant-pack-layers` at 57c2c1c6f0 (PR #170). The script reproduces
the boot-log capacities of both models to the digit (section "Method"). Rows
named "as served" use the shipped code. Rows named "what-if" change one named
piece, exist only in the planner, and need an engine change and a GPU boot
before anyone relies on them. Tok/s, quality and GPU residency are not measured
here; a claim about them carries `[INFERENCE]`.

The question: PR #168 (LeapQuant INT8 GDN state) and PR #170 (page packing) cut
the GDN state per head from 65,536 B to 37,024 B, yet the logged KV capacity of
the Flash-Next recipe moves only from 266,945 to 273,456 tokens (+2.44 %), and
the 27B at its recipe block even loses 1.1 %. This note finds what the pool is
spent on, sweeps the block size (the point of #170 on Flash-Next), and prices
every change that could turn the saved bytes into tokens.

## Credits

- **Algorithm:** LeapQuant, Yi Pan, Haocheng Xi, Kan Zhu, Xingyang Li, Yibo Wu,
  Mayank Mishra, Hongtao Zhang, William X. Zheng, Baris Kasikci, Song Han,
  Kurt Keutzer, Rishabh Iyer and Ion Stoica
  ([arXiv 2609.38166](https://arxiv.org/abs/2609.38166)). This note only counts
  the state blob the paper defines.
- **Record ring, cursor and per-window flush:** the ReplaySSM design by Dao AI
  Lab and NVIDIA ([blog](https://dao-lab.ai/blog/2026/replayssm/),
  [vllm#48018](https://github.com/vllm-project/vllm/pull/48018),
  [vllm#62004](https://github.com/vllm-project/vllm/pull/62004) by
  Johnny-Liou). The blob holds that ring, and the retained-input reserve of the
  planner follows from it.
- **Planner harness:** `tests/v1/core/test_leapquant_page_packing.py` of PR #170.
  The script imports it and swaps the model geometry. The planner and the
  platform rule it calls keep their upstream authors in git history.
- **Code:** the script and this note are original work for this fork. No code is
  taken from outside this fork.

## Summary

- **The pool is attention.** Flash-Next TP4, MTP2, KV 4,026,531,840 B per rank:
  92.1 % of the pool is attention KV at 14,144 B per token per rank (13 layers
  at 1,024 B main page plus 64 B compressed key per token). The state one
  request really needs is 117.9 MB flag off and 85.7 MB with LeapQuant: 2.9 %
  and 2.1 % of the pool. A request *holds* more, because state lives in whole
  block ids and every id costs a full attention block: 246.2 MB flag off at
  block 1,024 (48 % used), 152.1 MB with LeapQuant at 512 (56 % used).
- **Why the gain is small.** The log prints `262,144 x N / (attention ids +
  state ids)`. The 21 state ids of a LeapQuant request (17 flag off) each cost
  one block of attention, so the loss against the no-state ceiling of 284,672
  tokens is about `S x b` tokens: 17,408 flag off at 1,024, 10,752 at 512. The
  whole prize of deleting all state is +6.64 % over flag off. The shipped +2.44 %
  is 37 % of it. With state at its exact bytes in the same pool the ceiling is
  278,238 (+4.23 %).
- **Where the +2.44 % comes from.** Moving the flag-off recipe from 1,024 to the
  smallest block the platform keeps for FP32 (800) gives +2,797 tokens (43 % of
  the gain). FP32 state in a packed layout at block 512 (what-if) adds +1,677
  (26 %). The INT8 state adds the last +2,037 (31 %). The order matters. On the
  same packing planner the state format alone is worth +0.3 % to +1.2 % at
  blocks 128 to 1,024 (+0.75 % at 512). At exact bytes the INT8 saving is 2,337
  tokens (+0.88 % of the flag-off baseline), small because the state is only
  2.1 % of the pool.
- **The block floor is what INT8 buys.** The platform pads every GDN page to the
  attention page and raises the block until the attention page is at least the
  state page. The INT8 page (469,888 B) lets the block fall from 800 to 464.
  With padded pages the state ids stay at 21 for every block from 400 to 1,536,
  so capacity falls as the block grows: 464 gives 274,222 (+2.73 %), 512
  273,456 (+2.44 %), 1,024 263,090 (-1.44 %).
- **Packing (#170) does not add tokens at the shipped block.** For b <= 592 the
  packed layout equals the padded one (3 groups of 12 layers). At b >= 608 it
  merges to 2 groups of 18 (21 -> 16 ids), at b >= 1,200 to 1 group (11 ids).
  608 gives 273,846 (+2.59 %) and 640 gives 273,220 (+2.35 %), within +0.14 % and
  -0.09 % of the shipped 512. The five saved ids cut the state cost by only 512
  tokens at 640, because each id now costs 640 tokens; a larger pool remainder
  (7.37 MB, 512 tokens) and attention id rounding (256 tokens) take it back.
  Packing pays where the block is large: at 1,024 it turns -1.44 % into
  +0.37 %, and on the 27B recipe block it turns -1.11 % into +0.75 %. It never
  lowers the block floor, because the packed early return in the platform rule
  comes after the block raise.
- **What does add tokens** (planner numbers, none needs the speculative blocks
  removed): merging the PLE layer into a GDN group gives 276,045 at block 512
  (+3.41 %, 5 ids saved, 8x limit 24,576 -> 27,136). Lifting the platform floor
  under packing gives 275,784 at block 128 (+3.31 %) and 276,737 at 112
  (+3.67 %), with 12 GDN groups. A separate exact-bytes state pool gives 278,528
  at 512 (+4.34 %).
- **Reserve (needs a user decision, not taken here).** The 5 ids per GDN group are
  1 home + 2 retained + 2 draft. Retained 2 -> 1 equals async scheduling off
  (275,523, +3.21 %); no draft ids gives 277,622 (+4.00 %). The speculative
  blocks stay as they are in every row that is not marked "reserve".
- **At eight resident requests the effect is larger than the log shows.** The
  log counts the state of one request. The state of eight requests takes 48.9 %
  of the pool flag off and 30.2 % with LeapQuant at 512. The longest
  max-model-len at 8.0x concurrency goes 17,408 (flag off) -> 24,576 (shipped)
  -> 27,136 (PLE merged). Planner bound, not measured `[INFERENCE]`.
- **Recommendation.** Keep #168 at block 512 (boot-verified). Do the PLE merge
  next. Treat floor-lift plus packing at 128 to 256 and the state pool as
  optional, behind GPU evidence. The "Ranked options" table has the rest.

## Method

The script (`tools/rdna/leapquant_capacity.py`) imports the harness of PR #170
and drives the functions the engine runs at boot: `Platform._align_hybrid_block_size`,
`get_global_kv_cache_groups`, `get_kv_cache_configs` and the capacity formula of
`get_kv_cache_capacity` (`vllm/v1/core/kv_cache_utils.py`). It needs no GPU, no
weights and no network. A plan takes about 35 ms; a full run takes under a
minute on a CPU host.

- **Capacity formula.** A request at max-model-len consumes
  `cdiv(max_memory_usage_bytes, page)` ids from every group, all groups draw ids
  from one pool, and the log prints `int(num_blocks / ids_per_request x
  max_model_len)`. The "log tokens" in this note are that number. The log does
  not subtract the null block; the engine's "enough memory for one request"
  check does. A mode shows "-" in the tables when attention ids, state ids and
  the null block exceed the pool: padded LeapQuant from block 992 (except
  1,024, which fits with no spare id, concurrency 1.0036), flag off from 1,248.
- **Model per rank (Flash-Next, TP4).** 36 GDN layers, 12 attention layers plus
  one MTP draft attention layer (QSA: main page, compressed key page and key
  ring), one TP-replicated PLE short-conv layer. The specs are registered in the
  engine's order (key ring, compressed key, main KV per layer), which reproduces
  the group block sizes `(512, 512, 512, 512, 8, 512)` of the boot log.
- **Recipe values.** `tools/rdna/recipes/flashnext-leapquant-mtp2.env`: block 512,
  8 sequences, 2,048 batched tokens, KV 4,026,531,840 B, MTP 2, max-model-len
  262,144, `--replayssm-state-dtype leapquant`, `VLLM_USE_V2_MODEL_RUNNER=1`.
  The flag-off baseline is `flashnext-mtp2.env` (block 1,024).
- **State reserve per group.** `MambaSpec.max_memory_usage_bytes`
  (`vllm/v1/kv_cache_interface.py`) in align mode is
  `page x (1 + concurrent_batches + num_speculative_blocks +
  num_prefill_checkpoint_blocks)`. `concurrent_batches` is
  `vllm_config.max_concurrent_batches` with LeapQuant (`pp_size + 1` = 2 with
  async scheduling on the V2 runner, `vllm/config/vllm.py`) and 1 without.
  `num_speculative_blocks` is 2 (MTP2) and there are no prefill-checkpoint
  blocks, so LeapQuant reserves 5 ids per group and flag off 4. The GDN groups
  and the PLE group carry that reserve; the key ring takes 1 id.
- **Two extra views of the same plan.** "Longest max-model-len at 8.0x" is the
  largest max-model-len at which the planner still reports 8.0x concurrency (the
  recipe's `SEQS=8`). "Attention tokens left with 8 resident" is
  `(N - 8 S) x b`, the attention capacity left after eight requests hold their
  state ids. The log metric hides both, because it charges one request.
- **As served and what-if.** As-served rows run the real platform function, which
  raises a requested block to the floor. What-if rows replace only that function
  (`bypass_platform`: block kept, pages padded or packed as the mode says). The
  replacement equals the real function in every pair where the platform keeps
  the block. Other what-ifs patch one planner formula while a capacity is read
  (reserve, PLE merge, blob shape) and say so in their header. Nothing changes
  the engine.

Self-check against logged capacities (MB in this note are 10^6 bytes):

| model | block | mode | logged tokens | source | planner tokens |
|---|---|---|---|---|---|
| Flash-Next | 1024 | flag off | 266,945 | boot log | 266,945 |
| Flash-Next | 512 | LeapQuant #168 | 273,456 | boot log | 273,456 |
| Flash-Next | 512 | LeapQuant, 4 ids per group | 275,523 | boot log 2026-11-13 | 275,523 |
| Flash-Next | 640 | LeapQuant + packing | 273,220 | planner, not booted | 273,220 |
| 27B | 1024 | flag off | 328,658 | boot log | 328,658 |
| 27B | 1024 | LeapQuant #168 | 325,019 | boot log | 325,019 |
| 27B | 1024 | LeapQuant + packing | 331,129 | planner, not booted | 331,129 |
| 27B | 512 | LeapQuant #168 | 334,768 | planner, not booted | 334,768 |

The platform replacement matched the real platform in 130 (Flash-Next) and 183
(27B) block and mode pairs.

## Flash-Next TP4: pool and pages

Per rank, KV 4,026,531,840 B (3.75 GiB):

| item | bytes | note |
|---|---|---|
| attention, main page | 1,024 per token per layer | 1 KV head per rank, K and V |
| attention, compressed key page | 64 per token per layer | `tokens_per_state=4` |
| attention per token, 13 layers | 14,144 | 12 target layers + the MTP draft layer (1,088 B, 7.7 %) |
| block id bytes `A` | 14,144 x b | what one id costs in the pool, whatever it holds |
| GDN page, LeapQuant | 469,888 | conv 25,600 + 12 heads x 37,024 |
| GDN page, FP32 | 812,032 | conv 25,600 + 12 heads x 65,536 |
| PLE page (one layer, replicated) | 225,280 | |
| key ring page | 2,048 per layer | 8-token circular buffer |
| pool ceiling `KV / 14,144` | 284,683 tokens | zero state, no rounding |

Blob of one value head (37,024 B): INT8 residual `z` 16,384 (44.3 %), scales
1,024 (2.8 %), compensators 2,048 (5.5 %), record ring 17,552 (47.4 %), header 16.
Almost half of the blob is the record ring, which sets the size of any smaller
blob (option (e)).

### The block floor

The platform rule (`Platform._align_hybrid_block_size`,
`vllm/platforms/interface.py`, lines 1101 to 1127) computes
`attn_block_size = align x cdiv(state_page, align x attention_bytes_per_token_per_layer)`
with `align = max(backend kernel block, requested block)` (and the lcm of the
backend alignments), raises the block to it, asserts
`attention_page >= state_page`, then pads the state page up to the attention
page. LeapQuant packing returns after the raise, so it never lowers the floor.

| state | state page (B) | smallest block | attention page there (B) | padding per page (B) |
|---|---|---|---|---|
| FP32 (flag off) | 812,032 | 800 | 819,200 | 7,168 (0.9 %) |
| LeapQuant INT8 | 469,888 | 464 | 475,136 | 5,248 (1.1 %) |

At 448 the attention page is 458,752 B, below 469,888 B, so a request for 448
is served at 896. Blocks are multiples of 16: the QSA backend lists
`MultipleOf(16)` and `fa_rdna2.cu` needs a multiple of 8. A block such as 464
(16 x 29) is code-compatible with the runtime `PAGE_SIZE` kernels but not
qualified on GPU `[INFERENCE]`.

## Flash-Next TP4: block-size sweep

Tokens are the log metric. "-> 464" means the platform served 464 for the
requested block. The last column lifts the platform floor under packing
(what-if). "infeasible" means the planner rejects the plan: one block id cannot
hold the state pages, or the served block leaves no room for one max-length
request.

| requested block | flag off | LeapQuant #168 | LeapQuant + packing #170 | packing, platform rule lifted (what-if) |
|---|---|---|---|---|
| 16 | -> 800: 269,742 (+1.05%) | -> 464: 274,222 (+2.73%) | -> 464: 274,222 (+2.73%) | infeasible |
| 64 | -> 832: 269,228 (+0.86%) | -> 512: 273,456 (+2.44%) | -> 512: 273,456 (+2.44%) | 272,306 (+2.01%) |
| 128 | -> 896: 268,063 (+0.42%) | -> 512: 273,456 (+2.44%) | -> 512: 273,456 (+2.44%) | 275,784 (+3.31%) |
| 192 | -> 960: 266,648 (-0.11%) | -> 576: 271,486 (+1.70%) | -> 576: 271,486 (+1.70%) | 275,139 (+3.07%) |
| 256 | -> 1024: 266,945 (+0.00%) | -> 512: 273,456 (+2.44%) | -> 512: 273,456 (+2.44%) | 275,003 (+3.02%) |
| 320 | -> 960: 266,648 (-0.11%) | -> 640: 270,050 (+1.16%) | -> 640: 273,220 (+2.35%) | 275,468 (+3.19%) |
| 384 | -> 1152: 264,283 (-1.00%) | -> 768: 267,199 (+0.10%) | -> 768: 270,930 (+1.49%) | 273,975 (+2.63%) |
| 448 | -> 896: 268,063 (+0.42%) | -> 896: 264,648 (-0.86%) | -> 896: 268,930 (+0.74%) | 274,236 (+2.73%) |
| 464 | -> 928: 267,386 (+0.17%) | 274,222 (+2.73%) | 274,222 (+2.73%) | (as served) |
| 480 | -> 960: 266,648 (-0.11%) | 273,682 (+2.52%) | 273,682 (+2.52%) | (as served) |
| 496 | -> 992: 265,862 (-0.41%) | 273,106 (+2.31%) | 273,106 (+2.31%) | (as served) |
| 512 | -> 1024: 266,945 (+0.00%) | 273,456 (+2.44%) | 273,456 (+2.44%) | (as served) |
| 576 | -> 1152: 264,283 (-1.00%) | 271,486 (+1.70%) | 271,486 (+1.70%) | (as served) |
| 608 | -> 1216: 263,269 (-1.38%) | 270,824 (+1.45%) | 273,846 (+2.59%) | (as served) |
| 640 | infeasible | 270,050 (+1.16%) | 273,220 (+2.35%) | (as served) |
| 704 | infeasible | 268,797 (+0.69%) | 272,252 (+1.99%) | (as served) |
| 768 | infeasible | 267,199 (+0.10%) | 270,930 (+1.49%) | (as served) |
| 800 | 269,742 (+1.05%) | 266,650 (-0.11%) | 270,526 (+1.34%) | (as served) |
| 832 | 269,228 (+0.86%) | 266,033 (-0.34%) | 270,039 (+1.16%) | (as served) |
| 896 | 268,063 (+0.42%) | 264,648 (-0.86%) | 268,930 (+0.74%) | (as served) |
| 1024 | 266,945 (+0.00%) | 263,090 (-1.44%) | 267,926 (+0.37%) | (as served) |

Percentages are against the 266,945-token flag-off recipe at 1,024. The full
grid of every multiple of 16 is in the appendix.

Pages, padding, reserve and prefix-cache grain for the served blocks and the
requested set (128, 192, 256, 320, 384, 448, 512, 576, 608, 640, 768, 1,024
plus the smallest blocks 464 and 800). "State page: natural -> in plan" shows
the padding the platform forces. "GDN groups" is groups x layers. State ids are
GDN + PLE + ring per request. State held and needed are MB per request (held is
ids x `A`; needed is the bytes of the state itself).

| block | mode | attention page (B) | state page: natural -> in plan (B) | padding per page (B) | padded to | GDN groups (layers) | state ids GDN + PLE + ring | state held / needed (MB) | blocks N | tokens | vs 266,945-token flag off | prefix hit grain (tokens) | prefill tokens per step at 2,048 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | packing, floor lifted (what-if) | 131,072 | 469,888 -> 469,888 | 0 (0.0%) | none | 12 x 3 | 60 + 5 + 1 | 119.5 / 85.7 | 2224 | 275,784 | +3.31% | 128 | 2,048 |
| 192 | packing, floor lifted (what-if) | 196,608 | 469,888 -> 469,888 | 0 (0.0%) | none | 4 x 5 + 4 x 4 | 40 + 5 + 1 | 124.9 / 85.7 | 1482 | 275,139 | +3.07% | 192 | 1,920 |
| 256 | packing, floor lifted (what-if) | 262,144 | 469,888 -> 469,888 | 0 (0.0%) | none | 6 x 6 | 30 + 5 + 1 | 130.4 / 85.7 | 1112 | 275,003 | +3.02% | 256 | 2,048 |
| 320 | packing, floor lifted (what-if) | 327,680 | 469,888 -> 469,888 | 0 (0.0%) | none | 4 x 9 | 20 + 5 + 1 | 117.7 / 85.7 | 889 | 275,468 | +3.19% | 320 | 1,920 |
| 384 | packing, floor lifted (what-if) | 393,216 | 469,888 -> 469,888 | 0 (0.0%) | none | 4 x 9 | 20 + 5 + 1 | 141.2 / 85.7 | 741 | 273,975 | +2.63% | 384 | 1,920 |
| 448 | packing, floor lifted (what-if) | 458,752 | 469,888 -> 469,888 | 0 (0.0%) | none | 3 x 12 | 15 + 5 + 1 | 133.1 / 85.7 | 635 | 274,236 | +2.73% | 448 | 1,792 |
| 464 | LeapQuant #168 (padded pages) | 475,136 | 469,888 -> 475,136 | 5,248 (1.1%) | attention page | 3 x 12 | 15 + 5 + 1 | 137.8 / 85.7 | 613 | 274,222 | +2.73% | 464 | 1,856 |
| 464 | LeapQuant + packing #170 | 475,136 | 469,888 -> 469,888 | 0 (0.0%) | none | 3 x 12 | 15 + 5 + 1 | 137.8 / 85.7 | 613 | 274,222 | +2.73% | 464 | 1,856 |
| 512 | LeapQuant #168 (padded pages) | 524,288 | 469,888 -> 524,288 | 54,400 (11.6%) | attention page | 3 x 12 | 15 + 5 + 1 | 152.1 / 85.7 | 556 | 273,456 | +2.44% | 512 | 2,048 |
| 512 | LeapQuant + packing #170 | 524,288 | 469,888 -> 469,888 | 0 (0.0%) | none | 3 x 12 | 15 + 5 + 1 | 152.1 / 85.7 | 556 | 273,456 | +2.44% | 512 | 2,048 |
| 576 | LeapQuant #168 (padded pages) | 589,824 | 469,888 -> 589,824 | 119,936 (25.5%) | attention page | 3 x 12 | 15 + 5 + 1 | 171.1 / 85.7 | 494 | 271,486 | +1.70% | 576 | 1,728 |
| 576 | LeapQuant + packing #170 | 589,824 | 469,888 -> 469,888 | 0 (0.0%) | none | 3 x 12 | 15 + 5 + 1 | 171.1 / 85.7 | 494 | 271,486 | +1.70% | 576 | 1,728 |
| 608 | LeapQuant #168 (padded pages) | 622,592 | 469,888 -> 622,592 | 152,704 (32.5%) | attention page | 3 x 12 | 15 + 5 + 1 | 180.6 / 85.7 | 468 | 270,824 | +1.45% | 608 | 1,824 |
| 608 | LeapQuant + packing #170 | 622,592 | 469,888 -> 469,888 | 0 (0.0%) | none | 2 x 18 | 10 + 5 + 1 | 137.6 / 85.7 | 468 | 273,846 | +2.59% | 608 | 1,824 |
| 640 | LeapQuant #168 (padded pages) | 655,360 | 469,888 -> 655,360 | 185,472 (39.5%) | attention page | 3 x 12 | 15 + 5 + 1 | 190.1 / 85.7 | 444 | 270,050 | +1.16% | 640 | 1,920 |
| 640 | LeapQuant + packing #170 | 655,360 | 469,888 -> 469,888 | 0 (0.0%) | none | 2 x 18 | 10 + 5 + 1 | 144.8 / 85.7 | 444 | 273,220 | +2.35% | 640 | 1,920 |
| 768 | LeapQuant #168 (padded pages) | 786,432 | 469,888 -> 786,432 | 316,544 (67.4%) | attention page | 3 x 12 | 15 + 5 + 1 | 228.1 / 85.7 | 370 | 267,199 | +0.10% | 768 | 1,536 |
| 768 | LeapQuant + packing #170 | 786,432 | 469,888 -> 469,888 | 0 (0.0%) | none | 2 x 18 | 10 + 5 + 1 | 173.8 / 85.7 | 370 | 270,930 | +1.49% | 768 | 1,536 |
| 800 | flag off (FP32 state) | 819,200 | 812,032 -> 819,200 | 7,168 (0.9%) | attention page | 3 x 12 | 12 + 4 + 1 | 192.4 / 117.9 | 355 | 269,742 | +1.05% | 800 | 1,600 |
| 800 | LeapQuant #168 (padded pages) | 819,200 | 469,888 -> 819,200 | 349,312 (74.3%) | attention page | 3 x 12 | 15 + 5 + 1 | 237.6 / 85.7 | 355 | 266,650 | -0.11% | 800 | 1,600 |
| 800 | LeapQuant + packing #170 | 819,200 | 469,888 -> 469,888 | 0 (0.0%) | none | 2 x 18 | 10 + 5 + 1 | 181.0 / 85.7 | 355 | 270,526 | +1.34% | 800 | 1,600 |
| 1024 | flag off (FP32 state) | 1,048,576 | 812,032 -> 1,048,576 | 236,544 (29.1%) | attention page | 3 x 12 | 12 + 4 + 1 | 246.2 / 117.9 | 278 | 266,945 | +0.00% | 1024 | 2,048 |
| 1024 | LeapQuant #168 (padded pages) | 1,048,576 | 469,888 -> 1,048,576 | 578,688 (123.2%) | attention page | 3 x 12 | 15 + 5 + 1 | 304.2 / 85.7 | 278 | 263,090 | -1.44% | 1024 | 2,048 |
| 1024 | LeapQuant + packing #170 | 1,048,576 | 469,888 -> 469,888 | 0 (0.0%) | none | 2 x 18 | 10 + 5 + 1 | 231.7 / 85.7 | 278 | 267,926 | +0.37% | 1024 | 2,048 |

Rows for 128 to 448 are what-if: the platform serves them at the floor (464 or
higher). In packed mode the "attention page" is the layer page of the block
(`b x 1,024 B`); the block id still holds a whole group of packed state pages,
so nothing is padded. In padded mode the padding grows with the block: at 1,024
the LeapQuant page is padded by 123 % to 1,048,576 B, which is why #168 alone
loses 1.44 % there.

### What the sweep shows

- Padded LeapQuant state is 21 ids for every block from 400 to 1,536, so the
  state cost `S x b` grows linearly with the block and the capacity falls from
  274,222 at 464 to 263,090 at 1,024. INT8 helps only through the smaller block
  its page size allows.
- Packing changes the plan only from b = 598 (18 layers of 469,888 B fit one
  8,457,984 B group). Below that its rows equal the padded rows.
- Padded LeapQuant at 1,024 fits one max-length request with no spare id (277
  ids plus the null block in 278 blocks, concurrency 1.0036).
- The what-if column stays between 274.9k and 276.7k for blocks 80 to 304 and
  then declines to 274.2k at 448 and 273.2k at 640. It does not climb below 80:
  at 48 every GDN layer is its own group (36 groups, 186 state ids) and at 32
  one block id (469,888 B) is smaller than one state page.

## Why the capacity moves with the block, or not

The log capacity is

```text
tokens = 262,144 x N / (attention ids + state ids),   N = floor(KV / (14,144 x b))
```

For a block that divides 262,144 this is `N b / (1 + S b / 262,144)`. Three
things move with the block: `N b` (the pool in tokens, minus the remainder of
the division), the attention id rounding, and the state cost `S b`.

| config | block b | A (bytes) | N | pool remainder bytes (tokens-eq) | attention ids | attention rounding (tokens over 262,144) | state ids S | S x b (tokens-eq) | N x b | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| flag off | 1024 | 14,483,456 | 278 | 131,072 (9) | 256 | 0 | 17 | 17,408 | 284,672 | 266,945 |
| LeapQuant #168 | 512 | 7,241,728 | 556 | 131,072 (9) | 512 | 0 | 21 | 10,752 | 284,672 | 273,456 |
| LeapQuant #168 | 464 | 6,562,816 | 613 | 3,525,632 (249) | 565 | 16 | 21 | 9,744 | 284,432 | 274,222 |
| LeapQuant + packing | 608 | 8,599,552 | 468 | 1,941,504 (137) | 432 | 512 | 16 | 9,728 | 284,544 | 273,846 |
| LeapQuant + packing | 640 | 9,052,160 | 444 | 7,372,800 (521) | 410 | 256 | 16 | 10,240 | 284,160 | 273,220 |

- **1,024 -> 512 (flag off -> LeapQuant).** The state cost falls from 17,408 to
  10,752 tokens (-6,656): the smaller block makes each of the 21 ids cheaper
  than each of the 17 was. INT8 enters because it makes 512 legal; the id count
  went up (17 -> 21) because LeapQuant reserves one more retained id per group.
- **512 -> 464.** The ids stay at 21; the block shrinks by 48 tokens, so the
  state cost falls 1,008 tokens (10,752 -> 9,744). The pool loses 240 tokens of
  `N b` to a larger remainder (3.5 MB) and 16 to rounding. Net +766 tokens.
- **512 -> 640 packed.** Packing saves 5 ids, but each id now costs 640 tokens:
  `S b` falls only from 10,752 to 10,240 (+512 tokens). The pool remainder
  grows from 9 to 521 tokens-eq (444 blocks of 9,052,160 B leave a 7.37 MB
  tail), so `N b` falls from 284,672 to 284,160 (-512), and the attention ids
  round up from 409.6 to 410 (256 tokens charged). The three effects (+512,
  -512, -256) sum to about -256 tokens before the interplay and -236 exactly:
  273,220 against 273,456.
- **512 -> 608 packed.** The same five saved ids give `S b` = 9,728 (-1,024). The
  remainder is only 137 tokens-eq, the attention rounding costs 512. Net +390.
- **The state itself is the part no block removes.** With 5 copies of the state
  per layer (home, 2 retained, 2 draft) a request needs 85.7 MB, about 6,060
  tokens-eq of attention (85.7 MB / 14,144 B) whatever the block is. The padded
  21 ids at 512 hold 152.1 MB and cost 10,752 tokens-eq; a layout cannot go
  below the first number.
- **Small blocks.** A smaller block packs fewer layers per id, so the GDN groups
  multiply: 36 at b = 48, 18 at 80, 12 at 112, 9 at 144, 6 at 208, 4 at 304, 3
  at 400, 2 at 608, 1 at 1,200. `S = 5 G + 6` (5 ids per GDN group, 5 for the
  PLE group, 1 for the ring) grows with `G`, but `G` is about 1,196 / b, so
  `S b` stays near 5,980 + 6 b plus the rounding of layers per group: 7,392 at
  112, 7,488 at 208, 7,904 at 304 against 8,400 at 400 and 10,752 at 512. This
  is why the what-if curve is flat within 0.7 % from 80 to 304.

Group count against the block, with the best block per count (what-if below
464, as served above):

| GDN groups | groups x layers | first block with this count | tokens there | state ids | platform accepts the first block |
|---|---|---|---|---|---|
| 36 | 36 x 1 | 48 | 275,232 | 186 | no (below 464) |
| 18 | 18 x 2 | 80 | 276,521 | 96 | no (below 464) |
| 12 | 12 x 3 | 112 | 276,737 | 66 | no (below 464) |
| 9 | 9 x 4 | 144 | 276,707 | 51 | no (below 464) |
| 8 | 4 x 5 + 4 x 4 | 176 | 275,968 | 46 | no (below 464) |
| 6 | 6 x 6 | 208 | 276,494 | 36 | no (below 464) |
| 5 | 1 x 8 + 4 x 7 | 272 | 275,580 | 31 | no (below 464) |
| 4 | 4 x 9 | 304 | 276,003 | 26 | no (below 464) |
| 3 | 3 x 12 | 400 | 275,309 | 21 | no (below 464) |
| 2 | 2 x 18 | 608 | 273,846 | 16 | yes |
| 1 | 1 x 36 | 1200 | 270,122 | 11 | yes |

## Prefix-cache granularity and the prefill step

The block is the scheduler block, the hash block and the hit granularity in all
configurations here (`resolve_kv_cache_block_sizes`): a hit lands on a multiple
of the block. The scheduler ends every prefill chunk on a block boundary
(`_mamba_block_aligned_split`), so a step prefills `floor(2,048 / b) x b` tokens.

| block | hit grain (tokens) | expected re-prefill per hit `[INFERENCE]` | prefill tokens per step at 2,048 | vs 2,048 |
|---|---|---|---|---|
| 112 (what-if) | 112 | ~56 | 2,016 | -1.6 % |
| 128 (what-if) | 128 | ~64 | 2,048 | 0 |
| 256 (what-if) | 256 | ~128 | 2,048 | 0 |
| 464 | 464 | ~232 | 1,856 | -9.4 % |
| 512 (shipped) | 512 | ~256 | 2,048 | 0 |
| 608 | 608 | ~304 | 1,824 | -10.9 % |
| 640 | 640 | ~320 | 1,920 | -6.3 % |
| 800 | 800 | ~400 | 1,600 | -21.9 % |
| 1024 (flag off) | 1,024 | ~512 | 2,048 | 0 |

The re-prefill column assumes prefix lengths spread uniformly inside a block
(half a block lost per hit); it is arithmetic, not a measurement. A smaller
block refines hits, and the flag-off recipe at 1,024 loses twice as much per hit
as the shipped 512. A block that does not divide the batched-token budget loses
prefill tokens per step (a larger budget that is a multiple of the block, for
example 2,320 at 464, restores a full step at the cost of activation memory
`[INFERENCE]`). The 27B packing validation already used 2,560 for block 640.

A retained Mamba checkpoint is one block id in each Mamba group, so it costs
`(G + 1) x A` bytes (G GDN groups plus the PLE group): 57.9 MB flag off at
1,024, 29.0 MB at 512, 26.3 MB at 464, 27.2 MB at 640 packed and 23.5 MB at 128
with 12 GDN groups (what-if). The default `prefix_cache_retention_interval` is 0,
which keeps only semantic checkpoints (the replay boundary and shared-prefix
junctions), so this cost is paid per retained checkpoint, not per block
`[INFERENCE: residency not measured]`.

## Where the pool goes

Per request and for eight resident requests, Flash-Next TP4 (bytes held are ids
x `A`; needed is the state itself):

| config | owner | ids per max-length request | held MB | needed MB | % of KV pool | state reserve of 8 requests MB | 8 requests, % of pool |
|---|---|---|---|---|---|---|---|
| flag off, 1,024 | attention | 256 | 3,707.8 | 3,707.8 | 92.08% | | |
| flag off, 1,024 | GDN state (3 groups of 12) | 12 | 173.8 | 116.9 | 4.32% | 1,390.4 | 34.53% |
| flag off, 1,024 | PLE | 4 | 57.9 | 0.9 | 1.44% | 463.5 | 11.51% |
| flag off, 1,024 | ring | 1 | 14.5 | 0.0 | 0.36% | 115.9 | 2.88% |
| LeapQuant #168, 512 | attention | 512 | 3,707.8 | 3,707.8 | 92.08% | | |
| LeapQuant #168, 512 | GDN state (3 groups of 12) | 15 | 108.6 | 84.6 | 2.70% | 869.0 | 21.58% |
| LeapQuant #168, 512 | PLE | 5 | 36.2 | 1.1 | 0.90% | 289.7 | 7.19% |
| LeapQuant #168, 512 | ring | 1 | 7.2 | 0.0 | 0.18% | 57.9 | 1.44% |
| LeapQuant #168, 464 | attention | 565 | 3,708.0 | 3,707.8 | 92.09% | | |
| LeapQuant #168, 464 | GDN state (3 groups of 12) | 15 | 98.4 | 84.6 | 2.44% | 787.5 | 19.56% |
| LeapQuant #168, 464 | PLE | 5 | 32.8 | 1.1 | 0.81% | 262.5 | 6.52% |
| LeapQuant #168, 464 | ring | 1 | 6.6 | 0.0 | 0.16% | 52.5 | 1.30% |
| LeapQuant + packing, 608 | attention | 432 | 3,715.0 | 3,707.8 | 92.26% | | |
| LeapQuant + packing, 608 | GDN state (2 groups of 18) | 10 | 86.0 | 84.6 | 2.14% | 688.0 | 17.09% |
| LeapQuant + packing, 608 | PLE | 5 | 43.0 | 1.1 | 1.07% | 344.0 | 8.54% |
| LeapQuant + packing, 608 | ring | 1 | 8.6 | 0.0 | 0.21% | 68.8 | 1.71% |
| LeapQuant + packing, 640 | attention | 410 | 3,711.4 | 3,707.8 | 92.17% | | |
| LeapQuant + packing, 640 | GDN state (2 groups of 18) | 10 | 90.5 | 84.6 | 2.25% | 724.2 | 17.99% |
| LeapQuant + packing, 640 | PLE | 5 | 45.3 | 1.1 | 1.12% | 362.1 | 8.99% |
| LeapQuant + packing, 640 | ring | 1 | 9.1 | 0.0 | 0.22% | 72.4 | 1.80% |

The PLE layer needs 1.1 MB and holds 36.2 MB at 512, 5 ids for one short
convolution state: 0.9 % of the pool per request and 7.2 % for eight. The ring
holds 1 id.

Summary line per configuration:

| config | block | N | tokens | state share of a max-length request | held MB | needed MB | needed / held | attention tokens left with 8 resident | longest max-model-len at 8.0x |
|---|---|---|---|---|---|---|---|---|---|
| flag off | 1024 | 278 | 266,945 | 6.23% | 246.2 | 117.9 | 47.9% | 145,408 | 17,408 |
| LeapQuant #168 | 512 | 556 | 273,456 | 3.94% | 152.1 | 85.7 | 56.4% | 198,656 | 24,576 |
| LeapQuant #168 | 464 | 613 | 274,222 | 3.58% | 137.8 | 85.7 | 62.2% | 206,480 | 25,520 |
| LeapQuant + packing | 608 | 468 | 273,846 | 3.57% | 137.6 | 85.7 | 62.3% | 206,720 | 25,536 |
| LeapQuant + packing | 640 | 444 | 273,220 | 3.76% | 144.8 | 85.7 | 59.2% | 202,240 | 24,960 |

The state share of a max-length request equals the capacity loss against the
no-state ceiling (`S b / (262,144 + S b)`: 10,752 / 272,896 = 3.94 % at 512).

## Ceilings

What bounds each configuration, each against its own log tokens:

| config | block | tokens | reserve 4 ids per group | state at exact bytes, shared pool | separate state pool, exact bytes | GDN state = 0 bytes | pure attention (all state 0) | attention tokens with 8 resident, shared / separate pool |
|---|---|---|---|---|---|---|---|---|
| flag off | 1024 | 266,945 | 266,945 (+0.00%) | 275,901 (+3.35%) | 275,456 (+3.19%) | 279,218 (+4.60%) | 284,672 (+6.64%) | 145,408 / 217,088 |
| LeapQuant #168 | 512 | 273,456 | 275,523 (+0.76%) | 278,238 (+1.75%) | 278,528 (+1.85%) | 281,374 (+2.90%) | 284,672 (+4.10%) | 198,656 / 236,032 |
| LeapQuant #168 | 464 | 274,222 | 276,106 (+0.69%) | 277,987 (+1.37%) | 278,383 (+1.52%) | 281,426 (+2.63%) | 284,414 (+3.72%) | 206,480 / 236,176 |
| LeapQuant + packing | 608 | 273,846 | 275,693 (+0.67%) | 277,583 (+1.36%) | 277,921 (+1.49%) | 280,099 (+2.28%) | 283,989 (+3.70%) | 206,720 / 235,904 |
| LeapQuant + packing | 640 | 273,220 | 275,158 (+0.71%) | 277,473 (+1.56%) | 278,128 (+1.80%) | 279,788 (+2.40%) | 283,882 (+3.90%) | 202,240 / 236,160 |

- "State at exact bytes" charges the state its bytes (5 copies LeapQuant, 4
  flag off) in the same pool. The gap between it and the log tokens (+1.75 % at
  512) is what block granularity wastes: padding, unused block bytes, the PLE
  and ring ids. The INT8 saving itself is the gap between the two exact-bytes
  ceilings, 278,238 against 275,901: 2,337 tokens.
- "Separate state pool" gives state its exact bytes from a second pool and leaves
  attention the rest. At eight resident requests it leaves 236,032 attention
  tokens against 198,656 (+18.8 %).
- Attention KV is 92 % of the pool. This note does not size the lever on that
  92 % (the KV dtype); it is outside the question `[INFERENCE]`.

## Candidate unlocks

Each is a what-if on the planner unless it says "as served". Nothing here
removes a speculative block, except (c) and the "reserve" rows of (f), which are
reported as potential and need a user decision.

### (a) Smallest block

As served: 464 for LeapQuant (274,222) and 800 for flag off (269,742), see "The
block floor". Lifting the floor needs a small platform change: under
`leapquant_gdn_pack_enabled` skip the block raise and the
`attention_page >= state_page` assertion (lines 1109 to 1122 of
`vllm/platforms/interface.py`), because the packed planner fits pages by bytes.
The what-if rows use exactly that (`bypass_platform`). Blocks stay multiples of
16 (QSA kernel block `MultipleOf(16)`).

### (b) Layers per block

The group-count table above. Fewer groups need a bigger block (3 groups at 400,
2 at 608, 1 at 1,200); more groups allow a smaller one. Best tokens: 276,737 at
112 (12 groups), 276,707 at 144, 276,521 at 80, 276,494 at 208. A larger block
loses more than the group count wins back, because `S b` grows with `b`.

### (c) Reserve per group (potential, needs a user decision)

Each Mamba group reserves ids: 5 = 1 home + 2 retained + 2 draft with LeapQuant.
The retained count is `max_concurrent_batches` (2 with async scheduling on the
V2 runner), the draft count is `num_speculative_blocks`. This is not applied
anywhere in this note.

| config | block | 5 ids (today) | 4 ids: retained 1 | 3 ids: no draft | 2 ids |
|---|---|---|---|---|---|
| LeapQuant #168 | 512 | 273,456; 8x limit 24,576 | 275,523 (+0.76%); 26,624 | 277,622 (+1.52%); 28,672 | 279,754 (+2.30%); 30,720 |
| LeapQuant #168 | 464 | 274,222; 25,520 | 276,106 (+0.69%); 27,376 | 278,017 (+1.38%); 29,232 | 279,955 (+2.09%); 31,088 |
| LeapQuant + packing | 608 | 273,846; 25,536 | 275,693 (+0.67%); 27,360 | 277,564 (+1.36%); 29,184 | 279,461 (+2.05%); 31,008 |
| LeapQuant + packing | 640 | 273,220; 24,960 | 275,158 (+0.71%); 26,880 | 277,123 (+1.43%); 28,800 | 279,117 (+2.16%); 30,720 |

The 4-id row equals the flag-off reserve pattern (async scheduling off) and
reproduces the smoke boot of 2026-11-13 at 512 (275,523). The LeapQuant
document ([leapquant-gdn-2026-11-13.md](leapquant-gdn-2026-11-13.md)) explains
why two unresolved prefill chunks can retain three home states plus two draft
slots, and the kernels read only `spec_state_indices[:, 0]`. Removing draft ids
touches verify lengths and the V2 runner's shared spec-block parameters, which
is separate work and outside this change.

### (d) Merge the PLE layer into a GDN group

The planner buckets Mamba layers with `MambaSpec.is_uniform_with_collection`
(`vllm/v1/kv_cache_interface.py`), which requires equal `page_size_bytes`,
`tp_replicated` and `state_format` among others. The PLE layer (replicated, no
`state_format`, a 225,280 B page) never joins a GDN bucket, so it forms its own
group with 5 ids (36.2 MB held for 1.1 MB at 512). In padded mode a GDN group of
12 layers already has a free 13th layer slot (the attention group has 13
layers), so the PLE page fits at no extra bytes. The what-if (`ple_joins_gdn`)
is the simplest implementation: the PLE page is padded to the GDN page and the
37 Mamba layers form one bucket of equal pages. The predicate would need
`tp_replicated` and `state_format` relaxed. A group of mixed page sizes (the
real 225,280 B page) is not possible under the predicate, which demands one
`page_size_bytes`.

| block | mode | tokens | PLE merged | state ids | merged vs flag off at 1,024 | 8x limit |
|---|---|---|---|---|---|---|
| 464 | LeapQuant #168 | 274,222 | 276,582 (+0.86%) | 21 -> 16 | +3.61% | 25,520 -> 27,840 |
| 512 | LeapQuant #168 | 273,456 | 276,045 (+0.95%) | 21 -> 16 | +3.41% | 24,576 -> 27,136 |
| 608 | LeapQuant + packing | 273,846 | 273,846 (+0.00%) | 16 -> 16 | +2.59% | 25,536 |
| 640 | LeapQuant + packing | 273,220 | 276,465 (+1.19%) | 16 -> 11 | +3.57% | 24,960 -> 28,160 |
| 1024 | LeapQuant + packing | 267,926 | 272,943 (+1.87%) | | +2.25% | 18,432 -> 23,552 |

The merge needs a group with one more GDN page of free bytes
(`A - layers x 469,888 >= 469,888`). The padded layout has it at every served
block (a 13th layer slot). Packed layouts have it at 256, at 432 to 592 (3 x 12)
and from 640 (2 x 18); they do not at 112, 128, 144, 208, 304, 400 and 608 (at
608, 19 GDN pages are 8,927,872 B against a block of 8,599,552 B). The engine
change: relax the predicate, pad the PLE page, keep one block table for the
group, and bind the GDN and the short-conv metadata builders to the same KV
group (the V2 runner groups layers per backend inside a KV group
`[INFERENCE]`). The PLE layer keeps its home, retained and draft ids as part of
the group's ids. No speculative block is removed.

### (e) A smaller blob (numerics unqualified)

The 37,024 B blob of the shipped serving layout (window 32, rank 4) is 44.3 % INT8
residual and 47.4 % record ring. A smaller ring or window would shrink the page
and the smallest block. This table is arithmetic on the blob layout; the
numerics of those layouts are not qualified (the numerics note recommends window
32, rank 4) and the reserve is untouched.

| blob (window p, rank r) | bytes per head | state page (12 heads + conv) | smallest block | tokens there (#168) | tokens at 512 (#168) | best #170 block |
|---|---|---|---|---|---|---|
| p=32, r=4 (shipped) | 37,024 | 469,888 | 464 | 274,222 | 273,456 | 274,222 at 464 |
| p=24, r=4 | 32,896 | 420,352 | 416 | 275,009 | 273,456 | 275,303 at 544 |
| p=20, r=4 | 30,832 | 395,584 | 400 | 275,309 | 273,456 | 276,045 at 512 |
| p=16, r=4 | 28,768 | 370,816 | 368 | 276,072 | 273,456 | 276,112 at 480 |
| p=32, r=2 | 36,000 | 457,600 | 448 | 274,236 | 273,456 | 274,236 at 448 |
| p=16, r=2 | 27,744 | 358,528 | 352 | 276,517 | 273,456 | 276,582 at 464 |

A blob 25 % smaller (p=16, r=2) lowers the floor from 464 to 352 and gives
276,517 there: +3,061 tokens (+1.12 %) over the shipped 512 and +3.59 % over
flag off. Every smaller blob helps through the same two mechanisms, a lower
floor and cheaper ids; none is qualified numerically.

### (f) The levers together (what-if)

Cumulative columns: each column adds its lever to the previous one. Each cell is
tokens (vs flag off at 1,024) / longest max-model-len at 8.0x concurrency.

| block | packing, floor lifted | + PLE merged | + 4 ids per group | + 3 ids per group (no draft ids) |
|---|---|---|---|---|
| 112 | 276,737 (+3.67%) / 28,112 | 276,737 (+3.67%) / 28,112 | 278,240 (+4.23%) / 29,568 | 279,759 (+4.80%) / 31,024 |
| 128 | 275,784 (+3.31%) / 27,136 | 275,784 (+3.31%) / 27,136 | 277,490 (+3.95%) / 28,800 | 279,218 (+4.60%) / 30,464 |
| 144 | 276,707 (+3.66%) / 28,224 | 276,707 (+3.66%) / 28,224 | 278,193 (+4.21%) / 29,664 | 279,695 (+4.78%) / 31,104 |
| 208 | 276,494 (+3.58%) / 28,080 | 276,494 (+3.58%) / 28,080 | 277,994 (+4.14%) / 29,536 | 279,511 (+4.71%) / 30,992 |
| 256 | 275,003 (+3.02%) / 26,368 | 276,307 (+3.51%) / 27,648 | 277,887 (+4.10%) / 29,184 | 279,486 (+4.70%) / 30,720 |
| 304 | 276,003 (+3.39%) / 27,664 | 276,003 (+3.39%) / 27,664 | 277,564 (+3.98%) / 29,184 | 279,143 (+4.57%) / 30,704 |
| 400 | 275,309 (+3.13%) / 26,800 | 275,309 (+3.13%) / 26,800 | 276,945 (+3.75%) / 28,400 | 278,601 (+4.37%) / 30,000 |
| 464 | 274,222 (+2.73%) / 25,520 | 276,582 (+3.61%) / 27,840 | 278,017 (+4.15%) / 29,232 | 279,468 (+4.69%) / 30,624 |
| 512 | 273,456 (+2.44%) / 24,576 | 276,045 (+3.41%) / 27,136 | 277,622 (+4.00%) / 28,672 | 279,218 (+4.60%) / 30,208 |
| 608 | 273,846 (+2.59%) / 25,536 | 273,846 (+2.59%) / 25,536 | 275,693 (+3.28%) / 27,360 | 277,564 (+3.98%) / 29,184 |
| 640 | 273,220 (+2.35%) / 24,960 | 276,465 (+3.57%) / 28,160 | 277,785 (+4.06%) / 29,440 | 279,117 (+4.56%) / 30,720 |
| 1024 | 267,926 (+0.37%) / 18,432 | 272,943 (+2.25%) / 23,552 | 275,003 (+3.02%) / 25,600 | 277,095 (+3.80%) / 27,648 |

Blocks 112, 128, 144, 208, 304 and 400 show no gain from the merge because a
group there has less than one GDN page (469,888 B) free. At 128 (400,768 B free)
and 320 (297,088 B free) the real 225,280 B PLE page would fit where the padded
page does not, saving up to 5 ids: 640 tokens-eq at 128, 1,600 at 320
`[INFERENCE: arithmetic, not a planner run; the predicate does not allow a
group of mixed page sizes]`. The all-unlocked corner is about 279.8k (+4.8 %)
against the no-state ceiling of 284,672.

### (g) State format alone, FP32 against LeapQuant (what-if)

The same packing planner with the platform floor lifted. The FP32 rows keep the
flag-off reserve of 4 ids per group and the LeapQuant rows pay 5, so the
difference is the INT8 state net of the extra retained id.

| block | FP32: GDN groups | FP32: state ids | FP32 + packing | LeapQuant: GDN groups | LeapQuant: state ids | LeapQuant + packing | LeapQuant over FP32 |
|---|---|---|---|---|---|---|---|
| 64 | 36 x 1 | 149 | 274,679 | 36 x 1 | 186 | 272,306 | -0.86% |
| 112 | 36 x 1 | 149 | 267,513 | 12 x 3 | 66 | 276,737 | +3.45% |
| 128 | 18 x 2 | 77 | 274,356 | 12 x 3 | 66 | 275,784 | +0.52% |
| 176 | 12 x 3 | 53 | 274,716 | 4 x 5 + 4 x 4 | 46 | 275,968 | +0.46% |
| 256 | 9 x 4 | 41 | 273,712 | 6 x 6 | 36 | 275,003 | +0.47% |
| 400 | 6 x 6 | 29 | 272,093 | 3 x 12 | 21 | 275,309 | +1.18% |
| 464 | 1 x 8 + 4 x 7 | 25 | 272,363 | 3 x 12 | 21 | 274,222 | +0.68% |
| 512 | 1 x 8 + 4 x 7 | 25 | 271,419 | 3 x 12 | 21 | 273,456 | +0.75% |
| 608 | 4 x 9 | 21 | 270,824 | 2 x 18 | 16 | 273,846 | +1.12% |
| 640 | 4 x 9 | 21 | 270,050 | 2 x 18 | 16 | 273,220 | +1.17% |
| 800 | 3 x 12 | 17 | 269,742 | 2 x 18 | 16 | 270,526 | +0.29% |
| 1024 | 3 x 12 | 17 | 266,945 | 2 x 18 | 16 | 267,926 | +0.37% |

FP32 in the best layout (what-if) reaches 274,716 at block 176 (+2.91 %), 0.46 %
above the shipped LeapQuant at 512. The layout and the block floor are worth as
much as the state format.

## Ranked options (Flash-Next TP4)

Tokens are the planner's log metric at the recipe KV. "vs flag off" is against
266,945. The 8x limit is the longest max-model-len at 8.0x concurrency. Risk and
effort are judgments `[INFERENCE]`.

| rank | option | tokens | vs flag off | 8x limit | risk | effort | needs |
|---|---|---|---|---|---|---|---|
| 0 | flag off, block 1,024 (baseline) | 266,945 | +0.00 % | 17,408 | none | none | recipe `flashnext-mtp2` |
| 1 | LeapQuant #168, block 512 (shipped) | 273,456 | +2.44 % | 24,576 | boot-verified | done | recipe `flashnext-leapquant-mtp2` |
| 2 | PLE merged into a GDN group, block 512 | 276,045 | +3.41 % | 27,136 | medium-low: runner binds two backends to one KV group | small to medium: relax two terms of one predicate, pad the PLE page, share one block table | engine plus GPU boot |
| 3 | packing, platform floor lifted, block 128 | 275,784 | +3.31 % | 27,136 | medium: 12 GDN groups, an untested engine path, per-group overhead unknown | small platform change plus GPU | engine plus GPU boot |
| 3b | same at block 112 | 276,737 | +3.67 % | 28,112 | as above, block not a power of two | same | same |
| 4 | LeapQuant #168, block 464 | 274,222 | +2.73 % | 25,520 | low-medium: unqualified block, prefill step 1,856 (-9.4 %) | one recipe line | GPU boot |
| 5 | flag off, block 800 (no LeapQuant) | 269,742 | +1.05 % | 21,600 | prefill step 1,600 (-21.9 %) | one recipe line | recipe |
| 6 | reserve 4 ids per group (retained 1) at 512 | 275,523 | +3.21 % | 26,624 | async scheduling must be off | recipe | user decision |
| 7 | reserve 3 ids per group (no draft) at 512 | 277,622 | +4.00 % | 28,672 | high: verify lengths and V2 spec-block parameters | large | user decision |
| 8 | separate exact-bytes state pool at 512 | 278,528 | +4.34 % | (236,032 attention tokens left with 8 resident against 198,656 shared) | high: a second block pool across planner, worker and prefix-cache lookup | large | engine |
| 9 | everything unlocked (b=112, PLE, 3 ids) | 279,759 | +4.80 % | 31,024 | high | large | user decision |
| - | ceiling: no state at all | 284,672 | +6.64 % | | | | |

Rows 6, 7 and 9 need the user's decision on the speculative reserve and are not
recommended here. Rank is by net value after risk, not by price: row 4 is one
recipe line but buys +0.28 % over the shipped 512 for 9.4 % fewer prefill
tokens per step and an unqualified block, while row 2 buys +0.95 % with no block
or step change. Row 5 forgoes LeapQuant and spends 21.9 % of the prefill step.
Packing adds nothing to rows 1 and 4.

## Recommendation

1. Keep #168 at block 512 in the Flash-Next recipe (273,456, +2.44 %,
   boot-verified). Do not present packing as a Flash-Next capacity feature: at
   608 and 640 it lands within +0.14 % and -0.09 % of 512. It stays useful where
   the block is big (27B recipe at 1,024, or any recipe that needs a 1,024
   block: +0.37 % against -1.44 % for padded LeapQuant).
2. The next change to make is the **PLE merge**: +0.95 % over the shipped result
   (276,045, +3.41 % against flag off), 5 ids per request freed, and 27,136
   against 24,576 for the 8x limit, with the speculative blocks untouched. It is
   a small planner change (relax two terms of one predicate, pad the PLE page)
   plus a runner binding to qualify on GPU.
3. **Floor lift plus packing at block 112 to 256** is the second candidate
   (275.0k to 276.7k, +3.0 % to +3.7 %), the same range as the PLE merge at 512
   (276,045), not above it. Its extra value is a finer prefix-cache grain (128
   against 512); its cost is 6 to 12 GDN groups and a platform change. Take it
   only after a GPU boot shows many GDN groups are fine. In the what-if it does
   not stack with the PLE merge below block 400 (a group there has no spare GDN
   page); a mixed-size group could stack them `[INFERENCE]`.
4. Report both metrics when quoting the capacity: the log line (+2.44 %) and the
   longest max-model-len at the recipe's 8 sequences (+41 %: 17,408 to 24,576).
   The second is the one the INT8 state moves.
5. The reserve policy and the separate state pool are the user's call. Reserve
   4 ids gives +0.76 % at 512, no draft ids +1.52 %. The state pool gives +1.85 %
   over the shipped 512 at high cost.

What converts the INT8 saving into tokens: a smaller block (the floor from 800
to 464 is the whole mechanism of #168), fewer ids for the same bytes (the PLE
merge, fewer groups), or an allocator that spends bytes instead of ids (the
state pool). The saved bytes alone are worth 2,337 tokens (+0.88 %), because the
state is 2.1 % of the pool.

## Qwen3.8-27B AWQ, TP2 (short)

KV 12,000,000,000 B per rank, MTP2, 48 GDN layers, 16 attention layers plus the
draft, no PLE and no ring. Attention is 34,816 B per token per rank, and the
state pages are 939,776 B (LeapQuant) and 1,624,064 B (FP32). The floors are the
same as on Flash-Next: 464 for LeapQuant and 800 for flag off.

| configuration | block | tokens | vs flag off at 1,024 | 8x limit |
|---|---|---|---|---|
| flag off, recipe | 1024 | 328,658 | +0.00 % | 30,720 |
| flag off, best block | 800 | 331,535 | +0.88 % | 32,800 |
| LeapQuant #168, recipe block | 1024 | 325,019 | -1.11 % | 27,648 |
| LeapQuant #168 | 512 | 334,768 | +1.86 % | 35,328 |
| LeapQuant #168, smallest block | 464 | 335,363 | +2.04 % | 35,728 |
| LeapQuant + packing | 1024 | 331,129 | +0.75 % | 32,768 |
| LeapQuant + packing, best served block | 656 | 335,672 | +2.13 % | 36,080 |
| packing, floor lifted (what-if), best block | 112 | 335,950 | +2.22 % | 36,288 |

- At its recipe block (1,024) LeapQuant loses 1.11 % because padded pages hold
  534.8 MB of state ids for 225.5 MB of state (42 % used). The block change to 512
  is the fix: +1.86 %. Packing at 1,024 recovers +0.75 %, because 24 layers per
  group fit (2 groups, 10 ids) without padding. The exact-bytes shared-pool
  ceiling for LeapQuant at 512 is 336,266, so the layout is already 0.45 % short
  of it.
- The 27B has no PLE or ring, so there is no (d). Reserve variants at 512: 4 ids
  336,684, 3 ids 338,623, 2 ids 340,584 (potential, needs a user decision). The
  separate state pool gives 337,920 at 512.
- The pure attention ceiling is 344,064 (+4.69 % against flag off). INT8 over
  FP32 on the same packing planner is worth +0.2 % to +1.7 %.
- Exact state per request: 225.5 MB LeapQuant, 311.8 MB FP32. GDN held: 427.8 MB
  flag off, 534.8 MB LeapQuant at 1,024, 267.4 MB at 512, 356.5 MB packed at 1,024.

## KV sensitivity

Flash-Next at 6,000,000,000 B per rank (the "6 GB class" budget), same planner:

| configuration | block | tokens | vs flag off at 1,024 | 8x limit |
|---|---|---|---|---|
| flag off | 1024 | 397,537 | +0.00 % | 34,816 |
| flag off, best block | 800 | 402,713 | +1.30 % | 39,200 |
| LeapQuant #168 | 512 | 407,233 | +2.44 % | 41,984 |
| LeapQuant #168, smallest block | 464 | 408,873 | +2.85 % | 43,152 |
| LeapQuant + packing | 608 | 407,844 | +2.59 % | |
| packing, floor lifted (what-if) | 112 | 412,438 | +3.75 % | 45,584 |
| FP32 + packing, floor lifted (what-if) | 176 | 409,440 | +2.99 % | 43,648 |

The ratios hold: +2.44 % at 512 at both budgets, because the state cost
`S b / 262,144` does not depend on the pool size and only the remainder changes.
The pure-attention ceiling is 423,936 (+6.64 %). The flag-off block-800 step is
worth more here (+1.30 % against +1.05 %): it is 53 % of the LeapQuant gain at
6 GB against 43 % at the recipe budget.

## What the planner does not show

The following need a GPU, and nothing in this note claims them:

- **Blocks that are not powers of two** (464 and the what-if blocks 112, 144,
  176 and so on): the QSA and flash kernels take a runtime `PAGE_SIZE` (`//` and
  `%`), so they are code-compatible, but no boot or quality probe has run
  `[INFERENCE]`.
- **Many GDN groups** (6 to 12 at block 128 to 256): each group has a block
  table, a metadata builder and a state-index gather. The kernels read their
  state through per-group block tables, but the engine has not booted such a
  plan, and the per-step overhead is unknown `[INFERENCE]`.
- **The PLE merge** needs the runner to bind two backends to one KV group, and
  the PLE state to use the GDN group's ids.
- **Residency.** The planner's reserve is a bound. Measured peak HBM residency
  and the number of states actually held by eight requests are not known
  `[INFERENCE]`.
- **Throughput and prefix-hit rate.** The prefill step, the hit grain and the
  retained-checkpoint cost have the arithmetic shown above, not a measurement.
- **Attention KV.** The remaining 92 % of the pool is attention; this note does
  not analyze the KV dtype or the MTP draft layer (7.7 % of the per-token
  bytes).

## Reproduce

From the repository root, in a venv that imports vLLM, with no GPU visible:

```sh
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
  PYTHONPATH=. python tools/rdna/leapquant_capacity.py all
```

Other runs: `all --model 27b`, `all --kv-bytes 6000000000`, or one table:
`selfcheck`, `summary`, `sweep`, `anatomy`, `breakdown`, `ceilings`, `options`.
`--json out.json` writes every number of the run. `selfcheck` prints
`DIFFERENT` for any row where the planner and the logged capacity disagree. The
numbers in this note come from the runs `all`, `all --model 27b` and
`all --kv-bytes 6000000000` of 2026-11-15 on a CPU-only host with the tree of
PR #170 (no file of the tree was modified).

Related notes: [LeapQuant GDN](leapquant-gdn-2026-11-13.md),
[numerics and isolation](leapquant-gdn-numerics-and-isolation-2026-11-14.md),
[config guards](leapquant-config-guards-2026-11-14.md),
[page packing](leapquant-page-packing-2026-11-14.md) and
[PR 170 GPU validation](leapquant-pr170-gpu-validation-2026-11-15.md).

## Appendix: every multiple of 16 (Flash-Next TP4, KV 4,026,531,840 B)

Tokens, log metric. "-" is infeasible (the block cannot hold a state page, or one
262,144-token request does not fit) or not served by that mode. The what-if
columns lift the platform floor. State ids are per request (GDN + PLE + ring).
Beyond 1,232 only packing fits (1 group, 11 ids) and it declines from 269.2k at
1,248 to 266.5k at 1,536.

| block | A (MB) | blocks N | flag off | LeapQuant #168 | LeapQuant + packing #170 | packing, rule lifted (what-if) | GDN groups (what-if) | state ids/request (what-if) | FP32 + packing (what-if) | prefill tokens/step at 2,048 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.470 | 8569 | - | - | - | 268,120 | 36 x 1 | 186 | - | 2048 |
| 48 | 0.679 | 5930 | - | - | - | 275,232 | 36 x 1 | 186 | - | 2016 |
| 64 | 0.905 | 4448 | - | - | - | 272,306 | 36 x 1 | 186 | 274,679 | 2048 |
| 80 | 1.132 | 3558 | - | - | - | 276,521 | 18 x 2 | 96 | 272,244 | 2000 |
| 96 | 1.358 | 2965 | - | - | - | 274,940 | 18 x 2 | 96 | 269,880 | 2016 |
| 112 | 1.584 | 2541 | - | - | - | **276,737** | 12 x 3 | 66 | 267,513 | 2016 |
| 128 | 1.810 | 2224 | - | - | - | 275,784 | 12 x 3 | 66 | 274,356 | 2048 |
| 144 | 2.037 | 1976 | - | - | - | 276,707 | 9 x 4 | 51 | 272,917 | 2016 |
| 160 | 2.263 | 1779 | - | - | - | 275,949 | 9 x 4 | 51 | 271,768 | 1920 |
| 176 | 2.489 | 1617 | - | - | - | 275,968 | 4 x 5 + 4 x 4 | 46 | **274,716** | 1936 |
| 192 | 2.716 | 1482 | - | - | - | 275,139 | 4 x 5 + 4 x 4 | 46 | 273,782 | 1920 |
| 208 | 2.942 | 1368 | - | - | - | 276,494 | 6 x 6 | 36 | 272,917 | 1872 |
| 224 | 3.168 | 1270 | - | - | - | 275,826 | 6 x 6 | 36 | 271,995 | 2016 |
| 240 | 3.395 | 1186 | - | - | - | 275,378 | 6 x 6 | 36 | 274,164 | 1920 |
| 256 | 3.621 | 1112 | - | - | - | 275,003 | 6 x 6 | 36 | 273,712 | 2048 |
| 272 | 3.847 | 1046 | - | - | - | 275,580 | 1 x 8 + 4 x 7 | 31 | 272,838 | 1904 |
| 288 | 4.073 | 988 | - | - | - | 274,945 | 1 x 8 + 4 x 7 | 31 | 273,204 | 2016 |
| 304 | 4.300 | 936 | - | - | - | 276,003 | 4 x 9 | 26 | 272,629 | 1824 |
| 320 | 4.526 | 889 | - | - | - | 275,468 | 4 x 9 | 26 | 271,932 | 1920 |
| 336 | 4.752 | 847 | - | - | - | 275,137 | 4 x 9 | 26 | 271,437 | 2016 |
| 352 | 4.979 | 808 | - | - | - | 274,724 | 4 x 9 | 26 | 273,659 | 1760 |
| 368 | 5.205 | 773 | - | - | - | 274,204 | 4 x 9 | 26 | 273,096 | 1840 |
| 384 | 5.431 | 741 | - | - | - | 273,975 | 4 x 9 | 26 | 272,821 | 1920 |
| 400 | 5.658 | 711 | - | - | - | 275,309 | 3 x 12 | 21 | 272,093 | 2000 |
| 416 | 5.884 | 684 | - | - | - | 275,009 | 3 x 12 | 21 | 271,676 | 1664 |
| 432 | 6.110 | 658 | - | - | - | 274,666 | 3 x 12 | 21 | 271,211 | 1728 |
| 448 | 6.337 | 635 | - | - | - | 274,236 | 3 x 12 | 21 | 270,669 | 1792 |
| 464 | 6.563 | 613 | - | **274,222** | **274,222** | 274,222 | 3 x 12 | 21 | 272,363 | 1856 |
| 480 | 6.789 | 593 | - | 273,682 | 273,682 | 273,682 | 3 x 12 | 21 | 271,768 | 1920 |
| 496 | 7.015 | 573 | - | 273,106 | 273,106 | 273,106 | 3 x 12 | 21 | 271,134 | 1984 |
| 512 | 7.242 | 556 | - | 273,456 | 273,456 | 273,456 | 3 x 12 | 21 | 271,419 | 2048 |
| 528 | 7.468 | 539 | - | 272,771 | 272,771 | 272,771 | 3 x 12 | 21 | 272,771 | 1584 |
| 544 | 7.694 | 523 | - | 272,567 | 272,567 | 272,567 | 3 x 12 | 21 | 272,567 | 1632 |
| 560 | 7.921 | 508 | - | 271,773 | 271,773 | 271,773 | 3 x 12 | 21 | 271,773 | 1680 |
| 576 | 8.147 | 494 | - | 271,486 | 271,486 | 271,486 | 3 x 12 | 21 | 271,486 | 1728 |
| 592 | 8.373 | 480 | - | 271,183 | 271,183 | 271,183 | 3 x 12 | 21 | 271,183 | 1776 |
| 608 | 8.600 | 468 | - | 270,824 | 273,846 | 273,846 | 2 x 18 | 16 | 270,824 | 1824 |
| 624 | 8.826 | 456 | - | 270,447 | 273,541 | 273,541 | 2 x 18 | 16 | 270,447 | 1872 |
| 640 | 9.052 | 444 | - | 270,050 | 273,220 | 273,220 | 2 x 18 | 16 | 270,050 | 1920 |
| 656 | 9.278 | 433 | - | 269,616 | 272,856 | 272,856 | 2 x 18 | 16 | 269,616 | 1968 |
| 672 | 9.505 | 423 | - | 269,142 | 272,449 | 272,449 | 2 x 18 | 16 | 269,142 | 2016 |
| 688 | 9.731 | 413 | - | 268,648 | 272,023 | 272,023 | 2 x 18 | 16 | 268,648 | 1376 |
| 704 | 9.957 | 404 | - | 268,797 | 272,252 | 272,252 | 2 x 18 | 16 | 271,554 | 1408 |
| 720 | 10.184 | 395 | - | 268,256 | 271,776 | 271,776 | 2 x 18 | 16 | 271,065 | 1440 |
| 736 | 10.410 | 386 | - | 267,692 | 271,280 | 271,280 | 2 x 18 | 16 | 270,555 | 1472 |
| 752 | 10.636 | 378 | - | 267,811 | 271,480 | 271,480 | 2 x 18 | 16 | 270,738 | 1504 |
| 768 | 10.863 | 370 | - | 267,199 | 270,930 | 270,930 | 2 x 18 | 16 | 270,176 | 1536 |
| 784 | 11.089 | 363 | - | 267,298 | 271,106 | 271,106 | 2 x 18 | 16 | 270,336 | 1568 |
| 800 | 11.315 | 355 | **269,742** | 266,650 | 270,526 | 270,526 | 2 x 18 | 16 | 269,742 | 1600 |
| 816 | 11.542 | 348 | 269,103 | 265,965 | 269,899 | 269,899 | 2 x 18 | 16 | 269,103 | 1632 |
| 832 | 11.768 | 342 | 269,228 | 266,033 | 270,039 | 270,039 | 2 x 18 | 16 | 269,228 | 1664 |
| 848 | 11.994 | 335 | 268,557 | 265,311 | 269,381 | 269,381 | 2 x 18 | 16 | 268,557 | 1696 |
| 864 | 12.220 | 329 | 268,677 | 265,370 | 269,516 | 269,516 | 2 x 18 | 16 | 268,677 | 1728 |
| 880 | 12.447 | 323 | 268,801 | 265,431 | 269,657 | 269,657 | 2 x 18 | 16 | 268,801 | 1760 |
| 896 | 12.673 | 317 | 268,063 | 264,648 | 268,930 | 268,930 | 2 x 18 | 16 | 268,063 | 1792 |
| 912 | 12.899 | 312 | 268,160 | 264,689 | 269,042 | 269,042 | 2 x 18 | 16 | 268,160 | 1824 |
| 928 | 13.126 | 306 | 267,386 | 263,868 | 268,281 | 268,281 | 2 x 18 | 16 | 267,386 | 1856 |
| 944 | 13.352 | 301 | 267,475 | 263,897 | 268,385 | 268,385 | 2 x 18 | 16 | 267,475 | 1888 |
| 960 | 13.578 | 296 | 266,648 | 263,032 | 267,567 | 267,567 | 2 x 18 | 16 | 266,648 | 1920 |
| 976 | 13.805 | 291 | 266,726 | 263,047 | 267,662 | 267,662 | 2 x 18 | 16 | 266,726 | 1952 |
| 992 | 14.031 | 286 | 265,862 | - | 266,808 | 266,808 | 2 x 18 | 16 | 265,862 | 1984 |
| 1008 | 14.257 | 282 | 265,915 | - | 266,875 | 266,875 | 2 x 18 | 16 | 265,915 | 2016 |
| 1024 | 14.483 | 278 | 266,945 | 263,090 | 267,926 | 267,926 | 2 x 18 | 16 | 266,945 | 2048 |
| 1040 | 14.710 | 273 | 265,056 | - | 266,042 | 266,042 | 2 x 18 | 16 | 269,042 | 1040 |
| 1056 | 14.936 | 269 | 265,100 | - | 266,100 | 266,100 | 2 x 18 | 16 | 269,147 | 1056 |
| 1072 | 15.162 | 265 | 265,145 | - | 266,161 | 266,161 | 2 x 18 | 16 | 269,256 | 1072 |
| 1088 | 15.389 | 261 | 265,192 | - | 266,224 | 266,224 | 2 x 18 | 16 | 269,368 | 1088 |
| 1104 | 15.615 | 257 | 264,200 | - | 265,240 | 265,240 | 2 x 18 | 16 | 268,410 | 1104 |
| 1120 | 15.841 | 254 | 264,224 | - | 265,277 | 265,277 | 2 x 18 | 16 | 268,486 | 1120 |
| 1136 | 16.068 | 250 | 264,258 | - | 265,327 | 265,327 | 2 x 18 | 16 | 268,590 | 1136 |
| 1152 | 16.294 | 247 | 264,283 | - | 265,367 | 265,367 | 2 x 18 | 16 | 268,670 | 1152 |
| 1168 | 16.520 | 243 | 263,227 | - | 264,319 | 264,319 | 2 x 18 | 16 | 267,651 | 1168 |
| 1184 | 16.746 | 240 | 263,240 | - | 264,346 | 264,346 | 2 x 18 | 16 | 267,721 | 1184 |
| 1200 | 16.973 | 237 | 263,254 | - | 270,122 | 270,122 | 1 x 36 | 11 | 267,793 | 1200 |
| 1216 | 17.199 | 234 | 263,269 | - | 270,227 | 270,227 | 1 x 36 | 11 | 267,867 | 1216 |
| 1232 | 17.425 | 231 | 263,283 | - | 270,336 | 270,336 | 1 x 36 | 11 | 267,943 | 1232 |
