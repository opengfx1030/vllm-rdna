# FA-RDNA2 GQA decode kernel: validation on `rdna_extras` 1136bb952b (2026-11-11)

Decision: `VLLM_FA_RDNA2_GQA_DECODE` stays opt-in in the kernel (off when the
variable is unset) and stays on in the launcher for `ATTN=fa`
(`tools/rdna/rdna_launcher_common.sh`). PR 53, which made the launcher export 0,
is closed. Its premise was the comment in `csrc/rocm/fa_rdna2.cu` that the
grouped kernel produces garbage from the first decode token when a batch has two
or more decode rows. That report does not reproduce on the current kernel. It is
within tolerance of an fp32 reference in 66 of 66 kernel cases and bitwise
identical to the per-head kernel in all 66, and `27b-awq` at TP4 passes the
greedy, PPL, concurrency and mixed-batch probes with the switch on and off. The
switch makes the attention forward 1.4x to 2.6x faster at four or more rows and
takes the 16k c=8 ITL median from 63.1 to 52.9 ms. The kernel comment is
replaced in the same commit as this note.

## What the switch does

- `fa_rdna2_decode_paged` takes the grouped kernel
  (`fa_decode_paged_splitk_gqa_kernel_256`) when the switch is on (first
  character `1`, read once per process), head_dim is 256 in fp16,
  `rows × KV heads ≥ 4`, the K cache is the 5D x=8 layout, the group size
  `H_q / H_kv` is 2 to 8, `block_size % 8 == 0` and the K / V strides are the
  production ones. One CTA per (row, KV head, split) serves the whole group.
  Everything else runs the per-head kernel `fa_decode_paged_splitk_kernel_256`.
  Both share the combine kernel.
- Rows are query tokens: pure decode steps, MTP verify rows and the decode rows
  of a mixed batch (`RdnaAttentionImpl.forward` sends those through
  `fa_rdna2_decode_paged` with `cu_query_lens` while
  `VLLM_FA_RDNA2_SPLIT_DECODE=1` and the batch is at most 256 query × head
  rows). A TP4 rank of Qwen3.8-27B has 6 query heads and 1 KV head, so the
  kernel needs four rows. At TP2 (12 / 2 heads) two rows are enough.

## Which launches reach it

| Launch | Switch in the workers | Reaches the kernel |
|---|---|---|
| `27b-awq`, `27b-exl3` (TP4, `ATTN=fa`) | 1 (launcher) | With four or more rows. `27b-awq` sets `VLLM_RDNA_DECODE_STALL_MS=0`, so mixed steps are common. |
| Every `flashnext*` recipe (`ATTN=fa`) | 1 (launcher) | Never. The full-attention layers (and the MTP layer) are `Qwen4ExpQSAAttention`, which uses its own QSA backend. The serve logs of `flashnext-mtp0` and `flashnext-mtp2` list `RDNA_ATTN` as the argument and have no `Using RDNA_ATTN backend` line. |
| `full`, `dsv4-flash` (`ATTN=none`) | unset (kernel default 0) | No. |
| Other `serve_rdna.sh` launches (`ATTN` defaults to `fa`) | 1 (launcher) | If the model has head_dim 256 and a group size of 2 to 8. |
| `ATTN=triton` | not set | No. |

## History

| Date | Event |
|---|---|
| 2026-09-04 | The grouped kernel lands. Its commit message gives bit-identical output and 1.5 to 2.0x faster decode attention. |
| 2026-09-09 | The branch is compiled out during the true-FULL-graph hunt (27B-AWQ, TP2). The comment added then says `n_dec ≥ 2` went through this kernel and produced garbage from the first decode token while `n_dec = 1` (per-head kernel) was coherent. |
| 2026-09-10 | The hunt ends ([journal](../profiling/2026-09-10-truefull-journal.md)). Causes: the TRUE FULL dispatch design poisoned FULL graphs (16k c=1), and GDN conv/ssm decode state in reallocatable pool blocks let a decoding request read another request's state page at c=8. Both fixed (dispatch reverted, permanent per-slot state arenas). The journal does not name this kernel. The switch stays off. |
| 2026-09-12 | The same symptom at TP4 is traced to the RDNA2 W4A16 kernel under graph replay with FA-RDNA2 disabled, and that kernel is limited to TP ≤ 2 for the time being (`27b-awq` serves at TP4 now). |
| 2026-09-28 | PR 28 makes the grouped kernel opt-in again. Its GPU validation: coherent with 8 decode sequences, and the MTP=2 1k c=8 aggregate is 159.5 with the switch against 115.6 without ([comment](https://github.com/opengfx1030/vllm-rdna/pull/28#issuecomment-5862244508)). The launcher then exports 1 for `ATTN=fa`. PR 29 adds the `cu_query_lens` path, so verify and mixed rows can reach the kernel; its grouped-kernel coverage was a CPU emulation. |
| 2026-10-11 | PR 53 is opened from the 2026-09-09 comment. |
| 2026-11-01 | `27b-awq` at TP4 serves with the launcher default (greedy, PPL, concurrency probes pass, 16k c=8 ITL median 53.4 ms). |

That the `n_dec ≥ 2` garbage of 2026-09-09 was the state-page race is
[INFERENCE]: the race needs two decoding requests, and `n_dec = 1` is coherent.
The PR 28 validation of 2026-09-28 (8 decode sequences, coherent) is a run with
the switch on after the arena fix.

## Method

- Tree: `git archive` of `origin/rdna_extras` at 1136bb952b with the `.so` set of
  the 2026-11-06 14:42 build (`git diff 46868fb5f7 1136bb952b -- csrc` is empty
  and the last csrc commit, 2d0f6ccdf9, is from 14:31). `fa_rdna2.cu` and
  `rdna_attn.py` have the md5 of the tip. Venv `venv-7.14.0_0.32.0`, torch
  2.12.0+rocm7.14.0.
- Kernel level: `tools/rdna/port_v031/gqa_decode_check.py` drives
  `RdnaAttentionImpl.forward` (the production dispatch) on synthetic paged KV
  in the production layout (random fp16 K and V, random block tables, 1024,
  784 and 16 slot pages, padded block-table strides, padded graph rows) and
  compares with an fp32 reference. 66 cases: pure decode at TP4 / TP2 / TP1
  head counts, group sizes 2 to 8 and 12, MTP verify rows (q_len 2 to 4),
  mixed batches (1 to 8 decode rows plus a 512 to 2048 token prefill chunk) and
  a short extend. One process per setting on one V620 (bus d9, lease b4), 20
  repeats per case, then a 1000-repeat stress of six cases that reach the grouped
  kernel. An earlier run with the box-local original of the harness (bus d1,
  lease a2) gave the same 66 of 66 results and timings within 3 %.
- Which kernel ran: `rocprofv3 --kernel-trace` of five cases per setting, run
  with that box-local original (bus d1).
- Kernel speed: one `forward` captured in a graph and replayed, 5 windows of 100
  replays, median window, same inputs for both settings.
- Serving: `27b-awq` through `serve_validate.sh` (TP4, `FULL_AND_PIECEWISE`, W4A8,
  `BLOCK_SIZE=1024`, `SEQS=8`, `VLLM_RDNA_DECODE_STALL_MS=0`) on HIP 2 to 5
  (buses d9 dc df e2), lease b4. Boot A is the launcher default. Boot B passes
  `VLLM_FA_RDNA2_GQA_DECODE=0`, which is what PR 53 would make the default.
  Per boot: greedy probe, PPL probe, prefix probe, warm-up, concurrency probe
  (8 prompts together, churn, 1.5k longs), three cells inside a KFD
  `evicted_ms` window and a quiet window (`--temperature 0`, 1k/256 c=1 with 2
  requests, 1k/256 c=8 with 16, 16k/256 c=8 with 8), then
  `tools/rdna/port_v031/mixed_decode_probe.py` (8 prompts of about 10k tokens
  sent at once, 64 tokens each, every output compared with its own solo run on
  the same server) and the greedy probe again. c=1 has one row, below the gate,
  so it runs the per-head kernel in both boots and is the control for
  boot-to-boot drift.

## Results

### Serving: `27b-awq`, TP4, switch on (A) against off (B)

Both boots READY (A 120 s, B 100 s), no fault markers, no new amdgpu dmesg
faults, KFD `evicted_ms` +0 on all four workers for the whole run, GPU
temperature at most 75 C. `serve_rdna.sh PRINT=1` shows
`VLLM_FA_RDNA2_GQA_DECODE='1'` for A and `'0'` for B. (`/proc/<pid>/environ`
of a worker cannot show it: setproctitle overwrites it, and the probe reports
"unset" for both.) At most 8 requests ran at once in both boots.

| Check | A: switch 1 (launcher default) | B: switch 0 |
|---|---|---|
| Greedy probe, 3 prompts | PASS 3/3 | PASS 3/3 |
| PPL probe | 4.90, PASS | 4.90, PASS |
| Concurrency probe, 48 checks | PASS: 0 failing, 5 near-tie, 43 identical | PASS: 0 failing, 4 near-tie, 44 identical |
| Mixed probe, 8 prompts of about 10k tokens at once, each against its solo run | PASS: 0 failing of 8, 4 near-ties (margin at most 0.016) | PASS: 0 failing of 8, 3 near-ties (margin at most 0.031) |
| Greedy probe after the load | PASS 3/3 | PASS 3/3 |
| Prefix probe | FAIL: hits +0 | FAIL: hits +0 |

The near-ties are the same tokens in both boots (tokens 7, 29 and 34 of
mixed prompts 2, 4 and 6), so they come from batch composition in the
non-attention kernels, not from the switch. The prefix probe fails in both
boots: its default prompt is 1927 tokens, below the 2048-token scheduler chunk
(`hit granularity 2048 tokens` in the probe output), so no aligned checkpoint
exists [INFERENCE]. The 2026-11-01 run of the same recipe passed it with a longer
prompt (cold TTFT 10.65 s, `serve-v033-27b-awq-fixed`). It does not depend on
the switch.

Cells, `--temperature 0`, one boot per setting:

| Cell | Metric | A: switch 1 | B: switch 0 | B / A |
|---|---|---|---|---|
| 1k/256 c=1 (1 row, below the gate: control) | ITL median | 21.93 ms | 21.93 ms | 1.000 |
| 1k/256 c=8 | ITL median | 36.04 ms | 37.60 ms | 1.043 |
| 1k/256 c=8 | ITL mean | 44.67 ms | 46.23 ms | 1.035 |
| 1k/256 c=8 | output tok/s | 135.7 | 132.1 | 0.973 |
| 16k/256 c=8 | ITL median | 52.90 ms | 63.09 ms | 1.193 |
| 16k/256 c=8 | ITL mean (includes the stalls of the 8 prefills) | 257.2 ms | 267.5 ms | 1.040 |
| 16k/256 c=8 | output tok/s | 14.87 | 14.54 | 0.978 |

The 16k c=8 median is the step time of eight decode rows on 16k of context;
the switch takes 10.2 ms off it, against 7.7 ms predicted from the per-layer
kernel timings below (16 full-attention layers); the remaining 2.5 ms is not
explained. At 1k c=8 the gap (1.6 ms) is about the size of the boot-to-boot
spread of earlier A/B notes (`hwq-abba-2026-11-06.md`), so only the 16k cell is
a speed claim.

### Kernel level: switch off against on

`gqa_decode_check.py`, 66 cases, 20 repeats each, one process per setting:

| | Switch 0 | Switch 1 |
|---|---|---|
| Cases with `max_abs` at most 5e-3 against the fp32 reference | 66 of 66 | 66 of 66 |
| Largest `max_abs` / `max_rel` over all cases | 9.49e-04 / 1.10e-03 | 9.49e-04 / 1.10e-03 |
| NaN, bad rows, non-zero padded rows | 0 | 0 |
| Repeats bitwise equal | 66 of 66 | 66 of 66 |
| Output of switch 1 bitwise equal to switch 0 | 66 of 66 | |

58 of the 66 cases have rows × KV heads ≥ 4, and 2 of those have group size 12,
which the grouped kernel does not take, so at most 56 reach it. A 1000-repeat
stress with the switch on of six cases that reach it (TP4 decode with 4 rows and
with 8 rows at 16k, TP4 MTP verify with 4 sequences × 3 tokens, TP4 mixed
batches with 6 and 8 decode rows plus a 2048-token chunk, TP2 mixed with 4
decode rows plus a chunk) gave the same output bits in all 1000 repeats, equal
to the 20-repeat run and so to the switch-0 output (`max_abs` at most 6.1e-05).

`rocprofv3 --kernel-trace`, switch 0 against 1 (earlier box-local copy of the
harness, same cases and the same `.so`):

| Case | Switch 0 | Switch 1 |
|---|---|---|
| TP4 decode, 8 rows | `fa_decode_paged_splitk_kernel_256` + combine | `fa_decode_paged_splitk_gqa_kernel_256` + combine |
| TP4 mixed: 4 decode rows + 2048-token prefill chunk | per-head + combine (+ the varlen prefill kernel) | grouped + combine (+ the varlen prefill kernel) |
| TP4 MTP verify: q_len 3 × 4 sequences | per-head + combine | grouped + combine |
| TP2 decode, 2 rows | per-head + combine | grouped + combine |
| TP4 decode, 3 rows (below the gate) | per-head + combine | per-head + combine |

### Kernel speed

One `RdnaAttentionImpl.forward` per layer, CUDA-graph replay, median of 5
windows of 100 replays, same inputs for both settings. 12 of the 41 cases are below the gate
and run the same kernel in both settings (0.99x to 1.01x). The other 29 are
1.40x to 2.55x faster with the switch (median 1.75x). Selected cases, in
microseconds per layer:

| Case | Switch 0 | Switch 1 | 0 / 1 |
|---|---|---|---|
| TP4 decode, 3 rows, 16k (below the gate) | 536.2 us | 538.8 us | 1.00x |
| TP4 decode, 4 rows, 1k | 61.9 us | 39.8 us | 1.55x |
| TP4 decode, 8 rows, 1k | 111.5 us | 73.6 us | 1.51x |
| TP4 decode, 16 rows, 1k | 270.4 us | 106.0 us | 2.55x |
| TP4 decode, 4 rows, 16k | 694.4 us | 394.5 us | 1.76x |
| TP4 decode, 8 rows, 16k | 1360.5 us | 881.5 us | 1.54x |
| TP4 decode, 16 rows, 16k | 2997.1 us | 1812.1 us | 1.65x |
| TP4 decode, 8 rows, 32k | 3013.5 us | 2157.3 us | 1.40x |
| TP4 decode, 16 rows, 32k | 5932.6 us | 3278.2 us | 1.81x |
| TP2 decode, 2 rows, 16k | 683.1 us | 396.2 us | 1.72x |
| TP2 decode, 8 rows, 16k | 2934.7 us | 1816.8 us | 1.62x |
| TP4 MTP verify, 4 sequences × 3 tokens, 16k | 1774.2 us | 860.8 us | 2.06x |
| TP4 MTP verify, 8 sequences × 3 tokens, 16k | 3750.7 us | 1764.2 us | 2.13x |

At 8 rows and 16k of context the switch saves 479 us per layer. Qwen3.8-27B has
16 full-attention layers, which is 7.7 ms per decode step (0.6 ms at 1k).

## Limits

- The kernel harness uses random fp16 K and V and random queries on one GPU. It
  has no sliding-window or fp8 / int8 KV cases. `Qwen3.8-27B` has neither
  (`sliding_window` is null, 16 of 64 layers are full attention, 24 query heads,
  4 KV heads, head_dim 256). A race that needs a data pattern or an occupancy
  from a full serving step can survive 66 cases × 20 repeats and the 1000-repeat
  stress. The serving probes (real activations, FULL graphs, mixed batches) are
  the second line, and they cover one model on one box.
- One boot per setting. The c=1 cell, which never reaches the kernel, is the
  control for boot-to-boot drift: identical in both boots (21.93 ms). The 1k
  c=8 gap is inside the spread seen in earlier A/B notes.
- The 2026-09-09 case (TP2, true FULL graphs, mixed batch) was not run as a
  server. The TP2 boots refused to start (KV for `max_model_len` 262144 needs
  8.19 GiB per rank, `KV=8e9` gives 7.42 GiB), and the run was stopped there.
  TP2 is covered at kernel level only (`27b_tp2_*` cases, including mixed
  batches with 2 and 4 decode rows) and by the TP4 mixed probe.
- `27b-exl3` was not booted. It shares the attention path with `27b-awq`
  [INFERENCE].
- Flash-Next recipes do not reach this kernel, so nothing here says anything about
  them.
- The kernel default (switch unset = off) is unchanged. `full` and `dsv4-flash`
  (`ATTN=none`) and direct launches of the kernel run the per-head kernel.
  Changing the default is a separate decision.

## Reproduce

Kernel level, one GPU, once per setting (`<n>` is a HIP device index):

```bash
HIP_VISIBLE_DEVICES=<n> VLLM_FA_RDNA2_GQA_DECODE=0 \
  python tools/rdna/port_v031/gqa_decode_check.py --tag off --repeat 20 --out off.json
HIP_VISIBLE_DEVICES=<n> VLLM_FA_RDNA2_GQA_DECODE=1 \
  python tools/rdna/port_v031/gqa_decode_check.py --tag on --repeat 20 --out on.json
python tools/rdna/port_v031/gqa_decode_check.py --compare off.json on.json
# 1000-repeat stress of one case that reaches the grouped kernel
HIP_VISIBLE_DEVICES=<n> VLLM_FA_RDNA2_GQA_DECODE=1 \
  python tools/rdna/port_v031/gqa_decode_check.py --only 27b_tp4_decode_n8_16k_equal \
  --repeat 1000 --out stress.json
# per-layer forward time, graph replay
HIP_VISIBLE_DEVICES=<n> VLLM_FA_RDNA2_GQA_DECODE=0 \
  python tools/rdna/port_v031/gqa_decode_check.py --bench --out bench_off.json
HIP_VISIBLE_DEVICES=<n> VLLM_FA_RDNA2_GQA_DECODE=1 \
  python tools/rdna/port_v031/gqa_decode_check.py --bench --out bench_on.json
python tools/rdna/port_v031/gqa_decode_check.py --compare-bench bench_off.json bench_on.json
```

Serving, one boot per setting. `serve_validate.sh` reads `RECIPE`, `MODEL`,
`GPUS`, `PORT`, `CELLS`, `RESET_PREFIX_CACHE`, `BENCH_EXTRA` and `LOAD_HOOK` from
the environment and passes its `KEY=VALUE` arguments on to `serve_rdna.sh`. The
hook runs after the cells and gets `URL`, `PORT`, `SERVED`, `MODEL` and `OUT`:

```bash
# hook.sh
python tools/rdna/port_v031/mixed_decode_probe.py --url "$URL" --model "$SERVED" \
  --reset-url "http://127.0.0.1:$PORT/reset_prefix_cache" \
  -n 8 --sentences 500 --max-tokens 64 --verbose

RECIPE=27b-awq MODEL=<27B AWQ snapshot dir> GPUS=2,3,4,5 PORT=18331 \
  CELLS="1024:256:1:2 1024:256:8:16 16384:256:8:8" RESET_PREFIX_CACHE=1 \
  BENCH_EXTRA="--temperature 0" LOAD_HOOK=$PWD/hook.sh \
  bash tools/rdna/port_v031/serve_validate.sh VLLM_SERVER_DEV_MODE=1 \
  [VLLM_FA_RDNA2_GQA_DECODE=0]
```

The TP4 boots in this note ran the box-local original of `mixed_decode_probe.py`.
The copy in the tree has the same logic; it drops the unused `--stagger` option
and the `sys.path` line (the script now sits next to `concurrency_probe.py`).

Which kernel ran: `rocprofv3 --kernel-trace --output-format csv -d <dir> -o trace
-- python tools/rdna/port_v031/gqa_decode_check.py --only <case> --repeat 1` and
look in the CSV for `fa_decode_paged_splitk_gqa_kernel_256` (grouped) against
`fa_decode_paged_splitk_kernel_256` (per-head).
