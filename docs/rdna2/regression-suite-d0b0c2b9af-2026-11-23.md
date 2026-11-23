# Regression suite on `rdna_extras` d0b0c2b9af (2026-11-23)

Dated record. After PRs 197 to 200 merged, six boots on the four-card slot (HIP 2 to 5, TP=4 unless noted) checked that nothing regressed. Only PR 200 changes runtime: spec-decode NaN counts are attributed to requests through `cu_num_logits`, not through request state indices. PRs 197 to 199 are docs, tools and a probe.

## The bug PR 200 fixed

`sum_nans_per_request` did `index_add_(0, expanded_idx_mapping, num_nans)` into a `num_reqs`-long tensor. `expanded_idx_mapping` holds request state indices (up to `max_num_seqs`), so a request in a slot at or above `num_reqs` wrote out of bounds: `HSA_STATUS_ERROR_EXCEPTION` code 0x1016 in `indexFuncSmallIndex<...ReduceAdd>`. The path runs when the sampler counts NaNs (`VLLM_RAISE_ON_LOGIT_NANS=1`, which `--monokernel` forces) with MTP. Every `--monokernel` MTP boot on `rdna_extras` crashed before the fix (unmodified `00f106c5b0`, 221 s after launch, four GPUs); a tree with the fix booted and served 96 of 96 bench requests with no fault and no KFD eviction.

## Suite

Tree `d0b0c2b9af`, PLE int4 for Flash-Next, baselines from the 2026-11-23 gate on `00f106c5b0`.

| Boot | Verdict | Numbers (baseline) |
| --- | --- | --- |
| `flashnext-mtp2` | PASS | greedy 3/3, PPL 4.87 (4.82), prefix 53.1 % at 1927 tokens, 1k/512 ITL 22.4 ms at c=1 (22.2), 45.8 ms at c=8 (45.0) |
| `flashnext-mtp2 --monokernel` | PASS | greedy 3/3, PPL 4.86, ITL 23.5 ms at c=1, 46.3 ms at c=8, MTP acceptance 2.56; crashed at boot before PR 200 |
| `27b-awq` | PASS | greedy 3/3, PPL 4.90, prefix 97.3 % at 16,843 tokens (22.6x), ITL 22.0 ms at c=1 (22.0), 36.5 ms at c=8 (36.2) |
| `27b-awq`, MTP=2, `VLLM_RAISE_ON_LOGIT_NANS=1` | PASS on 3 reruns | greedy 3/3, PPL 4.90, prefix 97.3 %, concurrency PASS, MTP active; the first run was marked INVALID (below) |
| `inkling2-mini-tp1`, one card | PASS | chat 3/3, 17.45 ms/token (17.4) |
| `inkling2-mini-tp1-dspark`, NaN check on | PASS | chat 3/3, 13.97 ms/token, draft 810, accepted 500 |

ITL moves of +0.8 to +1.8 % are inside boot-to-boot spread. No fault line (`rdna_ar` disabled or wedged, NaNs, `HSA_STATUS_ERROR`, Memory Fault, Queue error) in any log; peak junction 80 C. The kernel log is unreadable on the box, so dmesg faults are unknown, not zero.

## The one INVALID boot

The first `27b-awq` MTP=2 boot saw KFD queue evictions of 178 to 256 ms per worker (two bursts: 15:46:38 at the start of the backbone compile, 15:47:11 after the KV allocation) and 3 ms after READY; its probes all passed. Three reruns (two with the NaN check, one without) saw 0 ms. HI2 scoring jobs with a 44 GB peak RSS ran on the host around the first burst. [INFERENCE] a host-side effect of that job, not the merged code; the original window had no process log, so it is not proved.

Method: `serve_validate.sh` harness, window rules of the lease protocol. Pull the numbers from the run directories on the build host, not from this page.
