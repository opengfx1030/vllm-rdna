# Inkling-2-Mini PR receipts (branch feat/inkling2-merge, 5a5331fddb, base 703349599a)

Host par1-cs25, V620 gfx1030, venv-7.14.0_0.34.0-rebase-1e1cac5340 (torch 2.12.0+rocm7.14.0, HIP 7.14.60850).
Trees: `git archive` exports of each commit, built with `build_lq.sh`.

## 1. `_get_packed_kv_cache_groups`: differential test against the base

`tests/v1/core/test_inkling2_planner_differential.py`, CPU only. The base function is loaded with
`git show 703349599a:vllm/v1/core/kv_cache_planning.py` (or `KV_PLANNER_BASE_FILE`). Random layer sets
(MLA, full, sliding-window, Mamba, LeapQuant Mamba, circular buffer; both layouts; GDN pack switch off/on).
Whenever the base returns a valid partition, the head's groups must be identical; the head must always
return a valid partition (every layer once, no empty group).

```bash
KV_PLANNER_BASE_FILE=<base tree>/vllm/v1/core/kv_cache_planning.py \
  pytest tests/v1/core/test_inkling2_planner_differential.py -q -s     # 10 passed
```

| input shape (3000 layer sets each) | base outcome | head |
|---|---|---|
| balanced buckets | 1283 groups, 1680 None, 37 `StopIteration` | identical groups on the 1283; None on the 1680; valid groups on the 37 |
| single-size buckets | 1096 groups, 1904 None | identical groups on the 1096; None on the 1904 |
| unbalanced buckets | 856 groups, 1642 None, 332 `StopIteration`, 170 dropped layers | identical on the 856; valid groups where the base failed |

The base fails on balanced buckets only when a state (Mamba / circular-buffer) bucket is asked for more
groups than it has repeats (empty group, `StopIteration`); the head skips the empty group. Unbalanced
buckets lost the longer page-size list's surplus layers in the base (zip); the head keeps every layer.
Pinned cases: sliding-window bucket with heads [4]*6+[8]*2 -> groups of 2,3,3; [8,8,4] -> 1,2.

Pre-existing failures: `pytest tests/v1/core` on exports of the base and the head: 113 failed, 4 errors on
both, identical failed-test lists (head 1250 passed, base 1248: two added tests). They include the 30 in
`test_leapquant_page_packing.py` and the 1 in `test_kv_cache_planning.py`. Not fixed here.

## 2. A/B on a non-Inkling recipe

AWQ `flashnext-mtp2`, TP4, HIP 2,3,4,5, `tools/rdna/pr_receipt.sh MODE=ab`, base 703349599a vs head 5a5331fddb.

| gate | result |
|---|---|
| launch plans | identical |
| token-exact (`VLLM_RDNA2_MOE_DETERMINISTIC=1`, greedy) | PASS: 26 of 26 equal, 0 differing; concurrency probe 0 failing on both trees |
| speed, overlap, stall | not run: the head speed boot was cancelled; no verdict (a short A/B is scripted, see the PR) |

## 3. Suites (head build)

| suite | result |
|---|---|
| CPU: `tests/models/inkling2_loop`, `test_inkling2_loop_{blocks,moe}.py`, `test_inkling2_tiny_harness.py`, `tests/tools/rdna/test_inkling2_recipes.py` | 752 passed, 98 skipped (GPU/kernel gated), 0 failed |
| GPU serving-path, graph, kernel suites, tp1 smoke | not run in this receipt |
