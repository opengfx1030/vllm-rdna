# TP4 health micro launcher attempts

Attempts 1–4 each exited nonzero before reaching a collective. Their status,
boot, and source before/after records are preserved in the original
`mtp-failstop-collective-v1/results-attempt{1,2,3,4}/` directories. The
successful attempt 5 and its executed runner snapshot are in this bundle.

1. **Attempt 1 — child environment guard.** `torch.distributed.run` populated
   `LOCAL_RANK`, `RANK`, and `WORLD_SIZE`; the launcher treated these expected
   distributed variables as forbidden device selectors and aborted.
2. **Attempt 2 — physical selector mapping.** The launcher removed the
   duplicated `CUDA_VISIBLE_DEVICES=6,7,8,9` alias before vLLM's device mapping
   helper ran. That helper then interpreted logical ordinals 0–3 as the
   physical selection, so the independent mapping check correctly rejected
   the mismatch.
3. **Attempt 3 — queue probe ordering.** KFD queue ownership was queried before
   creating an HSA queue. The queue list was empty, so physical ownership could
   not be established and the run stopped before the collective.
4. **Attempt 4 — identifier domains.** The check compared physical ordinals
   6–9 directly with KFD topology IDs such as 56424. The values describe
   different identifier domains. The corrected check verifies both
   `physical_ordinal == rank + 6` and
   `topology_gpu_ids[physical_ordinal] == expected_kfd_gpu_id`.

The successful attempt retained the exact duplicate selector alias for the
platform mapping helper and independently proved each worker's device using
KFD topology and live queue ownership.
