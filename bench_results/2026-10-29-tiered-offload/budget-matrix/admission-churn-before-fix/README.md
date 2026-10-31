# Pre-fix admission-churn diagnostic

Source `dc96fc711f`, uniform96 with 90% live KV watermark. The first matrix
completed; long c8 measured 21.29 output tok/s, mean TTFT 134.76 s and mean
ITL 475.6 ms. There were 11 physical snapshots, totaling 14,019,985,408 bytes
stored across both ranks. These results are excluded from the corrected primary
comparison. The warmed repeat was deliberately interrupted after discovering
the policy issues; its failure receipt reflects that shutdown.

Two admission mistakes caused unnecessary work: a restored ready request could
trigger a fresh-admission eviction, and a detached snapshot could remain in the
priority queue for GPU-KV holders, reclaiming its own freed pages before fresh
admission. Fix `8709ab5142` excludes retained/blocked waiters from new-demand
triggers and moves detached snapshots behind fresh requests. Actual FCFS ordering,
positive LRU admission and blocked/adapter cases have regression coverage.

`raw-receipts.tar.gz` retains both original receipt directories, before/after
layer counters, benchmark outputs and compressed logs. `manifest.json` records
all member hashes. Remote original receipts also remain under the run root.
