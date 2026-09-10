
---

## Resolution (2026-09-10 evening)

**FIXED: 16k c=8 = 8/8 (x2) + 16k/1k c=8 = 8/8 + seq-after 3/3.**

Root causes, in order:
1. TRUE FULL skip-compiled-eager design poisoned FULL graphs (16k c=1
   Parisduct). Fixed by reverting 3 dispatch files to cafe95ef8 (FPP13).
2. Residual c=8 race: GDN conv/ssm decode state lived in reallocatable
   pool blocks; at high concurrency a decoding request read another
   request's state page (outputs contained other requests' prompt text —
   "s52" tags). The block-table duplicate ids (bt-debug, 3.3k) were
   stale-but-never-read residue; the actual read path was the pool-block
   GDN state pages.
3. Fix: PR #4 (cursoragent) cherry-picked — permanent per-slot conv/ssm
   state arenas (slot 0 = NULL sentinel, gather/scatter by decode BS,
   pre-capture arena alloc). One-line fix on top: register_buffer ->
   direct assignment (layer pre-declares the attr; torch KeyError).

Final config: rdna_extras = cafe95ef8 + c091c420b + 3 PR#4 commits +
74f47b6af. Production: 16k/1k c=8 8/8, TTFT 58-476s, prefill 94.6 tok/s,
decode 3.08 tok/s/req, agg 12.51 tok/s, 12e9 KV, TP=2, PYNCCL, FPP13
(FULL_AND_PIECEWISE piecewise execute), prefix caching ON.
