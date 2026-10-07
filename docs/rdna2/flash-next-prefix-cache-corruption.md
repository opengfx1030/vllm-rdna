# Flash-Next prefix-caching correctness bug (gfx1030)

Date: 2026-09-15. Model: Qwen3.8-Flash-Next-AWQ-W4A16, TP=4, V1 +
FULL_AND_PIECEWISE cudagraphs. 16k-in / 1k-out load at concurrency 16.

## Symptom

After a sustained 16k x 16 @ 1k-out load, short probes degrade to garbage:
the first generated token is correct, then the model collapses to `!`.

```
The capital of France is  -> ' Paris. The capital of Germany is Berlin...'   ok
1+1=                      -> '2, 2+2=4, 3+3=6, 4+4'                          ok
5+5=                      -> '1!!!!!!!!!!!!!!!!!!!'                         FAIL
The capital of Germany is -> ' Berlin!!!!!!!!!!!!!!!!!!!'                    FAIL
2+2=                      -> '4!!!!!!!!!!!!!!!!!!!'                         FAIL
The capital of Italy is   -> ' Rome!!!!!!!!!!!!!!!!!!!'                      FAIL   (intermittent)
```

Reproduce: fresh server = 0/18 probe failures; after the load = **11/18**.

## Isolating the trigger

| config | post-load probe failures |
|---|---:|
| prefix caching ON (production) | **11 / 18** |
| prefix caching OFF | **0 / 18** |

The GDN decode path is the same in both (HIP `gdn_decode_rdna2`, fp16 state),
so the GDN kernel is **not** the cause.

## Root cause

Prefix caching auto-enables the mamba cache mode `'align'` for
`Qwen4ExpForConditionalGeneration`:

```
[config.py:615] Mamba cache mode is set to 'align' for
Qwen4ExpForConditionalGeneration by default when prefix caching is enabled
```

The align mode copies mamba/GDN recurrent state across block boundaries
(`precopy_mamba_align_fused_kernel` / `postprocess_mamba_fused_kernel` in
`vllm/v1/worker/mamba_utils.py`). The copy decision logic is:

```python
aligned_new_computed = (new_num_computed // block_size) * block_size
needs_copy = aligned_new_computed >= num_tokens_running_state
accept_token_bias = aligned_new_computed - num_tokens_running_state
dest_block_idx = aligned_new_computed // block_size - 1
```

A wrong copy (wrong src/dest block, wrong token bias, or a skipped/duplicated
copy) leaves the request's recurrent state stale, which corrupts every decode
step after the first token — exactly the observed signature.

## Why it is only visible now

Before the singular/plural `copy_funcs` fix (commit `4224ce202`), the align
path raised `TypeError: tuple indices must be integers or slices, not
MambaAttentionBackendEnum` at `MambaCopyBuffers.create`, so the worker died
before it could produce wrong output. Fixing that made the align path run —
and exposed its latent state-copy bug.

## Immediate mitigation

Run the Flash-Next with `--no-enable-prefix-caching`. Verified: 0/18 probe
failures after the same load, and 16/16 requests successful
(68.96 out tok/s at 16k/1k c=16, TPOT 89.26 ms).

This costs the prefix-cache hit rate but restores correctness.

## Next step

Debug the align-mode state copy: instrument
`_copy_mamba_state_block` / `precopy_mamba_align_fused_kernel` to log
(src_block_idx, dest_block_idx, accept_token_bias, needs_copy) per request and
compare against the Python reference `postprocess_mamba` semantics on a
prefix-cache-hit trace.
