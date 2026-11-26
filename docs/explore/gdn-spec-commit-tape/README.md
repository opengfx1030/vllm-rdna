# Explore: GDN speculative verify via a commit tape (from goss)

**Status**: explore only. This note is **not dest**. It proposes no default,
no recipe change, and makes **no tok/s claim**. No code lands with it. Every
number below is either copied from a cited line or labeled as arithmetic or
unsourced.

Base: `rdna_extras` @ [`93c59de2b50ccea1b26f299e2ac5713861ffb2e3`](https://github.com/opengfx1030/vllm-rdna/commit/93c59de2b50ccea1b26f299e2ac5713861ffb2e3).
External reference: [AgentAtelier/goss](https://github.com/AgentAtelier/goss) @
[`0793d7cfd642e2fea6bf3324f99de3ac767c3316`](https://github.com/AgentAtelier/goss/tree/0793d7cfd642e2fea6bf3324f99de3ac767c3316)
(MIT).

> goss is a personal project that "relies extensively on AI coding agents"
> and "has not undergone extensive independent review"
> ([README.md L37-39](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/README.md#L37-L39)).
> Its speed numbers are the author's only and are not repeated here.

## What goss does

goss runs batch 1, fp32 GDN state, one request.

- **Verify is read-only.** `gdn_verify_readonly_f32_kernel` reads the committed
  state without writing it and records `k_norm`, `v`, `beta`, `g` per token
  ([fuedem/kernels_seq.hip L216-231](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_seq.hip#L216-L231)).
- **Tape layout**, per layer, `max_n <= 8`: conv inputs
  `[max_n × conv_dim]`, `k_norm [max_n × n_k_heads × head_dim]`,
  `v [max_n × n_v_heads × head_dim]`, `beta`/`g [max_n × n_v_heads]`, fp32,
  256-byte aligned
  ([kernels_seq.hpp L22-37](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_seq.hpp#L22-L37),
  [kernels_seq.hip L1667-1684](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_seq.hip#L1667-L1684)).
- **Commit** replays the `k_accept` accepted steps for **all GDN layers in two
  launches**: one conv-state kernel and one state kernel
  (`gdn_s_commit_all_f32_kernel`, [L339](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_seq.hip#L339);
  launcher [L1756-1777](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/kernels_seq.hip#L1756-L1777)),
  called from `GossEngine::commit_verify`
  ([fuedem/goss_engine.cpp L974-995](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/goss_engine.cpp#L974-L995)).
- goss's own audit: verify 74 VGPR, commit 44 VGPR, 0 LDS, no spills
  ([mooss/hsaco_audit.json L362-377](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/mooss/hsaco_audit.json#L362-L377)).
  Author's file, not re-measured.
- **Parity evidence**: `--test-rollback` runs **one** prompt,
  `verify(8) → commit(3) → 16` greedy steps vs a pure n=1 run
  ([fuedem/goss_run.cpp L266-329](https://github.com/AgentAtelier/goss/blob/0793d7cfd642e2fea6bf3324f99de3ac767c3316/fuedem/goss_run.cpp#L266-L329)).

## What dest does today

Links are at `93c59de2`.

- `gdn_spec_verify_rdna2` (hand port of `fused_sigmoid_gating_delta_rule_update`,
  `IS_SPEC_DECODING + INPLACE_FINAL_STATE`) loads slot
  `ssm_state_indices[n, num_accepted - 1]` and **every token writes its full
  state** to `ssm_state_indices[n, t]`; fp32 accumulation over fp16 inputs,
  same op order as Triton
  ([csrc/rocm/gdn_decode_rdna2.cu L288-306](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/gdn_decode_rdna2.cu#L288-L306);
  the per-token store at [L451-457](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/gdn_decode_rdna2.cu#L451-L457)).
  `GDN_K = 128` ([L22](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/gdn_decode_rdna2.cu#L22)).
- `mono_gdn_rdna2` fuses conv1d + delta rule + gated RMSNorm in one launch
  and has the same spec contract: per-token state slots, conv and SSM state
  from `num_accepted - 1`; padded slots (index <= 0) are untouched
  ([csrc/rocm/mono_gdn_rdna2.cu L1-30](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/csrc/rocm/mono_gdn_rdna2.cu#L1-L30)).
  A LeapQuant state variant follows (L31 ff.).
- `csrc/rocm/v1_spec_decode_rdna2.cu` is the spec-decode bookkeeping (input
  prep, rejection sampling); it does not touch GDN state. It matters here
  only as the place where per-request `num_accepted` is produced.

## Memory and bandwidth (arithmetic only)

Shapes: Qwen3.8-27B as recorded in
[docs/rdna2/leapquant-pr170-gpu-validation-2026-11-15.md L56](https://github.com/opengfx1030/vllm-rdna/blob/93c59de2b50ccea1b26f299e2ac5713861ffb2e3/docs/rdna2/leapquant-pr170-gpu-validation-2026-11-15.md#L56):
48 GDN layers, 16 key heads, 48 value heads, K = V = 128, conv kernel 4.
TP=1, fp16 state. At TP=2 every figure halves and the ratio is the same. An
fp32 state doubles `S`.

| quantity | formula | per layer | × 48 layers |
|---|---|---:|---:|
| SSM state `S` (one slot) | HV · V · K · 2 B | 1,572,864 B (1.50 MiB) | 72.0 MiB |
| tape per token, fp16 | (conv_dim + H·K + HV·V + 2·HV) · 2 B, conv_dim = 2·H·K + HV·V = 10,240 | 18,528 values = 37,056 B (36.2 KiB) | 1.70 MiB |
| tape per token, fp32 (goss) | same × 4 B | 74,112 B (72.4 KiB) | 3.39 MiB |

So one speculative token costs **72 MiB of state writes** per request today
vs **1.70 MiB** of fp16 tape (≈ 42×).

Per verify step with `T` tokens and `k` accepted, per request, ignoring the
small conv state:

- today: read `S` once + write `T · S` = `(1 + T) · S`;
- tape: verify reads `S` and writes `T` tape rows; commit reads `S`, reads `k`
  tape rows, writes `S` = `3 · S + (T + k) · t`.

| T | today | tape (k = T, worst case) | 
|---:|---:|---:|
| 2 | 216 MiB | 223 MiB |
| 4 | 360 MiB | 230 MiB |
| 8 | 648 MiB | 243 MiB |

Break-even is around `T = 2`; at `T = 4` the difference is ≈ 130 MiB, about
0.27 ms at the 512 GB/s peak (peak arithmetic, not measured). A variant that
folds commit into the *next* verify (replay from the tape in registers,
then verify) would be `2 · S + tape` — a dest-side idea, not in goss, unsourced.

State slots: today each request needs one SSM slot per in-flight speculative
token. How many slots the dest allocator reserves per request is
**unsourced here** (Python side, not read); the tape would let that drop to
the committed slot plus `B · T · t` of tape.

## Proposal

1. **Tape buffer** `[layers × max_draft × (conv inputs + k, v, beta, g)]`,
   indexed by the same per-request slot the state uses, so it fits the paged
   state (one tape page per active request, sized to `max_draft ≤ 8`).
2. **Read-only verify**: `gdn_spec_verify_rdna2` / `mono_gdn_rdna2` spec path
   stops storing per-token state and writes tape rows instead.
3. **Commit kernel**: one launch over (request, layer, value head) that loads
   the committed state, replays the request's own accepted prefix, writes
   the state once; one conv-state launch alongside. Two launches total per
   step, like goss.

## Risks

- **Bit parity.** Today the next step starts from the fp16 store of the fp32
  register state after `k` tokens. The replay must use identical inputs and
  op order, so the tape must hold what the kernel actually consumes (the
  fp16 inputs, or the exact fp32 intermediates), not re-rounded values.
  Note the spec math differs from the non-spec path (fp16-rounded beta,
  `mono_gdn_rdna2.cu` L21-23); the replay must use the spec math.
- **Continuous batching**: per-request `k`, padded/null slots, preemption
  between verify and commit; the commit must be graph-capturable with
  variable `k`.
- **Spec methods**: MTP, DSpark, DFlash and dynamic drafting
  ([#125](https://github.com/opengfx1030/vllm-rdna/pull/125)) all feed `num_accepted`; tree drafts would need a tape per branch.
- **LeapQuant state**: the quantised-state path needs its own replay/requant.
- goss's parity test is a single prompt at batch 1 with fp32 state.

## Exit criteria

- Greedy-identical outputs vs the current verify path on a multi-prompt set
  (several lengths, `c = 1` and `c > 1`, MTP and one draft-model method).
- State bytes per request and measured bandwidth receipts (rocprof) for
  verify + commit vs today, at `T ∈ {2, 4, 8}`.
- No change when spec decoding is off.

AI assistance was used to draft this note. A human must review every line
before this leaves draft. **Do not merge. Keep draft.**
