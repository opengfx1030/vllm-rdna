# RDNA serving recipes

Each `<name>.env` here is one serving configuration for
[`tools/rdna/serve_rdna.sh`](../serve_rdna.sh): shell `KEY=value` lines holding
only that configuration's deltas. The launcher owns the shared machinery (tree,
venv, ROCm, TunableOp profile, GPU selection, cache root, stale-process
teardown, the common vLLM environment block, capture ladder and argument
assembly).

```bash
bash tools/rdna/serve_rdna.sh RECIPE=<name> MODEL=/path/to/checkpoint [KEY=value ...]
PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=<name> MODEL=/path/to/checkpoint
```

`PRINT=1` assembles the environment, the TunableOp profile and the exact
command without launching. Precedence, lowest to highest: recipe defaults,
caller environment, trailing `KEY=value` arguments. The reference for every
key, with the measured numbers behind it, is
[`docs/rdna/serving.md`](../../../docs/rdna/serving.md).

## Layout and lookup rules

The directory is flat on purpose. The launcher resolves
`$RDSE_RECIPES/<name>.env`, rejects a name containing `/` or `..`, and a recipe
builds on another with `source "$RDSE_RECIPES/<name>.env"` (the launcher follows
those chains to snapshot the caller's overrides). Docs, tests, launcher wrappers
and recorded benchmark commands all name recipes by this bare name, so a recipe
is not moved into a subfolder or renamed without migrating every one of those
references.

## Naming convention

`<model>-<variant>`, lowercase, dash separated:

- `<model>` is the model family: `27b`, `flashnext`, `inkling2-mini`,
  `dsv4-flash`.
- `<variant>` is a sequence of the tokens that distinguish the recipe from its
  siblings, in this order: checkpoint format (`awq`, `exl3`, `gguf`, `hi2`,
  `leapquant`), parallelism (`tp<N>`, `pp<N>`, `tp4pp2`), speculation (`mtp0`,
  `mtp2`, `dflash2`, `dspark`, `dspark-adaptive`).
- A parallelism token is written whenever the model has recipes at more than one
  size (`inkling2-mini-tp1/2/4`, `27b-hi2-tp2`) and omitted when the model's one
  layout is the default (`flashnext-mtp2` is TP=4).
- A recipe that only changes a few keys of another sources it
  (`inkling2-mini-tp1-dspark` sources `inkling2-mini-tp1`) and carries the
  parent's name plus one token.

Two names predate the convention and are kept because launchers, docs and bench
commands use them:

| Name | What it is |
| --- | --- |
| `flashnext` | Flash-Next AWQ-W4A16, no MTP, with the agentic tool/reasoning stack, port 18094 on `0.0.0.0`, 6 sequences; `flashnext-mtp0` is the same model and speculation setting with the public defaults |
| `full` | Qwen3.8-27B AWQ-INT4 at TP=2 with breakable `FULL_AND_PIECEWISE` graphs ("full HIP"); the TP=4 sibling is `27b-awq` |

## Inventory

Values are the launcher's resolved plan (`PRINT=1`) with no overrides. `MAXLEN`
"model" means no `--max-model-len` (the model's own context). Cards is `TP` times
`PP`; the default device list is `0,1,2,3` and is overridden with
`HIP_VISIBLE_DEVICES`, `GPUIDS` or `PCI_BUSES`. Alternatives of one model share a
port on purpose: only one runs on a given set of cards at a time.

| Recipe | Model | TP x PP | Cards | Port | MAXLEN | SEQS | Speculation | Key switches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `27b-awq` | Qwen3.8-27B AWQ-INT4 | 4 x 1 | 4 | 18210 | model | 8 | none | W4A8 prefill, FA-RDNA2, prefix caching, 1024-token blocks |
| `27b-exl3` | Qwen3.8-27B EXL3 3.00 bpw, codebook `mul1` | 4 x 1 | 4 | 18105 | 20480 | 8 | none | `SERVE_ENTRY=api`, FA-RDNA2, decode-stall cap 250 ms |
| `27b-hi2-tp2` | Qwen3.8-27B HI2 3.0 bpw (`rdna_hint_flex`) | 2 x 1 | 2 | 18211 | 262144 | 8 | none | HI2 switch table, TunableOp profile `rocm7.14-rocblas5.5-hi2-27b-tp2` |
| `27b-hi2-tp2-mtp2` | same | 2 x 1 | 2 | 18211 | 262144 | 8 | MTP=2 | `27b-hi2-tp2` with `MTP=2` only |
| `full` | Qwen3.8-27B AWQ-INT4 | 2 x 1 | 2 | 18094 | 200000 | 6 | none | `ATTN=none`, GQA subgroup prefill, 16-token blocks, `HOST` empty |
| `dsv4-flash` | DeepSeek-V4-Flash K128 (fp8 dense, MXFP4 experts) | 4 x 1 | 4 | 18300 | 32768 | 8 | none | fp8 KV, breakable graphs, `ATTN=none` |
| `flashnext` | Flash-Next AWQ-W4A16 | 4 x 1 (EP4) | 4 | 18094 | model | 6 | none | tool/reasoning parsers, `HOST=0.0.0.0` |
| `flashnext-mtp0` | Flash-Next AWQ-W4A16 | 4 x 1 (EP4) | 4 | 18096 | 262144 | 8 | none | plain decode |
| `flashnext-mtp2` | Flash-Next AWQ-W4A16 | 4 x 1 (EP4) | 4 | 18096 | 262144 | 8 | MTP=2 | spec decode, decode-stall cap |
| `flashnext-dflash2` | Flash-Next AWQ-W4A16 + Inco DFlash2 drafter | 4 x 1 (EP4) | 4 | 18096 | 262144 | 8 | DFlash2, `DFLASH_TOKENS=3` | `--monokernel`, drafter downloaded beforehand |
| `flashnext-leapquant-mtp2` | Flash-Next AWQ-W4A16 | 4 x 1 (EP4) | 4 | 18096 | 262144 | 8 | MTP=2 | `REPLAYSSM_STATE_DTYPE=leapquant` (lossy INT8 GDN state), `--monokernel` |
| `flashnext-exl3` | Flash-Next EXL3 | 4 x 1 (EP4) | 4 | 18094 | model | 6 | none | `PLE=exl3`, `HOST=0.0.0.0`, tool/reasoning parsers |
| `flashnext-gguf-mtp0` | Flash-Next UD-Q4_K_XL GGUF | 4 x 1 | 4 | 18096 | 32768 | 8 | none | needs `GGUF_DIR` and the gguf plugin fork |
| `flashnext-pp3` | Flash-Next AWQ-W4A16 | 1 x 3 | 3 | 18096 | 262144 | 8 | none | partition 16,16,16, dense INT8 only, `KV=auto` |
| `flashnext-pp3-mtp2` | Flash-Next AWQ-W4A16 | 1 x 3 | 3 | 18097 | 262144 | 4 | MTP=2 | partition 17,17,14, needs the W4A16 drafter checkpoint |
| `flashnext-tp4pp2-mtp2` | Flash-Next AWQ-W4A16 | 4 x 2 (EP4) | 8 | 18099 | 262144 | 8 | MTP=2 | `PCI_BUSES` of the eight-card reference box |
| `flashnext-tp8-mtp2` | Flash-Next AWQ-W4A16 | 8 x 1 (EP8) | 8 | 18098 | 262144 | 8 | MTP=2 | `PCI_BUSES` of the eight-card reference box |
| `inkling2-mini-tp1` | Inkling-2-Mini | 1 x 1 | 1 | 18221 | 194000 | 16 | none | `ATTN=none`, one image per prompt, `prompt-tokens-details`, no TunableOp |
| `inkling2-mini-tp1-dspark` | Inkling-2-Mini | 1 x 1 | 1 | 18221 | 135000 | 16 | DSpark, 7 drafted, 2 verified (`b<=2`), off at `b>=3` | sources `inkling2-mini-tp1`; `ADAPTIVE=1` (`MAXLEN=125000`) + optional `DRAFT_VOCAB` built in |
| `inkling2-mini-tp2` | Inkling-2-Mini | 2 x 1 | 2 | 18222 | 720000 | 8 | DSpark, 7 drafted, 2 verified (`b<=2`), off at `b>=3` | one-shot HIP all-reduce, `MAXBAT=1024`; `ADAPTIVE=1` / `DRAFT_VOCAB`, `DSPARK=0` for 1M ctx |
| `inkling2-mini-tp4` | Inkling-2-Mini | 4 x 1 | 4 | 18223 | 1048576 | 8 | DSpark, 7 drafted, 2 verified (`b<=2`), off at `b>=3` | one-shot HIP all-reduce, 4096-tok prefill, full 1M ctx fits; `ADAPTIVE=1` / `DRAFT_VOCAB` / `DSPARK=0` |

The three `flashnext-pp3*`, `flashnext-tp4pp2-mtp2` and `flashnext-tp8-mtp2`
recipes assume cards of 29.98 GiB (V620). The two recipes with `PCI_BUSES` carry
the bus numbers of the reference eight-card box: on another host set `PCI_BUSES`
(rank order) or `HIP_VISIBLE_DEVICES` after `RECIPE=`.

## Launcher wrappers

The old per-model launcher scripts forward to `serve_rdna.sh` with a fixed
recipe: `serve_gfx1030_27b_dense.sh` (`27b-awq`), `serve_gfx1030_exl3_27b.sh`
(`27b-exl3`), `serve_gfx1030_flashnext.sh` (`flashnext`),
`serve_gfx1030_flashnext_mtp.sh` (`flashnext-mtp2`) and `serve_gfx1030_full.sh`
(`full`).

## Adding or changing a recipe

1. Name it after the convention above and write only the deltas; start the
   header with what the configuration is and cite only files that exist.
2. Add a row to the inventory table. The test
   `tests/tools/rdna/test_recipe_index.py` fails when a recipe and its row do not
   match, when a header cites a missing `docs/`, `tests/`, `tools/` or
   `tunableop/` file, or when a `source` line names a missing recipe.
3. Check the plan: `PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=<name> MODEL=/path`.
