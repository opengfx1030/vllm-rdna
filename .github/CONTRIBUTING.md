# Contributing to vLLM

You may find information about contributing to vLLM on [docs.vllm.ai](https://docs.vllm.ai/en/latest/contributing).

## RDNA fork: contribution contract

This fork (`opengfx1030/vllm-rdna`, default branch `rdna_extras`) has one rule for every pull request, from people and from agents, with no exceptions:

> Run the tests that match what your change can reach, on hardware where it runs, before and after your change on the same setup, and post the receipt in the pull request.

- The hardware is whatever the change is relevant to: an RDNA3-only change is validated on RDNA3, a gfx1030 path on gfx1030.
- "Before" is the base, the `rdna_extras` commit your branch is cut from. "After" is the head, your branch.
- Something you did not run is not a pass. Say that you did not run it.

Upstream's rules (the pointer above and `AGENTS.md`) still apply. The fork's code rules are in `docs/rdna/agent-rules.md`.

### What to run, by change

| Your change | Required |
|---|---|
| Docs, tools, CI | pre-commit; tests for the tool |
| Code gated to one arch (e.g. RDNA3 only) | everything below on that arch; proof the other arches are untouched |
| Shared code (runs on every platform) | CPU unit tests; receipt on your arch; list every caller and which platforms reach it |
| New or changed kernel | kernel test against an fp32/fp64 reference for every shape the dispatcher sends it; worst-element error reported; kernel-by-kernel before/after table with ISA counts (VGPR, SGPR, LDS, spills, private scratch); `tools/rdna/archdiff.sh` for `csrc/` changes; a bug fix adds a test that fails before and passes after |
| Default path, numerics should not change | token-exact gate + speed receipt |
| Default path, numerics change | token-exact gate (measure token divergence with `FAIL_FAST=0`) + quality gate + speed receipt |
| Lossy feature (int8 activations, int4 KV, ...) | opt-in only; token-exact gate (measure token divergence with `FAIL_FAST=0`) + quality gate (run even when known lossy, to measure loss) + task-eval A/B; turning it on in a recipe needs the task-eval A/B and maintainer sign-off |
| Contrastive / quality-enhancing decoding (e.g. LoopCD) | opt-in only; `alpha = 0` / disabled path `0-ULP` bit-identical to baseline; prefix-cache invariance; programmatic Math + Code task-eval A/B; concurrency + stall probes; DSpark/MTP acceptance rate when applicable |
| New model | full set on that model; if shared code changed, also on the existing recipes it reaches |
| Touches speculative decoding / drafter / verify | also run the MTP recipe and report acceptance length |

A change that fits more than one row owes all of them, and a change that reaches inference also owes the overlap check and the stall probe below. Run `pre-commit run --files <changed files>` for every change.

### Token-exact gate

Greedy output token hashes are equal between base and head at the same settings, and concurrent replies equal their solo runs (`tools/rdna/port_v031/concurrency_probe.py`, 0 FAIL). Both run with `VLLM_RDNA2_MOE_DETERMINISTIC=1`: the default MoE epilogues add partial sums in arrival order, which changes from run to run, and the flag makes the routed experts bit-identical between runs.

If the base does not repeat its own solo outputs, it cannot judge the head, and the receipt says INCONCLUSIVE instead of PASS.

Do not skip this gate with `EXACT=0` when numerics are meant to change or when the technique is known to be lossy. Run with `EXACT=1 FAIL_FAST=0` alongside `QUALITY=1` so the receipt still measures how many greedy outputs diverged (`equal base vs head`), lists which outputs differed, and verifies concurrent-vs-solo consistency via `concurrency_probe.py` on head. Differing base-vs-head hashes are expected for lossy techniques and supply the required token-loss measurement, not a waiver of the run.

### Quality gate

Both parts must pass.

- Noise-relative: `tools/rdna/port_v031/topk_kl.py` compares head against base, and base against a second base boot. The head's KL and top-1 disagreement must be within the base-versus-base floor: the head's KL must not exceed the upper bound of the floor's bootstrap interval, and its top-1 agreement must not fall below the lower bound of the floor's interval. Both intervals are reported.
- Fixed: the perplexity rise is at most 2 % (`tools/rdna/port_v031/quality_ab.py --max-ppl-rel 0.02`), the concurrency probe has 0 FAIL, and the greedy probes pass.

These boots use the recipe's production configuration, so that the floor measures the run-to-run noise of the configuration you ship.

Known-lossy techniques must still run the full quality gate even when KL or top-1 disagreement is expected to exceed the base-vs-base noise floor: quantifying the loss (KL, top-1 agreement, and perplexity delta) is required for review alongside the task-eval A/B. When a lossy change acts across decode steps where prompt-only scoring cannot observe it (such as recurrent-state or KV compression), also run `tools/rdna/port_v031/decode_kl.py`.

### Contrastive / quality-enhancing decoding modes (e.g. LoopCD)

Opt-in decoding modes that intentionally alter output logits to improve reasoning quality must satisfy:

- **Zero-regression when off**: the `alpha = 0` / disabled path must be `0-ULP` bit-identical to baseline (no numerical drift when off).
- **Opt-in switches and prefix-cache invariance**: gated behind per-request (`SamplingParams.extra_args`), environment, or recipe switches; mixing enabled and disabled requests must preserve prefix-cache invariance and pass `concurrency_probe.py` (0 FAIL) and `stall_probe.py`.
- **Programmatic Math + Code task-eval A/B**: run a reproducible, programmatically verified Math + Code suite (e.g. `tools/rdna/inkling2/eval_loopcd.py` — 20-problem math + Python coding suite with exact answer / unit-test verification) comparing baseline against enabled arms, reporting:
    - per-domain pass rates (Math, Code, Total) and net `+gain / -loss` regressions vs baseline,
    - mean output tokens and truncation count,
    - decode latency overhead, and
    - speculative acceptance rate (`tok/step`) under DSpark / MTP when applicable.

### Concurrency, async and stalls

Serving is concurrent and asynchronous: requests overlap, and decoders keep producing tokens while another request prefills. A change must not add a sequential step.

- Overlap: the c=4 and c=8 speed cells report every request's submit, first-token and last-token times, and the aggregate output tok/s at c=N divided by the aggregate at c=1 on the same boot. A cell fails when its requests run one after another (never two generating at once) or when the aggregate output at c=N is not above c=1.
- Stalls: `tools/rdna/port_v031/stall_probe.py` runs on every speed boot, with the scenarios `inject` and `reverse`, prefill lengths 4096 and 16384 and `--server-pid`.
    - Decoders: the recipe's `--max-num-seqs` less one, because the injected request needs the last slot.
    - Reverse windows (the decoders arrive behind a running prefill) are reported and are not part of the hard gate.
    - Hard gate: no starved decoder stream, and tokens from every decoder during every injected prefill.
    - Relative metrics: the longest stall, the stalled time of the worst decoder, the decode tokens during the prefill, the Jain fairness and the injected request's TTFT against its solo TTFT. They are reported for the reviewer with the ABBA verdict of the speed receipt (worse, better, or no measurable difference per metric) and never fail the run.
- Async: the diff is scanned for added host syncs and unchecked launches, and the lines are listed in the receipt for the reviewer. In `vllm/`: `synchronize()` (streams and events), `.item()`, `.cpu()`, `.tolist()` and blocking copies. In `csrc/`: launches on the null stream (a `<<<...>>>` launch with fewer than four launch parameters) and launches without `C10_CUDA_KERNEL_LAUNCH_CHECK`. The scan decides nothing; if the per-step path needs a sync, the pull request says where it measures why.

### Speed receipt

Cells, as prompt tokens / generated tokens:

| Cell | Concurrency |
|---|---|
| 1k / 512 | 1, 4, 8 |
| 16k / 1k | 1, 4, 8 |
| 32k / 2k | 1, 4 (c=8 is skipped) |

Per cell, per request and in aggregate: TTFT p50 and p99, prefill tok/s, ITL p50 and p99, decode tok/s, output tok/s.

- Production compiled mode of the recipe (`FULL_AND_PIECEWISE`), greedy (`--temperature 0`), `--ignore-eos`, a fixed seed, the prefix cache reset before every cell, and a warm-up after boot (`serve_validate.sh` runs the capture-ladder warm-up).
- A KFD eviction during a cell invalidates the cell. Rerun it.
- Base, then head, on the same machine, GPUs, model revision and environment.

To claim a win, or when any cell is worse, repeat on fresh boots in the order base, head, head, base (ABBA). A metric regressed only when both head runs are worse than both base runs, and improved only when both head runs beat both base runs. Anything else is "no measurable difference". A cell is a regression if any of its metrics regressed, and a win if none did and one improved. With one base run and one head run the table shows the direction only; that is not a verdict.

The script reports these verdicts. Whether a regression is acceptable is a decision for review.

### Long-context sweep

For recipes with `--max-model-len >= 131072`, `tools/rdna/pr_receipt.sh` defaults `LONG_CTX=auto` to append 64k and 128k cells (`65536:1024:1:2 65536:1024:2:4 131072:512:1:2`, exactness `65536:128:1:1`) filtered to `in_len + out_len <= --max-model-len`, and runs deterministic ~50%-depth 6-digit passkey retrieval prompts (`NEEDLE_LENS=32768,65536,131072`, capped by `--max-model-len`) in `tools/rdna/port_v031/quality_ab.py` whenever `QUALITY=1`. Set `LONG_CTX=0` to restrict to the 1k–32k matrix above, or `LONG_CTX=1` to force the 64k/128k sweep. Ready runtime PRs are checked by `tools/rdna/check_pr_contract.py` for at least one `>= 64k` cell in `### Speed` (or an explicit `long-context N/A` reason when `max-model-len <= 32k`) and passkey needle coverage in `### Quality` whenever `QUALITY=1` is run.

### Other cards untouched

RDNA code goes behind the arch helpers in `vllm/platforms/rocm.py` (`on_gfx10x()`, `on_gfx1x()`, ...), never behind a hard-coded arch string.

For `csrc` changes, compile for `gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1200`, diff the disassembly per function, and let only the intended functions differ. `tools/rdna/archdiff.sh` does it on the CPU (no GPU is used):

```bash
BASE=origin/rdna_extras HEAD=my-branch \
EXPECT='LLGemm1_kernel' MAY_DIFFER="gfx1030 gfx1100 gfx1200" \
    bash tools/rdna/archdiff.sh
```

It builds the translation units that changed, once for each revision and arch, and writes `archdiff.md` and `archdiff.json`: per arch, the functions that differ. Differences that come only from moved `__LINE__` constants and PC-relative offsets are normalized away. `EXPECT` is a regular expression for the demangled names of the functions that may differ, `MAY_DIFFER` narrows that to the arches you list, and without `EXPECT` no function may differ. Exit status 0: nothing differs beyond that. 1: something else differs. 2: the run could not finish. `bash tools/rdna/archdiff.sh --help` lists every setting.

Write down the platforms you ran and the ones you did not. Not run is not a pass.

### Receipt in the PR body

Every pull request follows `.github/PULL_REQUEST_TEMPLATE.md`:

- **Overview**: concise headline summary of the change and measured outcome, a **Duplicate-work check** against open PRs (`gh pr list --repo opengfx1030/vllm-rdna --state open`), and an **AI assistance** disclosure.
- **Claims**: concrete claims, including the default values of any new flags or environment variables.
- **Validation**: for kernel/`csrc` changes, a kernel-by-kernel before/after table with compiled ISA counts (VGPR, SGPR, LDS, spills, private scratch), plus copy-pasteable **How to Reproduce Locally** steps (incremental build, launch command + expected activation log marker, and `pr_receipt.sh` / test command).
- **Details**: design fit, arch guards (`on_gfx10x()`, `#ifdef`), current-stream launches + `C10_CUDA_KERNEL_LAUNCH_CHECK()`, non-regression scope, tradeoffs, and rejected experiments.
- **RDNA receipt**: paste `receipt.md` from `tools/rdna/pr_receipt.sh` in place of the receipt block, fill in every row of the 8-gate summary table, and fill in what the script cannot know. Neither `token-exact` nor `stall, hard gate` may be `NOT RUN` on a Ready PR that reaches inference; Ready runtime PRs must also report at least one `>= 64k` cell in `### Speed` (or `long-context N/A` for `<= 32k` models) and passkey needle results in `### Quality` when the quality gate is run:
    - Environment: base and head SHAs, GPU and gfx target, pinned clocks/power caps when set, ROCm, torch, recipe, model revision, environment overrides.
    - Change class and reach: the row of the table above, the callers and platforms the change reaches (platforms run and not run), and for `csrc` changes the `archdiff.md` result.
    - Exactness and quality: the results of the two gates (plus `decode_kl.py` and task-eval A/B for lossy features).
    - Speed: the table.
    - Concurrency and stalls: the overlap table, the stall probe results and the list of added syncs and launches.
    - Raw results: where the run directory is, so that a reviewer can read the logs.

`tools/rdna/check_pr_contract.py` (enforced by `.github/workflows/rdna-pr-contract.yml`) checks that these required sections and gates are populated based on the changed files (`csrc/`, `vllm/`, or `docs/tools/ci`). Draft PRs receive advisory output only; Ready PRs fail CI when required sections or gates are missing.

### One command

`tools/rdna/pr_receipt.sh` takes a base tree and a head tree, boots each through `tools/rdna/port_v031/serve_validate.sh`, runs the probes and writes the receipt tables:

```bash
BASE_TREE=~/trees/base HEAD_TREE=~/trees/head VENV=~/venv \
RECIPE=flashnext-mtp2 MODEL=/path/to/model GPUS=2,3,4,5 \
MODE=abba QUALITY=1 \
CHANGE_CLASS="default path, numerics should not change" \
REACHED="every caller of the changed op, and the platforms that reach it" \
NOT_RUN="gfx1100, gfx1200" \
    bash tools/rdna/pr_receipt.sh
```

- `BASE_TREE` and `HEAD_TREE` are built trees of the two commits: the Python sources and the built `vllm/_rocm_C*.so`, named by a `.sha` file or by being a git checkout. The same path twice is an A/A run, which shows the noise of the machine.
- `VENV` is the venv that runs the servers and the bench clients (`BASE_VENV` and `HEAD_VENV` set them apart). `RECIPE` is a file of `tools/rdna/recipes/` and `GPUS` the HIP devices it needs. Trailing `KEY=value` arguments are recipe overrides for both trees.
- `MODE=ab` boots base, then head: one run of each, so the speed table shows direction only. `MODE=abba` boots base, head, head, base and is what a speed claim needs. `QUALITY=1` adds the quality gate. Keep `EXACT=1` even when numerics are meant to change or the technique is known to be lossy: pass `QUALITY=1 FAIL_FAST=0` instead of `EXACT=0` so the receipt still measures token divergence and deterministic concurrency before continuing to the quality and speed boots (`EXACT=0` is only for re-running speed cells when an exactness receipt already exists).
- `CELLS` replaces the speed cells (`"in:out:conc[:prompts] ..."`) and `EXACT_CELLS` the cells whose greedy texts are hashed. A run with fewer cells than the table is a smoke test, and the receipt lists the cells it ran. `DRY_RUN=1` checks the inputs and prints the launch plans and the boots without booting.
- The stall probe runs on every speed boot: `STALL=0` skips it, `STALL_REPEATS` (default 3) repeats each scenario, `STALL_LENS` (default `4096,16384`) sets the prefill lengths and `STALL_DECODERS` the decoders (default: the recipe's `--max-num-seqs` less one). The speed boots save per-request times (`--save-detailed`) for the overlap check. The diff scan (`tools/rdna/pr_receipt_scan.py`) runs before the boots and needs no GPU.
- Boots run one at a time, and the exactness boots come first (with `VLLM_RDNA2_MOE_DETERMINISTIC=1`), then the speed boots in production mode. The base-versus-base floor of the quality gate is the second base boot of `MODE=abba`, and an extra base boot in `MODE=ab`.
- Results are in `$OUT` (the script header gives the default): `receipt.md`, `receipt.json`, `scan.md`, the raw `boots/<id>/` directories (`stall/` holds the stall probe), `plans/` and `quality/`. The exit status is 0 when every gate that ran passed (token-exact, quality, overlap and the hard stall gate), 1 when a gate or a boot failed, 2 for bad input and 3 when the run was aborted because the GPUs were busy or too hot.
- Plan for hours: the default matrix has eight cells per boot, every speed boot also runs the stall probe, and `MODE=abba` boots four times.

What it does not do:

- It does not reserve the GPUs. If you share the machine, take your reservation first. `serve_validate.sh` takes the boot lock and waits for other users' quiet windows when the machine has them.
- It does not run the kernel reference tests or the task evals that the table asks for. Run them yourself and add their output to the pull request.
- It does not build the trees. Build each commit in its own tree and venv first.
- It reports the speed and the relative stall verdicts by the ABBA rule (the direction only with `MODE=ab`) but does not fail on them. The added-sync scan never fails.
