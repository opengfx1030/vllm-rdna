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
| New or changed kernel | kernel test against an fp32/fp64 reference for every shape the dispatcher sends it; worst-element error reported; a bug fix adds a test that fails before and passes after |
| Default path, numerics should not change | token-exact gate + speed receipt |
| Default path, numerics change | quality gate + speed receipt |
| Lossy feature (int8 activations, int4 KV, ...) | opt-in only; quality gate + task-eval A/B; turning it on in a recipe needs the task-eval A/B and maintainer sign-off |
| New model | full set on that model; if shared code changed, also on the existing recipes it reaches |
| Touches speculative decoding / drafter / verify | also run the MTP recipe and report acceptance length |

A change that fits more than one row owes all of them, and a change that reaches inference also owes the overlap check and the stall probe below. Run `pre-commit run --files <changed files>` for every change.

### Token-exact gate

Greedy output token hashes are equal between base and head at the same settings, and concurrent replies equal their solo runs (`tools/rdna/port_v031/concurrency_probe.py`, 0 FAIL). Both run with `VLLM_RDNA2_MOE_DETERMINISTIC=1`: the default MoE epilogues add partial sums in arrival order, which changes from run to run, and the flag makes the routed experts bit-identical between runs.

If the base does not repeat its own solo outputs, it cannot judge the head, and the receipt says INCONCLUSIVE instead of PASS.

### Quality gate

Both parts must pass.

- Noise-relative: `tools/rdna/port_v031/topk_kl.py` compares head against base, and base against a second base boot. The head's KL and top-1 disagreement must be within the base-versus-base floor: the head's KL must not exceed the upper bound of the floor's bootstrap interval, and its top-1 agreement must not fall below the lower bound of the floor's interval. Both intervals are reported.
- Fixed: the perplexity rise is at most 2 % (`tools/rdna/port_v031/quality_ab.py --max-ppl-rel 0.02`), the concurrency probe has 0 FAIL, and the greedy probes pass.

These boots use the recipe's production configuration, so that the floor measures the run-to-run noise of the configuration you ship.

### Concurrency, async and stalls

Serving is concurrent and asynchronous: requests overlap, and decoders keep producing tokens while another request prefills. A change must not add a sequential step.

- Overlap: the c=4 and c=8 speed cells report every request's submit, first-token and last-token times, and the aggregate output tok/s at c=N divided by the aggregate at c=1 on the same boot. A cell fails when its requests run one after another (never two generating at once) or when the aggregate output at c=N is not above c=1.
- Stalls: `tools/rdna/port_v031/stall_probe.py` runs on every speed boot, with the scenarios `inject` and `reverse`, prefill lengths 4096 and 16384 and `--server-pid`. The decoders are the recipe's max concurrency less the slot of the injected request (`--max-num-seqs` minus one): the probe cannot inject into a full batch.
    - Hard gate: no starved decoder stream, and tokens from every decoder during every injected prefill.
    - Relative gate, by the ABBA rule: the longest stall, the total stalled time, the decode tokens during the prefill, the Jain fairness and the injected request's TTFT against its solo TTFT must not regress.
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

The pull request template has an "RDNA receipt" block. Its sections are the ones of the `receipt.md` that `tools/rdna/pr_receipt.sh` writes, so paste `receipt.md` in its place and fill in what the script cannot know:

- Environment: base and head SHAs, GPU and gfx target, ROCm, torch, recipe, model revision, environment overrides.
- Change class and reach: the row of the table above, the paths and platforms reached (platforms run and not run), and for `csrc` changes the `archdiff.md` result.
- Exactness and quality: the results of the two gates.
- Speed: the table.
- Concurrency and stalls: the overlap table, the stall probe results and the list of added syncs and launches.
- Raw results: where the run directory is, so that a reviewer can read the logs.

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
- `MODE=ab` boots base, then head: one run of each, so the speed table shows direction only. `MODE=abba` boots base, head, head, base and is what a speed claim needs. `QUALITY=1` adds the quality gate. `EXACT=0` skips the token-exact gate; use it when the numerics are meant to change, because the table then asks for the quality gate and a failed exactness check ends the run unless `FAIL_FAST=0`.
- `CELLS` replaces the speed cells (`"in:out:conc[:prompts] ..."`) and `EXACT_CELLS` the cells whose greedy texts are hashed. A run with fewer cells than the table is a smoke test, and the receipt lists the cells it ran. `DRY_RUN=1` checks the inputs and prints the launch plans and the boots without booting.
- The stall probe runs on every speed boot: `STALL=0` skips it, `STALL_REPEATS` (default 3) repeats each scenario, `STALL_LENS` (default `4096,16384`) sets the prefill lengths and `STALL_DECODERS` the decoders (default: the recipe's `--max-num-seqs` less one). The speed boots save per-request times (`--save-detailed`) for the overlap check. The diff scan (`tools/rdna/pr_receipt_scan.py`) runs before the boots and needs no GPU.
- Boots run one at a time, and the exactness boots come first (with `VLLM_RDNA2_MOE_DETERMINISTIC=1`), then the speed boots in production mode. The base-versus-base floor of the quality gate is the second base boot of `MODE=abba`, and an extra base boot in `MODE=ab`.
- Results are in `$OUT` (the script header gives the default): `receipt.md`, `receipt.json`, `scan.md`, the raw `boots/<id>/` directories (`stall/` holds the stall probe), `plans/` and `quality/`. The exit status is 0 when every gate that ran passed (token-exact, quality, overlap, the hard stall gate and, with `MODE=abba`, the relative stall gate), 1 when a gate or a boot failed, 2 for bad input and 3 when the run was aborted because the GPUs were busy or too hot.
- Plan for hours: the default matrix has eight cells per boot, every speed boot also runs the stall probe, and `MODE=abba` boots four times.

What it does not do:

- It does not reserve the GPUs. If you share the machine, take your reservation first. `serve_validate.sh` takes the boot lock and waits for other users' quiet windows when the machine has them.
- It does not run the kernel reference tests or the task evals that the table asks for. Run them yourself and add their output to the pull request.
- It does not build the trees. Build each commit in its own tree and venv first.
- It reports speed verdicts by the ABBA rule but does not fail on a speed regression. The relative stall gate does fail under `MODE=abba`; under `MODE=ab` it shows the direction only. The added-sync scan never fails.
