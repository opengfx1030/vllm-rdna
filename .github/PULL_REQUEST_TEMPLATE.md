<!-- markdownlint-disable -->

## Overview

<!-- Summarize the change in 1-2 lines. Link relevant issues and PRs. -->

## Claims

<!-- State the accomplishments in a few concrete bullets: speedups, feature support, bug fixes, etc. -->

## Validation

<!-- Show evidence that the claims are met by this change, and steps to reproduce validations. -->

## Details

<!-- Explain motivation, design decisions, root cause, tradeoffs, and limitations concisely. Link or collapse lengthy supporting material. -->

## RDNA receipt: head-sha against base-sha

<!-- RDNA fork: required for every pull request, see .github/CONTRIBUTING.md. Run tools/rdna/pr_receipt.sh and paste its receipt.md in place of this block (its sections are these sections); fill in what the script cannot know. Write "not run" for anything you did not run: not run is not a pass. -->

| gate | result |
|---|---|
| token-exact | <!-- PASS / FAIL / INCONCLUSIVE / NOT RUN --> |
| quality | <!-- PASS / FAIL / NOT RUN --> |
| overlap | <!-- c=4 and c=8 cells: PASS / FAIL / INVALID / NOT RUN --> |
| stall, hard gate | <!-- PASS / FAIL / INVALID / ERROR / NOT RUN --> |
| stall, relative gate | <!-- PASS / REGRESSION / direction only (MODE=ab) / NOT RUN --> |
| speed | <!-- cells: no measurable difference / win / REGRESSION / invalid --> |
| added syncs and launches | <!-- lines to review / none / NOT RUN --> |
| boots | <!-- all exited 0, or the boots that failed --> |

### Environment

<!-- Base and head SHA and tree, md5 of the built vllm/_rocm_C*.so, venv, torch and ROCm (HIP) versions, GPUs with gfx target and PCI bus, recipe and its overrides, model and revision, compilation config, host and date. -->

### Change class and reach

<!-- The row of the table in CONTRIBUTING.md (docs/tools/CI, one arch, shared code, kernel, default path, lossy feature, new model, speculative decoding); the callers and platforms the change reaches; the platforms run and the platforms not run; for csrc changes the result of tools/rdna/archdiff.sh. -->

### Exactness

<!-- Greedy output hashes base against head with VLLM_RDNA2_MOE_DETERMINISTIC=1, and the concurrency probe of base and head. -->

### Quality

<!-- topk_kl.py head against base within the base-against-base floor (KL and top-1 with their intervals), PPL rise at most 2 %, concurrency and greedy probes; for a lossy feature also the task-eval A/B. -->

### Speed

<!-- One row per cell: head against base for TTFT p50 and p99, prefill tok/s, ITL p50 and p99, decode tok/s, output tok/s, and the verdict (ABBA to claim a win or when a cell is worse). -->

#### Speculative acceptance length

<!-- Only when the recipe speculates or the change touches speculative decoding / drafter / verify. -->

### Concurrency and stalls

<!-- Serving must be concurrent and asynchronous: requests overlap, decoders keep producing tokens while others prefill, no change adds a sequential step. -->

#### Overlap

<!-- c=4 and c=8 cells, per boot: each request's submit, first-token and last-token times, the most requests generating at once, and the aggregate output tok/s against c=1. A cell whose requests run one after another, or whose aggregate output at c=N is not above c=1, fails. -->

#### Stall probe

<!-- stall_probe.py on every speed boot (inject and reverse, decoders = the recipe's max concurrency less the injected request's slot, prefill 4096 and 16384; a KFD eviction invalidates a window). Hard gate: no starved decoder stream, tokens from every decoder during every injected prefill. Relative gate (ABBA rule): longest stall, stalled time, decode tokens during the prefill, Jain fairness, injected TTFT against solo. -->

#### Added host syncs and launches

<!-- Added lines in the diff that match host-sync patterns (synchronize(, .item(), .cpu(), .tolist(), blocking copies) in vllm/, and kernel launches on the null stream or without a launch check in csrc/. For the reviewer; if a sync is needed on the per-step path, say where the PR measures why. -->

### Boots

<!-- Per boot: time to READY, probes, KFD evictions, fault markers, dmesg faults, exit status. -->

### Raw results

<!-- Host and directory of the run (boots/<id>/ with serve.log, summary.txt, bench-*.json, concurrency.json), and the command that produced it. -->

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [ ] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
