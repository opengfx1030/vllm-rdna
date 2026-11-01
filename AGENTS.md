# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 0. RDNA fork (read first)

This repository is the **opengfx1030** fork of vLLM for AMD **RDNA2 (gfx1030)**
and related RDNA GPUs. The default branch is `rdna_extras` (upstream v0.31.0 plus
the RDNA port). Sections 1-2 below are upstream's rules and still apply here.

1. Read the community wiki: https://blivioniag.github.io/gfx1030-wiki/
2. Prefer a **host venv** over Hub Docker images for Flash-Next or current kernels
   (published tags lag `rdna_extras`).
3. Serve through the in-tree launcher. Do **not** write a fresh `vllm serve`
   argv from memory:

```bash
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=<name>
```

- Dry-run: `PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=<name>` prints the
  environment, the TunableOp profile and the exact command.
- Recipes live in `tools/rdna/recipes/*.env`: `flashnext`, `flashnext-mtp0`,
  `flashnext-mtp2`, `flashnext-pp3`, `flashnext-pp3-mtp2`, `27b-awq`,
  `27b-exl3`, `dsv4-flash`, `full`. Override any key with a trailing
  `KEY=value`; pass extra vLLM flags with `EXTRA_ARGS='...'`
  (for example `EXTRA_ARGS=--monokernel`).
- Flash-Next PLE table: `PLE=bf16` (default, ~95 GB host RAM), `PLE=int4` or
  `PLE=int4-disk` with `PLE_QUANT_DIR=/path/to/ples_int4`.

### Common mistakes

- Building Hub/Dockerfile images with `VLLM_REF=main` and expecting Flash-Next:
  use a host checkout and the launcher.
- Pasting old env dumps that set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` on Flash-Next.
  Only the recipes that need it (`dsv4-flash`, `full`) set it.
- Using BF16 on RDNA2: gfx1030 has no BF16 dot instructions, serve with
  `--dtype float16`.
- Enabling AITER or CK backends: they are CDNA-only. Keep `VLLM_ROCM_USE_AITER=0`.

### RDNA code rules

- **Never break other platforms.** Gate RDNA changes with the helpers in
  `vllm/platforms/rocm.py` (`on_gfx10x()`, `on_gfx1x()`, `on_rdna()`), never
  globally and never with an exact `"gfx1030" in arch` string match. CUDA, CDNA
  (AITER), GCN and CPU paths must keep working.
- **FP16 only on gfx1030.** RDNA2 has no BF16 dot instructions; BF16 kernels
  either fail to compile or run emulated. Keep BF16 paths for other archs.
- **Wave32.** Size Triton/HIP kernels for 32-lane waves: `num_warps` <= 4 and
  RDNA-sized tiles, not CDNA (wave64, 128x128) configs.
- **Launch on the current stream.** HIP kernels capture
  `at::cuda::getCurrentCUDAStream()` and launch with `<<<grid, block, 0, stream>>>`,
  followed by `C10_CUDA_KERNEL_LAUNCH_CHECK()`. A plain `<<<grid, block>>>` goes to
  the null stream, races the model stream and fails silently.
- **Zero atomically-accumulated outputs.** Kernels that accumulate with atomics
  need `torch.zeros`, not `torch.empty`.
- **Never route to wvSplitK or LLMM1 on gfx10.** They build for gfx1030 but
  assert on the device.
- **Disable kernels through the registry.** Use `VLLM_DISABLED_KERNELS=<Class>`;
  do not add per-kernel `VLLM_DISABLE_*` env vars. Prefer flags that travel
  through the vLLM config (CLI / `--additional-config`) over env-only knobs:
  engine-core and worker processes do not always see the caller's environment.
- **Cudagraphs are the production path.** Serving runs `FULL_AND_PIECEWISE`;
  a change must stay correct both there and under `--enforce-eager`, and any
  speed change needs an A/B against the current branch.
- **Benchmarks report four numbers per cell:** prefill tok/s, decode tok/s per
  request, TTFT (s) and ITL (ms), for 16k/1k and 1k/512 at c=1 and c=8.

### Where RDNA docs live

| Topic | Location |
|---|---|
| Human wiki (setup, recipes, troubleshooting) | https://blivioniag.github.io/gfx1030-wiki/ |
| Serve launcher and recipes | `tools/rdna/serve_rdna.sh`, `tools/rdna/recipes/` |
| Kernel and V620 notes | `docs/rdna2/` |
| TunableOp profiles | `tunableop/` |

Do not invent tok/s or hardware claims. Prefer editing a recipe over hard-coding
host paths.

---

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

For work on this fork, also check its own issues and PRs (including `[backlog]`
drafts):

```bash
gh pr list --repo opengfx1030/vllm-rdna --state open --search "<short area keywords>"
gh issue list --repo opengfx1030/vllm-rdna --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.
- Before opening a PR (drafts included) or requesting re-review, run the [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md) skill and address its findings.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# Start with precompiled artifacts for an editable install:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

On this fork the precompiled artifacts do not help: they are CUDA builds. For
RDNA, build the ROCm extensions from source in a venv that has a ROCm PyTorch
(`VLLM_TARGET_DEVICE=rocm`, `PYTORCH_ROCM_ARCH=gfx1030`); the wiki has the full
recipe.

For C/C++ or CUDA changes, follow the
[incremental compilation workflow](docs/contributing/incremental_build.md) to
configure and perform incremental builds.

### Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **RDNA serving and tuning** (fork env vars, launcher modes, TunableOp):
  [`docs/rdna2/tuning.md`](docs/rdna2/tuning.md)
- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
