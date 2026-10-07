# rdna_ar two-shot TP all-reduce on gfx1030 — operational notes

Status: merged in `rdna_extras` (PR opengfx1030/vllm-rdna#22, merge point
`c782696a3`; gate fix `22d6e3462`; docs `0b8585723`). Validated on
4× Radeon PRO V620 (gfx1030), TP=4, 2026-09-27.

## What it is

Push-based tensor-parallel all-reduce for RDNA (`_rocm_C::rdna_ar_*`). Staging
buffers and flags live in **uncached device memory**: each rank pushes its
contribution into every peer's staging slot with posted PCIe writes, then polls
its own flag slots locally — no peer loads, and waiting generates no PCIe read
traffic (the failure mode the old pull-based custom all-reduce had).

Two kernels, selected per message size:

| Path | When | Shape |
| --- | --- | --- |
| one-shot | `bytes <= VLLM_RDNA_AR_ONESHOT_KB` (default 32 KiB) | push own slice to all peers → W-flag wait → fixed-order fp32 reduce |
| two-shot | larger, up to `VLLM_RDNA_AR_MAX_KB` | push reduce-scatter + push allgather; traffic ≈ 2(W-1)/W·N instead of (W-1)·N |

Messages above the gate keep falling through to CUSTOM/PYNCCL, so the gate
value is the only thing that decides how far rdna_ar reaches. With the default
gate (20480 KiB) a full 4096-token prefill batch at hidden 2560 fp16 stays on
rdna_ar, and `VLLM_FORCE_CUSTOM_ALL_REDUCE` is no longer needed on gfx10x.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `VLLM_RDNA_AR` | `0` (library), `1` (launchers) | opt-in; dispatch ahead of CUSTOM/RCCL |
| `VLLM_RDNA_AR_MAX_KB` | `64` | gate for the one-shot path; above it falls through to RCCL |
| `VLLM_RDNA_AR_ONESHOT_KB` | `64` | keep equal to MAX_KB: the two-shot range is disabled |
| `VLLM_RDNA_AR_ALGO` | `auto` | `auto` \| `oneshot` \| `twoshot` |
| `VLLM_RDNA_AR_BLOCKS` / `_PACE` | `0` / `0` | launch-width / push pacing |
| `VLLM_RDNA_AR_SPIN_CAP` | ~2 s | abort bound; a wedge writes `$VLLM_CACHE_ROOT/rdna_ar_wedged` |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | `0` | superseded by rdna_ar; keep off |

## Health check

A healthy boot logs, per worker:

```
rdna_ar: one-shot all-reduce active (handle 0, rank r/4, devices [...], pix=..., max 20480 KB, oneshot 32 KB, algo auto; blocks cap 0, pace 0)
Using ['RDNA_ONESHOT', 'PYNCCL'] all-reduce backends (in dispatch order) ...
```

The boot self-test runs one warm-up + 3 timed collectives at 2 KiB fp16,
8 KiB fp16, `MAX_KB/2` numel fp16 (= the gate size, i.e. a real two-shot call),
and 1 KiB bf16; it must pass on **all** ranks.

**Silent-fallback symptom** — if the self-test fails, the backend disables
itself and everything runs on RCCL:

```
WARNING rdna_ar: disabled -- boot self-test failed on some rank (... falling back to RCCL): [...]
Using ['PYNCCL'] all-reduce backends (in dispatch order) ...
```

Measured cost of that fallback: ~5–10 % decode (see below). Check for
`rdna_ar: disabled` in the serve log after every config change.

## Two-shot is disabled by default (2026-09-27, late)

Two-shot (`ONESHOT_KB` < size ≤ `MAX_KB`) is **not boot-safe under PCIe
load**: it intermittently loses the peer-flag handshake and wedges —

```
RuntimeError: rdna_ar wedged: rank 0 timed out after ~1953 ms of spinning
at collective #72: peer rank 1's flag never arrived ...
```

— raised by `rdna_ar_check()` during the startup warmup, which kills engine
init. The boot self-test passes, then a later warmup/serving call wedges; the
wedge marker (`$VLLM_CACHE_ROOT/rdna_ar_wedged`) then forces RCCL on subsequent
boots. The one-shot path (≤ 64 KiB) is unaffected and has run for weeks.

**Repro**: boot with `VLLM_RDNA_AR_MAX_KB=20480` (two-shot enabled) while a
co-tenant generates PCIe traffic; watch for a wedge in warmup. Re-enable the
wide gate only after the two-shot flag handshake (phases 2/4) is fixed.

The following small-gate failures are the same defect at the bottom of the
two-shot range:

| `VLLM_RDNA_AR_MAX_KB` | self-test two-shot trial | verdict |
| --- | --- | --- |
| 64 KiB | 64 KiB | **FAIL** — wrong result (26.6–28.6 vs expected 30.0), 3/3 launches |
| 128 KiB | 128 KiB | **FAIL** — spin-cap abort: "peer rank 1's flag never arrived … collective #10" (phase 4) + partial wrong result |
| 1 MiB | 1 MiB | PASS (but flaky under load) |
| 20 MiB | 20 MiB | PASS sometimes; **wedged warmup under co-tenant load (×2)** |

Keep the gate at the default `20480` (or at least ≥ 1 MiB). The failure is a
cross-call sync bug in the two-shot path near the bottom of its range; the
exact boundary between 128 KiB and 1 MiB is uncharacterised and a standalone
probe sweep is the planned follow-up.

## Measured (two-shot era — historical; keep disabled until fixed)

4× V620, TP=4, Flash-Next AWQ W4A16, Triton attn, MTP=0/2:

- two-shot 20 MB ≈ **5.7 ms per call** (boot self-test, all ranks agreeing)
- cold 16k/1k prefill **1755 tok/s vs 1686 tok/s on RCCL (+4 %)**
- decode at parity with the pre-PR one-shot path (1k/512 c=1: TPOT 23.26 ms /
  43 tok/s vs 25.75 ms / 38.8 tok/s on RCCL, +10 %)
- 16k/1k c=8 MTP=0 aggregate **77.3 tok/s** (75.9 pre-PR, 74.0 RCCL)
- full per-cell tables, correctness evidence, and the gate sweep:
  gfx1030_optimized `docs/rdna2/pr22-twoshot-validation-2026-09-27.md`

## Rebuilding after csrc changes (0.28.0 tree)

Only `csrc/rocm/rdna_allreduce.cu/.cuh` changed for this PR; the extension is
`_rocm_C`. Configure once, build the single target, swap the `.so`:

```bash
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030 MAX_JOBS=16 \
       CMAKE_BUILD_TYPE=RelWithDebInfo ROCM_HOME=/opt/rocm/core-7.14
mkdir -p build/temp.linux-x86_64-cpython-312 && cd build/temp.linux-x86_64-cpython-312
cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_TARGET_DEVICE=rocm \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
  -DVLLM_PYTHON_EXECUTABLE=$VENV/bin/python -DFETCHCONTENT_BASE_DIR=$PWD/../../.deps \
  -DCMAKE_JOB_POOLS:STRING=compile=16 -DROCM_PATH=/opt/rocm/core-7.14
ninja _rocm_C && cp _rocm_C.abi3.so ../../vllm/_rocm_C.abi3.so
```

Pitfall: **do not `source venv/bin/activate` during the configure** — with
`VIRTUAL_ENV` set, cmake's `find_package(Python)` fails with "missing:
Interpreter Development.Module". Keep the old `.so` as a backup before
overwriting (the editable install loads it straight from `vllm/`).
