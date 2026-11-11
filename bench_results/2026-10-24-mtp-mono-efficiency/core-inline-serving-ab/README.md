# Ordinary K2 inline-build serving A/B

`script-snapshots/run_one_arm.sh.txt` preserves the executed one-arm serving
driver bytes for this A/B. To inspect or rerun it, restore a copy as
`run_one_arm.sh`; it is a root-owned, one-arm-at-a-time driver for the
ordinary model K2 core build. It compares the original build with the isolated
force-inline `grid_barrier` build. Both arms use `MTP_DRAFTER=0`; this checks
that the regular model MTP2 path stays unchanged while ordinary K2 runs.

The candidate tree is expected at
`/home/chenco_adm/w4a8_runs/mono-large-m-2026-10-24/core-inline-serve-tree`.
It may differ from the original tree only in `csrc/rocm/mono_k2_rdna2.cu`
(the single `__forceinline__` attribute) and the rebuilt
`vllm/_rocm_C.abi3.so`. The exact source/tree checker bytes are preserved as
`script-snapshots/check_candidate_source.py.txt` and
`script-snapshots/check_candidate_tree.py.txt`.
The driver pins expected binary/source SHA256 values before starting each arm
and rechecks them after the server stops.

The standard validation gate is reused from the October 23 A/B. The executed
gate and sequence scripts are preserved under `script-snapshots/`. It runs PPL,
prefix-cache, greedy/concurrency, after-load and fault checks, then the four
1024/512 and 16384/1024 cells at c1/c8 with a cold prefix reset for each cell.
The driver additionally requires the ordinary K2 route, zero dense MTP K2
calls, no MTP drafter K1/K2 route, no errors, unchanged boot ID and unchanged
serving inputs. It takes the existing `/tmp/codex-mtp-mono-hip6-9.lock` and
uses `check_reserved_gpu.py`: other work on disjoint HIP2–5 is allowed; any
KFD owner overlapping HIP6–9 or using an ambiguous device mask blocks launch.

The experimental core was built with the direct K2 API and keeps the existing
`_k2` Python dispatch and TP all-reduce in the candidate tree. This A/B keeps
`k2_ar=false`, so the ordinary outer `tensor_model_parallel_all_reduce`
remains active in both arms. Do not enable `k2_ar` for this candidate.

Example, after parent has staged the tree, verified its build SHA, and selected
four unique ports/output directories, run one process per arm:

```bash
export EXPECTED_CANDIDATE_SO=<candidate-_rocm_C.abi3.so-sha256>
export EXPECTED_CANDIDATE_CU=<candidate-mono_k2_rdna2.cu-sha256>
export CANDIDATE_TREE=/home/chenco_adm/w4a8_runs/mono-large-m-2026-10-24/core-inline-serve-tree
export BASE_TREE=/home/chenco_adm/vllm-rdna-mtp-mono
export GATE_SCRIPT=/home/chenco_adm/w4a8_runs/mono-large-m-2026-10-24/core-inline-serving-ab/serve_validate_mono_ab_gate.sh
export MODEL=/home/chenco_adm/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
export PLE_DIR=/home/chenco_adm/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4
export VENV=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final

ARM=control PORT=18875 TAG=inline-01-control RUNS_ROOT=/home/chenco_adm/w4a8_runs/mono-inline-ab/01-control bash run_one_arm.sh
ARM=candidate PORT=18876 TAG=inline-02-candidate RUNS_ROOT=/home/chenco_adm/w4a8_runs/mono-inline-ab/02-candidate bash run_one_arm.sh
ARM=candidate PORT=18877 TAG=inline-03-candidate RUNS_ROOT=/home/chenco_adm/w4a8_runs/mono-inline-ab/03-candidate bash run_one_arm.sh
ARM=control PORT=18878 TAG=inline-04-control RUNS_ROOT=/home/chenco_adm/w4a8_runs/mono-inline-ab/04-control bash run_one_arm.sh
```

Run each line serially. Never reuse a run directory or port. The script only
checks a serving arm; it does not build, sync, merge or publish anything.

## Completed A/B result

The four arms completed on the same boot and passed the recorded serving
gates. Curated per-request and per-cell outputs, provenance, fixed-prompt
log-probability comparisons, and an ABBA repeat comparison are in
[`../core-inline-serving-reviewed/`](../core-inline-serving-reviewed/).
The repeat comparison is descriptive: there are two serving runs per variant,
so it does not claim statistical significance.

Two-run means show whole-run input/output throughput gains in all four cells:
29.6%, 6.6%, 4.2%, and 1.9% for short c1, short c8, long c1, long c8. Mean
per-request decode rate changes are +30.2%, +6.7%, +11.3%, and +9.3%; median
stream ITL changes are −17.5%, −11.5%, −17.4%, and −9.9%. Per-request prefill
rate changes are +0.2%, +3.1%, +0.1%, and −5.4%. These are derived from mean
request metrics; the TTFT-derived prefill wave proxy is not actual prefill
throughput.

Mean TTFT changes are −0.2%, −2.4%, −0.1%, and +4.2% (long c8 is about 0.77 s
slower). Acceptance length changes +0.170, −0.093, −0.287, and −0.065 tokens.
This candidate reduces ITL and improves decode/whole-run throughput, but
acceptance falls in three cells and long c8 TTFT increases. The results do not
support a claim that every user metric improves. Fixed-prompt token/log-prob
comparisons and run-to-run drift are retained in the reviewed evidence. Exact
analysis, provenance, repeat-comparison and evidence-export script bytes are
also retained as `.py.txt` snapshots under `script-snapshots/`; restore the
suffix before execution if reproduction is needed. Their digests are in
`script-snapshots/SHA256SUMS.txt`.

Serving gates do not establish exact native-MTP equivalence for every K2 row
count. A separate strict quantized-target rank-local comparison failed route-ID
checks at M3/M7/M24 and exceeded the M24 output tolerance (max error
0.19696044921875 vs 0.005). The result and its scope are recorded in
[`RESULTS.md`](RESULTS.md); the quantized mismatch remains unresolved.
