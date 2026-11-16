# Scoreboard: arm `conc-off`

**Window: quiet protocol (QUIET_HOLD=1, no image build at the end).**

- recipe `27b-hi2-tp2`, overrides `VLLM_RDNA_AR_HIER=0 TUNABLEOP_PREFER=`, GPUs 2,3 (PCI d9 dc), tree sha `b2d569c402`
- started 2026-11-15 15:29:05, build running at start/end: 0/0, QUIET_HOLD=1

## 1-2. Performance (per request, then aggregate)

| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean | ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |
|---|---:|---|---:|---:|---|---:|---:|---:|

## 3. Quality

- short PPL 5.730; long-context PPL - over passages of None tokens
- baseline `/home/chenco_adm/w4a8_runs/scoreboard/base`, floor (baseline booted again) `/home/chenco_adm/w4a8_runs/scoreboard/base2`

## 4. Validity

- READY after 181 s; greedy probe: RESULT PASS (3/3); after load: RESULT PASS (3/3)
- PPL probe: 5.73 RESULT PASS
- prefix cache: cold TTFT 4.76 s, warm TTFT 0.42 s, speedup 11.4x -> PASS; prefix cache: queries +6386, hits +3072, warm hit rate 96.2% (hit granularity 2048 tokens)
- concurrency: RESULT PASS: 0 failing, 2 near-tie, 0 soft, 0 solo-noise, 46 identical of 48 checks
- KFD evictions: all zero (1 windows); fault markers 0, dmesg faults 0
- bus d9: junction max 82 C, worst 5 s mean 79.6 C (limit 90), VRAM max 25444 MiB
- bus dc: junction max 77 C, worst 5 s mean 76.4 C (limit 90), VRAM max 25452 MiB
