# Dense MTP K2 smoke at 0.92 GPU memory target

This records a correctness and serving smoke for the experimental dense FP16
MTP expert route. It is not a performance comparison against the module path.

|Item|Result|
|---|---|
|Host boot|`1d5b3cf3-6460-449b-a56d-9b68b7d1d155`|
|GPUs|6–9, TP4/EP4|
|Runtime|frozen `venv-7.14.0_0.31.0-final`|
|Binary SHA-256|`6e8e486e548213d159236298a24b0f9ffbc18e49ccc234dcace7e8c1d59f4df3`|
|Model|Qwen3.8 Flash-Next AWQ snapshot `0939125b929543a783ce700c90e36dd1a575c00c`|
|Memory target|`--gpu-memory-utilization 0.92`|
|Startup|READY in 200 s|
|Model weights|20.62 GiB per rank|
|KV cache|271,746 tokens; 4.07 GiB reported available KV memory|
|Sampled steady VRAM use|89–90% on active GPUs after the concurrency probe|
|Dense route exercised|4 K2 layout logs and 4 dense K2 calls at M=8, one per rank|
|Greedy smoke|3/3 passed|
|PPL probe|4.81, passed (261 tokens)|
|Prefix-cache probe|passed; cold TTFT 5.81 s, warm 0.68 s (8.6×)|
|Concurrency/parity probe|passed; 0 failing, 14 near-ties, 1 soft divergence, 33/48 identical checks|
|After-load hook and probe|passed|
|Fault markers|0 in server log; 0 new amdgpu/KFD fault markers|

The prior M=8 dense K2 startup fault was traced to treating the packed
`[gate; up]` expert matrix as though the scratch row also had width `2I`.
The dense P5 writer now uses `I` for the up-half scratch offset and tile
count. This run exercised that path on all four ranks without a fault.

The 0.92 value is the vLLM memory-utilization target, not a promise that
92% of VRAM is allocated or that a particular context length fits. This run
reported 4.07 GiB available for KV cache. No throughput or comparative
performance claim is made here; the next gate is longer correctness and
performance comparison against the module path under the same recipe.

Raw `summary.txt`, `load_hook.txt`, `concurrency.txt`, `parity.json`, and
server logs are retained on `par1-cs25` under
`/home/chenco_adm/w4a8_runs/mtp-mono-ab/dense-mtp-build-1d5b/serve-dense-mtp-trace3-gmem092/`.
