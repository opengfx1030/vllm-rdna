# TRUE FULL gfx1030 handover — 2026-09-10

Pick up here. Do not re-litigate closed causes. Local tree is source of truth; rsync to `.176` before any remote bench.

## Goal

Qwen3.8-27B-AWQ-INT4 (cyankiwi `63768c10df38c0395e12ef49edac1bd539eaeeea`) on `.176`, TP=2, branch `rdna_extras` in `opengfx1030_vllm-rdna`. Production path is **TRUE FULL** HIP CUDA graphs (FA-RDNA2 + AWQ RDNA2 + HIP GDN/KV): **not** `--enforce-eager`, **not** PIECEWISE-only. Do not disable FULL for batch>1, GQA, FA, or HIP AWQ.

Required: prefix caching AND mixed decode+chunked prefill. `VLLM_FORCE_CUSTOM_ALL_REDUCE=1`, `VLLM_USE_RDNA2_FA=1`.

Greedy: `tools/probe_greedy_correctness.py` — Paris in first 32, `1+1=` starts with 2, Berlin in first 40, garbage bar (`duct`, `\ufffd`, `{{{{`, bang storms) on the **full** completion. Concurrent slots use **different** prompts. Sequential greedy after a concurrent burst must still PASS.

Log TTFT, prefill tok/s, decode tok/s for **1k/512** and **16k/1k** at c=1,4,8. Do **not** run 16k c=8 until 16k c=4 **and** seq-after are green.

Plan: `~/.grok/sessions/%2FUsers%2Fkletorch%2FProjects%2Finfrastructure%2Fgfx1030_optimized/01a081a5-aed9-76a3-a106-d02d65e07297/goal/plan.md`

---

## Hardware / paths

| Item | Value |
|------|--------|
| Host | `chenco_adm@192.168.1.176` |
| SSH | `-i ~/.ssh/id_ed25519_ansible` |
| Local tree | `opengfx1030_vllm-rdna/` branch `rdna_extras` |
| Remote mirror | `/home/chenco_adm/opengfx1030_vllm-rdna` (rsync, exclude `.git`) |
| Venv | `/home/chenco_adm/Apps/vllm/venv-7.14.0` (PyTorch 2.12.0+rocm7.14.0) |
| ROCm SDK | `/opt/rocm/core-7.14` |
| Model | `/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea` |
| Serve | `scripts/serve_gfx1030_full.sh` `PORT=18094` `HIP_VISIBLE_DEVICES=0,1` `MAX_MODEL_LEN=32768` `KV_CACHE_MEMORY=6000000000` `MAX_NUM_SEQS=8` `VLLM_ROCM_MIXED_LOG=1` |
| Logs | `/tmp/gfx1030_truefull_livetail/` on `.176` |
| Scratch (local evidence) | `/var/folders/nf/tws6rmcx2h5ghlrqnvrbgf6r0000gn/T/grok-goal-12c2c5d0c5aa/implementer` |
| Kill | **by PID only**. Never `pkill -f`. |

Hybrid math (do not re-derive): 64 layers = 48 GDN + 16 FA; hidden=5120, Q=24, KV=4, D=256; TP=2 → Q=12, KV=2. Unified page 784 tokens. `bytes_per_block` = 16 FA layers × 1,605,632 = 24.50 MiB. Scheduler charges FA pages + GDN pages from one pool (2-group). 16k c=8 may not fit 6e9 under 2-group; 16k c=1/4 do.

---

## What works (proven)

TRUE FULL backends on serve26/serve27 logs:

- `cudagraph_mode=FULL_AND_PIECEWISE`, capture `[1,2,4,8,16]`
- `enforce_eager=False`
- `RDNA_ATTN` (FA-RDNA2 HIP)
- `RDNA2W4A16LinearKernel`
- prefix caching + mamba `align`
- `VLLM_FORCE_CUSTOM_ALL_REDUCE=1` (see rdna_ar note below)

| Cell | Best proven | Serve | Notes |
|------|-------------|-------|--------|
| Sequential greedy | **3/3** | 26, 27 | Paris / `2…` / Berlin |
| Seq-after short | **3/3** | 26, 27 | Graphs healthy after short mixed |
| 1k/512 c=1 | **1/1** | 26, 27 | TTFT ~2.20s, prefill ~465, decode ~23.3 |
| 1k/512 c=4 | **4/4** serve26; **3/4** serve27 | | serve27 miss is `2000…` zeros on the 1023-token `1+1=` slot, **not** `duct` |
| 1k/512 c=8 + seq-after | **8/8** and **3/3** | **serve26 only** | Unbind graph pool `(0,1)` in `eager_alloc_isolation` |

serve26 1k/512 c=8 metrics: TTFT 12.52s, prefill 125 tok/s, decode 14.6 tok/s/req, agg 84.8.

HIP kernels themselves are numerically fine in eager. Prefill skip-compiled logits are fine (first generated token is the right city).

---

## What does not work

| Cell | Result | Pattern |
|------|--------|---------|
| 16k/1k c=1 | **0/1** `Parisduct` (serve26) | First token Paris, rest `duct`. TTFT 57.7s (whole 16k is skip-compiled ~284 prefill tok/s). Seq-after **0/3 duct** |
| 16k/1k c=4 | **0/4** | Later reqs duct from first token or immediately after |
| 16k/1k c=8 | not run | Gated on c=4 + seq-after |
| 1k/512 c=8 on **serve27** | **3/8** | First 3 PASS; reqs 3–7 `Romeduct`/`Madridduct`/`Ottawaduct`/… **Regression** from 128 MiB persist floor |

`rdna_ar` boot self-test fails (`mean 1.00, expected 3.0` — weak P2P). Collectives fall back to **PYNCCL**. `VLLM_FORCE_CUSTOM_ALL_REDUCE=1` is still set. This is a **perf** issue, not the `duct` mechanism.

`tools/probe_greedy_correctness.py` has **no** `--concurrency` flag. Concurrent greedy is the streaming bench (`bench_streaming_ttft.py`), not that probe.

---

## Root cause (do not re-open closed hypotheses)

**Class:** skip-compiled (NONE / `blocked_by_prefill`) `torch.zeros` / persist grow lands in HIP graph-private pool **`(0,1)`** and overwrites FULL decode workspace.

How to read the symptom:

1. Prefill/mixed runs eager skip-compiled → first token correct (Paris/Madrid/Rome).
2. FULL graph replay for decode token 2+ → `duct`.
3. After that, even a 5-token greedy ducts → graphs stay poisoned. **Not** “16k KV is wrong.”
4. KV at 16k c=1 was **~22%**. Not capacity.
5. Raw HIP `isCapturing` is **False** during isolation. Not a gfx1030 false-positive capture.

Pools:

| Pool | Id | Role |
|------|----|------|
| HIP graph private (`torch.cuda.graph(pool=None)`) | `(0,1)` | FULL persist/workspace. **Must not** take skip-compiled allocs |
| FULL capture MemPool | `(0,2)` | ATen temps during capture |
| Eager isolation MemPool | `(0,3)` | Skip-compiled prefill/mixed |

Isolation must **unbind `(0,1)+(0,2)+(0,3)` then `beginAllocate(0,3)`**. Unbinding only `(0,2)/(0,3)` left the thread on `(0,1)` — that was serve26’s 1k c=8 fix.

16k c=1 is **pure prefill**, not mixed `n_dec>=2`. Chunks are **1568 tokens** (`2×784` hybrid pages), then 704. 1k mixed never allocated that size.

FA persist `O` for one 784-page: `784×12×256 = 2,408,448` elements. Serve27 logged that grow **during 1k c=8**, inflated to 128 MiB by the floor → 3/8 duct.

AWQ C for Qwen3.8 TP=2 (`intermediate=17408`): 1k ≈ 35 MiB, 16k 1568-token chunk ≈ 55 MiB. Old 32 MiB pin does not cover that 1k→16k grow.

---

## Closed / do not ship

Do **not** bring these back:

- `VLLM_ROCM_NO_MIXED_BATCH=1`
- Prefix cache off
- Share GDN last-state with FA last `block_id`
- Immortal small persist / immortal 5.5 GiB (starved SiluAndMul)
- CPU `seq_lens` on **pure decode** / `seq_lens_cpu_upper_bound` on prefill
- Dim-0-only GDN pin
- Eager side stream (serve22: sequential 3/3 then 1k c=1 `Parisduct`)
- Recapture-after-mixed (dummy capture writes live KV)
- Dummy skip-compiled 2048 pretouch after capture (serve24 OOM, 0 B free on 10e9)
- **128 MiB persist floor + 2× need** (serve27: 1k c=8 3/8)

10e9 KV pin left 0 B free after capture. Bench pin is **6e9**. Dummy 2048 pretouch is cancelled.

---

## What each serve proved

| Serve | Change | 1k c=8 | 16k c=1 | Keep? |
|-------|--------|--------|---------|--------|
| 20/21 | GDN T/NT + FA q pretouch | mixed | `Parisduct` | pretouch yes; not sufficient for 16k |
| 22 | Eager side stream | n/a (c=1 died) | — | **reverted** |
| 23 | pretouch, 10e9 | first 7/8 phrasing; rerun 6/8 duct | — | 10e9 dropped |
| 24 | dummy 2048 pretouch | OOM | — | **removed** |
| 25 | 6e9, pretouch, no dummy | 2/8 duct | duct | 6e9 kept |
| **26** | **unbind `(0,1)`** in isolation + post-capture | **8/8 + seq-after 3/3** | **Parisduct** + seq-after 0/3 | **KEEP unbind** |
| **27** | 128 MiB persist floor, `expandable_segments:False`, finally-unbind `(0,1)`, CPU `seq_lens` on all prefill | **3/8 duct** | not run | **revert 128 MiB and False**; keep CPU `seq_lens` + finally-unbind |

serve26 PID was 838699 (dead). serve27 PID was 855563 (killed). **Port 18094 is free.**

---

## Code state at handover (local + rsynced)

Uncommitted on `rdna_extras` (do not commit unless the user authorizes):

| File | What |
|------|------|
| `vllm/v1/worker/gpu/cudagraph_utils.py` | Isolation unbinds `(0,1)/(0,2)/(0,3)` then `beginAllocate(0,3)`. After capture, `endAllocateToPool((0,1))`. **Finally also `_unbind_known()`** so FULL replay is not left on the eager pool. Pretouch GDN×1 + FA contig ×16 after freeze. |
| `vllm/v1/attention/backends/rdna_attn.py` | Eager FA tables pre-sized to `max_num_seqs`. FA contig q pin 2048×16×256. **CPU `seq_lens` copy on all prefill** (not only mixed) so varlen attends full prefix, not the 1568-token chunk. |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | T pin 2048, NT pin 64, `rearrange_fused` 2048×8192, `decode_ssm_save` dim0 16. Dual capture/eager scratch tables. |
| `vllm/utils/rocm_graph_keepalive.py` | `hip_stream_is_capturing()`, capture guard, 256 MiB hybrid keepalive math. |
| `csrc/rocm/rdna2_graph_keepalive.cuh` | Dual `Rdna2PersistBuf` capture vs eager + freeze. **Local/rsynced but .so may still be serve27:** eager grow pins **leading dim0 to 2048** when `need>65536` and `d0<2048`. **No 128 MiB floor.** `fprintf` `[rdna2_persist] eager grow need=…` |
| `scripts/serve_gfx1030_full.sh` | `expandable_segments:True` restored. KV default 7e9; benches use `KV_CACHE_MEMORY=6000000000`. |

Python isolation/`seq_lens` is live on the remote tree without rebuild. The **2048-row persist pin is in the header on disk; `_rocm_C.abi3.so` at handover is still serve27** (`71437656` bytes, mtime **Sep 10 10:43**). A rebuild was in flight (`pip install -e .` against venv-7.14.0, `PYTORCH_ROCM_ARCH=gfx1030`). Check:

```bash
ls -l /home/chenco_adm/opengfx1030_vllm-rdna/vllm/_rocm_C.abi3.so
# want mtime after 10:43 and size != 71437656 if the 2048-row rebuild landed
```

If the in-flight rebuild finished, launch serve28. If it did not, finish it (touch the `.cu` files that include the header, then `pip install -e . --no-build-isolation --no-deps` with the cold-start env in AGENTS.md). Confirm the new pin by a grow line like `need=2408448 grown=6291456` (784→2048 rows), **not** `grown=67108864` (128 MiB).

---

## Next session — exact steps

1. Confirm `.so` is the 2048-row pin, not the 128 MiB floor.
2. Launch:

```bash
cd /tmp
export VLLM_ROCM_MIXED_LOG=1
nohup env \
  MODEL=/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea \
  PORT=18094 HIP_VISIBLE_DEVICES=0,1 \
  MAX_MODEL_LEN=32768 KV_CACHE_MEMORY=6000000000 MAX_NUM_SEQS=8 \
  VLLM_ROCM_MIXED_LOG=1 \
  /home/chenco_adm/opengfx1030_vllm-rdna/scripts/serve_gfx1030_full.sh \
  > /tmp/gfx1030_truefull_livetail/serve28.log 2>&1 &
echo $!
```

3. Gating log: `RDNA_ATTN`, `RDNA2W4A16`, `FULL_AND_PIECEWISE`, `enable_prefix_caching`, mamba `align`, `enforce_eager=False`, `eager isolation HIP_is_capturing=False eager_pool=(0, 3) global_graph_pool=(0, 1)`, pretouch `gdn=1 fa_layers=16`. No 128 MiB grow.
4. Sequential greedy 3/3, then 1k/512 c=1, c=4, c=8, then seq-after. **Must restore serve26’s 1k c=8 8/8** before touching 16k.
5. Only then 16k/1k c=1 + seq-after; only then c=4 + seq-after; only then c=8.
6. Copy benches to scratch: `bench_1k512_c{1,4,8}.txt`, `bench_16k1k_c{1,4,8}.txt`, `probe_seq*.txt`.

If 1k c=8 is 8/8 again and 16k c=1 is still `Parisduct`, the remaining leak is **not** FA/AWQ persist dim0 (those are pinned to 2048). Hunt unnamed skip-compiled temps at 1568 / all-reduce staging on a **non-isolated thread** (RCCL proxy). Do **not** raise the persist floor again.

If 1k c=8 regresses, the 2048-row pin still grew during mixed — pretouch FA `O` and AWQ `C` **after freeze under isolation** (dummy `rdna2_persist_zeros` at `{2048,H,D}` and `{2048,N}`) so the first request never grows.

---

## Rebuild env (venv-7.14.0)

```bash
source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate
cd /home/chenco_adm/opengfx1030_vllm-rdna
export SETUPTOOLS_SCM_PRETEND_VERSION=0.20.1.dev99
export VLLM_TARGET_DEVICE=rocm VLLM_USE_PRECOMPILED=0
export PYTORCH_ROCM_ARCH=gfx1030 CMAKE_HIP_ARCHITECTURES=gfx1030 AMDGPU_TARGETS=gfx1030
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14 HIP_ROOT_DIR=/opt/rocm/core-7.14
export CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$PATH
export VLLM_PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python
export MAX_JOBS=16 CMAKE_BUILD_TYPE=RelWithDebInfo
export CPATH=/opt/rocm/core-7.14/include LIBRARY_PATH=/opt/rocm/core-7.14/lib
# Do NOT set LD_LIBRARY_PATH for the build
pip install -e . --no-build-isolation --no-deps
```

Runtime `LD_LIBRARY_PATH` is set by `serve_gfx1030_full.sh` (`_rocm_sdk_libraries` first).

---

## Rsync

```bash
rsync -avz \
  --exclude='.git/' --exclude='.deps/' --exclude='build/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.abi3.so' \
  --exclude='.cache/' --exclude='.triton/' --exclude='.vllm-cache/' --exclude='.tmp/' \
  -e "ssh -i $HOME/.ssh/id_ed25519_ansible" \
  ./opengfx1030_vllm-rdna/ \
  chenco_adm@192.168.1.176:/home/chenco_adm/opengfx1030_vllm-rdna/
```

Do not rsync `.so`. Remote produces it.

---

## Bench commands

```bash
source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate
export TOKENIZERS_PARALLELISM=false
cd /tmp
TOOLS=/home/chenco_adm/opengfx1030_vllm-rdna/tools
URL=http://127.0.0.1:18094
MODEL=.../snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea

python "$TOOLS/probe_greedy_correctness.py" --url "$URL/v1/completions" --model "$MODEL"
python "$TOOLS/bench_streaming_ttft.py" --url "$URL/v1/completions" --model "$MODEL" \
  --input-len 1024 --output-len 512 --concurrency 8 --tag "1k512-c8"
# 16k only after 1k c=8 8/8 and seq-after PASS
python "$TOOLS/bench_streaming_ttft.py" --url "$URL/v1/completions" --model "$MODEL" \
  --input-len 16384 --output-len 1024 --concurrency 1 --tag "16k1k-c1"
```

Gate 16k c=4 on `OUTPUT_CHECK 1/1` **and** `RESULT PASS` seq-after. Gate 16k c=8 on c=4 4/4 **and** seq-after PASS.

---

## One-paragraph status for the human

TRUE FULL HIP graphs work for sequential greedy and 1k/512 through c=8 **when skip-compiled allocs stay off graph pool `(0,1)`** (serve26). 16k/1k still dies because 1568-token skip-compiled chunks grow persist/scratch that 1k never sized; first token is correct, decode replay is `duct`, then all later greeds duct. A 128 MiB persist hammer (serve27) made 1k c=8 worse. Next: finish the 2048-row persist pin rebuild, restore serve26 isolation + expandable segments, re-prove 1k c=8 8/8, then 16k. Do not disable FULL, FA, AWQ, prefix cache, or mixed batch.
