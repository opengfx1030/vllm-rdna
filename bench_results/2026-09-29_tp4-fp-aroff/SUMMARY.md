# TP=4 W4A16 FULL-HIP, `RDNA_AR=0`, FULL_AND_PIECEWISE — bring-up result

**Date:** 2026-09-29
**Tree:** `vllm-rdna-0.28.0` @ `8de502574` (`w4a8-wiring`) → `par1-cs25:/home/chenco_adm/vllm-rdna-0.28.0`
**Model:** `cyankiwi/Qwen3.8-27B-AWQ-INT4` (compressed-tensors W4A16)
**Venv:** `/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0`
**Run dir (remote):** `/home/chenco_adm/w4a8_runs/2026-09-29_tp4-fp-aroff/`

## Verdict

- **Boot clean: NO.** The chassis hard-reset during the post-compile / cudagraph-capture phase. New `PCI SERR` logged.
- **Coherence: not measured** (server never reached READY).
- **16k/1k numbers: not measured** (never READY).
- **Fork AR is exonerated:** the reset reproduces with `VLLM_RDNA_AR=0`; active all-reduce was `['PYNCCL']` only.

## What ran (exact env/args)

Driver `tools/rdna2_028/fp_aroff_27b.sh` (new) → launcher `scripts/serve_gfx1030_27b_dense.sh` (new `RDNA_AR` knob):

```
MTP=0 W4A8=0 RDNA_AR=0 EAGER=0 ATTN=fa TP=4 PORT=18240
KV=8000000000 SEQS=8 MAXBAT=2048 CG_MODE=FULL_AND_PIECEWISE
VLLM_CACHE_ROOT=$T/cache/hip-fp-aroff
TORCHINDUCTOR_CACHE_DIR=$T/cache/hip-fp-aroff/inductor
TRITON_CACHE_DIR=$T/cache/hip-fp-aroff/triton
TORCH_EXTENSIONS_DIR=$T/cache/hip-fp-aroff/extensions
```

Effective serve args (from `serve.log`):
`--attention-backend RDNA_ATTN --tensor-parallel-size 4 --dtype float16 --block-size 1024
--kv-cache-memory-bytes 8000000000 --gpu-memory-utilization 0.9 --max-num-batched-tokens 2048
--max-num-seqs 8 --enable-prefix-caching --mamba-cache-mode align --language-model-only
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"compile_ranges_endpoints":[]}`
No `--max-model-len` (model default).

Caches were wiped (`rm -rf $T/cache/hip-fp-aroff`) before launch, so no contaminated W4A8 graph
could replay. `serve.log` confirms the fresh root: `cache/hip-fp-aroff/torch_compile_cache/...`.

## Timeline

| Time (CEST) | Event |
|---|---|
| 14:38:00 | serve launched (driver preflight: PCI SERR baseline = **12**, uptime 57 min) |
| 14:39:11 | weights loaded, 5.01 GiB/GPU (all 4 ranks) |
| 14:39:06 | all-reduce selection: active `['PYNCCL']` |
| 14:39:26 | Dynamo bytecode transform: 13.20 s |
| 14:40:14 | inductor graph for compile range (1, 2048): 46.87 s |
| 14:40:18 | `torch.compile took 65.40 s`; `[rdna2_persist] eager grow` ×4; Triton deprecation warnings |
| 14:40:12 → 14:48:13 | `shm_broadcast: No available shared memory broadcast block found in 60 seconds` every 60 s (worker busy post-compile) |
| 14:48:13 | last worker log line |
| ~14:46:47 → 14:50:42 | host hangs / resets; no clean-shutdown journal entries |
| **14:50:42** | **new SEL `e68 | Critical Interrupt | PCI SERR | Asserted`** |
| 14:51:25 | system boot (journal boot id 0) |

**Cold-compile duration:** inductor `torch.compile` = **65.40 s** (Dynamo 13.20 s + range compile 46.87 s);
model load ≈ 5 s. The reset came ≈10.5 min *after* compile finished, in the silent post-compile
worker phase (cudagraph capture / full-graph replay prep), i.e. the previous "the long compile window
resets the chassis" framing is too narrow — the inductor compile itself completed quickly.

## IPMI / SEL delta

- Before (14:38:00): `PCI SERR` = **12**, `Critical Interrupt` = 17, `Correctable ECC` = 10.
- After (post-reboot): `PCI SERR` = **13**.
- New entry: `e68 | 09/29/2026 | 02:50:42 PM CEST | Critical Interrupt | PCI SERR | Asserted`.
- **Note the id is reused** — the SEL already had an old `e68` (09/23, `PCI PERR`), so a set-diff by
  record id shows no new id; the reliable signal is the count 12→13 plus the new `14:50:42` timestamp.
- Unlike the earlier resets (e64+e65 and e66+e67 both paired with `Memory | Correctable ECC | CPU 0 DIMM 0`),
  **e68 has no paired ECC entry** — this reset was a pure PCI SERR. (Matches the DIMM-ECC-vs-fabric
  ambiguity: the DIMM is a suspect, but not proven for this event.)

## Fork-marker audit (over the partial `serve.log`)

| Marker | Expected | Observed |
|---|---|---|
| `RDNA2 W4A8 sdot4 path active` | absent | **absent** (0) |
| `W4A8-DEBUG` | absent | **absent** (0) |
| `rdna_ar:` lines | none (`VLLM_RDNA_AR=0`) | **none** (0) |
| `Custom allreduce force-enabled` | absent | **absent** (0) |
| active all-reduce backends | PYNCCL only | `Using ['PYNCCL'] all-reduce backends … out of potential backends: ['RDNA_ONESHOT', …]` |
| attention backend | RDNA_ATTN | `Using RDNA_ATTN backend` |

`RDNA_ONESHOT` appears only in the *potential* backends list, never in the active list — the knob
(`RDNA_AR=0` → `VLLM_RDNA_AR=0`) works and the fork PCIe-peer AR is definitively out of the picture.

## Conclusion

The TP=4 F&P hard reset is **not caused by the fork's PCIe-peer all-reduce.** It reproduces with the
AR path disabled and the only active collective is RCCL/PYNCCL. The reset occurs during the
cudagraph-capture/post-compile phase and is accompanied by a fabric-level `PCI SERR` (no DIMM ECC
companion this time). Next investigation axes: DIMM/fabric stability (the DIMM was already the
suspect), and the TP=4 FULL_AND_PIECEWISE capture path itself — no further runs were attempted per
the "a reset is evidence, stop and report" rule.

## Evidence files

`remote_logs/` (copied verbatim from the box):
- `driver.log`, `ipmi.log` — driver progress + pre-run IPMI snapshot
- `serve.log` — partial serve log (122 lines, through 14:48:13)
- `before.sel.txt` — 14:38:00 SEL snapshot (SERR=12)
- `after_reboot.sel.txt`, `after_reboot.elist.txt` — post-reboot SEL (SERR=13, includes e68)
- `../args.txt` — non-default args + all-reduce backend line

## Reproduction

```bash
rsync -avz --exclude='.git/' --exclude='cache/' --exclude='*.abi3.so' \
  -e "ssh -i ~/.ssh/id_ed25519_ansible" \
  ./vllm-rdna-0.28.0/ chenco_adm@par1-cs25:/home/chenco_adm/vllm-rdna-0.28.0/
ssh -i ~/.ssh/id_ed25519_ansible chenco_adm@par1-cs25 \
  'cd /home/chenco_adm/vllm-rdna-0.28.0 && TAG=2026-09-29_tp4-fp-aroff bash tools/rdna2_028/fp_aroff_27b.sh'
```

**Do not re-run without a hardware-stability answer** — this is the 4th reset in the same class today
(e63/e65/e67/e68).
