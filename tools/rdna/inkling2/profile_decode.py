#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile prefill / decode steps of the real Inkling-2-Mini checkpoint (1 GPU).

Boots the checkpoint like ``real_check.py`` (offline ``LLM``, eager, fp16) but with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the model runs in this process, then drives
the engine one ``LLMEngine.step()`` at a time:

* pass A  clean wall time per step (no instrumentation) + a forward-only timer that
          separates CPU launch time from GPU completion time;
* pass C  ``cProfile`` of 8 decode steps (python frames);
* pass B  ``torch.profiler`` (CPU + GPU), one session per step.  The model is
          annotated with ``record_function`` ranges (``I2:<group>.<part>``) and every
          GPU kernel is attributed to the innermost range enclosing its launch call
          (hip runtime event <-> kernel via the correlation id of the chrome trace).

Phases: ``prefill`` of N x 22-token prompts, then 8 consecutive ``decode`` steps, for
batch 1 and batch ``--batch`` (default 4).

On the native HIP path (the default on a GPU) the ranges are one per HIP op
(``I2:attn.*``, ``I2:glue.*``, ``I2:moe.*``, ``I2:lin.*``); ``--path reference``
profiles the eager torch path (the baseline).  ``--long-prefill N`` also times the
prefill step of one ``N``-token prompt.

    HIP_VISIBLE_DEVICES=2 python profile_decode.py --ckpt ~/hfcache/inkling2-mini \
        --out-dir ~/w4a8_runs/inkling2-work/Inkling2Profile/run1
"""

from __future__ import annotations

import os

# the model must live in this process: set before vllm is imported
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import contextlib
import cProfile
import functools
import io
import json
import pstats
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import regex as re
import torch
from torch.profiler import ProfilerActivity, profile, record_function

PROMPT_LEN = 22

# ---------------------------------------------------------------------------
# label -> report group
# ---------------------------------------------------------------------------

GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("MoE (route/align/gemm1/act/gemm2/other)", ("I2:moe.",)),
    (
        (
            "Attention linears (HIP: fused input pack + o_proj pack; reference: "
            "q/k/v/o/gate/rel/sink/dynconv)"
        ),
        ("I2:lin.", "I2:attn.lin_"),
    ),
    (
        (
            "Attention non-linear (HIP: kv_prep/attention/sconv; reference: "
            "conv_taps/sconv/kv_write/kv_read/softmax/qk glue)"
        ),
        ("I2:attn.",),
    ),
    (
        "Layer glue (reference: post-attn/mlp conv, residual adds, casts)",
        ("I2:layer.",),
    ),
    (
        (
            "Blocks glue (HIP: attn_res/msg gates/rmsnorm/embed/copies; reference: "
            "attn_res, msg_res, msg_gate, rmsnorm, embed)"
        ),
        ("I2:glue.", "I2:blk.", "I2:model."),
    ),
    ("lm_head + softcap", ("I2:lm_head",)),
    ("Sampler (GPUModelRunner.sample_tokens)", ("I2:sampler",)),
    ("Input prep (GPUModelRunner.prepare_inputs)", ("I2:prep_inputs",)),
    ("Unlabelled (runner glue outside the above)", ("unlabelled", "unmatched")),
]


def group_of(label: str) -> str:
    for name, prefixes in GROUPS:
        if any(label.startswith(p) for p in prefixes):
            return name
    return "Unlabelled (runner glue outside the above)"


# kernel-name category (view independent of the python label)
def kernel_category(name: str) -> str:
    n = name.lower()
    if "moe_nvfp4" in n or "moe_mxfp8" in n or "nvfp4" in n or "mxfp8" in n:
        return "moe quant gemm kernels"
    if "moe_align" in n or "moealign" in n or "sort" in n or "topk" in n:
        return "moe align / sort / topk"
    if n.startswith("cijk") or "tensile" in n or "rocblas" in n or "gemm" in n:
        return "rocBLAS/Tensile gemm"
    if "gemv" in n:
        return "rocBLAS gemv"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    if "reduce" in n or "softmax" in n or "norm" in n:
        return "reduce/softmax/norm"
    if any(s in n for s in ("index", "gather", "scatter", "embedding", "cat")):
        return "index/gather/scatter/cat"
    if "elementwise" in n or "unrolled" in n or "vectorized" in n or "copy" in n:
        return "elementwise/copy"
    return "other"


# ---------------------------------------------------------------------------
# instrumentation (record_function labels), installed only for the torch pass
# ---------------------------------------------------------------------------


def _wrap(owner, name: str, label: str) -> None:
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def wrapped(*a, **k):
        with record_function(label):
            return orig(*a, **k)

    setattr(owner, name, wrapped)


def install_labels_reference(model: torch.nn.Module) -> None:
    from vllm.model_executor.layers.fused_moe import moe_align_block_size as align_mod
    from vllm.models.inkling2_loop.amd import attention as A
    from vllm.models.inkling2_loop.amd import blocks as B
    from vllm.models.inkling2_loop.amd import model as M
    from vllm.models.inkling2_loop.amd import moe as E
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    # MoE: innermost label wins, so ``moe.other`` is the remainder of forward
    _wrap(E.Inkling2LoopMoE, "forward", "I2:moe.other")
    _wrap(E.Inkling2LoopTopKRouter, "forward", "I2:moe.router")
    _wrap(E.Inkling2LoopExperts, "_apply_gate", "I2:moe.act")
    _wrap(align_mod, "moe_align_block_size", "I2:moe.align")
    orig_gemm = E.Inkling2LoopMoE._gemm

    @functools.wraps(orig_gemm)
    def gemm(self, stage, *a, **k):
        with record_function("I2:moe.gemm1" if stage == "w13" else "I2:moe.gemm2"):
            return orig_gemm(self, stage, *a, **k)

    E.Inkling2LoopMoE._gemm = gemm

    # attention
    _wrap(A.Inkling2Attention, "forward", "I2:attn.other")
    _wrap(A.Inkling2Attention, "_attend", "I2:attn.attend")
    _wrap(A.Inkling2Attention, "_scaled_clip", "I2:attn.qk_glue")
    _wrap(B, "l2_clip", "I2:attn.qk_glue")
    _wrap(A, "kv_read", "I2:attn.kv_read")
    _wrap(A, "kv_write", "I2:attn.kv_write")
    _wrap(A, "conv_taps", "I2:attn.conv_taps")
    in_kv = {"v": False}
    orig_kvs = A.Inkling2Attention._kv_stream

    @functools.wraps(orig_kvs)
    def kv_stream(self, *a, **k):
        in_kv["v"] = True
        try:
            with record_function("I2:attn.kv_stream"):
                return orig_kvs(self, *a, **k)
        finally:
            in_kv["v"] = False

    A.Inkling2Attention._kv_stream = kv_stream
    for cls in (A.StaticSconv, A.DynamicSconv):
        orig_ft = cls.forward_taps

        def make(orig_ft):
            @functools.wraps(orig_ft)
            def ft(self, *a, **k):
                label = "I2:attn.sconv" if in_kv["v"] else "I2:layer.sconv"
                with record_function(label):
                    return orig_ft(self, *a, **k)

            return ft

        cls.forward_taps = make(orig_ft)
    for name, m in model.named_modules():
        if isinstance(m, A._Linear):
            if name.endswith(("q_proj.linear", "k_proj.linear", "v_proj.linear")):
                label = "I2:attn.lin_qkv"
            elif name.endswith("o_proj.linear"):
                label = "I2:attn.lin_o"
            elif name.endswith("o_proj.gate_proj"):
                label = "I2:attn.lin_ogate"
            elif name.endswith("relative_position_bias.linear"):
                label = "I2:attn.lin_rel"
            elif name.endswith("sink_proj"):
                label = "I2:attn.lin_sink"
            elif name.endswith("weight_proj"):
                label = "I2:attn.lin_dynconv"
            else:
                label = "I2:attn.lin_other"
            orig_fwd = m.forward

            def make_lin(orig_fwd, label):
                def fwd(x):
                    with record_function(label):
                        return orig_fwd(x)

                return fwd

            m.forward = make_lin(orig_fwd, label)

    # blocks / layer / model
    _wrap(B.BlockAttentionResidual, "forward", "I2:blk.attn_res")
    _wrap(B.MessageAttentionResidual, "forward", "I2:blk.msg_res")
    _wrap(B.MessageGate, "forward", "I2:blk.msg_gate")
    _wrap(B.Inkling2RMSNorm, "forward", "I2:blk.rmsnorm")
    _wrap(B.UnweightedRMSNorm, "forward", "I2:blk.unw_norm")
    _wrap(M.Inkling2DecoderLayer, "forward", "I2:layer.glue")
    _wrap(M, "conv_taps", "I2:layer.conv_taps")
    _wrap(M.Inkling2Model, "forward", "I2:model.glue")
    _wrap(M.Inkling2Model, "embed_input_ids", "I2:model.embed")
    _wrap(M.Inkling2LoopTextForCausalLM, "forward", "WIN:forward")
    _wrap(M.Inkling2LoopTextForCausalLM, "compute_logits", "I2:lm_head")

    # runner
    _wrap(GPUModelRunner, "sample_tokens", "I2:sampler")
    _wrap(GPUModelRunner, "prepare_inputs", "I2:prep_inputs")
    _wrap(GPUModelRunner, "execute_model", "WIN:execute_model")


class _Labelled:
    """A packed-linear callable inside a ``record_function`` range; attribute reads
    (``names``, ``seg_n``, ...) go to the pack."""

    def __init__(self, inner, label: str) -> None:
        self.inner, self.label = inner, label

    def __call__(self, *a, **k):
        with record_function(self.label):
            return self.inner(*a, **k)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def install_labels_hip(model: torch.nn.Module) -> None:
    """Ranges of the native serving path: one per HIP op (the innermost range owns a
    kernel), plus the python of layer / attention / MoE / model as ``*.other`` /
    ``*.glue`` so the CPU side of the launch chain is attributed too."""
    from vllm import _custom_ops as ops
    from vllm.models.inkling2_loop.amd import attention as A
    from vllm.models.inkling2_loop.amd import model as M
    from vllm.models.inkling2_loop.amd import moe as E
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    for name, label in {
        "inkling2_attention": "I2:attn.attention",
        "inkling2_kv_prep": "I2:attn.kv_prep",
        "inkling2_sconv_static": "I2:attn.sconv",
        "inkling2_glue_attn_res": "I2:glue.attn_res",
        "inkling2_glue_msg_gate_down": "I2:glue.msg_gate",
        "inkling2_glue_msg_gate_mix": "I2:glue.msg_gate",
        "inkling2_glue_rms_norm": "I2:glue.rmsnorm",
        "inkling2_glue_embed_norm": "I2:glue.embed",
        "inkling2_glue_copy_states": "I2:glue.copy",
        "inkling2_glue_moe_route": "I2:moe.route",
        "inkling2_glue_swiglu": "I2:moe.swiglu",
        "inkling2_linear_argmax_reduce": "I2:lm_head",
    }.items():
        _wrap(ops, name, label)

    orig_gemm = E.Inkling2LoopMoE._gemm

    @functools.wraps(orig_gemm)
    def gemm(self, stage, *a, **k):
        with record_function("I2:moe.gemm1" if stage == "w13" else "I2:moe.gemm2"):
            return orig_gemm(self, stage, *a, **k)

    E.Inkling2LoopMoE._gemm = gemm
    _wrap(E.Inkling2LoopMoE, "forward_hip", "I2:moe.other")
    _wrap(A.Inkling2Attention, "forward_hip", "I2:attn.other")
    _wrap(M.Inkling2DecoderLayer, "forward_hip", "I2:layer.glue")
    _wrap(M, "static_conv_hip", "I2:attn.sconv")
    _wrap(M.Inkling2Model, "forward_hip", "I2:model.glue")
    _wrap(M.Inkling2LoopTextForCausalLM, "forward", "WIN:forward")
    _wrap(M.Inkling2LoopTextForCausalLM, "compute_logits", "I2:lm_head")
    _wrap(M.Inkling2LoopTextForCausalLM, "greedy_token_ids", "I2:lm_head")
    for m in model.modules():
        if isinstance(m, A.Inkling2Attention):
            m._attn_in = _Labelled(m._attn_in, "I2:lin.attn_in")
            m._o_pack = _Labelled(m._o_pack, "I2:lin.o_proj")

    # runner
    _wrap(GPUModelRunner, "sample_tokens", "I2:sampler")
    _wrap(GPUModelRunner, "prepare_inputs", "I2:prep_inputs")
    _wrap(GPUModelRunner, "execute_model", "WIN:execute_model")


def install_labels(model: torch.nn.Module) -> None:
    """Labels for the path the loaded model runs (``serving``: native HIP kernels)."""
    if getattr(model, "serving", False):
        install_labels_hip(model)
    else:
        install_labels_reference(model)


class FwdTimer:
    """Pass A2: forward / compute_logits launch-return time vs synced time."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []
        self._cur: dict[str, float] = {}

    def install(self) -> None:
        from vllm.models.inkling2_loop.amd import model as M

        cls = M.Inkling2LoopTextForCausalLM
        self._orig = (cls.forward, cls.compute_logits)
        timer = self

        def timed(orig, key):
            def f(self, *a, **k):
                t0 = time.perf_counter()
                out = orig(self, *a, **k)
                t1 = time.perf_counter()
                torch.accelerator.synchronize()
                t2 = time.perf_counter()
                timer._cur[f"{key}_launch_ms"] = (t1 - t0) * 1e3
                timer._cur[f"{key}_total_ms"] = (t2 - t0) * 1e3
                if key == "logits":
                    timer.rows.append(timer._cur)
                    timer._cur = {}
                return out

            return f

        cls.forward = timed(self._orig[0], "fwd")
        cls.compute_logits = timed(self._orig[1], "logits")

    def uninstall(self) -> None:
        from vllm.models.inkling2_loop.amd import model as M

        cls = M.Inkling2LoopTextForCausalLM
        cls.forward, cls.compute_logits = self._orig


# ---------------------------------------------------------------------------
# engine driver
# ---------------------------------------------------------------------------


class Rig:
    def __init__(self, llm, seed: int = 1234) -> None:
        from vllm import SamplingParams

        self.llm = llm
        self.eng = llm.llm_engine
        self.rng = random.Random(seed)
        self.counter = 0
        self.SamplingParams = SamplingParams

    def _prompt(self, n: int | None = None) -> dict:
        ids = [self.rng.randrange(1000, 50000) for _ in range(n or PROMPT_LEN)]
        return {"prompt_token_ids": ids}

    def run_phase(
        self, nreq: int, ndec: int, ctx=None, prompt_len: int | None = None
    ) -> dict:
        """``nreq`` fresh requests: 1 prefill step + ``ndec`` decode steps.

        ``ctx(kind, i)`` returns a context manager wrapped around each step.
        """
        sp = self.SamplingParams(
            temperature=0.0, max_tokens=ndec + 16, ignore_eos=True, detokenize=False
        )
        ids = []
        for _ in range(nreq):
            self.counter += 1
            rid = f"p{self.counter}"
            self.eng.add_request(rid, self._prompt(prompt_len), sp)
            ids.append(rid)
        steps = [("prefill", 0)] + [("decode", j) for j in range(ndec)]
        rows = []
        for kind, i in steps:
            torch.accelerator.synchronize()
            cm = ctx(kind, i) if ctx else contextlib.nullcontext()
            with cm:
                t0 = time.perf_counter()
                outs = self.eng.step()
                t1 = time.perf_counter()
            ntok = sum(len(o.outputs[0].token_ids) for o in outs)
            rows.append(
                {"kind": kind, "i": i, "ms": (t1 - t0) * 1e3, "out_tokens": ntok}
            )
        self.eng.abort_request(ids)
        assert not self.eng.has_unfinished_requests()
        return {"nreq": nreq, "steps": rows}


def summarize_wall(label: str, phases: list[dict]) -> dict:
    pre = [r["ms"] for p in phases for r in p["steps"] if r["kind"] == "prefill"]
    dec = [r["ms"] for p in phases for r in p["steps"] if r["kind"] == "decode"]
    out = {
        "prefill_ms": pre,
        "decode_ms": dec,
        "prefill_median": statistics.median(pre) if pre else None,
        "decode_median": statistics.median(dec) if dec else None,
        "decode_mean": statistics.fmean(dec) if dec else None,
        "decode_min": min(dec) if dec else None,
        "decode_max": max(dec) if dec else None,
    }
    print(
        f"[{label}] prefill ms {[round(x, 1) for x in pre]}  "
        f"decode ms median {out['decode_median']:.1f} mean {out['decode_mean']:.1f} "
        f"min {out['decode_min']:.1f} max {out['decode_max']:.1f}"
        if dec
        else f"[{label}] prefill ms {[round(x, 1) for x in pre]}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# trace analysis
# ---------------------------------------------------------------------------

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def analyze_trace(path: Path) -> dict:
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    ev = [e for e in ev if e.get("ph") == "X"]
    steps = [e for e in ev if e["name"] == "STEP"]
    assert len(steps) == 1, f"{path}: {len(steps)} STEP ranges"
    st = steps[0]
    t0, t1 = st["ts"], st["ts"] + st["dur"]

    runtime = {}
    api_count: Counter = Counter()
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                runtime[corr] = e
            if t0 <= e["ts"] <= t1:
                api_count[e["name"]] += 1

    # innermost I2 label of every runtime call + exclusive CPU time per label
    per_tid: dict = defaultdict(list)
    for e in ev:
        if e.get("cat") == "user_annotation" and e["name"].startswith(("I2:", "WIN:")):
            per_tid[(e["pid"], e["tid"])].append(("a", e))
    for e in runtime.values():
        per_tid[(e["pid"], e["tid"])].append(("r", e))
    rt_label: dict = {}
    cpu_excl: Counter = Counter()
    cpu_cnt: Counter = Counter()
    win_dur: dict = {}
    for items in per_tid.values():
        items.sort(
            key=lambda it: (it[1]["ts"], 0 if it[0] == "a" else 1, -it[1]["dur"])
        )
        stack: list[list] = []  # [end, label, child_sum, dur]

        def pop_until(ts, stack=stack):
            while stack and stack[-1][0] <= ts:
                end, label, child, dur = stack.pop()
                if label.startswith("I2:"):
                    cpu_excl[label] += dur - child
                    cpu_cnt[label] += 1
                if stack:
                    stack[-1][2] += dur

        for kind, e in items:
            pop_until(e["ts"])
            if kind == "a":
                stack.append([e["ts"] + e["dur"], e["name"], 0.0, e["dur"]])
                if e["name"].startswith("WIN:") and t0 <= e["ts"] <= t1:
                    win_dur[e["name"]] = e["dur"]
            else:
                lab = "unlabelled"
                for s in reversed(stack):
                    if s[1].startswith("I2:"):
                        lab = s[1]
                        break
                rt_label[e["args"]["correlation"]] = lab
        pop_until(float("inf"))

    gpu = [e for e in ev if e.get("cat") in GPU_CATS]
    by_label: dict = defaultdict(lambda: [0.0, 0])
    by_name: dict = defaultdict(lambda: [0.0, 0])
    by_cat: dict = defaultdict(lambda: [0.0, 0])
    intervals = []
    n_kernel = n_memcpy = n_memset = 0
    for g in gpu:
        corr = g.get("args", {}).get("correlation")
        r = runtime.get(corr)
        ts = r["ts"] if r is not None else g["ts"]
        if not (t0 <= ts <= t1):
            continue
        if g["cat"] == "kernel":
            n_kernel += 1
        elif g["cat"] == "gpu_memcpy":
            n_memcpy += 1
        else:
            n_memset += 1
        lab = rt_label.get(corr, "unmatched" if r is None else "unlabelled")
        d = g["dur"]
        by_label[lab][0] += d
        by_label[lab][1] += 1
        by_name[g["name"]][0] += d
        by_name[g["name"]][1] += 1
        c = kernel_category(g["name"] if g["cat"] == "kernel" else g["cat"])
        by_cat[c][0] += d
        by_cat[c][1] += 1
        intervals.append((g["ts"], g["ts"] + d))
    intervals.sort()
    busy = 0.0
    cur_s = cur_e = None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    gpu_sum = sum(v[0] for v in by_label.values())
    return {
        "step_wall_ms": st["dur"] / 1e3,
        "forward_window_ms": win_dur.get("WIN:forward", 0.0) / 1e3,
        "gpu_kernel_sum_ms": gpu_sum / 1e3,
        "gpu_busy_union_ms": busy / 1e3,
        "n_kernel": n_kernel,
        "n_memcpy": n_memcpy,
        "n_memset": n_memset,
        "api_count": dict(api_count),
        "by_label": {k: [v[0] / 1e3, v[1]] for k, v in by_label.items()},
        "by_name": {k: [v[0] / 1e3, v[1]] for k, v in by_name.items()},
        "by_cat": {k: [v[0] / 1e3, v[1]] for k, v in by_cat.items()},
        "cpu_excl_ms": {k: v / 1e3 for k, v in cpu_excl.items()},
        "cpu_cnt": dict(cpu_cnt),
    }


def cpu_ops(prof) -> dict[str, list[float]]:
    from torch.autograd import DeviceType

    out = {}
    for e in prof.key_averages():
        if e.device_type != DeviceType.CPU:
            continue
        if e.key.startswith(("I2:", "WIN:", "STEP", "ProfilerStep")):
            continue
        out[e.key] = [e.self_cpu_time_total / 1e3, e.count]
    return out


def merge(dicts: list[dict]) -> dict:
    """Mean over sessions of the numeric / [ms, count] fields of ``analyze_trace``."""
    n = len(dicts)
    out: dict = {"sessions": n}
    for k in (
        "step_wall_ms",
        "forward_window_ms",
        "gpu_kernel_sum_ms",
        "gpu_busy_union_ms",
        "n_kernel",
        "n_memcpy",
        "n_memset",
    ):
        out[k] = sum(d[k] for d in dicts) / n
    out["step_wall_list"] = [d["step_wall_ms"] for d in dicts]
    for k in ("by_label", "by_name", "by_cat"):
        acc: dict = defaultdict(lambda: [0.0, 0.0])
        for d in dicts:
            for name, (ms, c) in d[k].items():
                acc[name][0] += ms / n
                acc[name][1] += c / n
        out[k] = dict(acc)
    for k in ("cpu_excl_ms", "api_count", "cpu_cnt"):
        acc2: Counter = Counter()
        for d in dicts:
            for name, v in d[k].items():
                acc2[name] += v / n
        out[k] = dict(acc2)
    ops: dict = defaultdict(lambda: [0.0, 0.0])
    for d in dicts:
        for name, (ms, c) in d.get("cpu_ops", {}).items():
            ops[name][0] += ms / n
            ops[name][1] += c / n
    out["cpu_ops"] = dict(ops)
    return out


def short(name: str, n: int = 100) -> str:
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= n else name[: n - 3] + "..."


def report(title: str, m: dict, wall_clean_ms: float | None) -> None:
    gpu = m["gpu_kernel_sum_ms"]
    print(f"\n## {title}")
    print(
        f"- sessions {m['sessions']}; profiled step wall {m['step_wall_ms']:.1f} ms "
        f"(min/max {min(m['step_wall_list']):.1f}/{max(m['step_wall_list']):.1f}); "
        f"model forward window {m['forward_window_ms']:.1f} ms"
    )
    if wall_clean_ms:
        print(f"- clean (unprofiled) step wall {wall_clean_ms:.1f} ms")
    print(
        f"- GPU: kernel+copy time sum {gpu:.1f} ms, "
        f"busy-union {m['gpu_busy_union_ms']:.1f} ms; "
        f"busy/profiled-wall {100 * m['gpu_busy_union_ms'] / m['step_wall_ms']:.1f}% "
        f"(idle {100 - 100 * m['gpu_busy_union_ms'] / m['step_wall_ms']:.1f}%)"
        + (
            f"; busy/clean-wall {100 * m['gpu_busy_union_ms'] / wall_clean_ms:.1f}%"
            if wall_clean_ms
            else ""
        )
    )
    print(
        f"- launches per step: {m['n_kernel']:.0f} kernels + "
        f"{m['n_memcpy']:.0f} memcpy + {m['n_memset']:.0f} memset; "
        f"mean kernel {1e3 * gpu / max(m['n_kernel'], 1):.1f} us"
    )
    api = sorted(m["api_count"].items(), key=lambda kv: -kv[1])[:8]
    print("- HIP API calls per step: " + ", ".join(f"{k} x{v:.0f}" for k, v in api))

    print("\n### GPU time by group (innermost I2 label)")
    grp: dict = defaultdict(lambda: [0.0, 0.0, 0.0])
    for lab, (ms, c) in m["by_label"].items():
        g = group_of(lab)
        grp[g][0] += ms
        grp[g][1] += c
    for lab, ms in m["cpu_excl_ms"].items():
        grp[group_of(lab)][2] += ms
    print(
        "| group | GPU ms/step | % GPU | kernels/step | CPU excl ms (labelled python) |"
    )
    print("|---|---:|---:|---:|---:|")
    for g, (ms, c, cpu) in sorted(grp.items(), key=lambda kv: -kv[1][0]):
        print(f"| {g} | {ms:.2f} | {100 * ms / gpu:.1f} | {c:.0f} | {cpu:.1f} |")

    print("\n### GPU time by label")
    print("| label | GPU ms/step | % GPU | kernels/step | CPU excl ms | calls |")
    print("|---|---:|---:|---:|---:|---:|")
    for lab, (ms, c) in sorted(m["by_label"].items(), key=lambda kv: -kv[1][0]):
        print(
            f"| {lab} | {ms:.2f} | {100 * ms / gpu:.1f} | {c:.0f} | "
            f"{m['cpu_excl_ms'].get(lab, 0.0):.1f} | {m['cpu_cnt'].get(lab, 0):.0f} |"
        )

    print("\n### GPU time by kernel category")
    print("| category | GPU ms/step | % GPU | kernels/step |")
    print("|---|---:|---:|---:|")
    for c, (ms, n) in sorted(m["by_cat"].items(), key=lambda kv: -kv[1][0]):
        print(f"| {c} | {ms:.2f} | {100 * ms / gpu:.1f} | {n:.0f} |")

    print("\n### Top 15 GPU kernels")
    print("| kernel | GPU ms/step | % GPU | launches/step | mean us |")
    print("|---|---:|---:|---:|---:|")
    for name, (ms, n) in sorted(m["by_name"].items(), key=lambda kv: -kv[1][0])[:15]:
        print(
            f"| `{short(name)}` | {ms:.2f} | {100 * ms / gpu:.1f} | {n:.0f} | "
            f"{1e3 * ms / max(n, 1):.1f} |"
        )

    print("\n### Top 15 CPU ops (self CPU time, torch.profiler)")
    print("| op | self CPU ms/step | calls/step | mean us |")
    print("|---|---:|---:|---:|")
    for name, (ms, n) in sorted(m["cpu_ops"].items(), key=lambda kv: -kv[1][0])[:15]:
        print(
            f"| `{short(name, 70)}` | {ms:.2f} | {n:.0f} | {1e3 * ms / max(n, 1):.1f} |"
        )
    print(flush=True)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--clean-reps", type=int, default=3)
    ap.add_argument("--prefill-reps", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--keep-traces", action="store_true")
    ap.add_argument(
        "--long-prefill",
        type=int,
        default=0,
        help="also time the prefill step of one prompt of this many tokens",
    )
    ap.add_argument(
        "--path",
        choices=["reference", "hip"],
        help="VLLM_INKLING2_PATH (default: the native HIP kernels on a GPU)",
    )
    args = ap.parse_args()
    if args.path:
        os.environ["VLLM_INKLING2_PATH"] = args.path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM

    t0 = time.time()
    llm = LLM(
        model=args.ckpt,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(args.batch, 2),
        skip_tokenizer_init=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 0},  # text only
    )
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)
    holder: dict = {}

    def grab(m):
        holder["m"] = m

    llm.apply_model(grab)
    model = holder["m"]
    print("model class:", type(model).__name__, flush=True)
    wdt = {
        n: str(p.dtype)
        for n, p in model.named_parameters()
        if re.search(
            r"layers\.(0|6)\.self_attn\.(q_proj|o_proj)\.linear|lm_head|mlp\.gate\.weight",
            n,
        )
    }
    print("param dtypes:", wdt, flush=True)
    print(
        "torch",
        torch.__version__,
        "device",
        torch.cuda.get_device_name(0),
        "pci",
        hex(torch.cuda.get_device_properties(0).pci_bus_id),
        flush=True,
    )

    rig = Rig(llm)
    nd = args.decode_steps
    batches = [1, args.batch] if args.batch != 1 else [1]

    # warm-up (allocator, kernels for each batch shape)
    for b in batches:
        rig.run_phase(b, 3)
    print("warm-up done", flush=True)

    results: dict = {"args": vars(args), "clean": {}, "fwd": {}, "profile": {}}

    # ---- pass A1: clean wall per step --------------------------------------
    print("\n=== pass A1: clean step wall ===", flush=True)
    for b in batches:
        phases = [rig.run_phase(b, nd) for _ in range(args.clean_reps)]
        results["clean"][b] = summarize_wall(f"clean batch={b}", phases)

    if args.long_prefill:
        n = args.long_prefill
        print(f"\n=== long prefill: one {n}-token prompt ===", flush=True)
        phases = [
            rig.run_phase(1, 0, prompt_len=n) for _ in range(max(args.clean_reps, 3))
        ]
        results["long_prefill"] = summarize_wall(f"long prefill {n} tokens", phases)
        median = results["long_prefill"]["prefill_median"]
        results["long_prefill"]["tokens_per_s"] = n / (median / 1e3)
        print(f"long prefill: {results['long_prefill']['tokens_per_s']:.0f} tokens/s")

    # ---- pass A2: forward launch vs completion -----------------------------
    print("\n=== pass A2: forward timer (launch-return vs synced) ===", flush=True)
    ft = FwdTimer()
    ft.install()
    try:
        for b in batches:
            ft.rows.clear()
            phases = [rig.run_phase(b, nd) for _ in range(2)]
            # rows: per step order prefill, decode*nd, per phase
            pre, dec = [], []
            for pi in range(2):
                chunk = ft.rows[pi * (nd + 1) : (pi + 1) * (nd + 1)]
                pre.append(chunk[0])
                dec.extend(chunk[1:])

            def med(rows, k):
                return statistics.median(r[k] for r in rows)

            res = {}
            for kind, rows in (("prefill", pre), ("decode", dec)):
                res[kind] = {
                    k: med(rows, k)
                    for k in (
                        "fwd_launch_ms",
                        "fwd_total_ms",
                        "logits_launch_ms",
                        "logits_total_ms",
                    )
                }
                wall = statistics.median(
                    r["ms"] for p in phases for r in p["steps"] if r["kind"] == kind
                )
                res[kind]["step_wall_ms_with_timer"] = wall
                res[kind]["engine_other_ms"] = (
                    wall - res[kind]["fwd_total_ms"] - res[kind]["logits_total_ms"]
                )
                print(
                    f"[fwd batch={b} {kind}] "
                    + json.dumps({k: round(v, 2) for k, v in res[kind].items()}),
                    flush=True,
                )
            results["fwd"][b] = res
    finally:
        ft.uninstall()

    # ---- pass C: cProfile of decode steps ----------------------------------
    print("\n=== pass C: cProfile, batch 1 decode ===", flush=True)
    pr = cProfile.Profile()

    @contextlib.contextmanager
    def cprof_ctx(kind, i):
        if kind == "decode":
            pr.enable()
        try:
            yield
        finally:
            if kind == "decode":
                pr.disable()

    rig.run_phase(1, nd, ctx=cprof_ctx)
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.sort_stats("tottime").print_stats(15)
    print(f"--- top 15 python frames by tottime over {nd} decode steps (cProfile) ---")
    print(buf.getvalue(), flush=True)
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.sort_stats("cumtime").print_stats(
        r"inkling2_loop|gpu/model_runner|moe_align|llm_engine|scheduler", 15
    )
    print(f"--- top 15 inkling2/runner frames by cumtime over {nd} decode steps ---")
    print(buf.getvalue(), flush=True)

    # ---- pass B: torch.profiler --------------------------------------------
    print("\n=== pass B: torch.profiler ===", flush=True)
    install_labels(model)
    sessions: dict = defaultdict(list)  # (batch, kind) -> [analysis dict]
    kept: set = set()
    counter = {"n": 0}

    def make_ctx(b):
        @contextlib.contextmanager
        def ctx(kind, i):
            counter["n"] += 1
            path = out_dir / f"trace_b{b}_{kind}{i}_{counter['n']}.json"
            with (
                profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_stack=False,
                ) as prof,
                record_function("STEP"),
            ):
                yield
            prof.export_chrome_trace(str(path))
            a = analyze_trace(path)
            a["cpu_ops"] = cpu_ops(prof)
            sessions[(b, kind)].append(a)
            key = (b, kind)
            if args.keep_traces or key not in kept:
                kept.add(key)
            else:
                path.unlink()

        return ctx

    # throwaway session (profiler init)
    rig.run_phase(1, 1, ctx=make_ctx(0))
    sessions.pop((0, "prefill"), None)
    sessions.pop((0, "decode"), None)
    for b in batches:
        rig.run_phase(b, nd, ctx=make_ctx(b))
        for _ in range(args.prefill_reps - 1):
            rig.run_phase(b, 0, ctx=make_ctx(b))
        for kind in ("prefill", "decode"):
            m = merge(sessions[(b, kind)])
            results["profile"][f"b{b}_{kind}"] = m
            clean = results["clean"][b][
                "prefill_median" if kind == "prefill" else "decode_median"
            ]
            report(
                f"batch={b} {kind} ({PROMPT_LEN}-token prompts)"
                if kind == "prefill"
                else f"batch={b} decode (mean of {nd} steps)",
                m,
                clean,
            )

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"results: {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
