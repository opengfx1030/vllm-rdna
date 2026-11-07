#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-truncated end-to-end numerics check of Flash-Next (G13).

Run the same truncated checkpoint (``trunc_checkpoint.py``) on two GPUs
separately, dump the per-layer hidden states, then compare the dumps. The first
diverging layer (and, inside it, the attention or the MoE half) names the op to
blame (docs/rdna2/rdna3-flashnext-port-2026-11-07.md section 12 step G13,
section 3 per-op table).

    # V620 (gfx1030) and W7800 (gfx1100, TP1), one process each, never together
    HIP_VISIBLE_DEVICES=<v620 hip id> python tools/rdna/rdna3/trunc_e2e.py run \
        --model ~/w4a8_runs/rdna3_port/trunc/ckpt-L4 --out v620-L4.pt
    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/trunc_e2e.py run \
        --model ~/w4a8_runs/rdna3_port/trunc/ckpt-L4 --out w7800-L4.pt
    python tools/rdna/rdna3/trunc_e2e.py compare v620-L4.pt w7800-L4.pt

``run`` makes one prefill-only forward (greedy, ``max_tokens=1``) per prompt
through an in-process ``vllm.LLM`` (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and
captures, from sampled token rows (first, last and eight evenly spaced), as
float32:

* ``out``: each decoder layer's output with the delayed hyper-connection
  combine materialized (the ``[T, hc_count * hidden]`` multi-stream state that
  the next layer consumes);
* ``attn`` / ``mlp``: the GDN-or-QSA output and the MoE output of each layer
  (``--no-sublayers`` to skip), the finer localization inside a layer;
* ``final``: the single-stream state after the final mixer, the lm_head input.

It also stores the last-token top-20 logprobs and the greedy token, and with
``--ppl`` the per-token prompt logprobs of a short text (perplexity probe).
The environment is the caller's: ``HIP_VISIBLE_DEVICES`` is never set here and
must already be set, plus whatever recipe variables the run needs. Hooks need
the eager path (``--enforce-eager``, the default).

``compare`` prints per-layer relative L2 / max abs differences, the first layer
above ``--rtol`` (relative L2, default 2e-3), how fp16 noise grows with depth,
and the logprob / token agreement. Exit 1 when the first divergence is in
layer 0 or a greedy token differs, 2 on unusable input.
"""

import argparse
import functools
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import regex as re
import torch

DUMP_VERSION = 1
TOP_K = 20
N_MID_ROWS = 8
DEFAULT_RTOL = 2e-3
ENV_PREFIXES = (
    "VLLM_",
    "HIP_",
    "HSA_",
    "GPU_",
    "ROCR_",
    "CUDA_VISIBLE",
    "OMP_",
    "NCCL_",
    "RCCL_",
    "TORCH_",
    "PYTORCH_",
)

# --- prompts -----------------------------------------------------------------

BUILTIN_TEXT = (
    "The history of numerical computing is a history of rounding errors. Early "
    "machines had word lengths of a few dozen bits and every library author "
    "learned, usually the hard way, that adding a long list of numbers in the "
    "order it was written down gives a different answer than adding it in the "
    "order of increasing magnitude. Compensated summation, pairwise reduction "
    "and fused multiply-add instructions each removed one class of surprise and "
    "introduced another: a result that was bit-identical on one processor "
    "stopped being bit-identical on the next generation of the same product "
    "line. Engineers who ship neural network kernels live in this world every "
    "day, because a matrix multiplication is nothing but a very long sum.\n\n"
    "A transformer decoder layer applies a normalization, an attention or "
    "recurrent mixing block, another normalization and a feed-forward block, "
    "and adds each result back onto a running residual stream. Because the "
    "stream is only ever added to, an error introduced in an early layer is "
    "carried unchanged through every later layer, while the later layers add "
    "their own rounding noise on top. When two implementations of the same "
    "model disagree, the layer at which their hidden states first separate "
    "beyond the expected half-precision noise is the place to look, and the "
    "kind of layer tells which kernel is guilty.\n\n"
    "In a mixture of experts the router picks a handful of experts for each "
    "token from several hundred, and the chosen experts' outputs are summed "
    "with the routing weights. A tiny difference in the router logits can flip "
    "a choice between two nearly tied experts and change the output of a token "
    "by far more than rounding noise would; a numerics test therefore has to "
    "distinguish an expected, rare flip from a systematic error in a "
    "quantized matrix multiplication, which shows up on every token and grows "
    "with the depth of the network.\n\n"
    "Linear attention layers keep a fixed-size state that is updated token by "
    "token with a gated delta rule. Their kernels process the sequence in "
    "chunks, building a small triangular system per chunk, solving it, and "
    "carrying the state from chunk to chunk; the chunked and the recurrent "
    "forms are mathematically identical and numerically close but not equal. "
    "Full attention layers, by contrast, look at every earlier token through a "
    "softmax whose maximum and normalizer must be tracked in higher precision "
    "to stay stable over long contexts.\n\n"
    "A good end-to-end test feeds a few short and long prompts through a model "
    "that is small enough to run on every device under test, records the state "
    "after every layer, and reports the first layer where two devices part "
    "ways. The report is a table: the relative error of each layer's output, "
    "the largest absolute difference, and the agreement of the final token "
    "probabilities. A smooth, slow growth of the relative error with depth is "
    "the signature of half-precision rounding; a sudden step at one layer is "
    "the signature of a bug in the operation that layer is made of."
)
BUILTIN_PPL_TEXT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards "
    "jump quickly. Sphinx of black quartz, judge my vow. A journey of a "
    "thousand miles begins with a single step, and a single misplaced decimal "
    "point can undo a year of careful work. In the morning the harbor was "
    "quiet, the boats rocked gently against their ropes, and the fishermen "
    "carried their nets down to the water while the gulls circled overhead."
)
# (name, start offset in the built-in token stream, number of tokens)
BUILTIN_PROMPTS = (
    ("short20", 0, 20),
    ("medium96", 50, 96),
    ("long320", 120, 320),
    ("long600", 0, 600),
)


def sample_rows(num_tokens: int, n_mid: int = N_MID_ROWS) -> list[int]:
    """Row indices to keep: first, last and ``n_mid`` evenly spaced between."""
    last = num_tokens - 1
    rows = {0, last}
    rows.update(round(i * last / (n_mid + 1)) for i in range(1, n_mid + 1))
    return sorted(rows)


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    """``all`` or a list such as ``0,3,7`` / ``0-3,11``."""
    if spec == "all":
        return list(range(num_layers))
    ids: set[int] = set()
    for part in spec.split(","):
        m = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if not m:
            raise ValueError(f"bad --layers-capture item {part!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        ids.update(range(lo, hi + 1))
    bad = sorted(i for i in ids if i >= num_layers)
    if bad:
        raise ValueError(f"--layers-capture {bad} outside 0..{num_layers - 1}")
    return sorted(ids)


def _tile(ids: list[int], count: int) -> list[int]:
    return (ids * (count // len(ids) + 1))[:count]


def builtin_prompts(tokenizer: Any) -> list[dict[str, Any]]:
    stream = tokenizer.encode(BUILTIN_TEXT, add_special_tokens=False)
    prompts = []
    for name, start, count in BUILTIN_PROMPTS:
        prompts.append(
            {"name": name, "token_ids": _tile(stream[start:] or stream, count)}
        )
    return prompts


def load_prompts(spec: str, tokenizer: Any) -> list[dict[str, Any]]:
    """``builtin`` or a JSONL of ``{"name", "prompt" | "token_ids"[, "ppl"]}``."""
    if spec == "builtin":
        return builtin_prompts(tokenizer)
    prompts: list[dict[str, Any]] = []
    for n, line in enumerate(Path(spec).read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        if "token_ids" in item:
            ids = [int(t) for t in item["token_ids"]]
        elif "prompt" in item:
            ids = tokenizer.encode(item["prompt"], add_special_tokens=False)
        else:
            raise ValueError(f"{spec}:{n + 1}: need 'prompt' or 'token_ids'")
        prompts.append(
            {
                "name": str(item.get("name", f"p{len(prompts)}")),
                "token_ids": ids,
                "ppl": bool(item.get("ppl", False)),
            }
        )
    if not prompts:
        raise ValueError(f"{spec}: no prompts")
    return prompts


# --- worker side (runs inside the engine process; closures must stay plain) ---


def _device_info(model: Any) -> dict[str, Any]:
    p = next(model.parameters())
    info: dict[str, Any] = {
        "model_class": type(model).__name__,
        "param_device": str(p.device),
        "param_dtype": str(p.dtype),
        "decoder_layers": sum(
            type(m).__name__ == "Qwen4ExpDecoderLayer" for m in model.modules()
        ),
    }
    if p.is_cuda:
        prop = torch.cuda.get_device_properties(p.device)
        info["device_name"] = prop.name
        info["gcn_arch"] = getattr(prop, "gcnArchName", "")
    return info


def _install_capture(model: Any, layer_ids: list[int], sublayers: bool) -> list[int]:
    """Hook the decoder layers and the final mixer. Returns the hooked layer ids."""
    layers = {
        m.layer_idx: m
        for m in model.modules()
        if type(m).__name__ == "Qwen4ExpDecoderLayer"
    }
    if not layers:
        raise RuntimeError("no Qwen4ExpDecoderLayer in the model")
    missing = [i for i in layer_ids if i not in layers]
    if missing:
        raise RuntimeError(f"layers {missing} not in the model (has {sorted(layers)})")
    mixers = [
        m
        for name, m in model.named_modules()
        if name.endswith("hyper_connection_mixer")
    ]
    if len(mixers) != 1:
        raise RuntimeError(f"expected one hyper_connection_mixer, got {len(mixers)}")
    state: dict[str, Any] = {"armed": False, "num_tokens": 0, "rows": [], "calls": {}}
    state["out"] = {}
    state["attn"] = {}
    state["mlp"] = {}
    state["final"] = None
    model._trunc_e2e = state

    def take(key: str, idx: int, tensor: Any) -> None:
        if not state["armed"] or tensor.shape[0] < state["num_tokens"]:
            return
        rows = torch.as_tensor(state["rows"], device=tensor.device)
        part = tensor[: state["num_tokens"]].index_select(0, rows)
        state[key][idx] = part.float().cpu()
        state["calls"][(key, idx)] = state["calls"].get((key, idx), 0) + 1

    def first_tensor(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    def layer_hook(idx: int) -> Any:
        def hook(module: Any, args: Any, output: Any) -> None:
            if not state["armed"]:
                return
            if isinstance(output, tuple) and len(output) == 3:
                hidden, block_output, injection = output
                output = module.mlp_hyper_connection.combine(
                    hidden, block_output, injection
                )
            take("out", idx, output)

        return hook

    def sub_hook(key: str, idx: int) -> Any:
        def hook(module: Any, args: Any, output: Any) -> None:
            if state["armed"]:
                take(key, idx, first_tensor(output))

        return hook

    for idx in layer_ids:
        layer = layers[idx]
        layer.register_forward_hook(layer_hook(idx))
        if sublayers:
            attn = getattr(layer, "linear_attn", None) or getattr(
                layer, "self_attn", None
            )
            if attn is not None:
                attn.register_forward_hook(sub_hook("attn", idx))
            layer.mlp.register_forward_hook(sub_hook("mlp", idx))

    mixer = mixers[0]
    original = mixer.combine_and_mix

    def combine_and_mix(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if state["armed"] and result[1].shape[0] >= state["num_tokens"]:
            rows = torch.as_tensor(state["rows"], device=result[1].device)
            sampled = result[1][: state["num_tokens"]].index_select(0, rows)
            state["final"] = sampled.float().cpu()
        return result

    mixer.combine_and_mix = combine_and_mix
    return sorted(layer_ids)


def _arm_capture(model: Any, num_tokens: int, rows: list[int]) -> None:
    state = model._trunc_e2e
    state.update(armed=True, num_tokens=num_tokens, rows=rows, calls={}, final=None)
    for key in ("out", "attn", "mlp"):
        state[key] = {}


def _collect_capture(model: Any) -> dict[str, Any]:
    state = model._trunc_e2e
    state["armed"] = False
    multi = sorted(k for k, n in state["calls"].items() if n > 1)
    return {
        "out": state["out"],
        "attn": state["attn"],
        "mlp": state["mlp"],
        "final": state["final"],
        "multi_call": multi,
    }


# --- run ---------------------------------------------------------------------


def _stack(by_layer: dict[int, Any], ids: list[int]) -> Any:
    if not by_layer:
        return None
    if sorted(by_layer) != ids:
        raise RuntimeError(f"captured layers {sorted(by_layer)} != hooked {ids}")
    return torch.stack([by_layer[i] for i in ids])


def _top_logprobs(logprobs: dict[int, Any]) -> tuple[list[int], list[float]]:
    pairs = sorted(
        ((lp.logprob, tid) for tid, lp in logprobs.items()), key=lambda p: (-p[0], p[1])
    )[:TOP_K]
    return [t for _, t in pairs], [float(lp) for lp, _ in pairs]


def _prompt_logprobs(token_ids: list[int], prompt_logprobs: Any) -> list[float]:
    if prompt_logprobs is None:
        raise RuntimeError("engine returned no prompt_logprobs")
    out = []
    for pos in range(1, len(token_ids)):
        entry = prompt_logprobs[pos]
        if entry is None or token_ids[pos] not in entry:
            raise RuntimeError(f"no prompt logprob for position {pos}")
        out.append(float(entry[token_ids[pos]].logprob))
    return out


def cmd_run(args: argparse.Namespace) -> int:
    if not any(
        os.environ.get(v)
        for v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    ):
        print(
            "error: set HIP_VISIBLE_DEVICES for this run (the tool never picks a "
            "GPU: HIP 0 is the W7800, V620s are HIP 1-4 / 6-9)",
            file=sys.stderr,
        )
        return 2
    if not args.enforce_eager:
        print(
            "warning: forward hooks do not fire inside compiled / graph-captured "
            "code; expect 'no rows captured' unless the model runs eagerly",
            file=sys.stderr,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if args.tp > 1:
        # apply_model ships a function to the TP worker processes.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "language_model_only": not args.with_vision,
        "seed": 0,
        "disable_log_stats": True,
    }
    for item in args.engine_arg:
        key, _, value = item.partition("=")
        kwargs[key] = json.loads(value)
    started = time.time()
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    prompts = load_prompts(args.prompts, tokenizer)
    if args.ppl is not None:
        text = BUILTIN_PPL_TEXT if args.ppl == "builtin" else Path(args.ppl).read_text()
        prompts.append(
            {
                "name": "ppl",
                "token_ids": tokenizer.encode(text, add_special_tokens=False),
                "ppl": True,
            }
        )
    too_long = [p["name"] for p in prompts if len(p["token_ids"]) >= args.max_model_len]
    if too_long:
        print(f"error: prompts {too_long} exceed --max-model-len", file=sys.stderr)
        return 2

    info = llm.apply_model(_device_info)[0]
    num_layers = info["decoder_layers"]
    layer_ids = parse_layer_spec(args.layers_capture, num_layers)
    llm.apply_model(
        lambda m: _install_capture(m, layer_ids, args.sublayers)
    )  # hooks go in after the profile / warm-up forwards: those are not recorded

    records = []
    for prompt in prompts:
        ids = prompt["token_ids"]
        rows = sample_rows(len(ids))
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=TOP_K,
            prompt_logprobs=0 if prompt.get("ppl") else None,
            seed=0,
        )
        arm = functools.partial(_arm_capture, num_tokens=len(ids), rows=rows)
        llm.apply_model(arm)
        request = TokensPrompt(prompt_token_ids=ids)
        out = llm.generate([request], params, use_tqdm=False)[0]
        captured = llm.apply_model(_collect_capture)[0]
        if not captured["out"]:
            print(
                f"error: no rows captured for {prompt['name']}: run with "
                "--enforce-eager",
                file=sys.stderr,
            )
            return 1
        sample_logprobs = out.outputs[0].logprobs
        assert sample_logprobs is not None
        top_ids, top_lps = _top_logprobs(sample_logprobs[0])
        record: dict[str, Any] = {
            "name": prompt["name"],
            "num_tokens": len(ids),
            "token_ids": ids,
            "rows": rows,
            "layer_ids": layer_ids,
            "out": _stack(captured["out"], layer_ids),
            "attn": _stack(captured["attn"], layer_ids),
            "mlp": _stack(captured["mlp"], layer_ids),
            "final": captured["final"],
            "greedy_token": int(out.outputs[0].token_ids[0]),
            "top_ids": top_ids,
            "top_logprobs": top_lps,
            "prompt_logprobs": None,
            "multi_call": captured["multi_call"],
        }
        if prompt.get("ppl"):
            record["prompt_logprobs"] = _prompt_logprobs(ids, out.prompt_logprobs)
        records.append(record)
        print(
            f"{prompt['name']}: {len(ids)} tokens, greedy "
            f"{record['greedy_token']} {tokenizer.decode([record['greedy_token']])!r} "
            f"logprob {top_lps[0]:.4f}, rows {len(rows)}, layers {len(layer_ids)}"
        )
        if record["multi_call"]:
            print(
                f"  warning: {prompt['name']}: layers forwarded more than once "
                f"(chunked prefill or recompute?): {record['multi_call'][:4]}",
                file=sys.stderr,
            )

    import vllm

    meta = {
        "model": str(args.model),
        "dtype": args.dtype,
        "tp": args.tp,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "layer_ids": layer_ids,
        "device": info,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "vllm": getattr(vllm, "__version__", None),
        "vllm_file": getattr(vllm, "__file__", None),
        "env": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)
        },
        "argv": sys.argv,
        "seconds": round(time.time() - started, 1),
    }
    dump = {"version": DUMP_VERSION, "meta": meta, "prompts": records}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, out_path)
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


# --- compare -----------------------------------------------------------------


class Row:
    """One comparison point: a layer output or an attention / MoE sub-output."""

    def __init__(self, label: str, layer: int, rel: list[float], maxabs: float):
        self.label = label
        self.layer = layer
        self.rel = rel  # relative L2, one entry per prompt
        self.maxabs = maxabs

    @property
    def worst(self) -> float:
        return max(self.rel)


class Report:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.rows: list[Row] = []
        self.final_rel: list[float] = []
        self.token_a: list[int] = []
        self.token_b: list[int] = []
        self.top_overlap: list[int] = []
        self.top_maxdiff: list[float] = []
        self.ppl: list[tuple[str, float, float, float, float]] = []
        self.rtol = DEFAULT_RTOL
        self.notes: list[str] = []

    def first_divergence(self) -> Row | None:
        return next((r for r in self.rows if not r.worst <= self.rtol), None)

    def token_mismatches(self) -> list[str]:
        return [
            n for n, a, b in zip(self.prompts, self.token_a, self.token_b) if a != b
        ]

    def exit_code(self) -> int:
        first = self.first_divergence()
        if first is not None and first.layer < 1:
            return 1
        return 1 if self.token_mismatches() else 0


def rel_l2(a: Any, b: Any) -> list[float]:
    """Relative L2 of ``a - b`` per leading index; inf when either is non-finite."""
    a = a.double().reshape(a.shape[0], -1)
    b = b.double().reshape(b.shape[0], -1)
    num = torch.linalg.vector_norm(a - b, dim=1)
    den = torch.linalg.vector_norm(a, dim=1)
    finite = torch.isfinite(a).all(dim=1) & torch.isfinite(b).all(dim=1)
    out = []
    for n, d, ok in zip(num.tolist(), den.tolist(), finite.tolist()):
        if not ok:
            out.append(math.inf)
        elif d > 0:
            out.append(n / d)
        else:
            out.append(0.0 if n == 0 else math.inf)
    return out


def max_abs(a: Any, b: Any) -> list[float]:
    diff = (a.double() - b.double()).abs().reshape(a.shape[0], -1)
    diff = torch.nan_to_num(diff, nan=math.inf, posinf=math.inf)
    return diff.max(dim=1).values.tolist()


def compare_dumps(a: dict[str, Any], b: dict[str, Any], rtol: float) -> Report:
    """Compare two ``run`` dumps. Raises ValueError when they are not comparable."""
    for d in (a, b):
        if d.get("version") != DUMP_VERSION:
            raise ValueError(f"unsupported dump version {d.get('version')!r}")
    pa = {p["name"]: p for p in a["prompts"]}
    pb = {p["name"]: p for p in b["prompts"]}
    names = [n for n in pa if n in pb]
    if not names:
        raise ValueError("the dumps have no prompt in common")
    report = Report()
    report.rtol = rtol
    report.prompts = names
    for n in names:
        if pa[n]["token_ids"] != pb[n]["token_ids"] or pa[n]["rows"] != pb[n]["rows"]:
            raise ValueError(f"prompt {n}: token ids / sampled rows differ")
    for n in sorted(set(pa) ^ set(pb)):
        report.notes.append(f"prompt {n} only in one dump: skipped")

    layer_ids = sorted(
        set.intersection(*(set(p["layer_ids"]) for p in (*pa.values(), *pb.values())))
    )
    if not layer_ids:
        raise ValueError("the dumps have no captured layer in common")
    # (key, label suffix) in forward order inside a layer: attn, mlp, output
    kinds = (("attn", ".attn"), ("mlp", ".mlp"), ("out", ""))
    for layer in layer_ids:
        for key, suffix in kinds:
            rel_per_prompt: list[float] = []
            worst_abs = 0.0
            usable = True
            for n in names:
                ta, tb = pa[n][key], pb[n][key]
                if ta is None or tb is None:
                    usable = False
                    break
                ia = pa[n]["layer_ids"].index(layer)
                ib = pb[n]["layer_ids"].index(layer)
                x, y = ta[ia : ia + 1], tb[ib : ib + 1]
                rel_per_prompt.append(rel_l2(x, y)[0])
                worst_abs = max(worst_abs, max_abs(x, y)[0])
            if usable:
                report.rows.append(
                    Row(f"L{layer:02d}{suffix}", layer, rel_per_prompt, worst_abs)
                )
    for n in names:
        fa, fb = pa[n]["final"], pb[n]["final"]
        report.final_rel.append(
            rel_l2(fa[None], fb[None])[0]
            if fa is not None and fb is not None
            else math.nan
        )
        report.token_a.append(pa[n]["greedy_token"])
        report.token_b.append(pb[n]["greedy_token"])
        ids_a, lp_a = pa[n]["top_ids"], pa[n]["top_logprobs"]
        ids_b, lp_b = pb[n]["top_ids"], pb[n]["top_logprobs"]
        report.top_overlap.append(len(set(ids_a) & set(ids_b)))
        common = set(ids_a) & set(ids_b)
        da, db = dict(zip(ids_a, lp_a)), dict(zip(ids_b, lp_b))
        report.top_maxdiff.append(
            max((abs(da[t] - db[t]) for t in common), default=math.nan)
        )
        la, lb = pa[n].get("prompt_logprobs"), pb[n].get("prompt_logprobs")
        if la is not None and lb is not None:
            diffs = [abs(x - y) for x, y in zip(la, lb)]
            report.ppl.append(
                (
                    n,
                    math.exp(-sum(la) / len(la)),
                    math.exp(-sum(lb) / len(lb)),
                    max(diffs),
                    sum(diffs) / len(diffs),
                )
            )
    return report


def growth_summary(report: Report) -> list[str]:
    """How the relative error of the layer outputs grows with depth."""
    outs = [r for r in report.rows if not r.label.endswith((".attn", ".mlp"))]
    finite = [(r.layer, r.worst) for r in outs if 0 < r.worst < math.inf]
    lines = []
    if len(finite) >= 2:
        (l0, e0), (l1, e1) = finite[0], finite[-1]
        per_layer = (e1 / e0) ** (1 / (l1 - l0)) - 1 if l1 > l0 else 0.0
        lines.append(
            f"relative L2 of the layer output: {e0:.2e} at layer {l0} -> {e1:.2e} at "
            f"layer {l1} (x{e1 / e0:.1f}, {per_layer:+.1%} per layer, geometric)"
        )
        steps = [
            (finite[i][1] / finite[i - 1][1], finite[i][0])
            for i in range(1, len(finite))
            if finite[i][1] > 1e-5
        ]
        if steps:
            ratio, layer = max(steps)
            lines.append(
                f"largest step-up between consecutive layers: x{ratio:.1f} "
                f"at layer {layer}"
            )
    return lines


def format_report(report: Report, name_a: str, name_b: str) -> str:
    out = [f"A = {name_a}", f"B = {name_b}", f"rtol (relative L2) = {report.rtol:g}"]
    out += report.notes
    legend = ", ".join(f"p{i}={n}" for i, n in enumerate(report.prompts))
    out.append(f"prompts: {legend}")
    head = "label    " + "".join(
        f"{'p' + str(i):>10s}" for i in range(len(report.prompts))
    )
    out.append(head + "   worst relL2    max|d|  flag")
    first = report.first_divergence()
    for r in report.rows:
        cells = "".join(f"{v:10.2e}" for v in r.rel)
        flag = (
            "<-- FIRST" if r is first else (">" if not r.worst <= report.rtol else "")
        )
        out.append(f"{r.label:8s}{cells}   {r.worst:10.2e} {r.maxabs:9.2e}  {flag}")
    cells = "".join(f"{v:10.2e}" for v in report.final_rel)
    out.append(f"{'final':8s}{cells}")
    out.append("")
    if first is None:
        out.append("first divergence: none (every point within rtol)")
    else:
        what = {".attn": "attention (GDN / QSA)", ".mlp": "MoE block"}.get(
            first.label[3:], "layer output (first point of the layer is the attention)"
        )
        out.append(f"first divergence: {first.label} ({what}), relL2 {first.worst:.2e}")
    out += growth_summary(report)
    out.append("")
    out.append("prompt   tokenA  tokenB  top20-overlap  max|dlogprob|  match")
    for i, n in enumerate(report.prompts):
        same = "yes" if report.token_a[i] == report.token_b[i] else "NO"
        out.append(
            f"{'p' + str(i):8s}{report.token_a[i]:7d} {report.token_b[i]:7d} "
            f"{report.top_overlap[i]:10d}/{TOP_K}  "
            f"{report.top_maxdiff[i]:12.3e}  {same}"
        )
    for n, ppl_a, ppl_b, worst, mean in report.ppl:
        out.append(
            f"ppl probe {n}: A {ppl_a:.4f} B {ppl_b:.4f} "
            f"(rel {abs(ppl_a - ppl_b) / ppl_a:.2e}), per-token |dlogprob| "
            f"max {worst:.3e} mean {mean:.3e}"
        )
    verdict = []
    if first is not None and first.layer < 1:
        verdict.append("first divergence is in layer 0")
    if report.token_mismatches():
        verdict.append(f"greedy token differs on {report.token_mismatches()}")
    out.append("RESULT: " + ("FAIL: " + "; ".join(verdict) if verdict else "PASS"))
    return "\n".join(out)


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        a = torch.load(args.a, map_location="cpu", weights_only=True)
        b = torch.load(args.b, map_location="cpu", weights_only=True)
        report = compare_dumps(a, b, args.rtol)
    except Exception as e:  # unreadable / foreign / wrong-version dump files
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_report(report, str(args.a), str(args.b)))
    code = report.exit_code()
    if args.strict and code == 0 and report.first_divergence() is not None:
        return 1
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="forward the prompts and dump hidden states")
    run.add_argument("--model", required=True, type=Path, help="truncated checkpoint")
    run.add_argument("--prompts", default="builtin", help="JSONL file or 'builtin'")
    run.add_argument("--out", required=True, type=Path, help="dump file (.pt)")
    run.add_argument("--dtype", default="float16")
    run.add_argument("--tp", type=int, default=1)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="eager forward (default; required for the hooks to fire)",
    )
    run.add_argument(
        "--layers-capture", default="all", help="'all' or e.g. 0,3 / 0-3,11"
    )
    run.add_argument(
        "--sublayers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also capture each layer's attention and MoE outputs",
    )
    run.add_argument(
        "--ppl",
        nargs="?",
        const="builtin",
        default=None,
        metavar="TEXT_FILE",
        help="add a perplexity probe prompt (built-in text, or a text file)",
    )
    run.add_argument("--max-num-seqs", type=int, default=1)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument(
        "--with-vision", action="store_true", help="load the vision tower too"
    )
    run.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="extra vllm.LLM keyword, e.g. block_size=1024 (repeatable)",
    )
    run.set_defaults(func=cmd_run)
    cmp = sub.add_parser("compare", help="compare two dumps")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    cmp.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="relative L2 above which a layer counts as diverged (default 2e-3)",
    )
    cmp.add_argument(
        "--strict", action="store_true", help="exit 1 on any divergence above rtol"
    )
    cmp.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
