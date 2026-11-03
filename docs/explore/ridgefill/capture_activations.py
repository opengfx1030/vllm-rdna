#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture fp32 activations and refit RidgeFill maps for the W4A16 checkpoint.

RidgeFill predicts each late layer's block input from the four-stream split
state: ``x_hat_l = A_l x_S + b_l``.  The split state is the residual entering
layer 24 (10240 wide on every TP rank).  The fit target is the block input to
each layer 24-47 (2560 wide).  This script captures both in fp32 and solves
the normal equations.  It is the real-capture counterpart to
``example_capture_fit.py``; that file stays the CPU synthetic demo.

Two modes:

``--synthetic``
    Build a small Qwen4Exp-shaped model on CPU with random weights, capture
    its split state and layer inputs, fit maps, and report errors.  No
    checkpoint and no GPU.  This is the CPU smoke test.

``--model PATH``
    Load the W4A16 checkpoint with vLLM (``enforce_eager=True``), register
    forward hooks on layer 24 and the attention modules of layers 24-47, run
    the prompts, capture fp32 tensors, fit maps on the GPU, and write the
    bundle.  Requires the dest torch 2.12 / ROCm 7.14 tree.

Capture policy
--------------
*   Split layer 24, tail 2048, map layers 24-47.  These match
    ``config.RidgeConfig`` and the paper's Flash-Next split.
*   Capture all prompt tokens.  The fit uses every row.  ``--exclude-tail``
    drops the last ``tail`` tokens from the solve when a run wants the exact
    tail out of the fit; the default keeps them.
*   Defaults: 8 prompts, 4096 tokens each, so about 32k fit rows.  The fit
    row count is printed; it is the only input the fit needs.
*   fp32 capture.  The map is stored as fp16 for the serve GEMM, with the
    fp32 fit error reported per layer.

Outputs in ``--output`` (default ``~/w4a8_runs/lanes/ridgefill/capture``)::

    capture.pt       split_state [T, 10240] fp32, layer_inputs {24: [T, 2560]}
    maps.pt          {layer_idx: (A [2560, 10240] fp16, b [2560] fp16)}
    ridgefill.json   manifest: pins, per-layer fit error, map sha, shapes

Run the pin check before serving::

    python3 docs/explore/ridgefill/capture_activations.py --synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "vllm" / "model_executor" / "layers" / "ridgefill"

# Dest serve pins.  Do not bump these from this explore draft.
DEST_TORCH = "2.12"
DEST_ROCM = "7.14"

# Flash-Next defaults from config.RidgeConfig.
SPLIT = 24
TAIL = 2048
DIN = 10240
DOUT = 2560
LAM = 0.03
NUM_LAYERS = 48


def _load(name: str):
    """Load one RidgeFill module without importing vllm's package __init__."""
    import importlib.util

    path = PKG / f"{name}.py"
    mod_name = f"ridgefill_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the repo venv has torch
        raise SystemExit("torch is required for this capture script") from exc
    return torch


def _check_dest_runtime(torch) -> None:
    """Refuse a capture outside dest's torch 2.12 / ROCm 7.14 pins."""
    version = torch.__version__
    hip = getattr(torch.version, "hip", None)
    if not str(version).startswith(DEST_TORCH):
        raise SystemExit(f"capture torch must be {DEST_TORCH}.x, got {version}")
    if hip is None or not str(hip).startswith(DEST_ROCM):
        raise SystemExit(f"capture ROCm must be {DEST_ROCM}.x, got {hip}")


class _SyntheticLayer:
    """Small layer shaped like the Flash-Next split layer.

    ``attn_in`` is the 10240 -> 2560 hyperconnection mix.  ``self_attn`` and
    ``mlp`` are the two consumers whose inputs we capture.  ``out`` writes the
    attention + MLP sum back to the four-stream state.
    """

    def __init__(self, torch, din: int, dout: int) -> None:
        nn = torch.nn
        self.attn_in = nn.Linear(din, dout, bias=False)
        self.self_attn = nn.Linear(dout, dout, bias=False)
        self.mlp = nn.Linear(dout, dout, bias=False)
        self.out = nn.Linear(dout, din, bias=False)

    def __call__(self, hidden_states):
        block_input = self.attn_in(hidden_states)
        attn_out = self.self_attn(block_input)
        mlp_out = self.mlp(block_input)
        return hidden_states + self.out(attn_out + mlp_out)


class _SyntheticModel:
    """A stack of ``_SyntheticLayer``.  CPU only."""

    def __init__(self, torch, din: int, dout: int, layers: int) -> None:
        self.layers = [_SyntheticLayer(torch, din, dout) for _ in range(layers)]

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _CaptureHooks:
    """Forward hooks on the split layer and the late attention modules.

    The split hook records the 10240-wide ``hidden_states`` entering layer 24.
    The attention hook records each layer's 2560-wide ``block_input``.  Both
    are copied to CPU as fp32 so the fit does not hold GPU memory.
    """

    def __init__(self, split: int, end: int) -> None:
        self.split = split
        self.end = end
        self.split_state: list = []
        self.layer_inputs: dict[int, list] = {i: [] for i in range(split, end)}

    def _split_hook(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is not None:
            self.split_state.append(hidden.detach().float().cpu())

    def _attn_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None:
                self.layer_inputs[layer_idx].append(hidden.detach().float().cpu())

        return hook

    def _layers(self, model):
        # Qwen4ExpForCausalLM.model.layers and Qwen4ExpModel.layers both exist.
        inner = getattr(model, "model", model)
        return inner.layers

    def register(self, model) -> None:
        layers = self._layers(model)
        if len(layers) < self.end:
            raise SystemExit(
                f"model has {len(layers)} layers, capture wants {self.end}"
            )
        for layer_idx in range(self.split, self.end):
            layer = layers[layer_idx]
            if layer_idx == self.split:
                layer.register_forward_pre_hook(self._split_hook, with_kwargs=True)
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                attn = getattr(layer, "linear_attn", None)
            if attn is None:
                raise SystemExit(
                    f"layer {layer_idx} has neither self_attn nor linear_attn"
                )
            attn.register_forward_pre_hook(self._attn_hook(layer_idx), with_kwargs=True)


def _fit_error(pred, ys, denom, rows: int) -> dict[str, float]:
    """Absolute and relative fit error of one layer's prediction."""
    diff = (pred - ys).abs()
    return {
        "max_abs": float(diff.max().item()),
        "rel_rms": float((diff.square().mean().sqrt() / denom).item()),
        "rows": int(rows),
    }


def _fit_maps(torch, split_state, layer_inputs, lam: float):
    """Fit one map per late layer; return them and the fit error.

    Same estimator as ``ridgefill.math.fit_ridge``: center x and y, add
    ``alpha = lam * tr(G) / din`` to the centered Gram ``G``, solve
    ``A = (Yc^T Xc) G^-1``, and set ``b = mu_y - A mu_x``. The map is stored
    fp16 for the serve GEMM; fp32 is the fit and the error.
    """
    if not split_state:
        raise SystemExit("no split state captured")
    xs = torch.cat(split_state, dim=0)
    rows, din = xs.shape
    if rows == 0:
        raise SystemExit("captured split state has no rows")
    if din != DIN or xs.shape[1] != DIN:
        raise SystemExit(f"split state must be {DIN} wide, got {tuple(xs.shape)}")
    if xs.device.type != "cpu":
        raise SystemExit("the fit runs on the captured CPU copies")
    mu_x = xs.mean(dim=0)
    xc = xs - mu_x
    gram = xc.T @ xc
    trace = torch.diagonal(gram).sum()
    alpha = lam * float(trace) / float(din)
    gram = gram + alpha * torch.eye(din, dtype=xs.dtype, device=xs.device)
    # The ridge Gram is SPD; Cholesky is faster and more stable than a general
    # solve on a 10240-wide matrix.
    chol = torch.linalg.cholesky(gram)
    maps: dict[int, tuple] = {}
    errors: dict[int, dict[str, float]] = {}
    for layer_idx, chunks in layer_inputs.items():
        ys = torch.cat(chunks, dim=0)
        if ys.shape[0] != rows:
            raise SystemExit(
                f"layer {layer_idx} rows {ys.shape[0]} != split rows {rows}"
            )
        if ys.shape[1] != DOUT:
            raise SystemExit(f"layer {layer_idx} is {ys.shape[1]} wide, want {DOUT}")
        mu_y = ys.mean(dim=0)
        yc = ys - mu_y
        cross = yc.T @ xc
        a = torch.cholesky_solve(cross.T.contiguous(), chol).T.contiguous()
        b = (mu_y - a @ mu_x).contiguous()
        denom = ys.square().mean().sqrt().clamp(min=1e-12)
        # The serve GEMM sees the fp16 map, so report the fp16 error too:
        # a good fp32 fit can still lose a digit to the half rounding.
        a16 = a.half().float()
        b16 = b.half().float()
        entry = _fit_error(xs @ a.T + b, ys, denom, rows)
        fp16_error = _fit_error(xs @ a16.T + b16, ys, denom, rows)
        entry["rel_rms_fp16"] = fp16_error["rel_rms"]
        errors[layer_idx] = entry
        maps[layer_idx] = (a.half().cpu(), b.half().cpu())
    return maps, errors


def _save_bundle(torch, output: Path, split_state, layer_inputs, maps, errors, meta):
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split_state": torch.cat(split_state, dim=0),
            "layer_inputs": {
                k: torch.cat(v, dim=0) for k, v in layer_inputs.items() if v
            },
        },
        output / "capture.pt",
    )
    torch.save(maps, output / "maps.pt")
    digest = hashlib.sha256((output / "maps.pt").read_bytes()).hexdigest()
    manifest = {
        "format": "ridgefill-w4a16-fp16-v1",
        "model": {
            "vocab_sha256": meta.get("vocab_sha256", ""),
            "encodings": meta.get("encodings", {}),
        },
        "split": SPLIT,
        "tail": TAIL,
        "din": DIN,
        "dout": DOUT,
        "lambda": LAM,
        "map_dtype": "fp16",
        "map_sha256": digest,
        "layers": {
            str(k): {
                "shape": [int(v[0].shape[0]), int(v[0].shape[1])],
                "fit": errors[k],
            }
            for k, v in maps.items()
        },
    }
    (output / "ridgefill.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    # The serve pins file, independent of the manifest: the same checkpoint
    # pins, in the shape ``loader.require_bundle`` reads.
    (output / "pins.json").write_text(
        json.dumps(
            {
                "vocab_sha256": manifest["model"]["vocab_sha256"],
                "encodings": manifest["model"]["encodings"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return digest


def _synthetic(args) -> int:
    torch = _torch()
    torch.manual_seed(7)
    din, dout, layers = args.synthetic_din, args.synthetic_dout, args.synthetic_layers
    model = _SyntheticModel(torch, din, dout, layers)
    hooks = _CaptureHooks(split=args.split, end=layers)
    hooks.register(model)
    prompts = max(1, args.prompts)
    for _ in range(prompts):
        rows = max(1, args.tokens)
        hidden = torch.randn(rows, din)
        with torch.no_grad():
            model(hidden)
    maps, errors = _fit_maps(torch, hooks.split_state, hooks.layer_inputs, args.lam)
    output = Path(args.output)
    digest = _save_bundle(
        torch,
        output,
        hooks.split_state,
        hooks.layer_inputs,
        maps,
        errors,
        {"vocab_sha256": "synthetic", "encodings": {}},
    )
    for layer_idx in sorted(errors):
        e = errors[layer_idx]
        print(
            f"layer {layer_idx:2d} rows {e['rows']} "
            f"max_abs {e['max_abs']:.6g} rel_rms {e['rel_rms']:.6g}"
        )
    print(f"synthetic capture rows {sum(len(c) for c in hooks.split_state)}")
    print(f"map sha256 {digest}")
    print(f"wrote {output}")
    return 0


def _real(args) -> int:
    torch = _torch()
    _check_dest_runtime(torch)
    # The capture is the exact model; the flag must stay off so the session
    # does not try to load a map that does not exist yet.
    os.environ["VLLM_RIDGEFILL"] = "0"
    dump = Path(args.output) / "dump"
    dump.mkdir(parents=True, exist_ok=True)
    # The worker extension reads this variable in every TP worker.
    os.environ["VLLM_RIDGEFILL_CAPTURE"] = str(dump)

    from vllm import LLM, SamplingParams

    # TP>1 on a 32 GB V620 holds the W4A16 serve; TP=1 does not (the bf16
    # checkpoint is ~180 GB on disk, ~45 GB packed). The map is replicated,
    # so the TP=2 fit is the TP=4 serve map.
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        # The flashnext recipe runs TP4/EP4: without EP the WNA16 MoE's static
        # group scales need 640 / TP divisible by 128 (160 and 320 are not),
        # so a bare TP=2 or TP=4 load refuses to start.
        enable_expert_parallel=args.tp > 1,
        # One decode token after the prompt must fit the model length.
        max_model_len=max(args.tokens, 4096) + 1,
        max_num_seqs=1,
        # One prompt per request, and one prefill chunk per prompt: the split
        # state is that prompt's tokens, not a batch of prompts. The probe can
        # then drop the one-row decode step by row count.
        max_num_batched_tokens=max(args.tokens, 2048),
        disable_log_stats=True,
        worker_extension_cls=(
            "vllm.model_executor.layers.ridgefill.capture_probe.RidgeFillProbe"
        ),
    )
    llm.collective_rpc("ridgefill_capture", args=(args.split, args.end))
    prompts = _token_prompts(llm, args)
    params = SamplingParams(max_tokens=1, temperature=0.0)
    for prompt in prompts:
        llm.generate([prompt], params)
    llm.collective_rpc("ridgefill_save", args=(str(dump),))

    captured = torch.load(dump / "rank0.pt", map_location="cpu")
    split_state = captured["split"]
    layer_inputs = captured["inputs"]
    # Keep the prefill forwards only: a one-prompt decode step has one row.
    keep = [i for i, x in enumerate(split_state) if x.shape[0] > 1]
    if not keep:
        raise SystemExit("captured no prefill forward")
    split_state = [split_state[i] for i in keep]
    layer_inputs = {k: [v[i] for i in keep] for k, v in layer_inputs.items()}
    empty = [k for k, v in layer_inputs.items() if not v]
    if empty:
        raise SystemExit(f"captured no layer input for layers {empty}")
    if args.exclude_tail:
        tail = args.tail
        split_state = [x[:-tail] if x.shape[0] > tail else x[:0] for x in split_state]
        layer_inputs = {
            k: [x[:-tail] if x.shape[0] > tail else x[:0] for x in v]
            for k, v in layer_inputs.items()
        }
    maps, errors = _fit_maps(torch, split_state, layer_inputs, args.lam)
    output = Path(args.output)
    meta = _read_pins(args)
    digest = _save_bundle(torch, output, split_state, layer_inputs, maps, errors, meta)
    for layer_idx in sorted(errors):
        e = errors[layer_idx]
        print(
            f"layer {layer_idx:2d} rows {e['rows']} "
            f"max_abs {e['max_abs']:.6g} rel_rms {e['rel_rms']:.6g}"
        )
    print(f"map sha256 {digest}")
    print(f"wrote {output}")
    return 0


def _token_prompts(llm, args) -> list:
    """Exactly ``args.tokens`` token ids per prompt, from the text prompts."""
    from vllm.inputs import TokensPrompt

    tokenizer = llm.get_tokenizer()
    prompts = []
    for text in _read_prompts(args):
        ids = tokenizer.encode(text)
        if not ids:
            raise SystemExit("a prompt tokenized to zero tokens")
        while len(ids) < args.tokens:
            ids = ids + ids
        prompts.append(TokensPrompt(prompt_token_ids=ids[: args.tokens]))
    return prompts


def _read_prompts(args) -> list[str]:
    if args.prompts_file:
        path = Path(args.prompts_file)
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        texts = [t for t in texts if t]
        if not texts:
            raise SystemExit(f"{path} has no prompts")
        return texts[: args.prompts]
    # Default long-document prompts.  Replace with a real corpus for the
    # measured fit; this keeps the script runnable without a dataset.
    base = (
        "RidgeFill predicts the block input of each late layer from the four "
        "streams entering the split layer.  The fit is closed form and the "
        "decode path is unchanged.  A long document prompt exercises the "
        "filled region because the tail stays exact.  "
    )
    return [(base * 40)[: args.tokens] for _ in range(args.prompts)]


def _read_pins(args) -> dict:
    if args.pins:
        path = Path(args.pins)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "vocab_sha256": data.get("vocab_sha256", ""),
                "encodings": data.get("encodings", {}),
            }
    # Best-effort: a W4A16 checkpoint ships quant_config.json with encodings.
    model_dir = Path(args.model)
    quant = model_dir / "quant_config.json"
    encodings = {}
    if quant.exists():
        try:
            data = json.loads(quant.read_text(encoding="utf-8"))
            encodings = data.get("encodings", {})
        except (OSError, json.JSONDecodeError):
            encodings = {}
    vocab = model_dir / "tokenizer.json"
    vocab_sha = ""
    if vocab.exists():
        vocab_sha = hashlib.sha256(vocab.read_bytes()).hexdigest()
    return {"vocab_sha256": vocab_sha, "encodings": encodings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="CPU smoke test with a small Qwen4Exp-shaped model",
    )
    parser.add_argument("--model", default="", help="W4A16 checkpoint path")
    parser.add_argument(
        "--output",
        default=str(Path.home() / "w4a8_runs/lanes/ridgefill/capture"),
        help="capture directory",
    )
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--prompts-file", default="")
    parser.add_argument("--pins", default="", help="checkpoint pins JSON")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--end", type=int, default=NUM_LAYERS)
    parser.add_argument("--tail", type=int, default=TAIL)
    parser.add_argument("--lam", type=float, default=LAM)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--exclude-tail",
        action="store_true",
        help="drop the last --tail tokens from the fit",
    )
    parser.add_argument("--synthetic-din", type=int, default=64)
    parser.add_argument("--synthetic-dout", type=int, default=16)
    parser.add_argument("--synthetic-layers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.synthetic or not args.model:
        if not args.synthetic:
            print("no --model; running the synthetic CPU smoke test", file=sys.stderr)
        return _synthetic(args)
    return _real(args)


if __name__ == "__main__":
    raise SystemExit(main())
