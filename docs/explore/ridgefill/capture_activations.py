#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture activations and refit RidgeFill maps for the W4A16 checkpoint.

RidgeFill predicts each late layer's block input from the four-stream split
state: ``x_hat_l = A_l x_S + b_l``. The split state is the materialized HC
state entering layer ``split`` (10240 wide on every TP rank): the capture
applies the split layer's pending combine of layer ``split - 1``'s MoE output,
as the runtime does (``capture_probe.split_state``). The fit target is the
block input (attention-module input) of each later layer (2560 wide).

Modes:

``--synthetic``
    A small Qwen4Exp-shaped CPU model with random weights and the delayed HC
    combine. Captures every row, fits, and writes the legacy bundle. No
    checkpoint, no GPU: the CPU smoke test, and the only mode that runs
    without a corpus.

``--model PATH``
    Legacy raw capture: vLLM (``enforce_eager=True``), every fp32 row of
    ``--prompts`` prompts of ``--tokens`` tokens, one prefill forward each,
    fit on the host. Needs a corpus: ``--prompts-file`` (one document per
    line) or ``--corpus-json`` (``ppl_long``/``make_fit_corpus`` windows). A
    document shorter than ``--tokens`` tokens is refused, never tiled: the
    2026-11-03 capture fitted one placeholder sentence repeated to 4096
    tokens because this script fell back to it without a corpus.

``--stats`` (with ``--model``, or with ``--synthetic`` for a CPU dry run)
    The large capture. ``--corpus-json`` windows run in stats mode: rank 0
    adds every prefill forward's sums into fp32 on its GPU and folds them
    into fp64 once per prompt (``--fold``, ``capture_probe``), and writes
    cumulative snapshots after the ``--snapshots`` prompt counts.
    ``--held-json`` windows then run in raw mode: fp16 rows of every
    ``--raw-stride``-th position and of the positions below ``--raw-head``.
    ``fit_stats.py`` fits the maps from the snapshots and measures them on
    the held-out rows. ``--splits`` captures each split's state against the
    layers at or after it.

The stats capture runs the serve's scheduler shape (the ``flashnext`` recipe
with MTP off): chunked prefill with ``--max-batched-tokens`` 2048,
``--block-size`` 1024, one sequence at a time, prefix caching off, RidgeFill
off, eager (the hooks need it). Prompts are the corpus token ids as
``TokensPrompt`` with ``max_tokens=1``; the probe drops the one-row decode
forward, and the driver checks that each prompt's forwards cover its rows
once, in order.

Outputs of ``--stats`` in ``--output``::

    capture_meta.json   corpus files and shas, engine args, env, per-prompt
                        forwards and seconds, snapshot and row-file sizes
    stats_pNNN.pt       cumulative fp64 sums after NNN fit prompts,
                        8 * (sum_s (din^2 + 2 din + n_s din dout) + 3 n dout)
                        bytes with n_s the layers at or after split s and n
                        the layers after the first split: 9.2 GB for splits
                        24,36 and 5.9 GB for split 24 (Flash-Next)
    rows_held.pt        held-out fp16 rows: x per split, y per layer,
                        positions, prompt index; 163848 bytes a row for
                        splits 24,36, 5.4 GB for 8 x 16K at stride 4
    rows_head.pt        fp16 rows of the fit positions below --stats-head
                        (0.67 GB for 64 prompts x 64 positions)
    pins.json           the checkpoint pins for the fitted bundles

Legacy outputs (``--synthetic``, ``--model`` without ``--stats``)::

    capture.pt       split_state [T, 10240] fp32, layer_inputs {24: [T, 2560]}
    maps.pt          {layer_idx: (A [2560, 10240] fp16, b [2560] fp16)}
    ridgefill.json   manifest: pins, per-layer fit error, map sha, shapes
    pins.json        the pins, in the shape ``loader.require_bundle`` reads

Run the CPU smoke test::

    python3 docs/explore/ridgefill/capture_activations.py --synthetic
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "vllm" / "model_executor" / "layers" / "ridgefill"
PROBE = "vllm.model_executor.layers.ridgefill.capture_probe.RidgeFillProbe"

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

CAPTURE_FORMAT = "ridgefill-capture-v2"
# Environment recorded with a stats capture (no secrets: *TOKEN* is skipped).
ENV_PREFIXES = (
    "VLLM_",
    "HSA_",
    "HIP_",
    "GPU_",
    "ROCR_",
    "PYTORCH_",
    "TORCH_",
    "OMP_",
    "NCCL_",
    "RCCL_",
    "FLASH_",
    "HF_",
    "TRANSFORMERS_",
    "TOKENIZERS_",
)


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


def _synthetic_classes(torch):
    """The CPU model the synthetic modes capture.

    It is a real ``torch.nn.Module`` stack shaped like Flash-Next's decoder:
    ``din // dout`` hyperconnection streams, the attention module called with
    ``hidden_states=``, and each layer's MLP output left pending, so the
    next layer's ``attn_hyper_connection.combine`` adds it first (the delayed
    combine ``capture_probe.split_state`` materializes). The capture hooks
    register on it the same way they register on the served model.
    """
    nn = torch.nn

    class SyntheticHC(nn.Module):
        """``mix`` reads the streams into one block input and one injection
        weight per stream; ``combine`` adds a block's output back into every
        stream with those weights."""

        def __init__(self, din: int, dout: int) -> None:
            super().__init__()
            if din % dout:
                raise ValueError(f"din {din} is not a multiple of dout {dout}")
            self.streams = din // dout
            self.dout = dout
            self.read = nn.Linear(din, dout, bias=False)
            self.inject = nn.Linear(din, self.streams, bias=False)

        def mix(self, hidden_states):
            block_input = self.read(hidden_states)
            injection = torch.sigmoid(self.inject(hidden_states))
            return hidden_states, block_input, injection

        def combine(self, hidden_states, block_output, injection):
            rows = hidden_states.shape[0]
            streams = hidden_states.reshape(rows, self.streams, self.dout)
            streams = streams + injection.unsqueeze(-1) * block_output.unsqueeze(1)
            return streams.reshape(rows, -1)

    class SyntheticAttention(nn.Module):
        def __init__(self, dout: int) -> None:
            super().__init__()
            self.proj = nn.Linear(dout, dout, bias=False)

        def forward(self, hidden_states, positions=None):
            return torch.tanh(self.proj(hidden_states))

    class SyntheticLayer(nn.Module):
        """One decoder layer with Flash-Next's delayed MoE combine."""

        def __init__(self, din: int, dout: int) -> None:
            super().__init__()
            self.attn_hyper_connection = SyntheticHC(din, dout)
            self.mlp_hyper_connection = SyntheticHC(din, dout)
            self.self_attn = SyntheticAttention(dout)
            self.mlp = nn.Linear(dout, dout, bias=False)

        def forward(
            self,
            hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=None,
        ):
            hc = self.attn_hyper_connection
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
            hidden_states, block_input, injection = hc.mix(hidden_states)
            attn_out = self.self_attn(hidden_states=block_input, positions=positions)
            hidden_states = hc.combine(hidden_states, attn_out, injection)
            hidden_states, block_input, injection = self.mlp_hyper_connection.mix(
                hidden_states
            )
            return hidden_states, torch.tanh(self.mlp(block_input)), injection

    class SyntheticModel(nn.Module):
        """A stack of ``SyntheticLayer``.  CPU only.

        Called with hidden states (legacy synthetic mode) or with token ids
        and positions (stats dry run: a token embedding plus a sine of the
        position, expanded to every stream like ``Qwen4ExpModel.forward``).
        """

        def __init__(self, din: int, dout: int, layers: int, vocab: int = 0) -> None:
            super().__init__()
            self.streams = din // dout
            self.layers = nn.ModuleList(
                [SyntheticLayer(din, dout) for _ in range(layers)]
            )
            self.embed = nn.Embedding(vocab, dout) if vocab > 0 else None
            freqs = 1.0 / (100.0 ** (torch.arange(dout, dtype=torch.float32) / dout))
            self.register_buffer("freqs", freqs, persistent=False)

        def forward(self, hidden_states=None, positions=None, input_ids=None):
            if input_ids is not None:
                if self.embed is None:
                    raise ValueError("token ids need a vocab")
                hidden_states = self.embed(input_ids)
                if positions is not None:
                    angle = positions.float().unsqueeze(-1) * self.freqs
                    hidden_states = hidden_states + 0.5 * torch.sin(angle)
                hidden_states = hidden_states.repeat(1, self.streams)
            block_output = injection = None
            for layer in self.layers:
                hidden_states, block_output, injection = layer(
                    hidden_states=hidden_states,
                    prev_block_output=block_output,
                    prev_injection=injection,
                    positions=positions,
                )
            return self.layers[-1].mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )

    return SyntheticModel


class _CaptureHooks:
    """Forward hooks on the split layer and the late attention modules.

    The split hook records the materialized 10240-wide state entering layer
    ``split`` (``capture_probe.split_state``: the pending combine applied).
    The attention hook records each layer's 2560-wide block input. Both are
    copied to CPU as fp32 so the fit does not hold GPU memory.
    """

    def __init__(self, split: int, end: int) -> None:
        self.split = split
        self.end = end
        self.split_state: list = []
        self.layer_inputs: dict[int, list] = {i: [] for i in range(split, end)}
        self._split_state = _load("capture_probe").split_state

    def _split_hook(self, module, args, kwargs):
        hidden = self._split_state(module, args, kwargs)
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
    first = next(iter(maps.values()))
    manifest = {
        "format": "ridgefill-w4a16-fp16-v1",
        "model": {
            "vocab_sha256": meta.get("vocab_sha256", ""),
            "encodings": meta.get("encodings", {}),
        },
        "split": int(meta.get("split", SPLIT)),
        "tail": int(meta.get("tail", TAIL)),
        "din": int(first[0].shape[1]),
        "dout": int(first[0].shape[0]),
        "lambda": float(meta.get("lambda", LAM)),
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
    model = _synthetic_classes(torch)(din, dout, layers)
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
        {
            "vocab_sha256": "synthetic",
            "encodings": {},
            "split": args.split,
            "tail": args.tail,
            "lambda": args.lam,
        },
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
    os.environ["VLLM_RDNA_RIDGEFILL"] = "0"
    dump = Path(args.output) / "dump"
    dump.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    # TP>1 on a 32 GB V620 holds the W4A16 serve; TP=1 does not (the bf16
    # checkpoint is ~180 GB on disk, ~45 GB packed). The map is replicated,
    # so the TP=2 fit is the TP=4 serve map.
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        # serve_rdna.sh forces --dtype float16; the HC prefill kernels
        # require fp16 and a bf16 load dies in the profile run.
        dtype="float16",
        tensor_parallel_size=args.tp,
        # The flashnext recipe runs TP4/EP4: without EP the WNA16 MoE's static
        # group scales need 640 / TP divisible by 128 (160 and 320 are not),
        # so a bare TP=2 or TP=4 load refuses to start.
        enable_expert_parallel=args.tp > 1,
        # One decode token after the prompt must fit the model length.
        max_model_len=max(args.tokens, 4096) + 1,
        # Text-only capture: the ViT profile run allocates a 64 GiB SDPA
        # buffer on a 30 GiB V620 and OOMs before the first prompt.
        language_model_only=True,
        max_num_seqs=1,
        # One prompt per request, and one prefill chunk per prompt: the split
        # state is that prompt's tokens, not a batch of prompts. The probe can
        # then drop the one-row decode step by row count. The budget is twice
        # the prompt so the scheduler does not split it to leave decode room.
        max_num_batched_tokens=max(args.tokens * 2, 4096),
        # A repeated prefix would hit the prefix cache and capture only the
        # new rows; the fit needs every prompt's full prefill.
        enable_prefix_caching=False,
        disable_log_stats=True,
        worker_extension_cls=PROBE,
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
    if split_state[0].shape[1] != DIN:
        raise SystemExit(
            f"split state must be {DIN} wide, got {split_state[0].shape[1]}"
        )
    for layer_idx, chunks in layer_inputs.items():
        if chunks[0].shape[1] != DOUT:
            raise SystemExit(
                f"layer {layer_idx} is {chunks[0].shape[1]} wide, want {DOUT}"
            )
    maps, errors = _fit_maps(torch, split_state, layer_inputs, args.lam)
    output = Path(args.output)
    meta = {**_read_pins(args), "split": args.split, "tail": args.tail}
    meta["lambda"] = args.lam
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
    """``args.prompts`` prompts of exactly ``args.tokens`` corpus token ids.

    A shorter document is refused, never tiled: repeating text is the
    placeholder corpus this check exists to keep out of a fit.
    """
    from vllm.inputs import TokensPrompt

    docs: list[list[int]] = []
    if args.corpus_json:
        _, windows = _read_windows(args.corpus_json)
        docs = [list(w["ids"]) for w in windows]
    else:
        tokenizer = llm.get_tokenizer()
        docs = [tokenizer.encode(text) for text in _read_prompts(args)]
    prompts = []
    for i, ids in enumerate(docs[: args.prompts]):
        if len(ids) < args.tokens:
            raise SystemExit(
                f"document {i} has {len(ids)} tokens, fewer than --tokens "
                f"{args.tokens}; pass longer documents or a smaller --tokens"
            )
        prompts.append(TokensPrompt(prompt_token_ids=ids[: args.tokens]))
    if len(prompts) < args.prompts:
        raise SystemExit(
            f"the corpus has {len(prompts)} documents, want {args.prompts}"
        )
    return prompts


def _read_prompts(args) -> list[str]:
    """The documents of ``--prompts-file``, one per non-empty line."""
    if not args.prompts_file:
        raise SystemExit("a real capture needs --prompts-file or --corpus-json")
    path = Path(args.prompts_file)
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [t for t in texts if t]
    if not texts:
        raise SystemExit(f"{path} has no prompts")
    return texts


def _read_windows(path: str) -> tuple[dict, list[dict]]:
    """A ``ppl_long``/``make_fit_corpus`` windows file: ``meta`` and windows.

    Every window must hold ``meta.length`` ids whose ``ppl_long.ids_sha``
    matches its ``sha``, so a truncated or edited corpus file is refused.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from ppl_long import ids_sha

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta, windows = data.get("meta", {}), data.get("windows", [])
    if not windows:
        raise SystemExit(f"{path} has no windows")
    length = int(meta.get("length", 0))
    for i, win in enumerate(windows):
        ids = win.get("ids") or []
        if length and len(ids) != length:
            raise SystemExit(f"{path} window {i} has {len(ids)} ids, want {length}")
        if "sha" in win and ids_sha(ids) != win["sha"]:
            raise SystemExit(f"{path} window {i} does not match its sha")
    return meta, windows


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


def _ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _SyntheticLLM:
    """A TP-shaped stand-in for ``vllm.LLM`` over the synthetic model.

    Each fake rank is a ``RidgeFillProbe`` worker holding its own copy of one
    model, so the probe's hooks, RPC methods, rank-0 sums and per-rank
    checksums run as they do in the TP workers. ``generate`` runs a prompt
    the way the serve's scheduler does with chunked prefill: ``chunk``-row
    prefill forwards with absolute positions, then one single-row decode
    forward for the sampled token.
    """

    def __init__(self, torch, model, tp: int, chunk: int, probe) -> None:
        class Worker(probe.RidgeFillProbe):
            def __init__(self, rank: int, model) -> None:
                self.rank = rank
                self.model_runner = types.SimpleNamespace(model=model)

        self.torch = torch
        self.chunk = int(chunk)
        self.workers = [Worker(rank, copy.deepcopy(model)) for rank in range(tp)]

    def collective_rpc(self, method: str, args: tuple = ()) -> list:
        return [getattr(worker, method)(*args) for worker in self.workers]

    def generate(self, prompts: list, params=None, use_tqdm: bool = False) -> list:
        torch = self.torch
        for prompt in prompts:
            ids = torch.tensor(prompt["prompt_token_ids"], dtype=torch.long)
            rows = int(ids.numel())
            steps = [(s, min(rows, s + self.chunk)) for s in range(0, rows, self.chunk)]
            with torch.no_grad():
                for worker in self.workers:
                    model = worker.model_runner.model
                    for start, end in steps:
                        model(
                            input_ids=ids[start:end], positions=torch.arange(start, end)
                        )
                    model(input_ids=ids[-1:], positions=torch.tensor([rows]))
        return []


def _check_chunks(done: list, rows: int, role: str, index: int) -> list:
    """The forwards of one prompt cover its rows once, in order, on every rank."""
    chunks = [tuple(c) for c in done[0]["chunks"]]
    for other in done[1:]:
        if [tuple(c) for c in other["chunks"]] != chunks:
            raise SystemExit(
                f"{role} {index}: rank {other['rank']} saw forwards "
                f"{other['chunks']}, rank 0 saw {chunks}"
            )
    expect = 0
    for first, count in chunks:
        if first != expect:
            raise SystemExit(
                f"{role} {index}: a forward starts at position {first}, "
                f"expected {expect} (forwards {chunks})"
            )
        expect += count
    if expect != rows:
        raise SystemExit(f"{role} {index}: forwards cover {expect} of {rows} rows")
    return chunks


def _snapshot_entry(results: list, prompts: int, path: Path) -> dict:
    """Rank 0's snapshot facts and the largest rank deviation of ``sum x``.

    The split state is replicated, so every rank's fp64 ``sum x`` should
    match rank 0's; the deviation is relative to rank 0's largest entry.
    """
    lead = results[0]
    if "bytes" not in lead:
        raise SystemExit("rank 0 wrote no snapshot")
    deviation = 0.0
    for other in results[1:]:
        if other["rows"] != lead["rows"]:
            raise SystemExit(
                f"rank {other['rank']} saw {other['rows']} rows, rank 0 {lead['rows']}"
            )
        for split, values in other["check"].items():
            ref = lead["check"][split]
            scale = max(abs(v) for v in ref) or 1.0
            worst = max(abs(p - q) for p, q in zip(values, ref))
            deviation = max(deviation, worst / scale)
    return {
        "prompts": prompts,
        "file": path.name,
        "bytes": int(lead["bytes"]),
        "rows": int(lead["rows"]),
        "forwards": int(lead["forwards"]),
        "folds": int(lead.get("folds", 0)),
        "device_sums": bool(lead.get("device_sums", False)),
        "hook_seconds": [round(float(r["seconds"]), 3) for r in results],
        "rank_sum_x_max_rel": deviation,
    }


def _run_stats(llm, args, end: int, fit, held, make_prompt, params, record) -> int:
    """Drive the probe: fit windows in stats mode, held-out windows in raw mode."""
    out = Path(args.output)
    meta_path = out / "capture_meta.json"
    splits = sorted(set(_ints(args.splits)))
    snapshots = sorted(set(_ints(args.snapshots)))
    if not splits or splits[0] < 0 or splits[-1] >= end:
        raise SystemExit(f"--splits {args.splits} must lie in [0, {end})")
    if not snapshots or snapshots[0] < 1 or snapshots[-1] > len(fit):
        raise SystemExit(
            f"--snapshots {args.snapshots} must lie in [1, {len(fit)}] fit prompts"
        )
    record["begin"] = llm.collective_rpc(
        "ridgefill_stats_begin",
        args=(
            splits,
            end,
            args.raw_stride,
            args.raw_head,
            args.stats_head,
            args.fold_threads,
            args.fold == "prompt",
        ),
    )
    record.update(prompts=[], snapshots=[], rows={}, complete=False)
    started = time.perf_counter()

    def run(mode: str, role: str, index: int, window: dict) -> None:
        llm.collective_rpc("ridgefill_stats_mode", args=(mode, index))
        start = time.perf_counter()
        llm.generate([make_prompt(window["ids"])], params, use_tqdm=False)
        seconds = time.perf_counter() - start
        done = llm.collective_rpc("ridgefill_stats_mode", args=("off", -1))
        chunks = _check_chunks(done, len(window["ids"]), role, index)
        record["prompts"].append(
            {
                "role": role,
                "index": index,
                "book": window.get("book"),
                "title": window.get("title"),
                "offset": window.get("offset"),
                "sha": window.get("sha"),
                "tokens": len(window["ids"]),
                "forward_rows": [count for _, count in chunks],
                "seconds": round(seconds, 3),
                "fold_seconds": round(float(done[0].get("fold_seconds", 0.0)), 3),
            }
        )
        print(
            f"{role} {index:3d}: {len(window['ids'])} tokens in {len(chunks)} "
            f"forwards, {seconds:.1f} s",
            flush=True,
        )

    for index, window in enumerate(fit):
        run("stats", "fit", index, window)
        if index + 1 in snapshots:
            path = out / f"stats_p{index + 1:03d}.pt"
            start = time.perf_counter()
            results = llm.collective_rpc("ridgefill_stats_snapshot", args=(str(path),))
            entry = _snapshot_entry(results, index + 1, path)
            entry["write_seconds"] = round(time.perf_counter() - start, 3)
            record["snapshots"].append(entry)
            _write_json(meta_path, record)
            print(
                f"snapshot {path.name}: {entry['rows']} rows, "
                f"{entry['bytes'] / 1e9:.2f} GB, rank sum x max rel "
                f"{entry['rank_sum_x_max_rel']:.3g}",
                flush=True,
            )
    seconds = record.setdefault("seconds", {})
    seconds["fit"] = round(time.perf_counter() - started, 3)
    if args.stats_head > 0:
        _save_rows(llm, out, "head", record)
    start = time.perf_counter()
    for index, window in enumerate(held):
        run("raw", "held", index, window)
    seconds["held"] = round(time.perf_counter() - start, 3)
    _save_rows(llm, out, "held", record)
    seconds["capture"] = round(time.perf_counter() - started, 3)
    record["complete"] = True
    _write_json(meta_path, record)
    print(f"wrote {meta_path}")
    return 0


def _save_rows(llm, out: Path, kind: str, record: dict) -> None:
    path = out / f"rows_{kind}.pt"
    lead = llm.collective_rpc("ridgefill_stats_save_rows", args=(kind, str(path)))[0]
    if "bytes" not in lead:
        raise SystemExit(f"rank 0 wrote no {kind} rows")
    record["rows"][kind] = {
        "file": path.name,
        "rows": int(lead["rows"]),
        "bytes": int(lead["bytes"]),
    }
    print(f"rows {path.name}: {lead['rows']} rows, {lead['bytes'] / 1e9:.2f} GB")


def _stats_output(args) -> Path:
    out = Path(args.output)
    if (out / "capture_meta.json").exists():
        raise SystemExit(f"{out} already holds a capture; pick a new --output")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _corpus_record(path: str, meta: dict, windows: list) -> dict:
    return {
        "path": str(Path(path).resolve()),
        "sha256": _file_sha256(path),
        "windows": len(windows),
        "meta": meta,
    }


def _stats_record(args, torch, synthetic: bool) -> dict:
    env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(ENV_PREFIXES)
        and ("TOKEN" not in key or key.startswith("TOKENIZERS_"))
    }
    return {
        "format": CAPTURE_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "synthetic": synthetic,
        "model": args.model,
        "tree": str(ROOT),
        "torch": str(torch.__version__),
        "hip": str(getattr(torch.version, "hip", None)),
        "splits": sorted(set(_ints(args.splits))),
        "tail": args.tail,
        "raw_stride": args.raw_stride,
        "raw_head": args.raw_head,
        "stats_head": args.stats_head,
        "fold": args.fold,
        "argv": sys.argv,
        "env": env,
    }


def _random_windows(torch, count: int, tokens: int, vocab: int, seed: int) -> list:
    gen = torch.Generator().manual_seed(seed)
    return [
        {"book": i, "ids": torch.randint(vocab, (tokens,), generator=gen).tolist()}
        for i in range(count)
    ]


def _stats_synthetic(args) -> int:
    """The stats pipeline on the synthetic model, with fake TP ranks (CPU)."""
    torch = _torch()
    torch.manual_seed(7)
    probe = _load("capture_probe")
    out = _stats_output(args)
    vocab = max(2, args.synthetic_vocab)
    din, dout, layers = args.synthetic_din, args.synthetic_dout, args.synthetic_layers
    model = _synthetic_classes(torch)(din, dout, layers, vocab=vocab)
    record = _stats_record(args, torch, synthetic=True)
    if args.corpus_json:
        meta, fit = _read_windows(args.corpus_json)
        record["fit"] = _corpus_record(args.corpus_json, meta, fit)
    else:
        fit = _random_windows(torch, args.prompts, args.tokens, vocab, 11)
    if args.held_json:
        meta, held = _read_windows(args.held_json)
        record["held"] = _corpus_record(args.held_json, meta, held)
    else:
        held = _random_windows(torch, args.held_prompts, args.tokens, vocab, 13)
    end = min(args.end, layers)
    record.update(end=end, din=din, dout=dout, layers=layers, tp=args.tp)
    pins = {"vocab_sha256": "synthetic", "encodings": {}}
    _write_json(out / "pins.json", pins)
    record["pins"] = pins
    llm = _SyntheticLLM(torch, model, max(1, args.tp), args.max_batched_tokens, probe)
    record["engine"] = {
        "synthetic": True,
        "tensor_parallel_size": max(1, args.tp),
        "max_num_batched_tokens": args.max_batched_tokens,
    }

    def make_prompt(ids: list) -> dict:
        return {"prompt_token_ids": [int(i) % vocab for i in ids]}

    return _run_stats(llm, args, end, fit, held, make_prompt, None, record)


def _engine_args(args, length: int) -> dict:
    """The capture engine: the flashnext serve shape with MTP and RidgeFill off."""
    engine = {
        "model": args.model,
        # The hooks need eager forwards; the serve compiles the same ops.
        "enforce_eager": True,
        # serve_rdna.sh forces --dtype float16; the HC prefill kernels
        # require fp16 and a bf16 load dies in the profile run.
        "dtype": "float16",
        "tensor_parallel_size": args.tp,
        # Without EP the WNA16 MoE's static group scales need 640 / TP
        # divisible by 128, so a bare TP=4 load refuses to start.
        "enable_expert_parallel": args.tp > 1,
        # The serve's MAXLEN keeps its block and cache layout; the default
        # is the longest window plus the sampled token.
        "max_model_len": max(args.max_model_len, length + 1),
        # Text-only: the ViT profile run OOMs a 30 GiB V620.
        "language_model_only": True,
        # The serve's scheduler shape: one sequence, chunked prefill in
        # --max-batched-tokens steps, --block-size KV blocks, no prefix cache.
        "max_num_seqs": 1,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.max_batched_tokens,
        "block_size": args.block_size,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "worker_extension_cls": PROBE,
    }
    if args.kv_cache_bytes > 0:
        engine["kv_cache_memory_bytes"] = args.kv_cache_bytes
    if args.attention_backend:
        engine["attention_backend"] = args.attention_backend
    return engine


def _release_boot_lock(fd: int) -> None:
    """Drop the boot lock the launcher took, once the engine is up."""
    if fd < 0:
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _stats_real(args) -> int:
    torch = _torch()
    _check_dest_runtime(torch)
    # The capture is the exact model.
    os.environ["VLLM_RDNA_RIDGEFILL"] = "0"
    out = _stats_output(args)
    fit_meta, fit = _read_windows(args.corpus_json)
    held_meta, held = _read_windows(args.held_json)
    record = _stats_record(args, torch, synthetic=False)
    record["fit"] = _corpus_record(args.corpus_json, fit_meta, fit)
    record["held"] = _corpus_record(args.held_json, held_meta, held)
    record.update(end=args.end, tp=args.tp)
    pins = _read_pins(args)
    if not pins["vocab_sha256"]:
        raise SystemExit(
            "no vocab pin: pass --pins or a checkpoint with tokenizer.json"
        )
    _write_json(out / "pins.json", pins)
    record["pins"] = pins

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    length = max(len(w["ids"]) for w in fit + held)
    engine = _engine_args(args, length)
    record["engine"] = engine
    start = time.perf_counter()
    llm = LLM(**engine)
    record["seconds"] = {"boot": round(time.perf_counter() - start, 3)}
    _release_boot_lock(args.boot_lock_fd)
    params = SamplingParams(max_tokens=1, temperature=0.0)

    def make_prompt(ids: list) -> TokensPrompt:
        return TokensPrompt(prompt_token_ids=[int(i) for i in ids])

    return _run_stats(llm, args, args.end, fit, held, make_prompt, params, record)


def _require_corpus(args) -> None:
    """Only ``--synthetic`` runs without a real corpus."""
    if args.synthetic:
        return
    if args.stats:
        if not args.corpus_json or not args.held_json:
            raise SystemExit(
                "--stats needs --corpus-json (fit windows) and --held-json "
                "(held-out windows)"
            )
        return
    if not (args.prompts_file or args.corpus_json):
        raise SystemExit(
            "a real capture needs --prompts-file or --corpus-json; only "
            "--synthetic runs without a corpus"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="CPU run on a small Qwen4Exp-shaped model (no corpus needed)",
    )
    parser.add_argument("--model", default="", help="W4A16 checkpoint path")
    parser.add_argument(
        "--output",
        default=str(Path.home() / "w4a8_runs/lanes/ridgefill/capture"),
        help="capture directory",
    )
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--prompts-file", default="", help="one document per line")
    parser.add_argument(
        "--corpus-json",
        default="",
        help="windows file (ppl_long / make_fit_corpus); the fit windows of --stats",
    )
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
    parser.add_argument("--synthetic-vocab", type=int, default=4096)
    stats = parser.add_argument_group("statistics capture (--stats)")
    stats.add_argument("--stats", action="store_true", help="the large capture")
    stats.add_argument("--held-json", default="", help="held-out windows file")
    stats.add_argument(
        "--held-prompts",
        type=int,
        default=8,
        help="random held-out windows without --held-json (--synthetic only)",
    )
    stats.add_argument(
        "--splits", default=str(SPLIT), help="comma list; each split's state"
    )
    stats.add_argument("--snapshots", default="8,16,32,64", help="fit prompt counts")
    stats.add_argument("--raw-stride", type=int, default=4)
    stats.add_argument("--raw-head", type=int, default=64)
    stats.add_argument("--stats-head", type=int, default=64)
    stats.add_argument("--max-batched-tokens", type=int, default=2048)
    stats.add_argument("--block-size", type=int, default=1024)
    stats.add_argument(
        "--max-model-len", type=int, default=0, help="0: longest window + 1"
    )
    stats.add_argument("--kv-cache-bytes", type=int, default=4026531840)
    stats.add_argument("--attention-backend", default="RDNA_ATTN")
    stats.add_argument(
        "--fold-threads", type=int, default=0, help="rank 0 CPU threads (0: keep)"
    )
    stats.add_argument(
        "--fold",
        choices=("prompt", "forward"),
        default="prompt",
        help="fold rank 0's fp32 device sums into fp64 once per prompt "
        "(falls back to every forward if they do not fit) or every forward",
    )
    stats.add_argument(
        "--boot-lock-fd",
        type=int,
        default=-1,
        help="flock'd fd the launcher holds; released once the engine is up",
    )
    args = parser.parse_args(argv)
    if not args.synthetic and not args.model:
        raise SystemExit("pass --model PATH, or --synthetic for the CPU run")
    _require_corpus(args)
    if args.stats:
        return _stats_synthetic(args) if args.synthetic else _stats_real(args)
    if args.synthetic:
        return _synthetic(args)
    return _real(args)


if __name__ == "__main__":
    raise SystemExit(main())
