# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension that captures RidgeFill activations inside each worker.

A TP>1 serve runs the model in worker processes, so hooks registered in the
driver never see a forward. ``worker_extension_cls`` injects this class into
every worker; the driver then drives it with ``collective_rpc``:

    llm.collective_rpc("ridgefill_capture", args=(split, end))
    llm.generate(prompts, params)
    llm.collective_rpc("ridgefill_save", args=(dump_dir,))

Each worker writes ``rank<rank>.pt`` with the fp32 split state and the
fp32 block inputs of the late layers, one entry per forward. The driver
fits on rank 0, whose tensors and the map are the same on every TP rank
(the hyperconnection residual is replicated, not sharded).
"""

from __future__ import annotations

from pathlib import Path


class RidgeFillProbe:
    """Extension with two new RPC methods; it overrides no worker method."""

    def _ridgefill_layers(self):
        model = self.model_runner.model  # type: ignore[attr-defined]
        return getattr(model, "model", model).layers

    def ridgefill_capture(self, split: int, end: int) -> None:
        """Hook the split layer and the late attention modules."""
        layers = self._ridgefill_layers()
        if len(layers) < end:
            raise RuntimeError(f"model has {len(layers)} layers, capture wants {end}")
        self._ridgefill_split: list = []
        self._ridgefill_inputs: dict[int, list] = {
            layer: [] for layer in range(split, end)
        }

        def split_hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is not None:
                self._ridgefill_split.append(hidden.detach().float().cpu())

        def input_hook(layer: int):
            def hook(module, args, kwargs):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if hidden is not None:
                    self._ridgefill_inputs[layer].append(
                        hidden.detach().float().cpu()
                    )

            return hook

        for layer in range(split, end):
            module = layers[layer]
            if layer == split:
                module.register_forward_pre_hook(split_hook, with_kwargs=True)
            attn = getattr(module, "self_attn", None)
            if attn is None:
                attn = getattr(module, "linear_attn", None)
            if attn is None:
                raise RuntimeError(f"layer {layer} has no attention module")
            attn.register_forward_pre_hook(input_hook(layer), with_kwargs=True)

    def ridgefill_save(self, path: str) -> None:
        """Write this rank's captures; the driver reads rank 0."""
        import torch

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        rank = int(self.rank)  # type: ignore[attr-defined]
        torch.save(
            {
                "split": self._ridgefill_split,
                "inputs": self._ridgefill_inputs,
            },
            out / f"rank{rank}.pt",
        )
