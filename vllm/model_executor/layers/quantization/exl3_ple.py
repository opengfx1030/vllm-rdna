# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather rows out of an EXL3 n-gram table without materializing it.

The table is stored as per-shard trellis/suh/svh. A lookup decodes only the
16-column tiles that contain the requested rows. ``hadamard="in"`` matches
the quantizer (Hadamard on the input axis). ``hadamard="both"`` matches
exllamav3, which also mixes each 128-row group.
"""

import torch

from vllm.model_executor.layers.quantization.exl3 import _exl3_hadamard


class Exl3NgramTable:
    layout = "exl3"
    is_fp8 = False

    def __init__(self, hadamard: str, n_rows: int, n_parts: int,
                 width: int) -> None:
        self.hadamard = hadamard
        self.n_rows = int(n_rows)
        self.n_parts = int(n_parts)
        self.width = int(width)
        self.shards: dict[int, dict] = {}
        self.bits: int | None = None

    def _span(self, index: int) -> tuple[int, int]:
        shard = (self.n_rows + self.n_parts - 1) // self.n_parts
        start = index * shard
        rows = max(0, min(shard, self.n_rows - start))
        return start, rows

    def add(self, index: int, kind: str, tensor: torch.Tensor) -> None:
        start, rows = self._span(index)
        rec = self.shards.setdefault(index, {"start": start, "rows": rows})
        rec[kind] = tensor.detach().cpu().contiguous()
        if kind == "trellis":
            words = int(tensor.shape[-1])
            if words % 16 or not 1 <= words // 16 <= 8:
                raise ValueError(f"EXL3 n-gram trellis width {words}")
            self.bits = words // 16

    def _decode_tiles(self, trellis, n0: int, n1: int, device):
        sub = trellis[:, n0:n1].to(device=device, dtype=torch.int16).contiguous()
        k = sub.shape[0] * 16
        n = (n1 - n0) * 16
        raw = torch.zeros(k, n, dtype=torch.float16, device=device)
        torch.ops._rocm_C.exl3_decode_trellis_rdna2(
            sub, raw, int(self.bits), 0)
        return raw

    def _rows_from_raw(self, raw, suh, svh, local_rows, device):
        """raw is [K, N] for the decoded tile slice. Return [R, K]."""
        suh = suh.to(device=device, dtype=torch.float16).reshape(-1)
        svh = svh.to(device=device, dtype=torch.float16).reshape(-1)
        cols = raw[:, local_rows].transpose(0, 1).contiguous()
        cols = svh[local_rows].unsqueeze(1) * cols * suh.unsqueeze(0)
        out = torch.zeros_like(cols)
        _exl3_hadamard(cols, out, None, None, 1.0)
        return self._crop_width(out)

    def gather_into(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        if self.bits is None:
            raise RuntimeError("EXL3 n-gram table has no trellis shards")
        device = out.device
        ids = ids.reshape(-1).long()
        gathered = torch.empty(ids.shape[0], self.width,
                               dtype=torch.float32, device=device)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for pos, row in enumerate(ids.tolist()):
            shard_rows = (self.n_rows + self.n_parts - 1) // self.n_parts
            index = row // shard_rows
            rec = self.shards.get(index)
            if rec is None or "trellis" not in rec:
                raise KeyError(f"EXL3 n-gram shard {index} was not loaded")
            local = row - rec["start"]
            by_shard.setdefault(index, []).append((pos, local))
        for index, hits in by_shard.items():
            rec = self.shards[index]
            trellis = rec["trellis"]
            locals_ = [local for _, local in hits]
            if self.hadamard == "both":
                groups = sorted({local // 128 for local in locals_})
                for group in groups:
                    n0 = group * 8
                    n1 = n0 + 8
                    n1 = min(n1, trellis.shape[1])
                    raw = self._decode_tiles(trellis, n0, n1, device)
                    block = self._finish_both(raw, rec["suh"], rec["svh"])
                    base = group * 128
                    for pos, local in hits:
                        if local // 128 != group:
                            continue
                        gathered[pos] = block[local - base].float()
            else:
                tiles = sorted({local // 16 for local in locals_})
                for tile in tiles:
                    raw = self._decode_tiles(trellis, tile, tile + 1, device)
                    cols = [local % 16 for _, local in hits if local // 16 == tile]
                    # unique columns
                    uniq = sorted(set(cols))
                    block = self._rows_from_raw(
                        raw, rec["suh"], rec["svh"],
                        torch.tensor(uniq, device=device), device)
                    col_of = {c: i for i, c in enumerate(uniq)}
                    for pos, local in hits:
                        if local // 16 != tile:
                            continue
                        gathered[pos] = block[col_of[local % 16]].float()
        out.copy_(gathered.to(out.dtype))

    def _finish_both(self, raw, suh, svh):
        """exllamav3 reconstruct of one 128-row group. raw is [K, N]."""
        device = raw.device
        suh = suh.to(device=device, dtype=torch.float16).reshape(-1)
        svh = svh.to(device=device, dtype=torch.float16).reshape(-1)
        k, n = raw.shape
        wh = raw.transpose(0, 1).contiguous()
        wh = svh[:n].unsqueeze(1) * wh * suh.unsqueeze(0)
        k_side = torch.zeros_like(wh)
        _exl3_hadamard(wh, k_side, None, None, 1.0)
        if n % 128 != 0:
            return self._crop_width(k_side)
        src = k_side.transpose(0, 1).contiguous()
        dst = torch.zeros_like(src)
        _exl3_hadamard(src, dst, None, None, 1.0)
        return self._crop_width(dst.transpose(0, 1).contiguous())

    def _crop_width(self, rows: torch.Tensor) -> torch.Tensor:
        """Drop Hadamard padding. N-gram rows are 160 wide, stored as 256."""
        width = rows.shape[-1]
        if width == self.width:
            return rows
        if width < self.width:
            raise ValueError(
                f"EXL3 n-gram row width {width} is narrower than head dim {self.width}"
            )
        return rows[:, : self.width].contiguous()
