# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather rows out of an EXL3 n-gram table without materializing it.

Two layouts:

- ``Exl3NgramTable``: per-shard trellis/suh/svh tiles. A lookup decodes only
  the 16-column tiles that contain the requested rows. ``hadamard="in"``
  matches the quantizer (Hadamard on the input axis). ``hadamard="both"``
  matches exllamav3, which also mixes each 128-row group.
- ``Exl3NgramRingTable``: exllamav3's ``exl3_ngram_trellis`` format
  (``ngram_embedding.safetensors`` next to an exllamav3 >= 1.4.5 checkpoint).
  Every row is one tail-biting trellis ring over the mul1 codebook plus an
  fp16 scale, and ``row = mul1[state] * scale + head_bias[head]``.
"""

import ctypes
import logging
import os
import re

import numpy as np
import torch

from vllm.model_executor.layers.quantization.exl3 import _exl3_hadamard


class Exl3NgramTable:
    layout = "exl3"
    is_fp8 = False

    def __init__(self, hadamard: str, n_rows: int, n_parts: int, width: int) -> None:
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
        assert self.bits is not None, "EXL3 n-gram trellis not loaded"
        torch.ops._rocm_C.exl3_decode_trellis_rdna2(sub, raw, int(self.bits), 0)
        return raw

    def _rows_from_raw(self, raw, suh, svh, local_rows, device):
        """Raw is [K, N] for the decoded tile slice. Return [R, K]."""
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
        gathered = torch.empty(
            ids.shape[0], self.width, dtype=torch.float32, device=device
        )
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
                        raw,
                        rec["suh"],
                        rec["svh"],
                        torch.tensor(uniq, device=device),
                        device,
                    )
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


_MUL1 = 0x83DCD12D
_logger = logging.getLogger(__name__)


def exl3_side_file(model_config, name: str, download_dir: str | None = None):
    """Path of a file exllamav3 ships beside an EXL3 checkpoint, outside the
    model index (``ngram_embedding.safetensors``,
    ``mtp_hyper_connection_mixer_patch.safetensors``); None if absent."""
    if getattr(model_config, "quantization", None) != "exl3":
        return None
    model = model_config.model
    if os.path.isdir(model):
        path = os.path.join(model, name)
        return path if os.path.isfile(path) else None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            model, name, revision=model_config.revision, cache_dir=download_dir
        )
    except Exception:
        return None


def exl3_side_weights(name: str):
    """(name, tensor) pairs of an EXL3 side file, empty when there is none."""
    from safetensors import safe_open

    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config()
    path = exl3_side_file(config.model_config, name, config.load_config.download_dir)
    if path is None:
        return
    with safe_open(path, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
            yield key, f.get_tensor(key)


def mul1_codebook() -> torch.Tensor:
    """All 65536 decoded mul1 values (fp16), bit-exact with decode_3inst<2>."""
    s = torch.arange(65536, dtype=torch.int64)
    prod = (s * _MUL1) & 0xFFFFFFFF
    bsum = (
        (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    )
    k_inv = torch.tensor([0x1EEE], dtype=torch.int32).to(torch.int16)
    k_bias = torch.tensor([0xC931 - 0x10000], dtype=torch.int32).to(torch.int16)
    k_inv = k_inv.view(torch.float16).float()
    k_bias = k_bias.view(torch.float16).float()
    return ((1024 + bsum).float() * k_inv + k_bias).to(torch.float16)


class Exl3NgramRingTable:
    """exllamav3 ``exl3_ngram_trellis`` n-gram table, mmapped from disk.

    A row is ``1 + width * K / 16`` int16 words: word 0 holds the fp16 scale,
    the rest a ``width * K``-bit stream where bits ``[i*K, (i+1)*K)`` are the
    low K bits of position i's 16-bit trellis state. The state's higher bits
    are the symbols of the preceding positions (mod width), K bits each.

    Same sidecar interface as ``_PleQuantTable`` (vllm/v1/ple_offload).
    """

    ROWS_PER_SHARD = 2_500_012
    layout = "exl3_ring"
    is_fp8 = False

    def __init__(
        self,
        path: str,
        total_rows: int,
        width: int,
        head_offsets: torch.Tensor | None = None,
        head_vocab_sizes: torch.Tensor | None = None,
        layer_multipliers: torch.Tensor | None = None,
    ) -> None:
        from safetensors import safe_open

        # Kept open: the shard tensors are mmap-backed views of the file.
        self._file = safe_open(path, framework="pt")  # noqa: SIM115
        shard_re = re.compile(r"\.shard_(\d+)\.trellis$")
        shards: dict[int, str] = {}
        aux: dict[str, str] = {}
        for name in self._file.keys():  # noqa: SIM118 - safe_open is not a dict
            m = shard_re.search(name)
            if m:
                shards[int(m.group(1))] = name
            else:
                aux[name.rsplit(".", 1)[-1]] = name
        if not shards or sorted(shards) != list(range(len(shards))):
            raise ValueError(f"{path}: no contiguous shard_N.trellis tensors")
        self._q = [self._file.get_tensor(shards[n]) for n in range(len(shards))]
        rows = sum(t.shape[0] for t in self._q)
        if rows != total_rows:
            raise ValueError(
                f"{path}: table has {rows} rows, layer expects {total_rows}"
            )
        if any(t.shape[0] != self.ROWS_PER_SHARD for t in self._q):
            raise ValueError(f"{path}: shards are not {self.ROWS_PER_SHARD} rows")
        self.words = int(self._q[0].shape[1])
        self.K = (self.words - 1) * 16 // width
        if self.words != 1 + width * self.K // 16:
            raise ValueError(f"{path}: {self.words} words is not a {width}-wide ring")
        self.width = width
        self.n_shards = len(self._q)
        self.quant_dir = os.path.dirname(path)
        self.path = path

        self.head_offsets = self._file.get_tensor(aux["head_offsets"]).long()
        self.head_vocab_sizes = self._file.get_tensor(aux["head_vocab_sizes"]).long()
        self.layer_multipliers = self._file.get_tensor(aux["layer_multipliers"]).long()
        self.head_bias = self._file.get_tensor(aux["head_bias"]).float()
        expected = {
            "head_offsets": head_offsets,
            "head_vocab_sizes": head_vocab_sizes,
            "layer_multipliers": layer_multipliers,
        }
        for key, want in expected.items():
            if want is None:
                continue
            got = getattr(self, key)
            if not torch.equal(got, torch.as_tensor(want).to("cpu", torch.long)):
                raise ValueError(
                    f"{path}: {key} does not match the layer's hash layout "
                    f"(checkpoint {got.tolist()}, "
                    f"layer {torch.as_tensor(want).tolist()})"
                )

        self.codebook = mul1_codebook().float()
        bit = torch.arange(width) * self.K
        self._w0 = bit >> 4
        self._w1 = (self._w0 + 1) % (self.words - 1)
        self._shift = bit & 15
        self._q_np = [t.numpy().view(np.ndarray) for t in self._q]
        self._s_np: list[np.ndarray] = []
        _logger.info(
            "EXL3 n-gram ring table: %d shards x %d rows, K=%d, mmapped from %s",
            self.n_shards,
            self.ROWS_PER_SHARD,
            self.K,
            path,
        )

    def populate_page_tables(self) -> bool:
        """madvise(MADV_POPULATE_READ) every shard mapping (see _PleQuantTable)."""
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError:
            return False
        for arr in self._q_np:
            base = arr.ctypes.data
            off = base & 4095
            if libc.madvise(
                ctypes.c_void_p(base - off), ctypes.c_size_t(arr.nbytes + off), 22
            ):
                return False
        return True

    def decode(self, raw: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Packed rings ``raw`` [N, words] int16 of rows ``ids`` -> [N, width]."""
        scales = raw[:, 0].contiguous().view(torch.float16).float()
        stream = raw[:, 1:].to(torch.int64) & 0xFFFF
        window = stream[:, self._w0] | (stream[:, self._w1] << 16)
        symbols = (window >> self._shift) & ((1 << self.K) - 1)
        states = torch.zeros_like(symbols)
        for j in range((15 + self.K) // self.K):
            states |= torch.roll(symbols, j, dims=1) << (j * self.K)
        heads = torch.searchsorted(self.head_offsets, ids, right=True) - 1
        heads = heads.clamp_(0, self.head_bias.shape[0] - 1)
        return self.codebook[states & 0xFFFF] * scales[:, None] + self.head_bias[heads]

    def gather_into(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        ids = ids.reshape(-1).long().cpu()
        shard = ids // self.ROWS_PER_SHARD
        local = ids - shard * self.ROWS_PER_SHARD
        order = torch.argsort(shard)
        s_sorted, l_sorted = shard[order], local[order]
        uniq, counts = torch.unique_consecutive(s_sorted, return_counts=True)
        raw = torch.empty(ids.shape[0], self.words, dtype=torch.int16)
        pos = 0
        for s, c in zip(uniq.tolist(), counts.tolist()):
            raw[pos : pos + c] = self._q[s].index_select(0, l_sorted[pos : pos + c])
            pos += c
        rows = self.decode(raw, ids[order])
        out[order.to(out.device)] = rows.to(device=out.device, dtype=out.dtype)
