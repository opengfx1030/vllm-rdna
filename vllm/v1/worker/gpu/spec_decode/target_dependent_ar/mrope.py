# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


class MropeInputBuffers:
    """Model coordinates, separate from logical KV and sampling positions."""

    def __init__(
        self,
        num_dims: int,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ) -> None:
        self.num_dims = num_dims
        # Prefix views stay contiguous for each captured token count.
        self._prefill = torch.zeros(
            num_dims * max_num_tokens, dtype=torch.int64, device=device
        )
        self._decode = torch.zeros(
            num_dims * max_num_reqs, dtype=torch.int64, device=device
        )
        self._deltas = torch.zeros(
            (num_dims, max_num_reqs), dtype=torch.int64, device=device
        )

    def prepare_prefill(
        self,
        target_positions: torch.Tensor,
        last_token_indices: torch.Tensor,
        logical_positions: torch.Tensor,
    ) -> None:
        self.get_prefill_positions(target_positions.shape[1]).copy_(target_positions)
        num_reqs = last_token_indices.numel()
        deltas = self._deltas[:, :num_reqs]
        torch.index_select(target_positions, 1, last_token_indices, out=deltas)
        # Reuse decode output as scratch before the next decode overwrites it.
        last_logical = self._decode[:num_reqs]
        torch.index_select(logical_positions, 0, last_token_indices, out=last_logical)
        # A prompt's final delta is not valid during a partial multimodal prefill.
        deltas.sub_(last_logical.unsqueeze(0))
        self._deltas[:, num_reqs:].zero_()

    def get_prefill_positions(self, num_tokens: int) -> torch.Tensor:
        return self._prefill[: self.num_dims * num_tokens].view(
            self.num_dims, num_tokens
        )

    def get_decode_positions(self, logical_positions: torch.Tensor) -> torch.Tensor:
        num_tokens = logical_positions.numel()
        positions = self._decode[: self.num_dims * num_tokens].view(
            self.num_dims, num_tokens
        )
        torch.add(
            logical_positions.unsqueeze(0).expand(self.num_dims, -1),
            self._deltas[:, :num_tokens],
            out=positions,
        )
        return positions
