# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention / expert disaggregation across an RDNA3 fast tier and RDNA2.

Default off. ``RoutedExperts.forward_modular`` checks ``VLLM_HETERO_MOE``
and does not import this package unless that variable is ``1``.
"""

from vllm.distributed.hetero_moe.gate import hetero_moe_enabled

__all__ = ["hetero_moe_enabled"]
