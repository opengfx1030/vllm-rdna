# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import Qwen4ExpConfig


class Qwen4ExpMTPConfig(Qwen4ExpConfig):
    """Standalone MTP view retaining the Qwen4Exp backbone configuration.

    A distinct class prevents AutoConfig registration from changing the target
    config's model_type when target and draft are loaded in the same process.
    """

    model_type = "qwen4_exp_mtp"
