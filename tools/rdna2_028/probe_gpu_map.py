# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print per-HIP-device free/used VRAM to pick a co-tenant-safe GPU set."""

import torch

n = torch.cuda.device_count()
print(f"device_count={n}")
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    p = torch.cuda.get_device_properties(i)
    print(
        f"HIP{i} bdf={getattr(p, 'pci_bus_id', '?')} "
        f"free={free / 2**30:.1f}GiB used={(total - free) / 2**30:.1f}GiB "
        f"{p.gcnArchName}"
    )
