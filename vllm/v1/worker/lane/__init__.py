# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-lane engine, worker side (S3).

``streams`` owns the four dedicated-queue CU-masked streams, ``fences`` the
cross-lane end-of-batch events (F1), and ``runtime`` the per-lane threads,
inboxes and response queues.
"""
