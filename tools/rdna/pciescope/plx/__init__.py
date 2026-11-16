# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Broadcom PEX88096 ("Atlas", PCI 1000:c010) switch telemetry for pciescope.

Layout (every module is stdlib only; this ``__init__`` stays import-light so
``pciescope.model`` may import ``pciescope.plx.metrics`` without a cycle):

``regs``     register offsets and bit fields (documented with their source)
``pcie``     decoders for the standard capabilities read through BAR0
``ports``    PEX port number <-> BDF <-> GPU map built from the topology
``helper``   root-side BAR0 reader (a long-lived ``sudo -n`` child process);
             the only code that can write, and only the gated perf sequence
``access``   client of the helper (or in-process reader when already root)
``perf``     PerformancePAK counter layout and decode
``metrics``  metric registry rows of the ``plx.*`` namespace
``cli``      ``python -m pciescope.plx`` (ports, status, dump, perf-*)
"""
