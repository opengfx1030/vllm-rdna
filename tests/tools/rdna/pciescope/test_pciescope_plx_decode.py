# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEX88096 window decoders against the capture of par1-cs25 and lspci.

The register windows were read through BAR0 (read-only mmap); the expected values
come from ``lspci -vvv`` of the same functions, an independent decoder that reads
configuration space.  Every PEX function (36 of them) is compared field by field,
so a wrong bit position in ``pciescope.plx.pcie`` fails here.
"""

from __future__ import annotations

import plx_fakes
import pytest
from pciescope.plx import helper, pcie, regs
from pciescope.plx.ports import build_port_maps


@pytest.fixture(scope="module")
def capture():
    topo, chips, _ = plx_fakes.load_capture()
    expect = plx_fakes.load_lspci()
    rows = []  # (bdf, port, window)
    for cm in build_port_maps(topo):
        usp = f"0000:{cm.usp_bdf}"
        for p in cm.ports:
            rows.append((p.bdf, p.port, chips[usp][p.port]))
    return rows, expect


def _get(win):
    return lambda off: win.get(off, 0)


def test_every_pex_function_is_covered(capture):
    rows, expect = capture
    assert len(rows) == len(expect) == 36
    assert {r[0] for r in rows} == set(expect)


def test_static_fields_match_lspci(capture):
    rows, expect = capture
    for bdf, port, win in rows:
        st = pcie.decode_static(_get(win))
        e = expect[bdf]
        assert (st.vendor, st.device) == (regs.VENDOR_ID, regs.DEVICE_ID), bdf
        assert st.port_number == e["lnkcap_port"] == port, bdf
        assert st.max_speed_gts == e["lnkcap_speed"], bdf
        assert st.max_width == e["lnkcap_width"], bdf
        assert st.dl_active_reporting == e["lnkcap_dlactrep"], bdf
        assert st.mps_supported == e["devcap_mps"], bdf
        offsets = set(st.caps.std.values()) | set(st.caps.ext.values())
        assert offsets == set(e["caps"]), bdf


def test_port_types(capture):
    rows, expect = capture
    for bdf, _, win in rows:
        st = pcie.decode_static(_get(win))
        title = expect[bdf]["express"]
        want = (
            "endpoint"
            if "Endpoint" in title
            else "upstream"
            if "Upstream" in title
            else "downstream"
        )
        assert st.port_type_name == want, bdf


def _dynamic(win):
    st = pcie.decode_static(_get(win))
    vals = {name: win.get(off, 0) for name, off in pcie.dynamic_offsets(st)}
    return st, pcie.decode_dynamic(st, vals)


def test_link_state_matches_lspci(capture):
    rows, expect = capture
    for bdf, _, win in rows:
        st, ls = _dynamic(win)
        e = expect[bdf]
        assert ls.speed_gts == e["lnksta_speed"], bdf
        assert ls.width == e["lnksta_width"], bdf
        assert ls.training == e["lnksta_train"], bdf
        assert ls.bw_mgmt_status == e["lnksta_bwmgmt"], bdf
        assert ls.autonomous_bw_status == e["lnksta_abwmgmt"], bdf
        if st.dl_active_reporting:
            assert ls.link_up == e["lnksta_dlactive"], bdf
        else:
            assert ls.link_up == (e["lnksta_width"] > 0), bdf
        assert ls.target_speed_gts == e["lnkctl2_target"], bdf


def test_settings_match_lspci(capture):
    rows, expect = capture
    for bdf, _, win in rows:
        _, ls = _dynamic(win)
        e = expect[bdf]
        assert ls.mps == e["devctl_mps"], bdf
        assert ls.mrrs == e["devctl_mrrs"], bdf
        assert ls.aspm == e["lnkctl_aspm"], bdf
        if "acs_ctl_names" in e:
            want = sum(
                1 << bit
                for bit, name in pcie.ACS_BITS.items()
                if name in e["acs_ctl_names"]
            )
            assert ls.acs_ctl == want, bdf
        else:
            assert ls.acs_ctl is None, bdf


def _bits(mask):
    return [b for b in range(32) if mask >> b & 1]


def test_error_registers_match_lspci(capture):
    rows, expect = capture
    for bdf, _, win in rows:
        _, ls = _dynamic(win)
        e = expect[bdf]
        assert ls.dev_status == e["devsta"], bdf
        if "uncorr_bits" not in e:  # a function without the AER capability
            assert ls.uncorr_status is None and ls.corr_status is None, bdf
        else:
            assert _bits(ls.uncorr_status) == e["uncorr_bits"], bdf
            assert _bits(ls.corr_status) == e["corr_bits"], bdf
        if "lane_err" in e:
            assert ls.lane_err == e["lane_err"], bdf
        if "dpc_triggered" in e:
            assert bool(ls.dpc_status) == e["dpc_triggered"], bdf


def test_aer_and_devsta_bit_tables_follow_the_pcie_spec():
    assert pcie.UNCORR_BITS == {
        4: "DLP",
        5: "SDES",
        12: "TLP",
        13: "FCP",
        14: "CmpltTO",
        15: "CmpltAbrt",
        16: "UnexpectedCmplt",
        17: "RxOF",
        18: "MalfTLP",
        19: "ECRC",
        20: "UnsupReq",
        21: "ACSViol",
        22: "UncorrIntErr",
        23: "BlockedTLP",
        24: "AtomicOpBlocked",
        25: "TLPBlockedErr",
        26: "PoisonTLPBlocked",
        27: "DMWrReqBlocked",
        28: "IDECheck",
        29: "MisroutedIDETLP",
        30: "PCRC_CHECK",
        31: "TLPXlatBlocked",
    }
    assert pcie.CORR_BITS == {
        0: "RxErr",
        6: "BadTLP",
        7: "BadDLLP",
        8: "Rollover",
        12: "Timeout",
        13: "NonFatalErr",
        14: "CorrIntErr",
        15: "HeaderOF",
    }
    assert pcie.DEVSTA_BITS == {
        0: "CorrErr",
        1: "NonFatalErr",
        2: "FatalErr",
        3: "UnsupReq",
    }
    assert pcie.bit_names(1 << 4 | 1 << 20, pcie.UNCORR_BITS) == ["DLP", "UnsupReq"]
    assert pcie.bit_names(1 | 1 << 6, pcie.CORR_BITS) == ["RxErr", "BadTLP"]
    assert pcie.bit_names(1 << 1 | 1 << 2, pcie.UNCORR_BITS) == []  # reserved bits


def test_equalization_matches_lspci(capture):
    rows, expect = capture
    for bdf, _, win in rows:
        _, ls = _dynamic(win)
        e = expect[bdf]
        if ls.speed_gts >= 16.0 and "phy16_complete" in e:
            assert ls.eq_complete == e["phy16_complete"], bdf
        elif ls.speed_gts >= 8.0:
            assert ls.eq_complete == e["lnksta2_eq_complete"], bdf
        else:
            assert ls.eq_complete is None, bdf


def test_degraded_flags(capture):
    rows, _ = capture
    for bdf, port, win in rows:
        _, ls = _dynamic(win)
        # every GPU link of the box runs at its LnkCap maximum (16 GT/s x16)
        assert not ls.degraded, (bdf, port)


def test_serial_number_decodes_like_lspci(capture):
    rows, _ = capture
    serials = {pcie.decode_static(_get(w)).serial for _, port, w in rows}
    assert serials == {"00-80-5e-10-f9-57-20-61", "00-80-5e-10-fa-53-6b-2a"}


def test_dynamic_offsets_are_aligned_and_never_sensitive(capture):
    rows, _ = capture
    for bdf, _, win in rows:
        st = pcie.decode_static(_get(win))
        for name, off in pcie.dynamic_offsets(st):
            assert off % 4 == 0 and off < regs.PORT_WINDOW_SIZE, (bdf, name)
            assert off not in regs.READ_SENSITIVE, (bdf, name)


def test_walk_caps_is_bounded_on_a_loop():
    # a standard capability list that points to itself must terminate
    win = {0x04: 0x0010_0000, 0x34: 0x40, 0x40: 0x0000_4001}
    caps = pcie.walk_caps(_get(win))
    assert caps.std == {1: 0x40}
    # an extended list that loops back to 0x100 as well
    win[0x100] = 0x1_0001 | (0x100 << 20)
    assert pcie.walk_caps(_get(win)).ext == {1: 0x100}


def test_enabled_ports_from_masks():
    assert helper.enabled_ports([0x00010001, 0x00010001, 0x00010001, 0x3]) == [
        0,
        16,
        32,
        48,
        64,
        80,
        116,
        117,
    ]
    assert helper.enabled_ports([0, 0, 0, 0]) == []


def test_dpc_reason_text():
    assert pcie.dpc_reason(0) == "not triggered"
    assert pcie.dpc_reason(0b0_01_1) == "ERR_NONFATAL"
    assert pcie.dpc_reason(0b0_10_1) == "ERR_FATAL"
    assert pcie.dpc_reason(0b0_00_1) == "unmasked uncorrectable error"
    assert pcie.dpc_reason(0b1_11_1 | 1 << 5) == "software trigger"
