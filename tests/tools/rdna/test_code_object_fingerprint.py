# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Classification logic of tools/rdna/rdna3/code_object_fingerprint.py (stdlib only)."""

import argparse
import importlib.util
import json
import struct
from pathlib import Path

import pytest

TOOL = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "rdna"
    / "rdna3"
    / "code_object_fingerprint.py"
)


@pytest.fixture(scope="module")
def fp():
    spec = importlib.util.spec_from_file_location("code_object_fingerprint", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words(*values: int) -> bytes:
    return b"".join(struct.pack("<I", v) for v in values)


def test_empty_kernels_are_stub_only(fp):
    s_endpgm, s_code_end, s_nop = 0xBF810000, 0xBF9F0000, 0xBF800000
    assert fp._stub_only(b"")
    assert fp._stub_only(_words(s_endpgm, s_nop, s_nop, s_endpgm, s_code_end, 0, 0))


def test_any_other_instruction_is_not_a_stub(fp):
    v_add_f32 = 0x06000000  # a VOP2 word: real code
    assert not fp._stub_only(_words(0xBF810000, v_add_f32, 0xBF810000))


def test_a_trailing_partial_word_is_ignored(fp):
    assert fp._stub_only(_words(0xBF810000) + b"\xff\xff")


def _entry(text="t", rodata="r", note="n", kernels=("k",), stub_only=False):
    return {
        "text": text,
        "rodata": rodata,
        "note": note,
        "kernels": list(kernels),
        "stub_only": stub_only,
    }


def _compare(fp, tmp_path, capsys, base, new, strict=False):
    a, b = tmp_path / "base.json", tmp_path / "new.json"
    a.write_text(json.dumps(base))
    b.write_text(json.dumps(new))
    args = argparse.Namespace(base=str(a), new=str(b), strict=strict)
    with pytest.raises(SystemExit) as exit_info:
        fp.cmd_compare(args)
    return exit_info.value.code, capsys.readouterr().out


def test_identical_is_clean(fp, tmp_path, capsys):
    objs = {"a.o": _entry(), "b.o": _entry(stub_only=True)}
    code, out = _compare(fp, tmp_path, capsys, objs, objs)
    assert code == 0
    assert "IDENTICAL objects: 2" in out


def test_real_code_change_fails(fp, tmp_path, capsys):
    base = {"a.o": _entry()}
    new = {"a.o": _entry(text="other")}
    code, out = _compare(fp, tmp_path, capsys, base, new)
    assert code == 1
    assert "CHANGED a.o: text" in out


def test_change_from_stub_to_real_code_fails(fp, tmp_path, capsys):
    base = {"a.o": _entry(stub_only=True)}
    new = {"a.o": _entry(text="other", stub_only=False)}
    code, out = _compare(fp, tmp_path, capsys, base, new)
    assert code == 1
    assert "CHANGED a.o" in out


def test_stub_only_churn_is_reported_but_passes(fp, tmp_path, capsys):
    base = {"m.o": _entry(stub_only=True, kernels=("k1",))}
    new = {"m.o": _entry(text="x", stub_only=True, kernels=("k1", "k2", "k3"))}
    code, out = _compare(fp, tmp_path, capsys, base, new)
    assert code == 0
    assert "STUB-CHURN m.o: text, kernels (kernel symbols +2)" in out
    code, _ = _compare(fp, tmp_path, capsys, base, new, strict=True)
    assert code == 1


def test_object_set_changes_fail(fp, tmp_path, capsys):
    base = {"a.o": _entry(), "gone.o": _entry()}
    new = {"a.o": _entry(), "added.o": _entry()}
    code, out = _compare(fp, tmp_path, capsys, base, new)
    assert code == 1
    assert "ONLY-IN-BASE gone.o" in out
    assert "ONLY-IN-NEW added.o" in out


def test_missing_stub_flag_in_an_old_oracle_counts_as_changed(fp, tmp_path, capsys):
    old = _entry()
    del old["stub_only"]
    code, out = _compare(
        fp, tmp_path, capsys, {"a.o": old}, {"a.o": _entry(text="x", stub_only=True)}
    )
    assert code == 1
    assert "CHANGED a.o" in out
