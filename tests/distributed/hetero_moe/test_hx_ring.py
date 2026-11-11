# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hx step ring (``ring.py``): roundtrip, torn reads, overrun,
close, open failures and a producer/consumer pair over 20,000 steps."""

import struct
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from vllm.distributed.hetero_moe import hx_proto as P  # noqa: E402
from vllm.distributed.hetero_moe import ring as ring_mod  # noqa: E402
from vllm.distributed.hetero_moe.ring import (  # noqa: E402
    ENTRY_BYTES,
    HEADER_BYTES,
    RING_MAGIC,
    RingOverrun,
    StepDesc,
    StepRing,
    _check,
)

HEAD_OFF = 16
CLOSED_OFF = 12
CHECK_OFF = 24  # inside an entry


def entry_off(n: int, depth: int) -> int:
    return HEADER_BYTES + ((n - 1) % depth) * ENTRY_BYTES


@pytest.fixture
def pair(tmp_path):
    """(writer, reader) of a depth-8 ring."""
    path = tmp_path / "ring.bin"
    writer = StepRing.create(path, depth=8)
    reader = StepRing.open(path, timeout_s=5.0)
    yield writer, reader
    reader.close()
    writer.close()


def test_roundtrip_many_steps_with_every_field(pair):
    writer, reader = pair
    assert writer.writable and not reader.writable
    assert writer.depth == reader.depth == 8
    sent = []
    for n in range(1, 41):  # five times the depth: slots are reused
        epoch, T, bucket = 1000 + 7 * n, (n * 5) % 48 + 1, (0, 3, 6, 12)[n % 4]
        opcode = P.HX_OPCODE_MOE if n % 5 else 2
        assert writer.post(epoch, T, bucket, opcode) == n
        sent.append(StepDesc(n, epoch, T, bucket, opcode))
        assert reader.head == n
        assert reader.get(n) == sent[-1]
    # the last `depth` steps are still readable, in any order
    for desc in reversed(sent[-8:]):
        assert reader.get(desc.step_no) == desc


def test_default_opcode_is_moe_and_u64_epoch_survives(pair):
    writer, reader = pair
    epoch = (1 << 64) - 5
    writer.post(epoch, 2048, 0)
    desc = reader.get(1)
    assert desc == StepDesc(1, epoch, 2048, 0, P.HX_OPCODE_MOE)


def test_get_of_a_step_not_posted_yet_is_none(pair):
    writer, reader = pair
    assert reader.head == 0
    assert reader.get(1) is None
    writer.post(1, 3, 3)
    assert reader.get(1) is not None
    assert reader.get(2) is None
    assert reader.get(500) is None


def test_reader_cannot_post_and_writer_close_is_noop_for_reader(pair):
    writer, reader = pair
    with pytest.raises(RuntimeError, match="read-only"):
        reader.post(1, 3, 3)
    reader.close_writer()  # a reader has no say in closing
    assert not writer.closed and not reader.closed


def test_create_validates_depth_and_truncates_an_earlier_file(tmp_path):
    with pytest.raises(ValueError, match="depth"):
        StepRing.create(tmp_path / "x.bin", depth=1)
    path = tmp_path / "ring.bin"
    first = StepRing.create(path, depth=4)
    for n in range(1, 6):
        first.post(n, 3, 3)
    first.close_writer()
    first.close()
    second = StepRing.create(path, depth=4)  # a new run of the hub
    reader = StepRing.open(path, timeout_s=5.0)
    try:
        assert reader.head == 0 and not reader.closed
        assert reader.get(1) is None
        assert path.stat().st_size == HEADER_BYTES + 4 * ENTRY_BYTES
    finally:
        reader.close()
        second.close()


def test_closed_after_close_writer(pair):
    writer, reader = pair
    writer.post(1, 3, 3)
    assert not reader.closed
    writer.close_writer()
    assert reader.closed and writer.closed
    assert reader.get(1) is not None  # what was posted stays readable


# ---- torn and not-yet-visible entries ----------------------------------------------


@pytest.fixture
def fake_sleep(monkeypatch):
    """Replace ``time`` inside ring.py: ``sleep`` counts calls and runs a hook (a
    stand-in for the writer finishing its store between two reader retries)."""
    state = SimpleNamespace(calls=0, hook=None)

    def sleep(_seconds):
        state.calls += 1
        if state.hook is not None:
            state.hook(state.calls)

    monkeypatch.setattr(
        ring_mod, "time", SimpleNamespace(sleep=sleep, monotonic=time.monotonic)
    )
    return state


def test_torn_entry_is_retried_until_repaired(pair, fake_sleep):
    writer, reader = pair
    writer.post(77, 5, 6)
    off = entry_off(1, 8) + CHECK_OFF
    good = bytes(reader._mm[off : off + 8])
    reader._mm[off : off + 8] = bytes(b ^ 0xFF for b in good)  # torn: check mismatch

    def repair(call: int) -> None:
        if call == 3:
            reader._mm[off : off + 8] = good

    fake_sleep.hook = repair
    assert reader.get(1) == StepDesc(1, 77, 5, 6, P.HX_OPCODE_MOE)
    assert fake_sleep.calls == 3


@pytest.mark.parametrize("field_off", [0, 8, 16, 20, 22], ids=lambda o: f"off{o}")
def test_any_field_flip_breaks_the_check_word(pair, fake_sleep, field_off):
    writer, reader = pair
    writer.post(77, 5, 6)
    off = entry_off(1, 8) + field_off
    reader._mm[off] ^= 0x01
    with pytest.raises(RuntimeError, match="never became consistent"):
        reader.get(1)
    assert fake_sleep.calls == 100  # bounded retry, never a silent misread


def test_head_visible_before_the_entry_is_retried(pair, fake_sleep):
    writer, reader = pair
    struct.pack_into("<Q", reader._mm, HEAD_OFF, 1)  # head stored, entry not yet
    assert reader.head == 1

    def entry_lands(call: int) -> None:
        if call == 2:
            writer.post(9, 4, 3)

    fake_sleep.hook = entry_lands
    # the writer's own post() also moves head to 1: same value
    assert reader.get(1) == StepDesc(1, 9, 4, 3, P.HX_OPCODE_MOE)
    assert fake_sleep.calls == 2


def test_entry_never_written_raises_after_the_bounded_retry(pair, fake_sleep):
    _, reader = pair
    struct.pack_into("<Q", reader._mm, HEAD_OFF, 1)
    with pytest.raises(RuntimeError, match="entry 1 never became consistent"):
        reader.get(1)
    assert fake_sleep.calls == 100


# ---- overrun -----------------------------------------------------------------------


def test_reader_behind_by_more_than_the_depth_gets_ring_overrun(tmp_path):
    depth = 4
    writer = StepRing.create(tmp_path / "ring.bin", depth=depth)
    reader = StepRing.open(tmp_path / "ring.bin", timeout_s=5.0)
    try:
        for n in range(1, 11):
            writer.post(n, 3, 3)
        head = reader.head
        assert head == 10
        for n in range(1, head - depth + 1):  # 1..6: overwritten
            with pytest.raises(RuntimeError) as exc:
                reader.get(n)
            assert isinstance(exc.value, RingOverrun)
            assert f"step {n} was overwritten" in str(exc.value)
        for n in range(head - depth + 1, head + 1):  # 7..10: the last `depth` steps
            assert reader.get(n).step_no == n
    finally:
        reader.close()
        writer.close()


def test_overrun_boundary_is_exactly_the_depth(tmp_path):
    writer = StepRing.create(tmp_path / "ring.bin", depth=2)
    try:
        writer.post(1, 1, 1)
        writer.post(2, 1, 1)
        assert writer.get(1).step_no == 1  # head - n == depth - 1
        writer.post(3, 1, 1)  # slot of step 1 is now step 3
        with pytest.raises(RingOverrun):
            writer.get(1)
        assert writer.get(2).epoch_abs == 2
    finally:
        writer.close()


# ---- open --------------------------------------------------------------------------


def test_open_of_a_missing_file_times_out(tmp_path):
    path = tmp_path / "nope.bin"
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="did not appear") as exc:
        StepRing.open(path, timeout_s=0.3)
    assert str(path) in str(exc.value)
    assert 0.25 <= time.monotonic() - t0 < 10.0


def test_open_waits_for_the_writer_to_create_the_file(tmp_path):
    path = tmp_path / "late.bin"
    made = {}

    def create_later():
        time.sleep(0.15)
        made["writer"] = StepRing.create(path, depth=4)

    t = threading.Thread(target=create_later)
    t.start()
    try:
        reader = StepRing.open(path, timeout_s=30.0)
        assert reader.depth == 4
        reader.close()
    finally:
        t.join(30)
        made["writer"].close()


@pytest.mark.parametrize(
    "content",
    [b"", b"\0" * 10, b"\0" * HEADER_BYTES, b"x" * 200],
    ids=["empty", "short", "zero-header", "bad-magic"],
)
def test_open_of_a_file_that_is_not_a_ring_times_out(tmp_path, content):
    path = tmp_path / "junk.bin"
    path.write_bytes(content)
    with pytest.raises(TimeoutError):
        StepRing.open(path, timeout_s=0.2)


def test_open_of_a_ring_whose_size_does_not_match_its_depth_is_an_error(tmp_path):
    path = tmp_path / "bad.bin"
    header = struct.pack("<QIIQ", RING_MAGIC, 8, 0, 0).ljust(HEADER_BYTES, b"\0")
    path.write_bytes(header + bytes(4 * ENTRY_BYTES))  # depth says 8, room for 4
    with pytest.raises(RuntimeError, match="size/depth mismatch"):
        StepRing.open(path, timeout_s=5.0)


# ---- the check word ----------------------------------------------------------------


def test_check_changes_when_any_field_changes():
    base = dict(step_no=17, epoch=4242, T=24, bucket=12, opcode=1)
    ref = _check(**base)
    assert ref == _check(**base)  # deterministic
    seen = {ref}
    for name in base:
        changed = dict(base, **{name: base[name] + 1})
        value = _check(**changed)
        assert value != ref, name
        assert 0 <= value < 1 << 64
        seen.add(value)
    assert len(seen) == 1 + len(base)
    # swapping T and bucket (same sum) is also detected
    assert _check(17, 4242, 12, 24, 1) != ref


def test_check_has_no_collision_over_a_run_of_steps():
    values = {_check(n, n, 3, 3, 1) for n in range(1, 20_001)}
    assert len(values) == 20_000


# ---- concurrency -------------------------------------------------------------------


def test_producer_and_consumer_threads_over_20000_steps(tmp_path):
    """One writer thread, one reader thread, a 64-deep ring: every step arrives
    once, in order, with its own fields. The producer never runs more than 40
    steps ahead (a reader that falls 64 behind is an overrun by design; that case
    is tested above), so the test cannot flake on a loaded box."""
    total = 20_000
    window = 40
    path = tmp_path / "ring.bin"
    writer = StepRing.create(path, depth=64)
    reader = StepRing.open(path, timeout_s=5.0)
    consumed = [0]
    deadline = time.monotonic() + 120.0
    errors: list[BaseException] = []
    received: list[StepDesc] = []

    def expect(n: int) -> StepDesc:
        return StepDesc(n, 3 * n + 1, n % 2048 + 1, (0, 3, 6, 12, 24, 48)[n % 6], 1)

    def produce():
        try:
            for n in range(1, total + 1):
                while n - consumed[0] > window:
                    if time.monotonic() > deadline:
                        raise TimeoutError("consumer stalled")
                    time.sleep(0)
                d = expect(n)
                assert writer.post(d.epoch_abs, d.T, d.bucket, d.opcode) == n
            writer.close_writer()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def consume():
        try:
            n = 1
            while n <= total:
                desc = reader.get(n)
                if desc is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("producer stalled")
                    time.sleep(0)
                    continue
                received.append(desc)
                consumed[0] = n
                n += 1
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
    t0 = time.monotonic()
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(150)
        assert not any(t.is_alive() for t in threads)
        assert not errors, errors
        assert len(received) == total
        assert received == [expect(n) for n in range(1, total + 1)]
        assert reader.closed and reader.head == total
    finally:
        reader.close()
        writer.close()
    assert time.monotonic() - t0 < 60.0
