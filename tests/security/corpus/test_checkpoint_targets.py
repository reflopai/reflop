# SPDX-License-Identifier: Apache-2.0
"""Checkpoint targets and stalls in the worker (ADR-061).

reflopd may pass two descriptors, a RAM checkpoint (memfd) and a disk file;
the worker writes only the one SnapshotGo names and closes the other. When
the GPU has no room for a background copy, the cut is skipped unless reflopd
allows a stall; a stalled cut copies before any later request runs, and the
messages that arrive while it waits for SnapshotGo are served afterwards,
in order."""

import collections
import os
import threading

import pytest
import torch
from reflop.connection import default_limits
from reflop_worker._pb import capabilities_pb2 as kpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.core import Session
from reflop_worker.main import wait_for_go
from reflop_worker.safetensors import parse_header

H1, H2 = b"\x01" * 16, b"\x02" * 16


def session() -> Session:
    init = wpb.WorkerInit(seed=0)
    init.device.kind = gpb.DEVICE_CPU
    init.limits.CopyFrom(default_limits(kpb))
    return Session(init)


def _no_room(s):
    s.writer.copy_room = lambda items: "the GPU has 10 MiB free; a background copy of 64 MiB of state needs that much again"


def _fds(tmp_path, n=2):
    return [os.open(tmp_path / f"t{i}", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600) for i in range(n)]


def _open(fd) -> bool:
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


@pytest.mark.parametrize("target", [0, 1])
def test_writes_only_the_chosen_target(tmp_path, target):
    s = session()
    s.handles.put_new(H1, torch.ones(1000))
    fds = _fds(tmp_path)
    out = []
    s.snapshot_write(wpb.SnapshotWrite(request_id=3, checkpoint=True, generation=1, targets=2), fds,
                     lambda rid, n, _stall=False: s.writer.on_go(wpb.SnapshotGo(request_id=rid, proceed=True,
                                                                                 target=target)), out.append)
    s.writer.thread.join()
    assert out[-1].checkpoint_saved.generation == 1 and out[-1].checkpoint_saved.paused_ms == 0
    written = (tmp_path / f"t{target}").read_bytes()
    assert parse_header(memoryview(written), len(written))[1][H1.hex()]
    assert (tmp_path / f"t{1 - target}").stat().st_size == 0
    assert not any(_open(fd) for fd in fds)  # both closed: the worker keeps no descriptor


def test_a_target_out_of_range_fails_closed(tmp_path):
    s = session()
    s.handles.put_new(H1, torch.ones(4))
    fds = _fds(tmp_path)
    out = []
    s.snapshot_write(wpb.SnapshotWrite(request_id=3, checkpoint=True, targets=2), fds,
                     lambda rid, n, _stall=False: s.writer.on_go(wpb.SnapshotGo(request_id=rid, proceed=True, target=5)),
                     out.append)
    s.writer.thread.join()
    assert out[-1].error.message.startswith("checkpoint write failed")
    assert all((tmp_path / f"t{i}").stat().st_size == 0 for i in range(2))
    assert not any(_open(fd) for fd in fds)


def test_no_room_and_no_stall_skips(tmp_path):
    s = session()
    s.handles.put_new(H1, torch.ones(4))
    _no_room(s)
    out, cuts = [], []
    fds = _fds(tmp_path, 1)
    s.snapshot_write(wpb.SnapshotWrite(request_id=3, checkpoint=True, allow_stall=False), fds,
                     lambda *a: cuts.append(a), out.append, lambda gate: None)
    assert not cuts
    assert out[-1].checkpoint_saved.skipped
    assert "goes to the Owner's disk" in out[-1].checkpoint_saved.reason
    assert not _open(fds[0])


def test_stalled_cut_copies_before_any_later_request(tmp_path):
    """With no room on the GPU, the copy runs on the request thread: start()
    returns only when the state is written, so a rebind after it cannot be
    in the file and the old version is not kept alive."""
    s = session()
    s.handles.put_new(H1, torch.ones(1000))
    s.handles.put_new(H2, torch.full((3,), 2.0))
    _no_room(s)
    out, cuts, waited = [], [], []

    def cut(rid, n, stall=False):
        cuts.append((rid, n, stall))  # reflopd answers SnapshotGo later, on the control channel

    def wait_go(gate):
        waited.append(1)
        s.writer.on_go(wpb.SnapshotGo(request_id=3, proceed=True, target=1))

    fds = _fds(tmp_path)
    s.snapshot_write(wpb.SnapshotWrite(request_id=3, checkpoint=True, generation=2, targets=2, allow_stall=True), fds,
                     cut, out.append, wait_go)
    assert cuts and cuts[0][2] is True and waited
    assert s.writer.thread is None  # no background thread
    assert out and out[-1].checkpoint_saved.generation == 2 and out[-1].checkpoint_saved.paused_ms >= 1
    s.handles.replace(H1, s.handles.t[H1] * 5)  # the next request
    data = (tmp_path / "t1").read_bytes()
    start, hdr = parse_header(memoryview(data), len(data))
    _dt, _shape, b, e = hdr[H1.hex()]
    assert torch.equal(torch.frombuffer(bytearray(data[start + b:start + e]), dtype=torch.float32), torch.ones(1000))
    assert (tmp_path / "t0").stat().st_size == 0


def test_wait_for_go_keeps_other_messages_in_order():
    """While a stalled cut waits, reflopd may already have sent client
    requests: they are kept, in order, and SnapshotGo opens the gate."""
    gate = threading.Event()
    seen = []

    def on_go(g):
        seen.append(g.request_id)
        gate.set()

    msgs = collections.deque([
        wpb.WorkerEnvelope(stats_request=wpb.StatsRequest()),
        wpb.WorkerEnvelope(dataset_close=wpb.DatasetClose(dataset_id=b"d")),
        wpb.WorkerEnvelope(snapshot_go=wpb.SnapshotGo(request_id=7, proceed=True)),
        wpb.WorkerEnvelope(stats_request=wpb.StatsRequest()),
    ])
    pending = collections.deque()
    wait_for_go(msgs.popleft, on_go, pending, gate)
    assert seen == [7]
    assert [e.WhichOneof("body") for e in pending] == ["stats_request", "dataset_close"]
    assert len(msgs) == 1  # nothing read past SnapshotGo
