# SPDX-License-Identifier: Apache-2.0
"""Several threads sharing one session (#6, bench B6). Ops record one at a
time under the session's lock; before it, a flush by one thread while
another recorded left node indexes pointing into the old trace (IndexError)."""

from __future__ import annotations

import gc
import sys
import threading

import pytest
import reflop
import torch
from reflop import session

THREADS = 4
OPS = 200
TIMEOUT = 120


@pytest.fixture(scope="module")
def dev():
    d = reflop.device(23, local="cpu")
    yield d
    sess = session.by_index(23)
    if sess.lock.acquire(timeout=10):  # a thread stuck in a failed test may hold it
        sess.lock.release()
        sess.close()


def run_threads(fn, n=THREADS):
    """Run fn(i) in n threads started together; return the results in order
    and raise the first error a thread hit."""
    start = threading.Barrier(n)
    out, errs = [None] * n, []

    def body(i):
        try:
            start.wait()
            out[i] = fn(i)
        except BaseException as e:  # noqa: BLE001 - re-raised below
            errs.append(e)

    ts = [threading.Thread(target=body, args=(i,), daemon=True) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(TIMEOUT)
    assert not any(t.is_alive() for t in ts), "a thread is stuck (deadlock?)"
    if errs:
        raise errs[0]
    return out


def chain(dev, i, flush_every):
    g = torch.Generator().manual_seed(i)
    x = torch.randn(16, generator=g).to(dev)
    w = torch.randn(16, 16, generator=g).to(dev)
    reads = []
    for k in range(OPS):
        x = torch.tanh(w @ x) * 1.5 + 0.01 * k
        if flush_every and k % flush_every == 0:
            reads.append(x.sum().item())
    return x.cpu(), reads


@pytest.mark.parametrize("flush_every", [0, 7], ids=["record-only", "flushing"])
def test_threads_sharing_a_session_match_a_serial_run(dev, flush_every):
    want = [chain(dev, i, flush_every) for i in range(THREADS)]
    for _ in range(3):
        got = run_threads(lambda i: chain(dev, i, flush_every))
        for (gx, gr), (wx, wr) in zip(got, want, strict=True):
            assert torch.equal(gx, wx)
            assert gr == wr


def test_threads_that_flush_while_others_record(dev):
    # Two threads read a value every op (each read flushes the shared
    # trace); the other two record long chains and read only at the end.
    want = [chain(dev, i, 1 if i % 2 else 0) for i in range(THREADS)]
    got = run_threads(lambda i: chain(dev, i, 1 if i % 2 else 0))
    for (gx, gr), (wx, wr) in zip(got, want, strict=True):
        assert torch.equal(gx, wx) and gr == wr


def test_backward_with_a_hook_that_reads_a_value(dev):
    # PyTorch runs this device's backward on its own thread while the caller
    # waits; the caller holds no session lock, so the hook's read (an op and a
    # flush on that thread) does not deadlock, even with another thread
    # recording in the same session meanwhile.
    stop = threading.Event()

    def recorder(_):
        y = torch.ones(8).to(dev)
        n = 0
        while not stop.is_set():
            y = y * 1.0 + 1.0
            n += 1
        return n, y.sum().item()

    def trainer(_):
        try:
            seen = []
            for _ in range(5):
                x = torch.arange(4.0).to(dev).requires_grad_()
                h = x * 3
                h.register_hook(lambda g: seen.append(g.sum().item()))
                (h * h).sum().backward()
                assert x.grad.cpu().tolist() == [18.0 * v for v in range(4)]
            return seen
        finally:
            stop.set()

    (seen, (n, total)) = run_threads(lambda i: trainer(i) if i == 0 else recorder(i), n=2)
    assert seen == [2.0 * 3 * sum(range(4))] * 5
    assert total == 8 * (1 + n)


def test_many_dropped_tensors_from_several_threads(dev):
    # Storage.__del__ frees a handle on whichever thread runs the GC. No free
    # may be lost or sent twice: the worker ends up holding exactly the
    # handles of live tensors.
    sess = session.by_index(23)

    def churn(i):
        # Odd threads record and flush while every thread drops uploads.
        keep = []
        acc = torch.zeros(4).to(dev)
        for k in range(1500):
            t = torch.full((4,), float(k)).to(dev)  # an upload: a server handle
            if i % 2:
                acc = acc + t
                if k % 50 == 0:
                    acc.sum().item()
            if k % 100 == 0:
                keep.append(t)
            del t
        return keep, acc.cpu()

    def drop(i):
        # No ops, so the trace stays empty and a release that sees 4096
        # pending frees sends them itself, while odd threads hold the lock
        # for fetches (which do not send frees).
        keep = []
        for k in range(2500):
            t = torch.full((4,), float(k)).to(dev)
            if i % 2 and k % 10 == 0:
                assert sess.fetch_handles([t.reflop_handle])[0][0].item() == k
            if k % 500 == 0:
                keep.append(t)
            del t
        return keep, None

    sent = []
    send = sess._send_frees

    def counting_send():
        if len(sess.pending_free) >= 4096 and any(f.f_code.co_name == "release" for f in _frames()):
            sent.append(len(sess.pending_free))
        send()

    out = run_threads(churn)
    for i, (keep, acc) in enumerate(out):
        assert len(keep) == 15
        assert acc.tolist() == [float(sum(range(1500))) if i % 2 else 0.0] * 4
    sess.synchronize()
    sess._send_frees = counting_send
    try:
        out += run_threads(drop)
    finally:
        del sess._send_frees
    assert sent, "no release sent the pending frees"
    gc.collect()
    with sess.lock:
        sess._send_frees()
    sess.synchronize()
    worker = getattr(sess.conn, "worker", None)
    if worker is None:
        pytest.skip("needs the in-process worker to list the server's handles")
    live = {h for h, st in list(sess.storages.items()) if st.handle == h}
    assert len(live) >= 80 and sum(len(k) for k, _ in out) == 80
    assert set(worker.handles.t) == live


def _frames():
    f = sys._getframe(1)
    while f is not None:
        yield f
        f = f.f_back


def test_a_free_appended_during_a_send_is_kept(dev):
    # Storage.__del__ on another thread may append to the list it read just
    # before a send took the pending frees: that handle must stay pending.
    sess = session.by_index(23)
    t = torch.ones(2).to(dev)
    h = t.reflop_handle
    seen = sess.pending_free  # what a release on another thread holds
    with sess.lock:
        sess._send_frees()
        del t
        gc.collect()
        assert h in sess.pending_free
        sess._send_frees()
    seen.append(b"\x00" * 16)  # a late append from that release
    try:
        assert b"\x00" * 16 in sess.pending_free
    finally:
        sess.pending_free.clear()


def test_a_release_never_waits_for_the_lock(dev):
    # GC can run a release on a thread while another thread holds the lock
    # (and is waiting for the GIL): it must neither wait nor send.
    sess = session.by_index(23)
    sess.synchronize()
    held, done = threading.Event(), threading.Event()

    def holder():
        with sess.lock:
            held.set()
            done.wait(TIMEOUT)

    th = threading.Thread(target=holder)
    th.start()
    try:
        assert held.wait(TIMEOUT)
        fake = [b"\x01" * 16] * 5000
        sess.pending_free.extend(fake[:-1])
        sess.release(fake[-1])  # 5000 pending, empty trace: would send if it could
        assert len(sess.pending_free) == 5000
    finally:
        done.set()
        th.join(TIMEOUT)
        sess.pending_free.clear()


def test_moves_between_two_sessions_in_both_directions(dev):
    # A move or a 0-dim join holds both sessions' locks, taken in index
    # order: two threads going opposite ways do not deadlock.
    other = reflop.device(25, local="cpu")

    def mover(i):
        src, dst = (dev, other) if i % 2 else (other, dev)
        x = torch.arange(4.0).to(src)
        s = torch.tensor(2.0).to(dst)
        for _ in range(100):
            y = x.to(dst).to(src)  # two moves
            x = y * s  # s joins src's session
            x = x / s
        return x.device, x.cpu().tolist()

    for d, vals in run_threads(mover):
        assert vals == [0.0, 1.0, 2.0, 3.0]
        assert d in (dev, other)
    session.by_index(25).close()  # not in a finally: a stuck thread holds its lock
