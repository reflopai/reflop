# SPDX-License-Identifier: Apache-2.0
"""reflop.save/load, automatic checkpoints and reflop.recover (ADR-028,
ADR-029), through the client API and the in-process worker."""

import gc
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import reflop
import torch
import torch.nn as nn
import torch.nn.functional as F
from reflop import session as S
from reflop_worker import checkpoint


@pytest.fixture
def owner_interval(monkeypatch):
    def set_(seconds):
        monkeypatch.setenv("REFLOP_LOCAL_CHECKPOINT_S", str(seconds))
    yield set_
    reflop.close()


def model_on(dev, capturable=False):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 3)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, capturable=capturable)
    x = torch.randn(32, 8).to(dev)
    y = torch.randint(0, 3, (32,)).to(dev)

    def step(x, y):
        opt.zero_grad(set_to_none=False)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        return loss
    return m, step, x, y


def take_checkpoint(sess):
    ck = sess.checkpoints
    n = ck.saved
    ck.start -= 10_000  # due now (tests do not wait an interval)
    reflop.step_boundary()
    reflop.synchronize()
    t0 = time.time()
    while ck.saved == n and time.time() - t0 < 30:
        time.sleep(0.01)
    assert ck.saved == n + 1


def test_owner_must_enable(owner_interval):
    owner_interval(0)
    with pytest.raises(reflop.ReflopError, match="has not enabled checkpoints"):
        reflop.device(local="cpu", checkpoint=True)


def test_interval_never_below_the_owners(owner_interval, capsys):
    owner_interval(600)
    with pytest.raises(reflop.ReflopError, match="allows one checkpoint every 10 min"):
        reflop.device(local="cpu", checkpoint=True, checkpoint_interval=60)
    reflop.device(local="cpu", checkpoint=True, checkpoint_interval=900)
    assert S.current().checkpoints.interval == 900
    err = capsys.readouterr().err
    assert "checkpoints on (one every 15 min)" in err and "doubles" in err


@pytest.mark.parametrize("mode", ["eager", "capture"])
def test_recover_continues_from_the_checkpoint(owner_interval, mode):
    owner_interval(1)
    dev = reflop.device(local="cpu", checkpoint=True)
    sess = S.current()
    m, step, x, y = model_on(dev, capturable=mode == "capture")
    fn = reflop.capture(step) if mode == "capture" else step
    for _ in range(4):
        fn(x, y)
        reflop.step_boundary()
    reflop.checkpoint_tag({"step": 4})
    take_checkpoint(sess)
    at_ckpt = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    for _ in range(3):
        fn(x, y)
        reflop.step_boundary()
    later = (torch.ones(2) * 5).to(dev)
    sess.conn.kill_worker()
    with pytest.raises(reflop.SessionLostError, match="reflop.recover"):
        fn(x, y)
        reflop.step_boundary()
        reflop.synchronize()
    assert reflop.recover() == {"step": 4}
    for k, v in m.state_dict().items():
        assert torch.equal(v.detach().cpu(), at_ckpt[k]), k
    with pytest.raises(reflop.SessionLostError, match="after the last checkpoint"):
        (later + 1).cpu()
    loss = fn(x, y)
    reflop.step_boundary()
    assert torch.isfinite(loss.detach().cpu())


def test_recovery_is_bit_identical_to_an_uninterrupted_run(owner_interval):
    """Replaying the same steps after recover() gives exactly the parameters
    of a run that never stopped (CPU, dropout: the RNG state is restored).

    capturable=True keeps AdamW's step counter on the server, so it is in the
    checkpoint; a plain AdamW counts steps on the Guest's machine (GUEST_GUIDE)."""
    owner_interval(1)
    dev = reflop.device(local="cpu", checkpoint=True)
    sess = S.current()
    m, step, x, y = model_on(dev, capturable=True)
    for _ in range(3):
        step(x, y)
        reflop.step_boundary()
    take_checkpoint(sess)
    for _ in range(3):
        step(x, y)
        reflop.step_boundary()
    straight = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    sess.conn.kill_worker()
    reflop.recover()
    for _ in range(3):
        step(x, y)
        reflop.step_boundary()
    for k, v in m.state_dict().items():
        assert torch.equal(v.detach().cpu(), straight[k]), k


def test_recover_without_a_checkpoint(owner_interval):
    owner_interval(1)
    reflop.device(local="cpu", checkpoint=True)
    S.current().conn.kill_worker()
    with pytest.raises(reflop.ReflopError, match="cannot recover"):
        reflop.recover()


def test_save_and_load(owner_interval):
    owner_interval(0)
    dev = reflop.device(local="cpu")
    m, step, x, y = model_on(dev)
    step(x, y)
    w = m[0].weight
    obj = {"model": m.state_dict(), "extra": {"view": w[:2, :3]}}
    n = reflop.save("run1", obj)
    assert n > 0
    assert [s["name"] for s in reflop.snapshots()] == ["run1"]
    got = reflop.load("run1")
    assert set(got) == {f"model.{k}" for k in m.state_dict()} | {"extra.view"}
    assert torch.equal(got["extra.view"].cpu(), w[:2, :3].detach().cpu())
    m2 = model_on(dev)[0]
    m2.load_state_dict({k.removeprefix("model."): v for k, v in got.items() if k.startswith("model.")})
    for k, v in m.state_dict().items():
        assert torch.equal(m2.state_dict()[k].cpu(), v.detach().cpu())
    reflop.delete_snapshot("run1")
    assert reflop.snapshots() == []


def test_save_refuses_what_it_cannot_store(owner_interval):
    owner_interval(0)
    dev = reflop.device(local="cpu")
    t = torch.ones(2).to(dev)
    with pytest.raises(TypeError, match="tensors only"):
        reflop.save("x", {"t": t, "step": 3})
    with pytest.raises(TypeError, match="not on the reflop device"):
        reflop.save("x", {"t": torch.ones(2)})
    with pytest.raises(ValueError, match="cannot be used"):
        reflop.save("x", {"__reflop__/meta": t})


def test_owner_turning_checkpoints_off_does_not_break_training(owner_interval):
    owner_interval(1)
    dev = reflop.device(local="cpu", checkpoint=True)
    sess = S.current()
    m, step, x, y = model_on(dev)
    sess.conn.checkpoint_interval_s = 0  # the Owner runs `reflopd checkpoints off`
    sess.checkpoints.start -= 10_000
    step(x, y)
    reflop.step_boundary()
    reflop.synchronize()
    t0 = time.time()
    while not sess.checkpoints.disabled and time.time() - t0 < 10:
        time.sleep(0.01)
    assert sess.checkpoints.disabled
    reflop.checkpoint_tag({"step": 1})  # still fine
    step(x, y)
    reflop.step_boundary()


@pytest.mark.parametrize("replace_worker", [False, True], ids=["close", "replace"])
def test_local_teardown_joins_before_releasing_partial_registration(owner_interval, monkeypatch, replace_worker):
    registered = []
    events = []

    def register(ptr, _size, _flags):
        registered.append(ptr)
        return int(len(registered) > 1)

    def unregister(ptr):
        events.append(ptr)
        return 0

    monkeypatch.setattr(checkpoint, "RING", 4096)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "cudart", lambda: SimpleNamespace(cudaHostRegister=register,
                                                                   cudaHostUnregister=unregister))
    reflop.device(local="cpu")
    sess = S.current()
    conn = sess.conn
    writer = conn.worker.writer
    writer.ring = checkpoint.Ring(torch.device("cuda:0"))
    assert not writer.ring.pinned  # second registration failed; the first is still live
    writer.thread = SimpleNamespace(join=lambda: events.append("joined"))

    if replace_worker:
        conn.new_session()
    else:
        sess.close()
    assert events == ["joined", registered[0]]
    assert writer.ring is None
    writer.close()
    assert events == ["joined", registered[0]]


def test_ring_keeps_only_failed_unregistration_for_retry(monkeypatch):
    monkeypatch.setattr(checkpoint, "RING", 4096)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    runtime = SimpleNamespace(cudaHostRegister=Mock(return_value=0),
                              cudaHostUnregister=Mock(side_effect=[1, 0, 0]))
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(torch.cuda, "check_error", Mock(side_effect=RuntimeError("unregister failed")))
    ring = checkpoint.Ring(torch.device("cuda:0"))
    first, second = (view.data_ptr() for view in ring.views)

    try:
        with pytest.raises(RuntimeError, match="unregister failed"):
            ring.close()
        assert checkpoint._registered_buffers[first] is ring.bufs[0]
        assert second not in checkpoint._registered_buffers
    finally:
        ring.close()

    assert first not in checkpoint._registered_buffers
    assert [call.args[0] for call in runtime.cudaHostUnregister.call_args_list] == [first, second, first]


def test_snapshot_drains_cuda_stream_after_write_error(monkeypatch):
    stream = Mock()
    monkeypatch.setattr(checkpoint, "RING", 4)
    monkeypatch.setattr(torch.cuda, "Stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    write = Mock(side_effect=[None, OSError("disk full")])
    monkeypatch.setattr(checkpoint, "_write_all", write)

    writer = checkpoint.Writer(SimpleNamespace(device=torch.device("cuda:0")))
    writer.ring = checkpoint.Ring(torch.device("cpu"))
    writer.ring.pinned = True
    tensor = torch.arange(8, dtype=torch.uint8)
    with pytest.raises(OSError, match="disk full"):
        writer._write(-1, [("value", tensor)], b"header", object())
    stream.synchronize.assert_called_once_with()


@pytest.mark.parametrize("checkpointing", [False, True], ids=["snapshot", "checkpoint"])
def test_reopening_local_cuda_preserves_saved_values(owner_interval, devname, checkpointing):
    if devname != "cuda":
        pytest.skip("requires CUDA")
    owner_interval(1)
    for i in range(5):
        dev = reflop.device(local="cuda", checkpoint=checkpointing)
        sess = S.current()
        expected = torch.arange(256 * 1024, dtype=torch.float32) + i + 1
        tensor = expected.to(dev)
        if checkpointing:
            reflop.checkpoint_tag({"iteration": i})
            take_checkpoint(sess)
            tensor.zero_()
            reflop.synchronize()
            sess.conn.kill_worker()
            assert reflop.recover() == {"iteration": i}
            restored = tensor
        else:
            reflop.save("weights", {"tensor": tensor})
            restored = reflop.load("weights")["tensor"]
        assert torch.equal(restored.cpu(), expected), f"session {i}"
        reflop.close()
        del sess, tensor, restored
        gc.collect()
