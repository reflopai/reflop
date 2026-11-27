# SPDX-License-Identifier: Apache-2.0
"""reflop.capture replays a step only when the step is one graph (GH#62).

Regression: a session that flushed inside a recorded step (a host read such
as .item(), or a trace past REFLOP_FLUSH_NODES) left only the work after the
last flush in the plan. Replays then ran that tail alone and silently
skipped the rest of the step; a step whose tail was empty (non-capturable
ASGD) never replayed and was reported as doing "no work on a reflop
device". Now a step that reads on the host on two calls in a row raises
CaptureError, and a long step stays one graph up to the server's max_nodes.
"""

from __future__ import annotations

import copy
import os

import pytest
import reflop
import torch
import torch.nn as nn
import torch.nn.functional as F
from reflop import session

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
INDEX = 62


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=INDEX, local=DEVICE)


@pytest.mark.parametrize("where", ["middle", "end"])
def test_a_host_read_inside_the_step_is_refused(dev, where):
    w, acc = torch.zeros(4).to(dev), torch.zeros(4).to(dev)
    seen = []

    def step(x):
        w.add_(x.to(dev))
        if where == "middle":
            seen.append(w.sum().item())
        acc.add_(1)
        if where == "end":
            seen.append(acc.sum().item())
        return acc

    s = reflop.capture(step)
    s(torch.ones(4))
    with pytest.raises(reflop.CaptureError, match="reads a tensor's value on the host"):
        s(torch.ones(4))
    # Both calls ran in full, eagerly.
    assert s.replays == 0 and len(seen) == 2
    assert w.cpu().tolist() == [2.0] * 4 and acc.cpu().tolist() == [2.0] * 4


def test_a_host_read_on_the_first_call_only_still_replays(dev):
    w = torch.zeros(4).to(dev)
    calls = []

    def step(x):
        w.add_(x.to(dev))
        if not calls:
            calls.append(w.sum().item())  # one-off warm-up read
        return w

    s = reflop.capture(step)
    for _ in range(6):
        s(torch.ones(4))
    assert s.replays >= 3 and w.cpu().tolist() == [6.0] * 4


def test_a_step_longer_than_the_flush_limit_replays_whole(dev, monkeypatch):
    monkeypatch.setattr(session, "FLUSH_NODES", 4)
    w = torch.zeros(4).to(dev)

    def step(x):
        x = x.to(dev)
        for _ in range(10):
            w.add_(x)
        return w

    s = reflop.capture(step)
    for _ in range(6):
        s(torch.ones(4))
    assert s.replays >= 3 and w.cpu().tolist() == [60.0] * 4


def test_a_step_longer_than_one_graph_stays_eager(dev, monkeypatch):
    sess = session.by_index(INDEX)
    monkeypatch.setattr(session, "FLUSH_NODES", 4)
    old = sess.limits.max_nodes
    sess.limits.max_nodes = 6  # client side only: the server keeps its own limit
    try:
        w = torch.zeros(4).to(dev)

        def step(x):
            x = x.to(dev)
            for _ in range(10):
                w.add_(x)
            return w

        s = reflop.capture(step)
        with pytest.warns(RuntimeWarning, match="more operations than the server accepts in one graph"):
            for _ in range(6):
                s(torch.ones(4))
        assert s.replays == 0 and w.cpu().tolist() == [60.0] * 4
    finally:
        sess.limits.max_nodes = old


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))


def _batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(8, 16, generator=g), torch.randint(0, 10, (8,), generator=g)


def test_non_capturable_asgd_is_refused(dev):
    m = _mlp().to(dev)
    opt = torch.optim.ASGD(m.parameters(), lr=0.01)
    x, y = _batch()

    def step(a, b):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(a.to(dev)), b.to(dev))
        loss.backward()
        opt.step()
        return loss

    s = reflop.capture(step)
    s(x, y)
    with pytest.raises(reflop.CaptureError, match="ASGD"):
        s(x, y)


def test_capturable_asgd_replays_and_matches_pytorch(dev):
    base = _mlp()
    x, y = _batch()
    m, ref = copy.deepcopy(base).to(dev), copy.deepcopy(base)
    opt = torch.optim.ASGD(m.parameters(), lr=0.01, capturable=True)
    ropt = torch.optim.ASGD(ref.parameters(), lr=0.01)  # native CPU ASGD is not capturable

    def step(a, b):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(a.to(dev)), b.to(dev))
        loss.backward()
        opt.step()
        return loss

    s = reflop.capture(step)
    for i in range(6):
        loss = s(x + i * 0.01, y)
        ropt.zero_grad(set_to_none=True)
        rloss = F.cross_entropy(ref(x + i * 0.01), y)
        rloss.backward()
        ropt.step()
        torch.testing.assert_close(loss.detach().cpu(), rloss.detach(), rtol=1e-4, atol=1e-5)
    assert s.replays >= 3
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), ref.state_dict()[k], rtol=1e-4, atol=1e-5)
    for p, q in zip(m.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(opt.state[p]["ax"].cpu(), ropt.state[q]["ax"], rtol=1e-4, atol=1e-5)


def test_lbfgs_under_capture_is_refused(dev):
    m = _mlp().to(dev)
    opt = torch.optim.LBFGS(m.parameters(), lr=0.1, max_iter=3)
    x, y = _batch()

    def step(a, b):
        a, b = a.to(dev), b.to(dev)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(a), b)
            loss.backward()
            return loss
        return opt.step(closure)

    s = reflop.capture(step)
    s(x, y)
    with pytest.raises(reflop.CaptureError, match="reads a tensor's value on the host"):
        s(x, y)


def test_a_copy_from_a_cpu_argument_is_replayed(dev):
    # Regression: b.copy_(x) was uploaded outside the step's graph, so replays
    # returned the right sum but left buf as the second recorded call wrote it.
    buf = torch.zeros(4).to(dev)

    def step(x):
        buf.copy_(x)
        return (buf * 2).sum()

    s = reflop.capture(step)
    for i in range(6):
        assert s(torch.full((4,), float(i))).item() == 8.0 * i
        assert buf.cpu().tolist() == [float(i)] * 4
    assert s.replays >= 3


def test_a_copy_from_a_changing_cpu_constant_stays_eager(dev):
    buf = torch.zeros(4).to(dev)
    consts = [torch.full((4,), float(i)) for i in range(6)]
    calls = []

    def step(x):
        buf.copy_(consts[len(calls)])  # not an argument: replays would freeze it
        calls.append(1)
        return (buf + x.to(dev)).sum()

    s = reflop.capture(step)
    with pytest.warns(RuntimeWarning, match="changes every call"):
        for i in range(6):
            assert s(torch.ones(4)).item() == 4.0 * i + 4
    assert s.replays == 0 and buf.cpu().tolist() == [5.0] * 4
