# SPDX-License-Identifier: Apache-2.0
"""copy_ between reflop tensors copies; copy_ from the CPU frees what it
replaces (GH#61).

Regression: after a flush, b.copy_(a) made a's server handle b's value
without a copy. The two tensors then shared one server tensor: freeing `a`
freed it under `b` ("UNKNOWN_HANDLE" at the next read; LBFGS hit this with
prev_flat_grad.copy_(flat_grad)), and a.add_(1) changed `b` as well. And
b.copy_(cpu_tensor) uploaded into a new handle and dropped b's old one
without freeing it: one server tensor leaked per copy (ASGD's eta and mu,
every step), and reflop.recover marked `b` lost through the stale handle.
"""

from __future__ import annotations

import copy
import gc
import os

import pytest
import reflop
import torch
import torch.nn as nn
import torch.nn.functional as F
from reflop import session

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
INDEX = 61


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=INDEX, local=DEVICE)


def _server_handles():
    return len(session.by_index(INDEX).conn.worker.handles)


@pytest.mark.parametrize("flush", [False, True])
def test_copy_then_changing_the_source_leaves_the_copy(dev, flush):
    a, b = torch.ones(4).to(dev), torch.zeros(4).to(dev)
    b.copy_(a)
    if flush:
        reflop.synchronize()
    a.add_(1)
    assert b.cpu().tolist() == [1.0] * 4
    assert a.cpu().tolist() == [2.0] * 4


def test_copy_then_changing_the_copy_leaves_the_source(dev):
    a, b = torch.ones(4).to(dev), torch.zeros(4).to(dev)
    b.copy_(a)
    reflop.synchronize()
    b.mul_(3)
    assert a.cpu().tolist() == [1.0] * 4 and b.cpu().tolist() == [3.0] * 4


def test_copy_outlives_its_source(dev):
    a, b = torch.ones(4).to(dev), torch.zeros(4).to(dev)
    b.copy_(a)
    del a
    gc.collect()
    reflop.synchronize()  # sends the free of a's handle
    assert (b * 2).cpu().tolist() == [2.0] * 4


def test_copy_into_a_fresh_tensor_and_into_itself(dev):
    a = torch.arange(4.0).to(dev)
    b = torch.empty(4, device=dev)
    b.copy_(a)
    a.copy_(a)
    del a
    gc.collect()
    reflop.synchronize()
    assert b.cpu().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_copy_from_cpu_frees_the_replaced_handle(dev):
    b = torch.zeros(1000).to(dev)
    reflop.synchronize()
    before = _server_handles()
    for i in range(20):
        b.copy_(torch.full((1000,), float(i)))
        assert (b * 2)[0].item() == 2.0 * i
    gc.collect()
    reflop.synchronize()
    assert _server_handles() == before
    assert b.cpu()[0].item() == 19.0


def test_copy_from_cpu_reads_the_old_value_first(dev):
    # The old handle is freed after the execute that still reads it.
    b = torch.ones(4).to(dev)
    reflop.synchronize()
    c = b * 5
    b.copy_(torch.zeros(4))
    assert c.cpu().tolist() == [5.0] * 4 and b.cpu().tolist() == [0.0] * 4


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))


@pytest.mark.parametrize("line_search", [None, "strong_wolfe"])
def test_lbfgs_matches_pytorch(dev, line_search):
    base = _mlp()
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(8, 16, generator=g), torch.randint(0, 10, (8,), generator=g)

    def run(device):
        m = copy.deepcopy(base).to(device)
        opt = torch.optim.LBFGS(m.parameters(), lr=0.1, max_iter=3, line_search_fn=line_search)
        a, b = x.to(device), y.to(device)
        losses = []
        for _ in range(5):
            def closure():
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(m(a), b)
                loss.backward()
                return loss
            losses.append(opt.step(closure).item())
        return losses, [p.detach().cpu() for p in m.parameters()]

    # The reference runs on the CPU: native CUDA backward cannot run next to
    # the reFLOP device (GH#55).
    want, want_p = run("cpu")
    got, got_p = run(dev)
    tol = dict(rtol=1e-3, atol=1e-4) if DEVICE == "cuda" else dict(rtol=0, atol=0)
    torch.testing.assert_close(torch.tensor(got), torch.tensor(want), **tol)
    for p, q in zip(got_p, want_p, strict=True):
        torch.testing.assert_close(p, q, **tol)
