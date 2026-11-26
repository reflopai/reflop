# SPDX-License-Identifier: Apache-2.0
"""An op the server lacks still runs when its decomposition reaches another
op the server lacks (GH#56).

Regression: with aten.threshold_backward disabled, its decomposition calls
torch.where(cond, 0, grad), i.e. where.ScalarSelf, whose
CompositeImplicitAutograd kernel calls to.dtype. Decompositions run below
autograd, so to.dtype reached __torch_dispatch__ instead of decomposing,
joined a view chain as a step the server does not have, and the read raised
"aten.to.dtype is not available on this server", under "auto" too. Now such
a view op decomposes on the client, and under "auto" a decomposition that
still fails this way runs the op on the client.
"""

from __future__ import annotations

import os
import warnings

import pytest
import reflop
import torch
from reflop import decomp, session
from reflop import tensor as rtensor
from reflop.trace import UnsupportedOperatorError

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
INDEX = 41
aten = torch.ops.aten


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=INDEX, local=DEVICE)


@pytest.fixture
def sess(dev):
    s = session.by_index(INDEX)
    before = set(s.enabled)
    yield s
    s.enabled.clear()
    s.enabled.update(before)


def _inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(16, generator=g), torch.randn(16, generator=g)


def _no_client_runs(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the op ran on the client")

    monkeypatch.setattr(rtensor, "_run_locally", refuse)


@pytest.mark.parametrize("mode", ["auto", "remote"])
def test_a_disabled_op_decomposes_on_the_server(dev, sess, monkeypatch, mode):
    grad, x = _inputs()
    want = aten.threshold_backward.default(grad, x, 0.0)
    sess.enabled.discard("aten.threshold_backward.default")
    _no_client_runs(monkeypatch)
    with reflop.placement(mode):
        got = aten.threshold_backward.default(grad.to(dev), x.to(dev), 0.0)
    assert isinstance(got, rtensor.RemoteTensor)
    assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_where_with_a_scalar_self(dev, sess, monkeypatch, dtype):
    # The CompositeImplicitAutograd op from the report, on its own and below
    # autograd, as a decomposition calls it: its kernel's to.dtype is a view
    # op the server does not run. (Above autograd PyTorch decomposes both
    # before reflop sees them.)
    assert not sess.op_enabled("aten.to.dtype")
    _, x = _inputs()
    x = x.to(dtype)
    want = torch.where(x <= 0, 0, x)
    _no_client_runs(monkeypatch)
    xr = x.to(dev)
    with reflop.placement("remote"), torch._C._AutoDispatchBelowAutograd():
        got = aten.where.ScalarSelf(xr <= 0, 0, xr)
    assert got.dtype == dtype
    assert torch.equal(got.cpu(), want)


def test_auto_runs_the_op_on_the_client_when_its_decomposition_fails(dev, sess, monkeypatch):
    # Every decomposition the server cannot finish raises
    # UnsupportedOperatorError. Under "auto" the op then runs on the client.
    grad, x = _inputs()
    want = aten.threshold_backward.default(grad, x, 0.0)
    sess.enabled.discard("aten.threshold_backward.default")
    calls = []

    def fails(*a, **k):
        calls.append(1)
        raise UnsupportedOperatorError("aten.made_up.default is not available on this server")

    monkeypatch.setitem(decomp.TABLE, aten.threshold_backward.default, fails)
    with warnings.catch_warnings(), reflop.placement("auto"):
        warnings.simplefilter("ignore")
        got = aten.threshold_backward.default(grad.to(dev), x.to(dev), 0.0)
    assert calls
    assert isinstance(got, rtensor.RemoteTensor)
    assert torch.equal(got.cpu(), want)


def test_remote_still_raises_when_a_decomposition_fails(dev, sess, monkeypatch):
    grad, x = _inputs()
    sess.enabled.discard("aten.threshold_backward.default")

    def fails(*a, **k):
        raise UnsupportedOperatorError("aten.made_up.default is not available on this server")

    monkeypatch.setitem(decomp.TABLE, aten.threshold_backward.default, fails)
    with reflop.placement("remote"), pytest.raises(UnsupportedOperatorError, match="made_up"):
        aten.threshold_backward.default(grad.to(dev), x.to(dev), 0.0)


def test_no_fallback_session_still_raises(dev, sess, monkeypatch):
    # allow_local_fallback=False keeps failing closed under "auto" too.
    grad, x = _inputs()
    monkeypatch.setattr(sess, "allow_local_fallback", False)
    sess.enabled.discard("aten.threshold_backward.default")

    def fails(*a, **k):
        raise UnsupportedOperatorError("aten.made_up.default is not available on this server")

    monkeypatch.setitem(decomp.TABLE, aten.threshold_backward.default, fails)
    with reflop.placement("auto"), pytest.raises(UnsupportedOperatorError, match="made_up"):
        aten.threshold_backward.default(grad.to(dev), x.to(dev), 0.0)


def test_other_errors_from_a_decomposition_are_not_swallowed(dev, sess, monkeypatch):
    grad, x = _inputs()
    sess.enabled.discard("aten.threshold_backward.default")

    def fails(*a, **k):
        raise ValueError("a bug in the decomposition")

    monkeypatch.setitem(decomp.TABLE, aten.threshold_backward.default, fails)
    with reflop.placement("auto"), pytest.raises(ValueError, match="a bug"):
        aten.threshold_backward.default(grad.to(dev), x.to(dev), 0.0)
