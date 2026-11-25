# SPDX-License-Identifier: Apache-2.0
"""Backward on PyTorch's autograd thread (ADR-053). PyTorch runs this
device's backward on a thread of its own while the caller waits: errors
raised there used to abort the Guest, and the caller's placement mode and
session did not reach it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import pytest
import reflop
import torch
from reflop import session
from reflop import tensor as T

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"


# ---------------------------------------------------------------- errors (child process)


@pytest.fixture(scope="module")
def child():
    env = dict(os.environ, CHILD_DEVICE=DEVICE,
               PYTHONPATH=os.pathsep.join([str(ROOT / "client/python"), str(ROOT / "server/worker")]))
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("autograd_errors_child.py"))],
                       capture_output=True, text=True, env=env, timeout=300)
    results, last = {}, None
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            if "start" in d:
                last = d["start"]
            else:
                results[d["name"]] = d["result"]
    return p.returncode, results, last, p.stderr[-2000:]


def result(child, name):
    rc, results, last, err = child
    assert name in results, f"the child process died (rc {rc}) in scenario {last!r}:\n{err}"
    return results[name]


BOOM = "Boom: boom in aten.threshold_backward.default"


def test_an_error_in_backward_raises_from_backward(child):
    assert result(child, "backward_raises") == BOOM
    assert result(child, "autograd_grad_raises") == BOOM


def test_a_gradient_from_a_failed_backward_raises_on_read(child):
    assert result(child, "grad_read_raises") == BOOM
    # One accumulated before keeps neither its old value nor a partial one.
    assert result(child, "accumulated_backward_raises") == BOOM
    assert result(child, "accumulated_grad_read_raises") == BOOM


def test_backward_from_a_client_tensor_raises_at_the_next_read(child):
    assert result(child, "cpu_root_backward") == "ok"
    assert result(child, "cpu_root_next_read") == BOOM


def test_hooks_and_factories_that_fail_in_backward(child):
    assert result(child, "hook_raises").startswith("ZeroDivisionError")
    assert result(child, "post_accumulate_hook_raises").startswith("ZeroDivisionError")
    assert result(child, "factory_fails").startswith("UnsupportedOperatorError: aten.zeros.default")


def test_the_session_works_after_failed_backward_passes(child):
    assert result(child, "recovers") == [1.0, 0.0, 1.0, 0.0]
    assert child[0] == 0, child[3]


# ---------------------------------------------------------------- placement and session (in process)


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=32, local=DEVICE)


def test_local_placement_keeps_backward_on_the_client(dev):
    sess = session.by_index(32)
    x = torch.tensor([1.0, -2.0, 3.0, -4.0]).to(dev).requires_grad_()
    y = torch.relu(x * 3).sum()
    sess.flush()
    with reflop.placement("local"):
        y.backward()
        assert len(sess.trace) == 0  # nothing recorded for the server
    assert x.grad.cpu().tolist() == [3.0, 0.0, 3.0, 0.0]


def test_remote_placement_does_not_fall_back_in_backward(dev, monkeypatch):
    sess = session.by_index(32)
    x = torch.tensor([1.0, -2.0, 3.0, -4.0]).to(dev).requires_grad_()
    ran_locally = []
    run = T._run_locally
    monkeypatch.setattr(T, "_run_locally", lambda *a, **k: ran_locally.append(a[1]) or run(*a, **k))
    monkeypatch.setattr(sess, "enabled", sess.enabled - {"aten.threshold_backward.default"})
    monkeypatch.setattr(T, "_decomp_table", dict)  # no decomposition: the client is the only other place
    with reflop.placement("remote"), pytest.raises(T.UnsupportedOperatorError):
        torch.relu(x).sum().backward()
    assert ran_locally == []
    x.grad = None
    with warnings.catch_warnings():  # auto: falls back on the client, as in forward
        warnings.simplefilter("ignore")
        torch.relu(x).sum().backward()
    assert x.grad.cpu().tolist() == [1.0, 0.0, 1.0, 0.0] and ran_locally


def test_tensors_created_in_backward_go_to_the_roots_session(monkeypatch):
    # clamp's backward builds where(mask, grad, 0); with the model's session
    # not the default, the 0 used to land in the default session (and cost a
    # move between sessions).
    d9 = reflop.device(index=33, local="cpu")
    reflop.device(index=34, local="cpu")  # now the default
    try:
        moves = []
        move = T._move_raw
        monkeypatch.setattr(T, "_move_raw", lambda *a: moves.append(a) or move(*a))
        x = torch.tensor([1.0, -2.0, 3.0]).to(d9).requires_grad_()
        torch.clamp(x, min=0).sum().backward()
        assert x.grad.device == d9 and x.grad.cpu().tolist() == [1.0, 0.0, 1.0]
        assert moves == []
    finally:
        for i in (34, 33):
            session.by_index(i).close()


def test_concurrent_backward_calls_in_two_sessions():
    # Both share PyTorch's one autograd thread for this device; the most
    # recent call's context applies, and results stay right.
    devs = [reflop.device(index=i, local="cpu") for i in (35, 36)]
    try:
        out, errs = {}, []
        start = threading.Barrier(2)

        def train(i):
            try:
                d = devs[i]
                g = torch.Generator().manual_seed(i)
                w = torch.randn(8, 4, generator=g).to(d).requires_grad_()
                xs = torch.randn(16, 8, generator=g)
                start.wait()
                for _ in range(20):
                    w.grad = None
                    torch.clamp(xs.to(d) @ w, min=0).pow(2).sum().backward()
                out[i] = (w.grad.device, w.grad.cpu())
            except BaseException as e:  # noqa: BLE001 - re-raised below
                errs.append(e)

        ts = [threading.Thread(target=train, args=(i,), daemon=True) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(120)
        assert not any(t.is_alive() for t in ts)
        if errs:
            raise errs[0]
        for i in range(2):
            g = torch.Generator().manual_seed(i)
            w = torch.randn(8, 4, generator=g).requires_grad_()
            torch.clamp(torch.randn(16, 8, generator=g) @ w, min=0).pow(2).sum().backward()
            assert out[i][0] == devs[i] and torch.allclose(out[i][1], w.grad, atol=1e-5)
    finally:
        for i in (36, 35):
            session.by_index(i).close()


def test_placement_inside_a_hook_does_not_stick_to_the_autograd_thread(dev):
    from reflop import _placement

    seen = []

    def hook(g):
        with reflop.placement("local"):
            pass
        seen.append(_placement.mode())

    for mode in ("remote", None):
        x = torch.ones(3).to(dev).requires_grad_()
        h = x * 2
        h.register_hook(hook)
        if mode is None:
            h.sum().backward()
        else:
            with reflop.placement(mode):
                h.sum().backward()
    assert seen == ["remote", "auto"]
