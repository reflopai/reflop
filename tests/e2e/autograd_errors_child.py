# SPDX-License-Identifier: Apache-2.0
"""Child process for test_autograd_thread.py: errors raised in reFLOP ops on
PyTorch's autograd thread used to abort the process (core dump), so these
scenarios run here, where an abort fails one test instead of the whole run.
Prints one JSON line per scenario: {"name": ..., "result": ...}."""

from __future__ import annotations

import json
import os
import sys

import reflop
import torch
from reflop import session
from reflop import tensor as T

DEVICE = os.environ.get("CHILD_DEVICE", "cpu")
dev = reflop.device(index=31, local=DEVICE)
sess = session.by_index(31)
_functional = T._functional


class Boom(RuntimeError):
    pass


def failing(name):
    def f(s, func, args, kwargs):
        if str(func) == name:
            raise Boom(f"boom in {name}")
        return _functional(s, func, args, kwargs)
    return f


def outcome(fn):
    try:
        fn()
        return "ok"
    except Exception as e:  # noqa: BLE001 - reported to the parent
        return f"{type(e).__name__}: {e}"


def report(name, result):
    print(json.dumps({"name": name, "result": result}), flush=True)


def leaf():
    return torch.tensor([1.0, -2.0, 3.0, -4.0]).to(dev).requires_grad_()


def scenario(name):
    print(json.dumps({"start": name}), flush=True)
    return name


T._functional = failing("aten.threshold_backward.default")

x = leaf()
report(scenario("backward_raises"), outcome(lambda: torch.relu(x).sum().backward()))
report(scenario("grad_read_raises"), outcome(lambda: x.grad.cpu()))
x.grad = None

report(scenario("autograd_grad_raises"), outcome(lambda: torch.autograd.grad(torch.relu(x).sum(), [x])))

# A gradient accumulated before must not keep its old value silently.
T._functional = _functional
y = leaf()
(y * 2).sum().backward()
T._functional = failing("aten.threshold_backward.default")
report(scenario("accumulated_backward_raises"), outcome(lambda: torch.relu(y).sum().backward()))
report(scenario("accumulated_grad_read_raises"), outcome(lambda: y.grad.cpu()))

# Backward from a client tensor does not pass through reFLOP's backward():
# it returns, and the session's next read raises the kept error.
z = leaf()
loss = torch.relu(z).cpu().sum()
report(scenario("cpu_root_backward"), outcome(loss.backward))
report(scenario("cpu_root_next_read"), outcome(reflop.synchronize))
T._functional = _functional


# Gradient hooks run on the autograd thread too.
def hook_raises():
    w = leaf()
    h = w * 2
    h.register_hook(lambda g: 1 / 0)
    h.sum().backward()


def post_accumulate_hook_raises():
    w = leaf()
    w.register_post_accumulate_grad_hook(lambda t: 1 / 0)
    (w * 2).sum().backward()


report(scenario("hook_raises"), outcome(hook_raises))
report(scenario("post_accumulate_hook_raises"), outcome(post_accumulate_hook_raises))


# A factory kernel that C++ calls directly, failing in backward.
class MakesZeros(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        return a * 1

    @staticmethod
    def backward(ctx, g):
        return g + torch.zeros(4, device=g.device)


def factory_fails():
    sess.enabled.discard("aten.zeros.default")
    try:
        MakesZeros.apply(leaf()).sum().backward()
    finally:
        sess.enabled.add("aten.zeros.default")


report(scenario("factory_fails"), outcome(factory_fails))

# After all that the session still works.
r = leaf()
torch.relu(r).sum().backward()
report(scenario("recovers"), r.grad.cpu().tolist())
sys.stdout.flush()
