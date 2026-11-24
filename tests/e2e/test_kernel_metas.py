# SPDX-License-Identifier: Apache-2.0
"""Ops whose CPU or CUDA kernels return other shapes or dtypes than
PyTorch's meta kernels say (#25 E, ADR-049).

The client records each output with the shape and dtype the kernel on the
session's device returns, and the worker infers the same, so the outputs
fetch, and bind into later graphs (a backward reading the forward's saved
statistics), as they do in PyTorch. Each case runs in plain PyTorch on the
local device and through reFLOP (the in-process worker, and a running
reflopd with REFLOP_E2E=1); outputs and gradients must agree in shape,
dtype and value, both as the client recorded them and as fetched.

Regressions: these calls failed with an internal error (the kernel output
did not match the worker's meta, E3) or a binding error (E1) in the next
graph, e.g. EmbeddingBag's backward, a first-layer BatchNorm, or
F.logsigmoid's backward on CUDA.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import reflop
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
A = torch.ops.aten
f16, bf16, f32, f64 = torch.float16, torch.bfloat16, torch.float32, torch.float64


@pytest.fixture(scope="module", params=["local", "server"])
def dev(request):
    """local: the in-process worker. server: a running reflopd (REFLOP_E2E=1,
    a paired client) on the same kind of device."""
    from reflop import session

    if request.param == "server" and os.environ.get("REFLOP_E2E") != "1":
        pytest.skip("needs a running reflopd (REFLOP_E2E=1)")
    # A new session, so it is the default: a tensor that a C++ backward
    # formula creates (ctc_loss's zero_infinity) goes to the default session
    # (BUGS.md #26).
    if request.param == "local":
        index, d = 21, reflop.device(index=21, local=DEVICE)
    else:
        index, d = 22, reflop.device(index=22, kind=DEVICE)
    assert session.current() is session.by_index(index)
    yield d
    session.by_index(index).close()


def _t(d, *shape, dtype=f32, seed=0, grad=False):
    t = torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(d, dtype)
    return t.requires_grad_() if grad else t


def _grads(loss, *ts):
    loss.float().sum().backward()
    return [t.grad for t in ts if t.grad is not None]


def _module(make, d, dtype=f32):
    torch.manual_seed(0)
    return make().to(d, dtype)


def _ebag(mode, pad=None, inc=False, psw=False, transposed=False):
    def case(d):
        torch.manual_seed(1)
        w = torch.randn(3, 10).to(d).requires_grad_() if transposed else torch.randn(10, 3).to(d).requires_grad_()
        idx = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], device=d)
        off = torch.tensor([0, 4] + ([8] if inc else []), device=d)
        weights = torch.rand(8).to(d) if psw else None
        out = F.embedding_bag(idx, w.t() if transposed else w, off, mode=mode, padding_idx=pad,
                              include_last_offset=inc, per_sample_weights=weights)
        return [out.detach(), *_grads(out, w)]
    return case


def _bn_first_layer(train):
    # BatchNorm as the first layer: the input needs no gradient, so the
    # backward computes no grad_input.
    def case(d):
        m = _module(lambda: nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2)), d)
        m.train(train)
        out = m(_t(d, 6, 4))
        return [out.detach(), *_grads(out, *m.parameters()), m[0].running_mean, m[0].running_var]
    return case


def _bn_eval(d):
    m = _module(lambda: nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4)), d).eval()
    x = _t(d, 2, 3, 6, 6, grad=True)
    out = m(x)
    return [out.detach(), *_grads(out, x, *m.parameters())]


def _norm(make, dtype, params, train=True, shape=(2, 4, 3)):
    # Half and bfloat16 inputs with parameters in the same dtype or float32
    # (the CPU kernels take float32 parameters for any of them).
    def case(d):
        m = _module(make, d, params)
        m.train(train)
        x = _t(d, *shape, dtype=dtype, grad=True)
        out = m(x)
        bufs = [b for b in m.buffers() if b.is_floating_point()]
        return [out.detach(), *_grads(out, x, *m.parameters()), *bufs]
    return case


def _loss(fn, backward=True):
    # A float64 target: the kernels compute in self's dtype (ADR-049).
    def case(d):
        x = _t(d, 3, 4, grad=backward)
        t = _t(d, 3, 4, dtype=f64, seed=1)
        out = fn(x, t)
        return [out.detach(), *(_grads(out, x) if backward else [])]
    return case


def _logsigmoid(d):
    x = _t(d, 3, 4, grad=True)
    out = F.logsigmoid(x)
    return [out.detach(), *_grads(out, x)]


def _ctc(d):
    lp = _t(d, 6, 3, 5, grad=True)
    out = F.ctc_loss(lp.log_softmax(2), torch.tensor([[1, 2, 3], [4, 1, 1], [2, 2, 2]], device=d), [6, 3, 1],
                     [2, 1, 0], reduction="sum", zero_infinity=True)
    return [out.detach(), *_grads(out, lp)]


def _direct(f):
    def case(d):
        out = f(d)
        return [t.detach() for t in (out if isinstance(out, (tuple, list)) else [out])]
    return case


def _prelu(d):
    m = _module(lambda: nn.PReLU(3), d)
    x = _t(d, 2, 3, 4, grad=True)
    out = m(x)
    return [out.detach(), *_grads(out, x, *m.parameters())]


def _prelu_wider_weight(d):
    # weight has more dims than self: both gradients have the broadcast shape.
    x, w = _t(d, 3, grad=True), _t(d, 1, 1, seed=1).abs().requires_grad_()
    out = A._prelu_kernel(x, w)
    return [out.detach(), *_grads(out, x, w)]


def _bn_saved_stats(train, dtype=f32, params=f32):
    # The saved statistics bind into the backward as the forward returned them.
    def case(d):
        x = _t(d, 4, 3, 5, dtype=dtype)
        w, b = _t(d, 3, dtype=params, seed=1), _t(d, 3, dtype=params, seed=2)
        rm, rv = _t(d, 3, dtype=params, seed=3), _t(d, 3, dtype=params, seed=4).abs()
        out, mean, invstd = A.native_batch_norm(x, w, b, rm, rv, train, 0.1, 1e-5)
        g = torch.ones_like(out)
        grads = A.native_batch_norm_backward(g, x, w, rm, rv, mean, invstd, train, 1e-5, [False, True, True])
        return [out, mean, invstd, rm, rv, *[t for t in grads if t is not None]]
    return case


CASES = {
    **{f"embedding_bag {mode}{' pad' if pad is not None else ''}{' last' if inc else ''}": _ebag(mode, pad, inc)
       for mode in ("sum", "mean", "max") for pad in (None, 2) for inc in (False, True)},
    "embedding_bag per_sample_weights": _ebag("sum", psw=True),
    "embedding_bag transposed weight": _ebag("sum", transposed=True),
    "embedding_bag transposed weight max": _ebag("max", transposed=True),
    "embedding_bag outputs": _direct(lambda d: A._embedding_bag(
        _t(d, 10, 3), torch.tensor([1, 2, 4, 5], device=d), torch.tensor([0, 2], device=d, dtype=torch.int32),
        False, 0)),
    "embedding_bag outputs max": _direct(lambda d: A._embedding_bag(
        _t(d, 10, 3), torch.tensor([1, 2, 4, 5], device=d), torch.tensor([0, 2], device=d), False, 2)),
    "batch_norm first layer": _bn_first_layer(True),
    "batch_norm first layer eval": _bn_first_layer(False),
    "batch_norm eval": _bn_eval,
    "batch_norm saved stats": _bn_saved_stats(True),
    "batch_norm saved stats eval": _bn_saved_stats(False),
    "batch_norm saved stats half": _bn_saved_stats(True, f16, f32),
    "batch_norm functional half": _direct(lambda d: A._native_batch_norm_legit_functional(
        _t(d, 4, 3, 5, dtype=f16), None, None, _t(d, 3, dtype=f16), _t(d, 3, dtype=f16).abs(), True, 0.1, 1e-5)),
    "batch_norm no_stats half": _direct(lambda d: A._native_batch_norm_legit.no_stats(
        _t(d, 4, 3, 5, dtype=f16), None, None, True, 0.1, 1e-5)),
    **{f"{name} {str(dt)[6:]} {str(p)[6:]} params{'' if train else ' eval'}": _norm(make, dt, p, train)
       for name, make in (("batch_norm", lambda: nn.BatchNorm1d(4)), ("layer_norm", lambda: nn.LayerNorm(3)),
                          ("group_norm", lambda: nn.GroupNorm(2, 4)))
       for dt in (f16, bf16) for p in (dt, f32) for train in (True, False)
       if train or name == "batch_norm"},
    "layer_norm outputs half": _direct(lambda d: A.native_layer_norm(_t(d, 2, 4, 3, dtype=f16), [3], None, None, 1e-5)),
    "group_norm outputs half": _direct(lambda d: A.native_group_norm(_t(d, 2, 4, 3, dtype=f16), None, None, 2, 4, 3,
                                                                     2, 1e-5)),
    "logsigmoid": _logsigmoid,
    "log_sigmoid_forward": _direct(lambda d: A.log_sigmoid_forward(_t(d, 3, 4))),
    "huber_loss float64 target": _loss(F.huber_loss, backward=False),
    "mse_loss float64 target": _loss(F.mse_loss),
    "smooth_l1_loss float64 target": _loss(F.smooth_l1_loss),
    "soft_margin_loss float64 target": _loss(F.soft_margin_loss),
    "l1_loss float64 target": _loss(F.l1_loss),
    "ctc_loss": _ctc,
    "prelu": _prelu,
    "prelu weight wider than self": _prelu_wider_weight,
    "threshold_backward mixed dtypes": _direct(lambda d: A.threshold_backward(_t(d, 5), _t(d, 5, dtype=f64, seed=1),
                                                                              0.0)),
    "pixel_unshuffle empty": _direct(lambda d: F.pixel_unshuffle(torch.ones(1, 1, 0, 4, device=d), 2)),
    "affine_grid_generator empty": _direct(lambda d: A.affine_grid_generator(_t(d, 2, 2, 3), [2, 3, 0, 4], False)),
    "view_copy transposed": _direct(lambda d: torch.view_copy(_t(d, 3, 4).t(), [12])),
}

# The CUDA kernels refuse float32 parameters for a half or bfloat16 input.
CPU_ONLY = {n for n in CASES if n.startswith(("layer_norm", "group_norm")) and n.endswith("float32 params")}


def _references(path):
    out = {n: [t.detach().cpu() for t in f(torch.device(DEVICE))]
           for n, f in CASES.items() if DEVICE == "cpu" or n not in CPU_ONLY}
    torch.save(out, path)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    # Plain PyTorch in a child process: once reflop has registered its
    # backend, a local CUDA backward in this process fails.
    path = tmp_path_factory.mktemp("kernel_metas") / "want.pt"
    code = ("import importlib.util as u, sys; s = u.spec_from_file_location('m', sys.argv[1]); "
            "m = u.module_from_spec(s); s.loader.exec_module(m); m._references(sys.argv[2])")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    subprocess.run([sys.executable, "-c", code, __file__, str(path)], env=env, check=True, timeout=600)
    return torch.load(path, weights_only=True)


@pytest.mark.parametrize("case", list(CASES))
def test_matches_pytorch(dev, reference, case):
    if case in CPU_ONLY and DEVICE != "cpu":
        pytest.skip("PyTorch refuses it on CUDA")
    want = reference[case]
    got = CASES[case](dev)
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert (g.dtype, tuple(g.shape)) == (w.dtype, tuple(w.shape)), f"output {i} as recorded"
        v = g.cpu()
        assert (v.dtype, tuple(v.shape)) == (w.dtype, tuple(w.shape)), f"output {i} as fetched"
        torch.testing.assert_close(v, w, equal_nan=True, msg=lambda m, i=i: f"output {i}: {m}")

