# SPDX-License-Identifier: Apache-2.0
"""conj() views match plain PyTorch: reads, arithmetic, writes and autograd.

conj() gives a view with PyTorch's conjugate bit set. reflop records it as a
materialized step in the view chain, so the values must match PyTorch whether
the read starts inside dispatch (x.cpu() flushes there) or after a
step boundary."""

from __future__ import annotations

import os

import pytest
import reflop
import torch
from reflop.connection import _as_worker_process
from reflop.trace import UnsupportedOperatorError

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
G = torch.Generator().manual_seed(7)


def cplx(*shape, dtype=torch.complex64):
    return torch.complex(torch.randn(*shape, generator=G), torch.randn(*shape, generator=G)).to(dtype)


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=12, local=DEVICE)


def close(a, b):
    a, b = a.cpu(), b.resolve_conj().resolve_neg()
    assert a.dtype == b.dtype and a.shape == b.shape
    tol = 1e-5 if a.dtype in (torch.complex64, torch.float32) else 1e-12
    assert torch.allclose(a, b, atol=tol, rtol=tol), (a, b)


CASES = {
    "conj": lambda z, m: z.conj(),
    "conj_add": lambda z, m: z.conj() + 0,
    "conj_mul": lambda z, m: z.conj() * 2,
    "conj_add_tensor": lambda z, m: z.conj() + z,
    "resolve_conj": lambda z, m: z.conj().resolve_conj(),
    "clone": lambda z, m: z.conj().clone(),
    "contiguous_t": lambda z, m: m.conj().t().contiguous(),
    "conj_conj": lambda z, m: z.conj().conj(),
    "real": lambda z, m: z.conj().real,
    "imag": lambda z, m: z.conj().imag,
    "imag_add": lambda z, m: z.conj().imag + 1,
    "view_as_real": lambda z, m: torch.view_as_real(z.conj().resolve_conj()),
    "slice_after": lambda z, m: z.conj()[1:3],
    "slice_before": lambda z, m: z[1:3].conj(),
    "select_after": lambda z, m: m.conj()[1],
    "sum": lambda z, m: z.conj().sum(),
    "abs": lambda z, m: z.conj().abs(),
    "angle": lambda z, m: z.conj().angle(),
    "matmul": lambda z, m: m.conj() @ m.mT,
    "matmul_both": lambda z, m: m.conj() @ m.conj().mT,
    "mH": lambda z, m: m.mH @ m,
    "adjoint": lambda z, m: m.adjoint().contiguous(),
    "vdot": lambda z, m: torch.vdot(z, z.conj()),
    "conj_physical": lambda z, m: torch.conj_physical(z),
}


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("read", ["in_dispatch", "after_boundary"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_conj_values(dev, case, read, dtype):
    z, m = cplx(5, dtype=dtype), cplx(3, 4, dtype=dtype)
    f = CASES[case]
    out = f(z.to(dev), m.to(dev))
    if read == "after_boundary":
        reflop.step_boundary()
    close(out, f(z.clone(), m.clone()))


def test_write_through_conj_view(dev):
    z = cplx(5)
    rz = z.to(dev)
    w, rw = z.conj(), rz.conj()
    w.add_(1 + 1j)
    rw.add_(1 + 1j)
    close(rz, z)
    close(rw, w)


def test_write_through_slice_of_conj_view(dev):
    z = cplx(4, 3)
    rz = z.to(dev)
    w, rw = z.conj(), rz.conj()
    w[1] = 5 - 2j
    rw[1] = 5 - 2j
    w[:, 0].mul_(2)
    rw[:, 0].mul_(2)
    close(rz, z)
    close(rw, w)


def test_conj_view_sees_base_writes(dev):
    z = cplx(5)
    rz = z.to(dev)
    w, rw = z.conj(), rz.conj()
    z.mul_(3 - 1j)
    rz.mul_(3 - 1j)
    close(rw, w)


def test_conj_of_offset_view(dev):
    z = cplx(6)
    rz = z.to(dev)
    w, rw = z[2:].conj(), rz[2:].conj()
    w[0] = 9j
    rw[0] = 9j
    close(rz, z)
    close(rw, w)


def test_conj_autograd(dev):
    z = cplx(5).requires_grad_(True)
    rz = z.detach().to(dev).requires_grad_(True)
    a, b = cplx(5, 3), cplx(5)
    for x, aa, bb in ((z, a, b), (rz, a.to(dev), b.to(dev))):
        loss = ((x.conj() * bb).abs().sum() + (x.conj() @ aa).abs().sum() + (x.conj() * bb).angle().sum()
                + torch.view_as_real(x.conj().resolve_conj()).pow(3).sum())
        loss.backward()
    close(rz.grad, z.grad)


@pytest.mark.xfail(strict=True, reason="select_backward's dim is validated against grad_output's rank, not "
                   "input_sizes (fails for real float tensors too)")
@pytest.mark.parametrize("part", ["real", "imag"])
def test_conj_part_autograd(dev, part):
    z = cplx(5).requires_grad_(True)
    rz = z.detach().to(dev).requires_grad_(True)
    for x in (z, rz):
        getattr(x.conj(), part).sum().backward()
    close(rz.grad, z.grad)


def test_reflop_tensors_never_report_conj(dev):
    w = cplx(3).to(dev).conj()
    assert not w.is_conj() and not w._rf_meta.is_conj()


def test_as_strided_on_conj_view_fails_closed(dev):
    z = cplx(6)
    rz = z.to(dev)
    with pytest.raises(UnsupportedOperatorError, match="as_strided on a conj"):
        rz[2:].conj().as_strided((2,), (1,), 1)
    close(rz[2:].conj().clone().as_strided((2,), (1,), 1), z[2:].conj().clone().as_strided((2,), (1,), 1))


def test_neg_view_fails_closed(dev):
    with pytest.raises(UnsupportedOperatorError, match="negative bit"):
        torch._neg_view(cplx(3).to(dev))


# --------------------------------------------------- in-process worker state


class _Wrapper(torch.Tensor):
    """A wrapper subclass: its __torch_dispatch__ runs in the state where
    reflop's own dispatch flushes (x.cpu())."""

    @staticmethod
    def __new__(cls, e):
        return torch.Tensor._make_wrapper_subclass(cls, e.shape, dtype=e.dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return cls.body()


def _conj_add():
    z = torch.tensor([1 + 2j, 3 - 4j])
    return torch.ops.aten.add.Tensor(torch.ops.aten._conj.default(z), 0)


def test_in_process_worker_runs_in_worker_dispatch_state():
    want = torch.tensor([1 - 2j, 3 + 4j])
    seen = {}

    def body():
        seen["plain"] = _conj_add()
        with _as_worker_process():
            seen["worker"] = _conj_add()
        seen["after"] = torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Conjugate)
        return torch.zeros(2, dtype=torch.complex64)

    _Wrapper.body = staticmethod(body)
    _Wrapper(torch.zeros(2, dtype=torch.complex64)) + 0
    # Inside dispatch PyTorch skips the conjugate kernel; the guard restores it
    # and puts the dispatch state back afterwards.
    assert not torch.equal(seen["plain"], want)
    assert torch.equal(seen["worker"], want)
    assert seen["after"]


def test_in_process_worker_ignores_caller_modes():
    from torch.utils._python_dispatch import TorchDispatchMode

    calls = []

    class Spy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            calls.append(func)
            return func(*args, **(kwargs or {}))

    with Spy():
        with _as_worker_process():
            torch.ones(2) + 1
        assert not calls
        torch.ones(2) + 1
    assert calls
