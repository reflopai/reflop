# SPDX-License-Identifier: Apache-2.0
"""Grid cases for the meta/kernel parity test (parity.py, #25 B, ADR-049).

The RNG, factory and reduction ops take a dtype argument or a value the
kernels check after allocating: the bounds of uniform and randint, the
probability of bernoulli, the fill of linspace's one step. Mutating a seed
reaches few of those, so GRID() builds them directly: every dtype, as the
output dtype and as the input's, on full and empty shapes (DTYPES), and
for the value checks, the edges of each dtype's range and inf and NaN
(VALUES)."""

from __future__ import annotations

import itertools
import math

import torch
from parity import DTYPES, SCALAR_TYPES, SPEC, Call, _base

A = "aten."
DTN = DTYPES + [None]
SHAPES = ([3, 4], [0], [3, 0])


def _call(op, **named):
    spec = SPEC[op]
    vals = []
    for sa, s in zip(spec.args, spec.fn._schema.arguments):
        if sa.name in named:
            v = named[sa.name]
        else:
            v = s.default_value
            if _base(sa.type).rstrip("?") == "ScalarType" and type(v) is int:
                v = SCALAR_TYPES[v]
        vals.append(v)
    return Call(op, vals, "grid " + " ".join(f"{k}={_label(v)}" for k, v in named.items()))


def _label(v):
    if isinstance(v, torch.Tensor):
        return f"{str(v.dtype)[6:]}{list(v.shape)}"
    if isinstance(v, list):
        return "[" + ",".join(_label(x) for x in v) + "]"
    if isinstance(v, torch.dtype):
        return str(v)[6:]
    return repr(v)


def _t(dt, shape, lo=-2.0, hi=2.0):
    g = torch.Generator().manual_seed(len(shape) * 31 + DTYPES.index(dt))
    x = torch.rand(shape, generator=g, dtype=torch.float64) * (hi - lo) + lo
    if dt == torch.bool:
        return x > (lo + hi) / 2
    if dt.is_complex:
        return torch.complex(x, x.flip(-1) if x.dim() else x).to(dt)
    if not dt.is_floating_point:
        return x.round().to(dt) if lo >= 0 or torch.iinfo(dt).min < 0 else x.abs().round().to(dt)
    return x.to(dt)


def _factories():
    for dt in DTN:
        for size in SHAPES:
            yield _call(A + "rand.default", size=size, dtype=dt)
            yield _call(A + "randn.default", size=size, dtype=dt)
            yield _call(A + "randint.default", high=5, size=size, dtype=dt)
            yield _call(A + "randint.low", low=-3, high=5, size=size, dtype=dt)
            yield _call(A + "normal.float_float", mean=0.0, std=1.0, size=size, dtype=dt)
        for n in (5, 1, 0):
            yield _call(A + "randperm.default", n=n, dtype=dt)
            yield _call(A + "eye.default", n=n, dtype=dt)
            yield _call(A + "eye.m", n=n, m=3, dtype=dt)
            yield _call(A + "linspace.default", start=0, end=4, steps=n, dtype=dt)
            yield _call(A + "logspace.default", start=0, end=4, steps=n, dtype=dt)
        for end in (5, 0, 2.5, True):
            yield _call(A + "arange.default", end=end, dtype=dt)
            yield _call(A + "arange.start", start=0, end=end, dtype=dt)
            yield _call(A + "arange.start_step", start=0, end=end, step=1, dtype=dt)


def _on_tensors():
    f32 = torch.float32
    for sdt in DTYPES:
        for shape in SHAPES:
            x = _t(sdt, shape)
            for dt in DTN:
                yield _call(A + "rand_like.default", self=x, dtype=dt)
                yield _call(A + "randn_like.default", self=x, dtype=dt)
                yield _call(A + "randint_like.default", self=x, high=5, dtype=dt)
                yield _call(A + "randint_like.low_dtype", self=x, low=0, high=2, dtype=dt)
                yield _call(A + "sum.default", self=x, dtype=dt)
                for dims in ([1], [0], [], None):
                    if dims and x.dim() <= max(dims):
                        continue
                    yield _call(A + "sum.dim_IntList", self=x, dim=dims, dtype=dt)
                    yield _call(A + "nansum.default", self=x, dim=dims, dtype=dt)
                yield _call(A + "cumsum.default", self=x, dim=0, dtype=dt)
                yield _call(A + "cumprod.default", self=x, dim=0, dtype=dt)
                for o in (2, 1, 0, 3, 2.5, math.inf, -math.inf, -1, -2.5, math.nan, True):
                    yield _call(A + "_foreach_norm.Scalar", self=[x], ord=o, dtype=dt)
                yield _call(A + "_foreach_norm.Scalar", self=[x, _t(f32, [2])], ord=2, dtype=dt)
                yield _call(A + "_foreach_norm.Scalar", self=[_t(f32, [2]), x], ord=1, dtype=dt)
            yield _call(A + "uniform.default", self=x)
            yield _call(A + "exponential.default", self=x)
            yield _call(A + "normal.Tensor_float", mean=x)
            yield _call(A + "normal.float_Tensor", std=_t(sdt, shape, 0, 2))
            yield _call(A + "normal.Tensor_Tensor", mean=x, std=_t(f32, shape, 0, 2))
            yield _call(A + "normal.Tensor_Tensor", mean=_t(f32, shape), std=_t(sdt, shape, 0, 2))
            yield _call(A + "bernoulli.p", self=x, p=0.5)
            yield _call(A + "bernoulli.default", self=_t(sdt, shape, 0, 1))
            yield _call(A + "bernoulli.Tensor", self=x, p=_t(f32, shape, 0, 1))
            yield _call(A + "bernoulli.Tensor", self=_t(f32, shape), p=_t(sdt, shape, 0, 1))
            for n, rep in ((1, True), (2, False), (5, True)):
                yield _call(A + "multinomial.default", self=_t(sdt, shape, 0.1, 1), num_samples=n, replacement=rep)
            for p, train in itertools.product((0.0, 0.5, 1.0), (True, None, False)):
                yield _call(A + "native_dropout.default", input=x, p=p, train=train)
            for mdt in DTYPES:
                yield _call(A + "native_dropout_backward.default", grad_output=x, mask=_t(mdt, shape, 0, 1), scale=2.0)
            for idt in DTYPES:
                for op in ("_softmax_backward_data", "_log_softmax_backward_data"):
                    yield _call(A + op + ".default", grad_output=x, output=_t(sdt, shape), dim=0, input_dtype=idt)


def _values():
    inf, nan = math.inf, math.nan
    fv = [0.0, 1.0, -1.0, 2.5, 70000.0, -70000.0, 1e10, 3.5e38, -3.5e38, 1e300, -1e300, inf, -inf, nan]
    floats = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128]
    for dt in floats:
        for shape in ([3], [0]):
            x = _t(dt, shape)
            for lo, hi in itertools.product(fv, fv):
                yield _call(A + "uniform.default", self=x, **{"from": lo}, to=hi)
            for p in (0.0, 1.0, -0.1, 1.1, nan, inf, -0.0, 1e-300):
                yield _call(A + "bernoulli.p", self=x, p=p)
            for lam in (1.0, 0.0, -1.0, nan, inf, 1e-300):
                yield _call(A + "exponential.default", self=x, lambd=lam)
            for std in (1.0, 0.0, -1.0, nan, inf):
                yield _call(A + "normal.Tensor_float", mean=x, std=std)
            for mean in (0.0, 70000.0, 1e10, 3.5e38, 1e300, inf, nan):
                yield _call(A + "normal.float_Tensor", mean=mean, std=_t(dt, shape, 0, 2))
    iv = [-2**63, -2**31 - 1, -129, -128, -1, 0, 1, 2, 3, 127, 128, 255, 256, 2049, 65504, 65505, 70000,
          2**24 + 1, 2**53 + 1, 2**62, 2**63 - 1]
    for dt in DTN:
        for size in ([3], [0]):
            for lo, hi in itertools.product(iv, iv):
                yield _call(A + "randint.low", low=lo, high=hi, size=size, dtype=dt)
            for hi in iv:
                yield _call(A + "randint.default", high=hi, size=size, dtype=dt)
            for mean, std in itertools.product((0.0, 70000.0, 3.5e38, 1e300, inf, nan), (1.0, 0.0, -1.0, nan, inf)):
                yield _call(A + "normal.float_float", mean=mean, std=std, size=size, dtype=dt)
        for a, b, step in itertools.product((0.0, 1.0, -1.0, 2.5, 70000.0, 1e10, inf, nan),
                                            (0.0, 5.0, -3.0, 2.5, 70000.0, 1e10, inf, nan), (1.0, -1.0, 0.0, 0.5)):
            yield _call(A + "arange.start_step", start=a, end=b, step=step, dtype=dt)
        for a, b in itertools.product((0.0, -1.0, 2.5, 70000.0, 3.5e38, 1e300, inf, nan), repeat=2):
            for n in (0, 1, 5):
                yield _call(A + "linspace.default", start=a, end=b, steps=n, dtype=dt)
                yield _call(A + "logspace.default", start=a, end=b, steps=n, dtype=dt)
        for base in (2.0, 1e300, -10.0, 0.0, nan):
            yield _call(A + "logspace.default", start=1.5, end=3.0, steps=1, base=base, dtype=dt)
            yield _call(A + "logspace.default", start=3.0, end=1.0, steps=2, base=base, dtype=dt)
    for sdt in (torch.int8, torch.uint8, torch.bool, torch.float16, torch.float32, torch.int64):
        for shape in ([3], [0]):
            x = _t(sdt, shape, 0, 1)
            for dt in (None, torch.int8, torch.float16, torch.bool):
                for lo, hi in itertools.product((-129, -1, 0, 1, 2, 256, 2049, 70000, 2**63 - 1), repeat=2):
                    yield _call(A + "randint_like.low_dtype", self=x, low=lo, high=hi, dtype=dt)
                for hi in (-1, 0, 1, 2, 128, 256, 2049, 70000, 2**63 - 1):
                    yield _call(A + "randint_like.default", self=x, high=hi, dtype=dt)


def GRID():
    return [*_factories(), *_on_tensors(), *_values()]
