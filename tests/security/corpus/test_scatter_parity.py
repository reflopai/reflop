# SPDX-License-Identifier: Apache-2.0
"""select_scatter, slice_scatter, diagonal_scatter and as_strided_scatter
accept exactly what PyTorch's kernels accept.

PyTorch's meta kernels for these check nothing, so a bad index, dims, offset
or src shape used to pass V4, and the device kernel allocated its output
before it raised (I3). Each case here runs the native kernel on CPU tensors
and the worker on the same graph: where PyTorch accepts, the worker returns
the same values; where PyTorch raises, the worker rejects the graph at
registration and keeps serving."""

from __future__ import annotations

import itertools

import pytest
import torch
from helpers import INTS, NONE, I, decode, graph, make_session, register, run, tin, tnode
from reflop_worker._pb import compute_pb2 as cpb
from test_corpus import ok_after

A = torch.ops.aten
SHAPE = (3, 4, 5)
G = torch.Generator().manual_seed(5)


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def sess(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return make_session(request.param)


def opt(v):
    return NONE if v is None else I(v)


def view_shape(f):
    try:
        return tuple(f(torch.empty(SHAPE)).shape)
    except Exception:  # noqa: BLE001 - an invalid view: try a src of some other shape
        return (4, 5)


def cases():
    out = []
    # (label, op, int args as graph Args, the same for the native call, src shape, src dtype, transposed self)
    for d, i in itertools.product(range(-4, 4), (-4, -1, 0, 2, 3, 5)):
        vs = view_shape(lambda t, d=d, i=i: t.select(d, i))
        for src in {vs, vs[:-1], vs + (1,), (1,) + vs[1:] if vs else (1,)}:
            out.append((f"select d={d} i={i} src={src}", "select_scatter", [I(d), I(i)], [d, i], src))
    for d, (st, en), step in itertools.product(range(-4, 4), [(None, None), (1, 3), (-2, None), (3, 99), (4, 1)],
                                               (1, 2, 0, -1)):
        vs = view_shape(lambda t, d=d, st=st, en=en, step=step: A.slice.Tensor(t, d, st, en, step))
        for src in {vs, vs[:-1] + (vs[-1] + 1,) if vs else (1,)}:
            out.append((f"slice d={d} {st}:{en}:{step} src={src}", "slice_scatter",
                        [I(d), opt(st), opt(en), I(step)], [d, st, en, step], src))
    for o, (d1, d2) in itertools.product((-5, -1, 0, 2, 6), [(0, 1), (1, 2), (2, 0), (-1, -3), (1, 1), (2, -1), (0, 3)]):
        vs = view_shape(lambda t, o=o, d1=d1, d2=d2: t.diagonal(o, d1, d2))
        for src in {vs, vs[:-1] + (vs[-1] + 1,)}:
            out.append((f"diagonal o={o} {d1},{d2} src={src}", "diagonal_scatter", [I(o), I(d1), I(d2)], [o, d1, d2],
                        src))
    strided = [((2, 2), (5, 1), 0), ((2, 2), (5, 1), 50), ((4, 5), (5, 1), 40), ((4, 5), (5, 1), 41), ((60,), (1,), 0),
               ((61,), (1,), 0), ((2, 2), (1, 1), 0), ((2, 2), (0, 1), 0), ((1, 2), (0, 1), 3), ((2, 2), (-1, 1), 4),
               ((2, 2), (5, 1), -1), ((2, 2), (5,), 0), ((0, 3), (1, 1), 59), ((0, 3), (1, 1), 61), ((3,), (20,), 19),
               ((3,), (20,), 20), ((2, 2), (5, 1), None)]
    for size, stride, off in strided:
        try:
            vs = tuple(torch.empty(SHAPE).as_strided(size, stride, off or 0).shape)
        except Exception:  # noqa: BLE001
            vs = tuple(size)
        for src in {vs, vs + (1,)}:
            out.append((f"as_strided {size} {stride} {off} src={src}", "as_strided_scatter",
                        [INTS(*size), INTS(*stride), opt(off)], [list(size), list(stride), off], src))
    full = []
    for label, op, gargs, nargs, src in out:
        for dt in (torch.float32, torch.float64):
            for tr in (False, True):
                full.append((f"{label} {dt} {'T' if tr else 'C'}", op, gargs, nargs, src, dt, tr))
    return full


CASES = cases()


def native(op, x, src, nargs):
    try:
        return getattr(A, op).default(x, src, *nargs)
    except Exception:  # noqa: BLE001 - PyTorch rejects this case
        return None


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_scatter_matches_pytorch(sess, case):
    _, op, gargs, nargs, src_shape, src_dt, transposed = case
    base_shape = SHAPE[::-1] if transposed else SHAPE
    base = torch.randn(base_shape, generator=G)
    src = torch.randn(src_shape, generator=G).to(src_dt)
    x = base.transpose(0, 2) if transposed else base
    want = native(op, x, src, nargs)

    nodes, self_ref = [], tin(0)
    if transposed:
        nodes.append(("aten.transpose.int", [tin(0), I(0), I(2)]))
        self_ref = tnode(0)
    nodes.append((f"aten.{op}.default", [self_ref, tin(1), *gargs]))
    g = graph(sess, [(torch.float32, base_shape), (src_dt, src_shape)], nodes, [(torch.float32, SHAPE)])

    if want is None:
        r = register(sess, g)
        assert r.WhichOneof("body") == "error", "PyTorch rejects this case; the worker registered it"
        assert r.error.stage in (cpb.STAGE_V3_SCHEMA, cpb.STAGE_V4_META), cpb.ValidationStage.Name(r.error.stage)
        ok_after(sess)
        return
    r = run(sess, g, [base, src])
    assert isinstance(r, cpb.ExecuteResult), f"PyTorch accepts this case; the worker rejected it: {r}"
    got = decode(r.returned[0])
    shared = _shared_addresses(op, nargs, x)
    if shared is None:
        assert torch.equal(got, want)
        return
    # The view writes several src elements to one location; which one lands
    # is up to the kernel (CPU and CUDA differ). Everything else must match.
    keep = torch.ones(want.numel(), dtype=torch.bool)
    keep[list(shared)] = False
    assert torch.equal(got.reshape(-1)[keep], want.reshape(-1)[keep])


def _shared_addresses(op, nargs, x):
    """For an as_strided_scatter whose view hits a location more than once,
    those locations as flat indices of the output; else None."""
    if op != "as_strided_scatter":
        return None
    size, stride, off = nargs
    out = x.clone()
    seen, shared = set(), set()
    for idx in itertools.product(*(range(n) for n in size)):
        addr = (off if off is not None else 0) + sum(i * s for i, s in zip(idx, stride))
        (shared if addr in seen else seen).add(addr)
    if not shared:
        return None
    # Map storage addresses of the clone to flat indices of the logical output.
    addr_of = torch.empty(out.numel(), dtype=torch.long)
    for flat, idx in enumerate(itertools.product(*(range(n) for n in out.shape))):
        addr_of[flat] = sum(i * s for i, s in zip(idx, out.stride()))
    return {flat for flat in range(out.numel()) if int(addr_of[flat]) in shared}


def test_both_outcomes_are_covered():
    acc = sum(native(op, torch.zeros(SHAPE[::-1]).transpose(0, 2) if tr else torch.zeros(SHAPE),
                     torch.zeros(s).to(dt), n) is not None for _, op, _, n, s, dt, tr in CASES)
    assert 100 < acc < len(CASES) - 100, (acc, len(CASES))

