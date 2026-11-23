# SPDX-License-Identifier: Apache-2.0
"""select_backward, diagonal_backward and slice_backward: dims address the
input described by input_sizes, not grad_output, which can have fewer dims.

V3 checks the dims against len(input_sizes). The meta kernel (V4) checks
index, offset and shapes, and the output limits apply, all before anything is
allocated. Every malformed graph must be rejected at registration, and the
worker must keep serving."""

from __future__ import annotations

import pytest
import torch
from helpers import INTS, I, decode, graph, make_session, register, run, tin
from reflop_worker._pb import compute_pb2 as cpb
from test_corpus import expect, ok_after

V3, V4 = cpb.STAGE_V3_SCHEMA, cpb.STAGE_V4_META
G = torch.Generator().manual_seed(3)


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def sess(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return make_session(request.param)


def sel(grad_shape, sizes, dim, index, out=None):
    return grad_shape, ("aten.select_backward.default", [tin(0), INTS(*sizes), I(dim), I(index)]), out or sizes


def diag(grad_shape, sizes, offset, d1, d2, out=None):
    return (grad_shape, ("aten.diagonal_backward.default", [tin(0), INTS(*sizes), I(offset), I(d1), I(d2)]),
            out or sizes)


def slc(grad_shape, sizes, dim, start, end, step, out=None):
    return (grad_shape, ("aten.slice_backward.default", [tin(0), INTS(*sizes), I(dim), I(start), I(end), I(step)]),
            out or sizes)


def build(sess, case):
    grad_shape, node, out = case
    return graph(sess, [(torch.float32, grad_shape)], [node], [(torch.float32, out)])


def native(op, grad, args):
    return getattr(torch.ops.aten, op).default(grad, *args)


# Every dim, both signs, on the last dims grad_output does not have.
VALID = (
    [sel(tuple(s for i, s in enumerate((3, 4, 5)) if i != d % 3), (3, 4, 5), d, 1) for d in range(-3, 3)]
    + [sel((), (7,), 0, 6), sel((), (7,), -1, -7), sel((3, 4), (3, 4, 2), 2, -1)]
    + [diag(None, (3, 4, 5), o, a, b) for a in range(-3, 3) for b in range(-3, 3) if a % 3 != b % 3
       for o in (-1, 0, 2)]
    + [slc((3, 4, 2), (3, 4, 5), d, 1, 3, 1) if d % 3 == 2 else None for d in (2, -1)]
)
VALID = [c for c in VALID if c is not None]


def _diag_grad_shape(sizes, offset, d1, d2):
    return tuple(torch.empty(sizes, device="meta").diagonal(offset, d1, d2).shape)


@pytest.mark.parametrize("case", range(len(VALID)))
def test_valid_backward_dims_match_torch(sess, case):
    grad_shape, (name, args), out = VALID[case]
    ints = [a.ints.values if a.HasField("ints") else a.i for a in args[1:]]
    if grad_shape is None:  # diagonal_backward: grad_output is the diagonal's shape
        grad_shape = _diag_grad_shape(list(ints[0]), *ints[1:])
    grad = torch.randn(grad_shape, generator=G)
    g = graph(sess, [(torch.float32, grad_shape)], [(name, args)], [(torch.float32, out)])
    r = run(sess, g, [grad])
    assert isinstance(r, cpb.ExecuteResult), r
    want = native(name.split(".")[1], grad, [list(ints[0]), *ints[1:]])
    assert torch.equal(decode(r.returned[0]), want)


REJECT = {
    # dims outside len(input_sizes)
    "select_dim_high": (sel((3, 4), (3, 4, 5), 3, 0), cpb.ERROR_INVALID_GRAPH, V3),
    "select_dim_low": (sel((3, 4), (3, 4, 5), -4, 0), cpb.ERROR_INVALID_GRAPH, V3),
    "select_dim_past_grad_rank_but_not_sizes": (sel((3, 4, 5, 6), (3, 4), 2, 0, out=(3, 4)),
                                                cpb.ERROR_INVALID_GRAPH, V3),
    "diagonal_dim1_high": (diag((3, 4), (3, 4, 5), 0, 3, 0), cpb.ERROR_INVALID_GRAPH, V3),
    "diagonal_dim2_low": (diag((3, 4), (3, 4, 5), 0, 0, -4), cpb.ERROR_INVALID_GRAPH, V3),
    "slice_dim_high": (slc((3, 4, 5), (3, 4, 5), 3, 0, 1, 1), cpb.ERROR_INVALID_GRAPH, V3),
    # index, offset and shapes are the meta kernel's
    "select_index_high": (sel((4, 5), (3, 4, 5), 0, 3), cpb.ERROR_SHAPE_MISMATCH, V4),
    "select_index_low": (sel((4, 5), (3, 4, 5), 0, -4), cpb.ERROR_SHAPE_MISMATCH, V4),
    "select_grad_shape": (sel((4, 6), (3, 4, 5), 0, 0), cpb.ERROR_SHAPE_MISMATCH, V4),
    "select_negative_size": (sel((4, 5), (-3, 4, 5), 0, 0, out=(3, 4, 5)), cpb.ERROR_SHAPE_MISMATCH, V4),
    "diagonal_same_dims": (diag((3, 4), (3, 4, 5), 0, 1, 1), cpb.ERROR_SHAPE_MISMATCH, V4),
    "diagonal_same_dims_wrapped": (diag((3, 4), (3, 4, 5), 0, 2, -1), cpb.ERROR_SHAPE_MISMATCH, V4),
    "diagonal_grad_shape": (diag((3, 5), (3, 4, 5), 0, 1, 2), cpb.ERROR_SHAPE_MISMATCH, V4),
    "slice_grad_shape": (slc((3, 4, 3), (3, 4, 5), 2, 1, 3, 1), cpb.ERROR_SHAPE_MISMATCH, V4),
}


@pytest.mark.parametrize("name", sorted(REJECT))
def test_malformed_backward_dims_rejected(sess, name):
    case, code, stage = REJECT[name]
    expect(register(sess, build(sess, case)).error, code, stage)
    ok_after(sess)


# The declared graph output is small, so the node's own output is what is
# over the limit (a large declared output is rejected earlier, at V2).
def test_input_sizes_over_the_numel_limit_rejected():
    s = make_session("cpu", max_tensor_numel=1 << 22)
    expect(register(s, build(s, sel((1 << 11, 1 << 11), (4, 1 << 11, 1 << 11), 0, 0, out=(1,)))).error,
           cpb.ERROR_RESOURCE_LIMIT, V4)
    ok_after(s)


def test_input_sizes_over_the_dims_limit_rejected():
    s = make_session("cpu", max_dims=5)
    sizes = (1,) * 6
    expect(register(s, build(s, sel((1,) * 5, sizes, 5, 0, out=(1,)))).error, cpb.ERROR_RESOURCE_LIMIT, V4)
    ok_after(s)
