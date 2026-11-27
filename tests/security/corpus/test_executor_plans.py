# SPDX-License-Identifier: Apache-2.0
"""The executor's per-node plans (execute.py, compile and run).

A simple node (one output, and no runtime check, pre or post fix-up,
written-argument copy, UNINIT zeroing, range propagation or native
implementation) runs on a shorter path in run(). That path must compute
what the full path computes, keep the output check (an output that differs
from what V4 validated fails closed, E3), and every other node must stay on
the full path: runtime checks (I4), integer ranges (ADR-014) and the rest.
"""

from __future__ import annotations

import pytest
import torch
from helpers import NAME, I, decode, graph, make_session, meta, register, run, tensor_binding, tin, tnode
from reflop_worker import validators
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker.ops_generated import OPS
from test_corpus import expect, ok_after

F32, I64 = torch.float32, torch.int64
GELU_NONE = gpb.Arg(enum_code=list(OPS[NAME["aten.gelu.default"]].enums["approximate"]).index("none"))


@pytest.fixture(params=["cpu", "cuda"])
def sess(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return make_session(request.param)


def ints(*v):
    return gpb.Arg(ints=gpb.Int64List(values=list(v)))


def tlist(*refs):
    return gpb.Arg(tensors=gpb.TensorList(items=[r.tensor for r in refs]))


def plans(sess, g):
    """The compiled plans of a registered and executed graph."""
    h = register(sess, g).graph_registered.graph_hash
    return sess.programs[h]._compiled[0]


# Every argument kind a node can take, on both paths.
MIXED = [
    ("aten.relu.default", [tin(0)]),                       # 0 simple: slot argument only
    ("aten.view.default", [tnode(0), ints(3, 2)]),         # 1 simple: float view
    ("aten.gelu.default", [tnode(0), GELU_NONE]),          # 2 simple: constant keyword argument
    ("aten.cat.default", [tlist(tnode(0), tin(2)), I(0)]),  # 3 simple: tensor list
    ("aten.view.default", [tin(1), ints(2, 2)]),           # 4 full: integer view (range)
    ("aten.view.default", [tnode(4), ints(4)]),            # 5 full: integer view (range)
    ("aten.index_select.default", [tnode(3), I(0), tnode(5)]),  # 6 full: runtime check
    ("aten.view.default", [tnode(0), ints(1, 6)]),         # 7 simple
    ("aten.unbind.int", [tnode(7), I(0)]),                  # 8 simple: the kernel returns a list
]
MIXED_OUT = [(1, 0), (2, 0), (3, 0), (5, 0), (6, 0), (8, 0)]
MIXED_META = [(F32, (3, 2)), (F32, (2, 3)), (F32, (4, 3)), (I64, (4,)), (F32, (4, 3)), (F32, (6,))]


def mixed(sess):
    return graph(sess, [(F32, (2, 3)), (I64, (4,)), (F32, (2, 3))], MIXED, MIXED_META, MIXED_OUT)


def mixed_inputs():
    g = torch.Generator().manual_seed(7)
    return [torch.randn(2, 3, generator=g), torch.tensor([3, 0, 2, 1]), torch.randn(2, 3, generator=g)]


def test_simple_and_full_nodes_compute_what_torch_does(sess):
    x, idx, y = mixed_inputs()
    r = run(sess, mixed(sess), [x, idx, y])
    assert isinstance(r, cpb.ExecuteResult), r
    # The reference runs on the session's device: CUDA's gelu differs from
    # the CPU's in the last bit.
    x, idx, y = (t.to(sess.device) for t in (x, idx, y))
    h = torch.relu(x)
    c = torch.cat([h, y], 0)
    want = [h.view(3, 2), torch.nn.functional.gelu(h), c, idx.view(2, 2).view(4),
            c.index_select(0, idx), h.view(1, 6).unbind(0)[0]]
    got = [decode(t) for t in r.returned]
    for w, o in zip(want, got, strict=True):
        assert o.dtype == w.dtype and torch.equal(o, w.cpu())


def test_only_simple_nodes_take_the_short_path(sess):
    g = mixed(sess)
    assert isinstance(run(sess, g, mixed_inputs()), cpb.ExecuteResult)
    ps = plans(sess, g)
    assert [p[0] for p in ps] == [True, True, True, True, False, False, False, True, True]
    # Plans with only positional slot arguments fill them from `pairs`.
    assert [p[6] is not None for p in ps if p[0]] == [True, True, True, False, True, True]
    # The integer views propagate ranges; index_select keeps its runtime check.
    assert ps[4][14] and ps[5][14] and ps[6][8]


def test_scalar_inputs_on_the_short_path(sess):
    # A scalar input as a positional (mul) and as a keyword argument (alpha).
    g = gpb.Graph(opset_version=1, device=sess.gdevice)
    g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_TENSOR, meta=meta(F32, (3,))))
    g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_SCALAR, meta=meta(torch.float64, ())))
    g.nodes.append(gpb.Node(op_id=NAME["aten.mul.Tensor"], args=[tin(0), tin(1)]))
    g.nodes.append(gpb.Node(op_id=NAME["aten.add.Tensor"], args=[tnode(0), tin(0), tin(1)]))
    g.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=1, output=0)))
    g.output_meta.append(meta(F32, (3,)))
    r = register(sess, g)
    assert r.WhichOneof("body") == "graph_registered", r
    for s in (2.5, -1.0):
        ex = cpb.Execute(graph_hash=r.graph_registered.graph_hash)
        ex.bindings.append(tensor_binding(torch.tensor([1.0, 2.0, 3.0])))
        b = cpb.Binding()
        b.scalar.f = s
        ex.bindings.append(b)
        ex.outputs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_RETURN))
        out = []
        sess.handle(cpb.Envelope(request_id=2, execute=ex), out.append)
        x = torch.tensor([1.0, 2.0, 3.0])
        assert decode(out[0].execute_result.returned[0]).tolist() == (x * s + s * x).tolist()
    ps = sess.programs[r.graph_registered.graph_hash]._compiled[0]
    assert [p[0] for p in ps] == [True, True] and [p[6] for p in ps] == [None, None]


def _break(sess, g, i, fn):
    """Replace node i's kernel in the compiled plan: the executor's own
    check must refuse what it returns."""
    ps = plans(sess, g)
    assert ps[i][0]
    ps[i] = ps[i][:2] + (fn,) + ps[i][3:]


@pytest.mark.parametrize("fn,message", [
    (lambda x: torch.zeros(5, device=x.device), "validated"),                   # wrong shape
    (lambda x: torch.zeros(2, 3, dtype=torch.float64, device=x.device), "validated"),  # wrong dtype
    (lambda x: (x.clone(), x.clone()), "produced 2 outputs"),                    # wrong count
])
def test_the_short_path_keeps_the_output_check(sess, fn, message):
    g = graph(sess, [(F32, (2, 3))], [("aten.relu.default", [tin(0)]), ("aten.neg.default", [tnode(0)])],
              [(F32, (2, 3))], [(1, 0)])
    x = torch.randn(2, 3)
    assert isinstance(run(sess, g, [x]), cpb.ExecuteResult)
    _break(sess, g, 0, fn)
    err = run(sess, g, [x])
    expect(err, cpb.ERROR_INTERNAL, cpb.STAGE_E3_EXECUTE)
    assert message in err.message and err.node_index == 0
    ok_after(sess)


def test_integer_views_keep_their_range(sess, monkeypatch):
    # ADR-014: the index's range is measured once on the host at upload and
    # carried through both views, so the bounds check needs no measurement.
    calls = []
    real = validators.measure

    def measure(t):
        calls.append(tuple(t.shape))
        return real(t)

    monkeypatch.setattr(validators, "measure", measure)
    x, idx, y = mixed_inputs()
    r = run(sess, mixed(sess), [x, idx, y])
    assert isinstance(r, cpb.ExecuteResult), r
    assert calls == [(4,)]  # the host copy of the bound index, nothing else


def test_an_out_of_range_index_still_fails_after_the_views(sess):
    x, _, y = mixed_inputs()
    err = run(sess, mixed(sess), [x, torch.tensor([0, 1, 2, 4]), y])
    expect(err, cpb.ERROR_INDEX_OUT_OF_BOUNDS, cpb.STAGE_E2_RUNTIME)
    assert err.node_index == 6
    ok_after(sess)
