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
from helpers import NAME, NONE, B, I, decode, graph, make_session, meta, register, run, tensor_binding, tin, tnode
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


# ------------------------------------------------- pooling indices (ADR-065)

def _counting_measure(monkeypatch):
    calls = []
    real = validators.measure

    def measure(t):
        calls.append(tuple(t.shape))
        return real(t)

    monkeypatch.setattr(validators, "measure", measure)
    return calls


def _pool_graph(sess, shape, k, s, p, d):
    """max_pool2d_with_indices, then its backward on the forward's indices."""
    out_shape = tuple(torch.ops.aten.max_pool2d_with_indices.default(
        torch.zeros(shape), [k, k], [s, s], [p, p], [d, d], False)[0].shape)
    pool = [ints(k, k), ints(s, s), ints(p, p), ints(d, d), B(False)]
    return graph(sess, [(F32, shape), (F32, out_shape)],
                 [("aten.max_pool2d_with_indices.default", [tin(0)] + pool),
                  ("aten.max_pool2d_with_indices_backward.default", [tin(1), tin(0)] + pool + [tnode(0, 1)])],
                 [(F32, shape)]), out_shape


def test_max_pool_indices_need_no_measurement(sess, monkeypatch):
    # Every window holds an input element, so the indices are positions in
    # the input and the backward's bounds check needs no device sync.
    g, out_shape = _pool_graph(sess, (2, 3, 8, 8), 3, 2, 1, 1)
    calls = _counting_measure(monkeypatch)
    x, go = torch.randn(2, 3, 8, 8), torch.randn(out_shape)
    r = run(sess, g, [x, go])
    assert isinstance(r, cpb.ExecuteResult), r
    assert calls == []
    _, idx = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1], [1, 1], False)
    want = torch.ops.aten.max_pool2d_with_indices_backward.default(go, x, [3, 3], [2, 2], [1, 1], [1, 1], False, idx)
    assert torch.equal(decode(r.returned[0]), want)


def test_max_pool_empty_window_indices_are_measured(sess, monkeypatch):
    # Kernel 2, padding 1, dilation 2 on a 1x1 input: the window holds only
    # padding, and both kernels return index 2, past the input. The rule
    # gives these indices no range; the measurement refuses them.
    x = torch.randn(1, 1, 1, 1)
    _, idx = torch.ops.aten.max_pool2d_with_indices.default(x, [2, 2], [1, 1], [1, 1], [2, 2], False)
    assert int(idx.max()) >= 1  # the kernel's behaviour this guards against
    g, out_shape = _pool_graph(sess, (1, 1, 1, 1), 2, 1, 1, 2)
    calls = _counting_measure(monkeypatch)
    err = run(sess, g, [x, torch.randn(out_shape)])
    expect(err, cpb.ERROR_INDEX_OUT_OF_BOUNDS, cpb.STAGE_E2_RUNTIME)
    assert err.node_index == 1 and calls
    ok_after(sess)


def test_adaptive_max_pool_indices_need_no_measurement(sess, monkeypatch):
    shape = (2, 3, 7, 5)
    g = graph(sess, [(F32, shape), (F32, (2, 3, 3, 4))],
              [("aten.adaptive_max_pool2d.default", [tin(0), ints(3, 4)]),
               ("aten.adaptive_max_pool2d_backward.default", [tin(1), tin(0), tnode(0, 1)])],
              [(F32, shape)])
    calls = _counting_measure(monkeypatch)
    r = run(sess, g, [torch.randn(shape), torch.randn(2, 3, 3, 4)])
    assert isinstance(r, cpb.ExecuteResult), r
    assert calls == []


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_window_rule_matches_the_kernels(dev):
    # Wherever the rule says every window holds an input element, both
    # kernels return indices inside the input (the rule may be stricter).
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import itertools

    from reflop_worker.execute import windows_cover_input

    A = torch.ops.aten
    covered = 0
    for h, k, s, p, d, ceil in itertools.product(range(1, 7), range(1, 4), range(1, 3), range(0, 2), range(1, 4),
                                                 (False, True)):
        try:
            out, idx = A.max_pool2d_with_indices.default(torch.full((1, 1, h, h + 1), float("-inf"), device=dev),
                                                         [k, k], [s, s], [p, p], [d, d], ceil)
        except RuntimeError:
            continue
        rule = (windows_cover_input(h, out.shape[-2], k, s, p, d)
                and windows_cover_input(h + 1, out.shape[-1], k, s, p, d))
        if rule:
            covered += 1
            assert 0 <= int(idx.min()) and int(idx.max()) < h * (h + 1), (h, k, s, p, d, ceil)
    assert covered > 100


def test_a_bound_that_does_not_fit_is_measured(sess):
    # argmax gives its output the bound [0, 15]; the actual index is 3, which
    # fits a 10-row table. The check measures instead of refusing on the bound.
    x = torch.zeros(16)
    x[3] = 1.0
    g = graph(sess, [(F32, (16,)), (F32, (10, 2))],
              [("aten.argmax.default", [tin(0), NONE, B(False)]),
               ("aten.view.default", [tnode(0), ints(1)]),
               ("aten.index_select.default", [tin(1), I(0), tnode(1)])],
              [(F32, (1, 2))])
    table = torch.arange(20.0).reshape(10, 2)
    r = run(sess, g, [x, table])
    assert isinstance(r, cpb.ExecuteResult), r
    assert decode(r.returned[0]).tolist() == [[6.0, 7.0]]
    x[3], x[12] = 0.0, 1.0
    expect(run(sess, g, [x, table]), cpb.ERROR_INDEX_OUT_OF_BOUNDS, cpb.STAGE_E2_RUNTIME)
