# SPDX-License-Identifier: Apache-2.0
"""Kernels that write to their arguments never change server tensors in
place (GH#49, ADR-058).

Regression: native_batch_norm in training mode overwrote the bound
running_mean and running_var, which clients up to 0.2.3 send for every
BatchNorm forward. _native_batch_norm_legit (training) did the same, and
atan2.out wrote into, and resized, the tensor bound to `out`. Checkpoint
cuts hold references to the current tensors as a consistent view, so a
later execute could change a checkpoint while it was written. Now the
kernel writes copies, the handles point at the copies when the execute
commits, and the original tensors stay as they were.
"""

from __future__ import annotations

import pytest
import torch
from helpers import NONE, B, F, I, graph, make_session, register, tin, tnode
from reflop_worker import writes
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker.errors import ReflopError
from reflop_worker.ops_generated import OPS
from test_corpus import expect, ok_after

A = torch.ops.aten
F32 = torch.float32
BN = "aten.native_batch_norm.default"
BN_LEGIT = "aten._native_batch_norm_legit.default"
HM, HV, HO = b"\x01" * 16, b"\x02" * 16, b"\x03" * 16


@pytest.fixture(params=["cpu", "cuda"])
def sess(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return make_session(request.param)


def _x():
    return torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(0))


def _stats():
    return torch.randn(3, generator=torch.Generator().manual_seed(1)), torch.rand(3) + 0.5


def _store(sess, h, t):
    sess.handles.put_new(h, t.to(sess.device).clone())
    return sess.handles.get(h)


def _execute(sess, g, bindings, outputs):
    r = register(sess, g)
    assert r.WhichOneof("body") == "graph_registered", r
    ex = cpb.Execute(graph_hash=r.graph_registered.graph_hash)
    for b in bindings:
        ex.bindings.append(cpb.Binding(handle=b) if isinstance(b, bytes) else _inline(b))
    for kind, h in outputs:
        ex.outputs.append(cpb.OutputSpec(kind=kind, handle=h or b""))
    e = []
    sess.handle(cpb.Envelope(request_id=2, execute=ex), e.append)
    return e[0]


def _inline(t):
    from helpers import tensor_binding

    return tensor_binding(t)


def _bn_graph(sess, op, training=True, then=()):
    x = (F32, (4, 3, 5))
    nodes = [(op, [tin(0), NONE, NONE, tin(1), tin(2), B(training), F(0.1), F(1e-5)])]
    nodes += list(then)
    return graph(sess, [x, (F32, (3,)), (F32, (3,))], nodes, [(F32, (4, 3, 5))], outputs=[(0, 0)])


DISCARD = (cpb.OutputSpec.KIND_DISCARD, None)


@pytest.mark.parametrize("op", [BN, BN_LEGIT])
def test_training_batch_norm_updates_the_handles_without_writing_in_place(sess, op):
    x, (rm, rv) = _x(), _stats()
    want_m, want_v = rm.clone(), rv.clone()
    want_out = getattr(A, op.split(".")[1]).default(x, None, None, want_m, want_v, True, 0.1, 1e-5)[0]
    old_m, old_v = _store(sess, HM, rm), _store(sess, HV, rv)
    e = _execute(sess, _bn_graph(sess, op), [x, HM, HV], [(cpb.OutputSpec.KIND_RETURN, None)])
    assert e.WhichOneof("body") == "execute_result", e
    # What a checkpoint cut holds is unchanged.
    assert torch.equal(old_m.cpu(), rm) and torch.equal(old_v.cpu(), rv)
    # The handles hold the statistics the in-place kernel would have left.
    new_m, new_v = sess.handles.get(HM), sess.handles.get(HV)
    assert new_m is not old_m and new_v is not old_v
    torch.testing.assert_close(new_m.cpu(), want_m)
    torch.testing.assert_close(new_v.cpu(), want_v)
    from helpers import decode

    torch.testing.assert_close(decode(e.execute_result.returned[0]), want_out, rtol=1e-4, atol=1e-5)


def test_eval_batch_norm_keeps_the_same_tensors(sess):
    x, (rm, rv) = _x(), _stats()
    old_m, old_v = _store(sess, HM, rm), _store(sess, HV, rv)
    e = _execute(sess, _bn_graph(sess, BN, training=False), [x, HM, HV], [DISCARD])
    assert e.WhichOneof("body") == "execute_result", e
    assert sess.handles.get(HM) is old_m and sess.handles.get(HV) is old_v
    assert torch.equal(old_m.cpu(), rm)


def test_a_later_node_reads_the_updated_statistics(sess):
    # In-place, a node after the kernel saw the new values. It still does.
    x, (rm, rv) = _x(), _stats()
    want_m = rm.clone()
    A.native_batch_norm.default(x, None, None, want_m, rv.clone(), True, 0.1, 1e-5)
    _store(sess, HM, rm), _store(sess, HV, rv)
    g = _bn_graph(sess, BN, then=[("aten.clone.default", [tin(1), NONE])])
    del g.outputs[:], g.output_meta[:]
    from helpers import meta

    g.outputs.append(tnode(1).tensor)
    g.output_meta.append(meta(F32, (3,)))
    e = _execute(sess, g, [x, HM, HV], [(cpb.OutputSpec.KIND_RETURN, None)])
    assert e.WhichOneof("body") == "execute_result", e
    from helpers import decode

    torch.testing.assert_close(decode(e.execute_result.returned[0]), want_m)


def test_two_batch_norms_on_the_same_statistics(sess):
    x, (rm, rv) = _x(), _stats()
    want_m, want_v = rm.clone(), rv.clone()
    for _ in range(2):
        A.native_batch_norm.default(x, None, None, want_m, want_v, True, 0.1, 1e-5)
    old_m = _store(sess, HM, rm)
    _store(sess, HV, rv)
    second = (BN, [tin(0), NONE, NONE, tin(1), tin(2), B(True), F(0.1), F(1e-5)])
    e = _execute(sess, _bn_graph(sess, BN, then=[second]), [x, HM, HV], [DISCARD])
    assert e.WhichOneof("body") == "execute_result", e
    assert torch.equal(old_m.cpu(), rm)
    torch.testing.assert_close(sess.handles.get(HM).cpu(), want_m)
    torch.testing.assert_close(sess.handles.get(HV).cpu(), want_v)


def test_one_handle_bound_as_both_statistics(sess):
    # The kernel writes the same memory twice; a single copy keeps that.
    # Which write lands last is the device kernel's: compute the expected
    # value on the session's device (the CPU and CUDA kernels differ).
    x, (rm, _) = _x(), _stats()
    want = rm.clone().to(sess.device)
    A.native_batch_norm.default(x.to(sess.device), None, None, want, want, True, 0.1, 1e-5)
    want = want.cpu()
    old = _store(sess, HM, rm)
    e = _execute(sess, _bn_graph(sess, BN), [x, HM, HM], [DISCARD])
    assert e.WhichOneof("body") == "execute_result", e
    assert torch.equal(old.cpu(), rm)
    torch.testing.assert_close(sess.handles.get(HM).cpu(), want)


def test_a_failed_execute_changes_no_handle(sess):
    # The batch norm runs, then a later node fails: nothing is committed.
    # (In place, the statistics had already changed.)
    x, (rm, rv) = _x(), _stats()
    old_m, old_v = _store(sess, HM, rm), _store(sess, HV, rv)
    fails = ("aten.index_select.default", [tin(1), I(0), tin(3)])
    g = _bn_graph(sess, BN, then=[fails])
    from helpers import meta
    from reflop_worker._pb import graph_pb2 as gpb

    g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_TENSOR, meta=meta(torch.int64, (1,))))
    e = _execute(sess, g, [x, HM, HV, torch.tensor([7])], [DISCARD])
    expect(e.error, cpb.ERROR_INDEX_OUT_OF_BOUNDS)
    assert sess.handles.get(HM) is old_m and sess.handles.get(HV) is old_v
    assert torch.equal(old_m.cpu(), rm) and torch.equal(old_v.cpu(), rv)
    ok_after(sess)


def test_an_explicit_rebind_of_the_same_handle_wins(sess):
    x, (rm, rv) = _x(), _stats()
    _store(sess, HM, rm), _store(sess, HV, rv)
    g = _bn_graph(sess, BN, then=[("aten.zeros_like.default", [tin(1), NONE, NONE, NONE, NONE, NONE])])
    del g.outputs[:], g.output_meta[:]
    from helpers import meta

    g.outputs.append(tnode(1).tensor)
    g.output_meta.append(meta(F32, (3,)))
    e = _execute(sess, g, [x, HM, HV], [(cpb.OutputSpec.KIND_REBIND, HM)])
    assert e.WhichOneof("body") == "execute_result", e
    assert torch.equal(sess.handles.get(HM).cpu(), torch.zeros(3))


def test_atan2_out_writes_a_copy(sess):
    old = _store(sess, HO, torch.zeros(4))
    g = graph(sess, [(F32, (4,)), (F32, (4,)), (F32, (4,))], [("aten.atan2.out", [tin(0), tin(1), tin(2)])],
              [(F32, (4,))])
    e = _execute(sess, g, [torch.ones(4), torch.ones(4), HO], [DISCARD])
    assert e.WhichOneof("body") == "execute_result", e
    assert torch.equal(old.cpu(), torch.zeros(4))
    torch.testing.assert_close(sess.handles.get(HO).cpu(), torch.full((4,), torch.pi / 4))


@pytest.mark.parametrize("n", [0, 2, 1000])
def test_atan2_out_may_not_resize_its_out(sess, n):
    # Before: the kernel resized the stored tensor to [4], growing or
    # shrinking its storage. The copy it now writes must keep the shape, as
    # V4's meta of that input (which the meta kernel resizes) must.
    old = _store(sess, HO, torch.zeros(n))
    g = graph(sess, [(F32, (4,)), (F32, (4,)), (F32, (n,))], [("aten.atan2.out", [tin(0), tin(1), tin(2)])],
              [(F32, (4,))])
    expect(register(sess, g).error, cpb.ERROR_SHAPE_MISMATCH, cpb.STAGE_V4_META)
    assert sess.handles.get(HO) is old and tuple(old.shape) == (n,) and not old.any()
    ok_after(sess)


def test_the_executor_refuses_a_resized_copy_too():
    # The V4 check above is the one Guests meet; this is the executor's own.
    class Node:
        index, op = 0, OPS[51]

    orig = torch.zeros(2)
    env = {"r": orig}
    with pytest.raises(ReflopError, match="resize"):
        writes.adopt(Node, [(orig, torch.zeros(4))], env, {}, cpb.STAGE_E3_EXECUTE)
    assert env["r"] is orig


def test_the_copies_count_toward_peak_memory(sess, monkeypatch):
    # [4096] statistics: their two copies are 2 x 16 KiB.
    from reflop_worker import cost

    def peak(eps):
        g = graph(sess, [(F32, (2, 4096, 1)), (F32, (4096,)), (F32, (4096,))],
                  [(BN, [tin(0), NONE, NONE, tin(1), tin(2), B(True), F(0.1), F(eps)])],
                  [(F32, (2, 4096, 1))])
        r = register(sess, g)
        assert r.WhichOneof("body") == "graph_registered", r
        return r.graph_registered.peak_memory_bytes

    counted = peak(1e-5)
    monkeypatch.setattr(cost, "written_bytes", lambda node, metas: 0)
    assert counted - peak(1e-3) == 2 * 4096 * 4  # another eps: not the cached graph


def test_every_op_that_writes_an_argument_is_known():
    # A new opset entry whose kernel writes an argument needs a look: add it
    # here once its copies are right (and its hidden writes to HIDDEN).
    found = {o.name for i, o in OPS.items() if o.fn is not None and writes.written(i) is not None
             and not o.flags & 512}  # BANNED ops never run
    assert found == {BN, BN_LEGIT, "aten.atan2.out"}
