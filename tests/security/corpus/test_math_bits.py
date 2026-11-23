# SPDX-License-Identifier: Apache-2.0
"""Worker outputs with PyTorch's conjugate or negative bit set.

aten._conj is in the opset, so any client can send it. Its output is a view
whose kernels apply the conjugation lazily; a raw byte copy drops it. Returned
values, stored handles and fetches must all carry the conjugated values."""

from __future__ import annotations

import os

import pytest
import torch
from helpers import call, graph, make_session, run, tin, tnode
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker.core import _host_copy
from reflop_worker.safetensors import tensor_bytes

Z = torch.tensor([[1 + 2j, 3 - 4j], [-5 + 6j, 7 - 8j]], dtype=torch.complex64)


@pytest.fixture
def sess():
    return make_session("cpu")


def _decode(td):
    from helpers import decode

    return decode(td)


def test_conj_return(sess):
    g = graph(sess, [(Z.dtype, Z.shape)], [("aten._conj.default", [tin(0)])], [(Z.dtype, Z.shape)])
    r = run(sess, g, [Z])
    assert isinstance(r, cpb.ExecuteResult), r
    assert torch.equal(_decode(r.returned[0]), Z.conj().resolve_conj())


def test_conj_then_transpose_return(sess):
    g = graph(sess, [(Z.dtype, Z.shape)],
              [("aten._conj.default", [tin(0)]), ("aten.t.default", [tnode(0)])], [(Z.dtype, Z.shape)])
    r = run(sess, g, [Z])
    assert torch.equal(_decode(r.returned[0]), Z.conj().t().resolve_conj())


def _store(sess, g, inputs, handle):
    from helpers import register, tensor_binding

    reg = register(sess, g)
    ex = cpb.Execute(graph_hash=reg.graph_registered.graph_hash)
    for t in inputs:
        ex.bindings.append(tensor_binding(t))
    ex.outputs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_STORE, handle=handle))
    e = call(sess, cpb.Envelope(request_id=3, execute=ex))[0]
    assert e.WhichOneof("body") == "execute_result", e


def test_stored_conj_holds_plain_values(sess):
    h = os.urandom(16)
    g = graph(sess, [(Z.dtype, Z.shape)], [("aten._conj.default", [tin(0)])], [(Z.dtype, Z.shape)])
    _store(sess, g, [Z], h)
    t = sess.handles.get(h)
    assert not t.is_conj() and torch.equal(t, Z.conj().resolve_conj())

    # A fetch copies the handle's raw bytes.
    frames = call(sess, cpb.Envelope(request_id=4, fetch=cpb.Fetch(handles=[h])))
    data = b"".join(f.fetch_chunk.data for f in frames)
    got = torch.frombuffer(bytearray(data), dtype=Z.dtype).reshape(Z.shape)
    assert torch.equal(got, Z.conj().resolve_conj())

    # Later graphs see a plain tensor: view_as_real rejects conj views.
    g2 = graph(sess, [(Z.dtype, Z.shape)], [("aten.view_as_real.default", [tin(0)])],
               [(torch.float32, (*Z.shape, 2))])
    reg = call(sess, cpb.Envelope(request_id=5, register_graph=cpb.RegisterGraph(graph=g2)))[0]
    ex = cpb.Execute(graph_hash=reg.graph_registered.graph_hash, bindings=[cpb.Binding(handle=h)],
                     outputs=[cpb.OutputSpec(kind=cpb.OutputSpec.KIND_RETURN)])
    e = call(sess, cpb.Envelope(request_id=6, execute=ex))[0]
    assert e.WhichOneof("body") == "execute_result", e
    assert torch.equal(_decode(e.execute_result.returned[0]), torch.view_as_real(Z.conj().resolve_conj()))


@pytest.mark.parametrize("make", [lambda: Z.conj(), lambda: torch._neg_view(Z), lambda: torch._neg_view(Z.conj()),
                                  lambda: Z.conj().t()])
def test_byte_copies_apply_math_bits(make):
    t = make()
    want = t.resolve_conj().resolve_neg().contiguous()
    got = torch.frombuffer(bytearray(tensor_bytes(t)), dtype=t.dtype).reshape(t.shape)
    assert torch.equal(got, want)
    assert torch.equal(_host_copy(t).view(t.dtype).reshape(t.shape), want)


def test_plain_outputs_are_stored_as_is():
    from reflop_worker.safetensors import resolved

    t = torch.arange(4)
    assert resolved(t) is t
