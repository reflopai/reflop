# SPDX-License-Identifier: Apache-2.0
"""What the graph fuzzer found (GH#4, ADR-044), as plain regression cases.
Replays of the original fuzz cases are in tests/security/fuzz/corpus."""

from __future__ import annotations

import pytest
import torch
from helpers import NAME, B, I, graph, make_session, register, run, tin, tnode
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from test_corpus import ok_after

GiB = 1 << 30


@pytest.fixture(scope="module")
def sess():
    return make_session("cpu")


def _call(sess, env):
    out = []
    sess.handle(env, out.append)  # nothing may escape: the test fails on any exception
    return out[-1]


def test_a_number_where_a_validator_reads_a_tensor(sess):
    # V3 allows a wrapped Python number for a Tensor argument; scatter's
    # validator called .dim() on a bool and an AttributeError escaped.
    g = graph(sess, [(torch.int64, (2,)), (torch.float32, (2,))],
              [("aten.scatter.src", [B(True), I(0), tin(0), tin(1)])], [(torch.float32, (2,))])
    r = register(sess, g)
    assert r.WhichOneof("body") == "error" and r.error.code != cpb.ERROR_INTERNAL, r
    ok_after(sess)


@pytest.mark.parametrize("shape", [[0, 2**62, 2**62], [4, 3, 0, 2**63 - 1]])
def test_an_empty_shape_whose_strides_overflow(sess, shape):
    # numel is 0, so the element limit passed; building the meta input then
    # raised "Stride calculation overflowed" out of Session.handle.
    g = graph(sess, [(torch.float32, shape)], [("aten.neg.default", [tin(0)])], [(torch.float32, shape)])
    r = register(sess, g)
    assert r.WhichOneof("body") == "error" and r.error.code == cpb.ERROR_RESOURCE_LIMIT, r
    assert r.error.stage == cpb.STAGE_V2_STRUCTURE
    ok_after(sess)


def test_new_empty_strided_with_a_sparse_stride(sess):
    # 18 elements reaching 2**31 * 2 into their storage: about 17 GB of
    # address space for a tensor V5 counted as 72 bytes.
    dev = gpb.Arg(device=sess.gdevice)
    g = graph(sess, [(torch.float32, (2, 3, 3))],
              [("aten.new_empty_strided.default", [tin(0), gpb.Arg(ints=gpb.Int64List(values=[2, 3, 3])),
                                                   gpb.Arg(ints=gpb.Int64List(values=[9, 3, 2**31])),
                                                   gpb.Arg(dtype=gpb.DTYPE_FLOAT32), gpb.Arg(layout=gpb.LAYOUT_STRIDED),
                                                   dev, gpb.Arg(none=True)])],
              [(torch.float32, (2, 3, 3))])
    r = register(sess, g)
    assert r.WhichOneof("body") == "error" and r.error.stage == cpb.STAGE_V4_META, r
    ok_after(sess)


def _ones(sess, shape):
    return ("aten.ones.default", [gpb.Arg(ints=gpb.Int64List(values=list(shape))), gpb.Arg(dtype=gpb.DTYPE_FLOAT32),
                                  gpb.Arg(none=True), gpb.Arg(device=sess.gdevice), gpb.Arg(none=True)])


def _peak(sess, g):
    r = register(sess, g)
    assert r.WhichOneof("body") == "graph_registered", r
    return r.graph_registered.peak_memory_bytes


def test_a_non_contiguous_view_output_counts_its_copy(sess):
    # Returning or fetching x.t() copies it whole (.contiguous()); V5 now
    # counts that copy, and x stays alive under its view.
    S = 1024 * 256 * 4
    t = graph(sess, [], [_ones(sess, (1024, 256)), ("aten.transpose.int", [tnode(0), I(0), I(1)])],
              [(torch.float32, (256, 1024))])
    assert _peak(sess, t) == 2 * S
    # A contiguous view (a slice along dim 0) is returned without a copy.
    s = graph(sess, [], [_ones(sess, (1024, 256)), ("aten.slice.Tensor", [tnode(0), I(0), I(0), I(512), I(1)])],
              [(torch.float32, (512, 256))])
    assert _peak(sess, s) == S
    ok_after(sess)


def test_a_view_keeps_its_base_alive(sess):
    # a -> a.view(...) -> v * 2: a's memory is in use until the mul has run.
    # V5 freed a at the view and counted 1x; the graph holds 2x.
    S = 1024 * 256 * 4
    g = graph(sess, [], [_ones(sess, (1024, 256)), ("aten.view.default", [tnode(0), gpb.Arg(ints=gpb.Int64List(
        values=[256, 1024]))]), ("aten.mul.Tensor", [tnode(1), gpb.Arg(f=2.0)])], [(torch.float32, (256, 1024))])
    assert _peak(sess, g) == 2 * S
    ok_after(sess)


def test_an_expanded_scalar_cannot_pass_the_rails():
    # The original finding: a 0-d input expanded to [2**31, 2] and returned
    # made the worker copy 16 GiB on the host past every rail. V5 now counts
    # the copy, so a device quota refuses it.
    from reflop.connection import default_limits
    from reflop_worker._pb import capabilities_pb2 as kpb
    from reflop_worker._pb import worker_pb2 as wpb
    from reflop_worker.core import Session

    init = wpb.WorkerInit(seed=0, device_memory_quota_bytes=1 * GiB)
    init.device.kind = gpb.DEVICE_CPU
    init.limits.CopyFrom(default_limits(kpb))
    s = Session(init)
    g = graph(s, [(torch.float32, ())],
              [("aten.expand.default", [tin(0), gpb.Arg(ints=gpb.Int64List(values=[2**31, 2])), B(False)])],
              [(torch.float32, (2**31, 2))])
    r = register(s, g)
    assert r.WhichOneof("body") == "error" and r.error.code == cpb.ERROR_QUOTA, r
    assert r.error.stage == cpb.STAGE_V5_SAFETY
    small = graph(s, [(torch.float32, ())],
                  [("aten.expand.default", [tin(0), gpb.Arg(ints=gpb.Int64List(values=[4, 2])), B(False)])],
                  [(torch.float32, (4, 2))])
    res = run(s, small, [torch.tensor(1.5)])
    assert isinstance(res, cpb.ExecuteResult), res
    _ = NAME
