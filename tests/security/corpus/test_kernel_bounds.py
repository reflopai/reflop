# SPDX-License-Identifier: Apache-2.0
"""Calls whose PyTorch kernels read or write out of bounds, found by the
meta/kernel parity test (#21, ADR-042). Each one used to pass V4 and crash
the worker (or return memory it does not own); now V3/V4 rejects it, the
worker keeps serving, and the call autograd makes still runs."""

from __future__ import annotations

import pytest
import torch
from helpers import make_session
from parity import Call, check, infer_outputs, worker
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker.graph import Ref
from test_corpus import ok_after

A = torch.ops.aten
L = torch.long


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def sess(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return make_session(request.param)


def r(*shape, dtype=torch.float32):
    return torch.randn(shape).to(dtype)


def _bag(mode):
    w, idx, off = r(6, 4), torch.tensor([1, 2, 4, 5, 4, 3]), torch.tensor([0, 2, 4])
    out, o2b, bag_size, maxi = A._embedding_bag.default(w, idx, off, False, mode)
    return out, idx, off, o2b, bag_size, maxi


def _pool2d_idx():
    out, idx = A.max_pool2d_with_indices.default(r(2, 3, 6, 6), [2, 2], [2, 2], [0, 0], [1, 1], False)
    return out, idx


def _pool3d_idx(x):
    return A.max_pool3d_with_indices.default(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False)


def unsafe():
    """(label, call) pairs PyTorch's CPU kernels do not check."""
    o, idx, off, o2b, bag_size, maxi = _bag(1)
    x3 = r(1, 2, 4, 4, 4)
    p3, i3 = _pool3d_idx(x3)
    _, ai3 = A.adaptive_max_pool3d.default(x3, [2, 2, 2])
    # Ascending values put every window's max at its far corner, so the deeper
    # windows' indices are >= 48, past a 4x4x3 self. Indices from random data
    # all fall inside it about once in 256 runs, and the call then runs safely.
    um3, ui3 = A.adaptive_max_pool3d.default(torch.arange(128.0).reshape(1, 2, 4, 4, 4), [2, 2, 2])
    return [
        ("adaptive avg pool 2d: grad_output has another batch size",
         Call("aten._adaptive_avg_pool2d_backward.default", [r(4, 6, 1, 3), r(3, 6, 1, 6)])),
        ("adaptive avg pool 2d: self has an empty dim",
         Call("aten._adaptive_avg_pool2d_backward.default", [r(2, 3, 2, 2), r(2, 3, 6, 0)])),
        ("adaptive avg pool 3d: self has an empty dim",
         Call("aten._adaptive_avg_pool3d_backward.default", [r(1, 2, 2, 2, 2), r(1, 2, 4, 4, 0)])),
        ("adaptive max pool 2d: self has an empty dim",
         Call("aten.adaptive_max_pool2d_backward.default",
              [r(2, 3, 3, 3), r(2, 3, 6, 0), torch.zeros(2, 3, 3, 3, dtype=L)])),
        ("adaptive max pool 3d: grad_output has another batch size",
         Call("aten.adaptive_max_pool3d_backward.default", [r(2, 2, 2, 2, 2), x3, ai3])),
        ("max pool 3d: grad_output has another batch size",
         Call("aten.max_pool3d_with_indices_backward.default",
              [r(2, 2, 2, 2, 2), x3, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False, i3])),
        ("max pool 2d: indices has another shape",
         Call("aten.max_pool2d_with_indices_backward.default",
              [_pool2d_idx()[0], r(2, 3, 6, 6), [2, 2], [2, 2], [0, 0], [1, 1], False,
               torch.zeros(2, 3, 3, 4, dtype=L)])),
        ("log_softmax backward: output has fewer dims than grad_output",
         Call("aten._log_softmax_backward_data.default", [r(3, 4, 5), r(4, 5), 1, torch.float32])),
        ("weight_norm: g shorter than v.size(dim)",
         Call("aten._weight_norm_interface.default", [r(4, 3), r(3), 0])),
        ("weight_norm: g empty", Call("aten._weight_norm_interface.default", [r(4, 3), r(4, 0), 0])),
        ("weight_norm backward: grad_w empty",
         Call("aten._weight_norm_interface_backward.default", [r(4, 0), r(4, 3), r(4, 1), r(4, 1), 0])),
        ("embedding_bag backward: bag_size empty",
         Call("aten._embedding_bag_backward.default",
              [o, idx, off, o2b, torch.zeros(0, dtype=L), maxi, 6, False, 1, False, None, -1])),
        ("embedding_bag backward: bag_size shorter than the bags",
         Call("aten._embedding_bag_dense_backward.default",
              [o, idx, o2b, bag_size[:2], maxi, 6, False, 1, None, -1])),
        ("embedding_bag: indices but no bags",
         Call("aten._embedding_bag.default", [r(6, 4), idx, torch.zeros(0, dtype=L), False, 2, False, None, False, -1])),
        ("avg pool 3d: self has another batch size",
         Call("aten.avg_pool3d_backward.default",
              [A.avg_pool3d.default(x3, [2, 2, 2]), r(2, 2, 4, 4, 4), [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True,
               None])),
        ("adaptive avg pool 3d: self has another batch size",
         Call("aten._adaptive_avg_pool3d_backward.default", [r(1, 2, 2, 2, 2), r(2, 2, 4, 4, 4)])),
        ("max pool 3d: indices beyond self", Call("aten.max_pool3d_with_indices_backward.default",
                                                  [p3, x3, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False, i3 + 64])),
        ("adaptive max pool 3d: self smaller than the indices", Call("aten.adaptive_max_pool3d_backward.default",
                                                                    [um3, r(1, 2, 4, 4, 3), ui3])),
    ]


UNSAFE = unsafe()


@pytest.mark.parametrize("case", UNSAFE, ids=[c[0] for c in UNSAFE])
def test_rejected_before_any_kernel(sess, case):
    _, call = case
    outs, err = infer_outputs(sess, call)
    if err is None:
        # Index values are data: the runtime bounds check (E2) refuses them
        # before the kernel launches.
        kind, got = worker(sess, call, outs)
        assert kind == "failed" and got.stage == cpb.STAGE_E2_RUNTIME, (kind, got)
    else:
        assert err.stage in (cpb.STAGE_V3_SCHEMA, cpb.STAGE_V4_META), cpb.ValidationStage.Name(err.stage)
    ok_after(sess)


def autograd_calls():
    """What autograd passes the same ops: these must keep running."""
    o, idx, off, o2b, bag_size, maxi = _bag(1)
    so, sidx, soff, so2b, sbag, smaxi = _bag(2)
    x2, x3 = r(2, 3, 6, 6), r(1, 2, 4, 4, 4)
    p2, i2 = _pool2d_idx()
    p3, i3 = _pool3d_idx(x3)
    am2, ai2 = A.adaptive_max_pool2d.default(x2, [3, 3])
    lsm = r(3, 4, 5).log_softmax(1)
    v, g = r(4, 3), r(4, 1)
    _, norms = A._weight_norm_interface.default(v, g, 0)
    vb, gb = r(4, 3, dtype=torch.bfloat16), r(1, 3, dtype=torch.bfloat16)
    _, normsb = A._weight_norm_interface.default(vb, gb, 1)
    return [
        Call("aten._adaptive_avg_pool2d_backward.default", [r(2, 3, 2, 2), x2]),
        Call("aten._adaptive_avg_pool2d_backward.default", [r(3, 2, 2), x2[0]]),
        Call("aten._adaptive_avg_pool3d_backward.default", [r(1, 2, 2, 2, 2), x3]),
        Call("aten.adaptive_max_pool2d_backward.default", [r(2, 3, 3, 3), x2, ai2]),
        Call("aten.max_pool2d_with_indices_backward.default",
             [r(*p2.shape), r(2, 3, 6, 6), [2, 2], [2, 2], [0, 0], [1, 1], False, i2]),
        Call("aten.max_pool3d_with_indices_backward.default",
             [r(*p3.shape), x3, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False, i3]),
        Call("aten.avg_pool2d_backward.default", [r(2, 3, 3, 3), x2, [2, 2], [2, 2], [0, 0], False, True, None]),
        Call("aten._log_softmax_backward_data.default", [r(3, 4, 5), lsm, 1, torch.float32]),
        Call("aten._softmax_backward_data.default", [r(3, 4, 5), lsm.exp(), 2, torch.float32]),
        Call("aten._weight_norm_interface.default", [v, g, 0]),
        Call("aten._weight_norm_interface_backward.default", [r(4, 3), v, g, norms, 0]),
        Call("aten._weight_norm_interface_backward.default", [r(4, 3, dtype=torch.bfloat16), vb, gb, normsb, 1]),
        Call("aten._embedding_bag_backward.default", [o, idx, off, o2b, bag_size, maxi, 6, False, 1, False, None, -1]),
        Call("aten._embedding_bag_dense_backward.default", [so, sidx, so2b, sbag, smaxi, 6, False, 2, None, -1]),
        Call("aten._embedding_bag.default", [r(6, 4), torch.zeros(0, dtype=L), torch.zeros(0, dtype=L), False, 0,
                                             False, None, False, -1]),
    ]


AUTOGRAD = autograd_calls()


@pytest.mark.parametrize("call", AUTOGRAD, ids=[f"{c.op}-{i}" for i, c in enumerate(AUTOGRAD)])
def test_autograd_shapes_still_run(sess, call):
    device = "cuda" if sess.gdevice.kind == gpb.DEVICE_CUDA else "cpu"
    outs, err = infer_outputs(sess, call)
    assert err is None, err
    assert check(sess, call, device) is None


def test_factory_without_device_uses_the_session_device(sess):
    # A None device used to leave the factory on the CPU, in meta inference
    # too: aten.full([2**31, 3]) allocated 25 GB while being validated.
    call = Call("aten.full.default", [[2**31, 3], 1.5, None, None, None, None])
    from parity import encode, graph_of
    from reflop_worker import meta

    inputs, nodes = encode(sess, call)
    prog = sess.builder.build(graph_of(sess, inputs, nodes, [(torch.float32, (2**31, 3))]))
    assert prog.nodes[0].args[4] == "session-device"
    meta.infer(prog, sess.limits)
    assert prog.metas[Ref(0, 0)].device.type == "meta"
    small = Call("aten.full.default", [[2, 3], 1.5, None, None, None, None])
    kind, got = worker(sess, small, [(torch.float32, (2, 3))])
    assert kind == "ran" and torch.equal(got[0], torch.full((2, 3), 1.5))


# ADR-043: found once the parity test covered losses, norms, distances, FFT,
# grid sampling and the fused optimizers.


def unsafe_043():
    a, b = r(2, 4, 3), r(2, 5, 3)
    cd = A._cdist_forward.default(a, b, 2.0, None)
    x2 = r(5, 3)
    pd = A._pdist_forward.default(x2, 2.0)
    x, w = r(4, 3, 5), r(3)
    g3, x3 = r(1, 2, 3, 3, 3), r(1, 2, 3, 4, 5)
    grid3 = r(1, 3, 3, 3, 3).tanh()
    p, gr = [r(3, 4), r(5)], [r(3, 4), r(5)]
    z = [torch.zeros(3, 4), torch.zeros(5)]
    st = [torch.ones(()), torch.ones(())]
    adam = [p, gr, z, z, [], st, 1e-3, 0.9, 0.999, 0.0, 1e-8, False, False, None, None]
    return [
        ("cdist backward: grad_output with an empty dim",
         Call("aten._cdist_backward.default", [r(2, 4, 0), a, b, 2.0, cd])),
        ("cdist backward: x1 with another feature size",
         Call("aten._cdist_backward.default", [r(*cd.shape), r(2, 4, 4), b, 2.0, cd])),
        ("pdist backward: empty grad_output", Call("aten._pdist_backward.default", [r(0), x2, 2.0, pd])),
        ("pdist backward: pdist shorter", Call("aten._pdist_backward.default", [r(10), x2, 2.0, pd[:9]])),
        ("rfft: a negative dim", Call("aten._fft_r2c.default", [r(3, 6), [-2, 1], 0, True])),
        ("irfft: a negative dim",
         Call("aten._fft_c2r.default", [torch.randn(3, 4, dtype=torch.complex64), [-2, 1], 2, 6])),
        ("batch_norm: a 0-d weight", Call("aten.native_batch_norm.default",
                                          [x, r(), None, torch.zeros(3), torch.ones(3), True, 0.1, 1e-5])),
        ("batch_norm: float64 running_mean beside float32 input",
         Call("aten.native_batch_norm.default", [x, w, None, torch.zeros(3, dtype=torch.float64), torch.ones(3),
                                                 True, 0.1, 1e-5])),
        ("grid_sampler_3d backward: an unknown interpolation mode",
         Call("aten.grid_sampler_3d_backward.default", [g3, x3, grid3, -1, 0, False, [True, True]])),
        ("layer_norm backward: grad_out with an empty dim",
         Call("aten.native_layer_norm_backward.default",
              [r(4, 3, 0), r(4, 3, 5), [5], r(4, 3, 1), r(4, 3, 1), r(5), r(5), [True, True, True]])),
        ("fused adam: exp_avgs shorter than the param",
         Call("reflop.fused_adam.default", [p, gr, [torch.zeros(3, 3), z[1]], z, [], st, *adam[6:]])),
        ("fused adam: a state step of two elements",
         Call("reflop.fused_adam.default", [p, gr, z, z, [], [torch.ones(2), st[1]], *adam[6:]])),
        ("fused adamw: max_exp_avg_sqs shorter, amsgrad",
         Call("reflop.fused_adamw.default", [p, gr, z, z, [torch.zeros(2), z[1]], st, *adam[6:11], True, False,
                                             None, None])),
        ("fused sgd: momentum buffer shorter",
         Call("reflop.fused_sgd.default", [p, gr, [torch.zeros(3, 3), z[1]], 0.0, 0.9, 0.1, 0.0, False, False,
                                           False, None, None])),
        ("fused adagrad: state_sums of another dtype",
         Call("reflop.fused_adagrad.default", [p, gr, [z[0].double(), z[1]], st, 1e-2, 0.0, 0.0, 1e-10, False,
                                               None, None])),
        ("amp unscale: an empty found_inf",
         Call("reflop.amp_unscale_and_check.default", [gr, torch.zeros(0), torch.tensor([0.5])])),
        ("binary_cross_entropy: input outside [0, 1]",
         Call("aten.binary_cross_entropy.default", [r(4, 5), torch.rand(4, 5), None, 1])),
        ("binary_cross_entropy: target outside [0, 1]",
         Call("aten.binary_cross_entropy.default", [torch.rand(4, 5), r(4, 5), None, 1])),
    ]


UNSAFE_043 = unsafe_043()


@pytest.mark.parametrize("case", UNSAFE_043, ids=[c[0] for c in UNSAFE_043])
def test_rejected_before_any_kernel_043(sess, case):
    test_rejected_before_any_kernel(sess, case)


def test_strict_ops_accept_what_autograd_passes(sess):
    # STRICT ops may refuse more than PyTorch, so a false rejection there is
    # not reported by the parity test; their seeds (calls PyTorch itself
    # makes) must all pass.
    from parity import STRICT, seeds

    deliberate = {"aten._ctc_loss_backward.Tensor"}  # lengths go as int lists (ADR-032)
    refused = []
    for c in seeds(refused=False):
        if c.op in STRICT and c.op not in deliberate:
            _, err = infer_outputs(sess, c)
            if err is not None:
                refused.append(f"{c.op}: {err.message[:200]}")
    assert not refused, "\n".join(refused)


def test_one_element_strided_views_read_back(sess):
    # reshape(-1) keeps the stride of a size-1 dim; reading x[::12] failed
    # with "self.stride(-1) must be 1 to view Float as Byte".
    for size, step in ((10, 12), (5, 7), (1, 3)):
        call = Call("aten.slice.Tensor", [torch.arange(float(size)), 0, None, None, step])
        kind, got = worker(sess, call, [(torch.float32, (1,))])
        assert kind == "ran" and got[0].tolist() == [0.0], (kind, got)
