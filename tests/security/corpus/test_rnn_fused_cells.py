# SPDX-License-Identifier: Apache-2.0
"""The `rnn_fused` cells (`_thnn_fused_lstm_cell`, `_thnn_fused_gru_cell` and
their backwards, ADR-067). test_meta_parity mutates their shapes, ranks and
dtypes; this covers what it does not: Guest-chosen values in a backward's
workspace, and the fused kernels' numbers against the decomposition the
client records when the Owner turns the capability off."""

from __future__ import annotations

import pytest
import torch
from helpers import make_session
from parity import Call, infer_outputs, worker
from reflop import decomp
from test_corpus import ok_after

A = torch.ops.aten
B, H = 5, 7


@pytest.fixture(scope="module")
def sess():
    if not torch.cuda.is_available():
        pytest.skip("the fused cells are CUDA kernels")
    return make_session("cuda")


CELLS = ("aten._thnn_fused_lstm_cell.default", "aten._thnn_fused_lstm_cell_backward_impl.default",
         "aten._thnn_fused_gru_cell.default", "aten._thnn_fused_gru_cell_backward.default")


def test_a_cpu_session_offers_neither_the_cells_nor_the_capability():
    """The cells have CUDA kernels only; `reflop status` must not list
    rnn_fused for a server CPU device."""
    from helpers import NAME

    cpu = make_session("cpu")
    assert "rnn_fused" not in cpu.caps
    assert not {NAME[n] for n in CELLS} & set(cpu.enabled)


def run(sess, op, args):
    call = Call(op, args)
    outs, err = infer_outputs(sess, call)
    assert err is None, err
    kind, got = worker(sess, call, outs)
    assert kind == "ran", (kind, got)
    return got


def r(*shape, dtype=torch.float32, g=None):
    return torch.randn(shape, generator=g).to(dtype)


# A backward's workspace is a graph value: a Guest can bind any tensor of the
# shape the meta kernel checks. The kernels only do arithmetic on it, so
# values the forward never produces give wrong gradients, not unsafe memory.
WILD = torch.tensor([float("inf"), float("-inf"), float("nan"), 3e38, -3e38, 0.0, -0.0, 1e-45])


def wild(*shape):
    n = torch.Size(shape).numel()
    return WILD.repeat(n // len(WILD) + 1)[:n].reshape(shape)


def test_lstm_backward_takes_any_workspace_values(sess):
    got = run(sess, "aten._thnn_fused_lstm_cell_backward_impl.default",
              [r(B, H), r(B, H), r(B, H), r(B, H), wild(B, 4 * H), True])
    assert [tuple(t.shape) for t in got] == [(B, 4 * H), (B, H), (4 * H,)]
    ok_after(sess)


def test_gru_backward_takes_any_workspace_values(sess):
    got = run(sess, "aten._thnn_fused_gru_cell_backward.default", [r(B, H), wild(B, 5 * H), True])
    assert [tuple(t.shape) for t in got] == [(B, 3 * H), (B, 3 * H), (B, H), (3 * H,), (3 * H,)]
    ok_after(sess)


LABELS = ["lstm", "lstm bias", "gru", "gru bias", "lstm backward", "lstm backward, no grad_cy", "gru backward"]


def case(label, dtype):
    """(op, args) for one labelled call. Built inside the test, after the
    CUDA fixture skipped where there is no GPU: the backward cases take their
    workspace from the CUDA forward kernels."""
    g = torch.Generator().manual_seed(0)
    lstm = [r(B, 4 * H, dtype=dtype, g=g), r(B, 4 * H, dtype=dtype, g=g), r(B, H, dtype=dtype, g=g)]
    lstm_bias = lstm + [r(4 * H, dtype=dtype, g=g), r(4 * H, dtype=dtype, g=g)]
    gru = [r(B, 3 * H, dtype=dtype, g=g), r(B, 3 * H, dtype=dtype, g=g), r(B, H, dtype=dtype, g=g)]
    gru_bias = gru + [r(3 * H, dtype=dtype, g=g), r(3 * H, dtype=dtype, g=g)]
    grads = [r(B, H, dtype=dtype, g=g) for _ in range(4)]
    if label == "lstm":
        return "aten._thnn_fused_lstm_cell.default", lstm + [None, None]
    if label == "lstm bias":
        return "aten._thnn_fused_lstm_cell.default", lstm_bias
    if label == "gru":
        return "aten._thnn_fused_gru_cell.default", gru + [None, None]
    if label == "gru bias":
        return "aten._thnn_fused_gru_cell.default", gru_bias
    if label.startswith("lstm backward"):
        _, cy, ws = A._thnn_fused_lstm_cell.default(*(t.cuda() for t in lstm))
        grad_cy = None if label.endswith("no grad_cy") else grads[1]
        return "aten._thnn_fused_lstm_cell_backward_impl.default", [grads[0], grad_cy, lstm[2], cy.cpu(), ws.cpu(), True]
    assert label == "gru backward", label
    _, gws = A._thnn_fused_gru_cell.default(*(t.cuda() for t in gru))
    return "aten._thnn_fused_gru_cell_backward.default", [grads[2], gws.cpu(), True]


def decomposed(op, args, dtype):
    fn = decomp.TABLE[getattr(A, op.split(".")[1]).default]
    out = fn(*(a.cuda().to(dtype) if isinstance(a, torch.Tensor) else a for a in args))
    return [t.cpu() for t in out if t is not None]


@pytest.mark.parametrize("label", LABELS)
def test_fused_cells_match_the_decomposition_fp32(sess, label):
    op, args = case(label, torch.float32)
    for fused, dec in zip(run(sess, op, args), decomposed(op, args, torch.float32), strict=True):
        torch.testing.assert_close(fused, dec, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("label", LABELS)
def test_fused_cells_in_bf16_stay_within_rounding(sess, label):
    """Under autocast the cells run in bf16. Fused and decomposed results
    differ there by rounding, in either direction; both stay within a few
    bf16 ulps (2**-7) of the float32 result, relative to its largest value."""
    op, args = case(label, torch.bfloat16)
    ref = decomposed(op, [a.float() if isinstance(a, torch.Tensor) else a for a in args], torch.float32)
    for fused, want in zip(run(sess, op, args), ref, strict=True):
        assert fused.dtype == torch.bfloat16
        err = (fused.float() - want).abs().max().item() / want.abs().max().item()
        assert err <= 3e-2, (label, err)
