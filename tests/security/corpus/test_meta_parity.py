# SPDX-License-Identifier: Apache-2.0
"""Every op call the worker accepts, PyTorch's kernel accepts too, with the
same values and output shapes; every call PyTorch rejects, the worker
rejects before a kernel runs (#21, #25 E, ADR-042, ADR-049).

The cases are seeds recorded from small PyTorch programs and their one-
argument mutations (parity.py), about 23,000 per device. The only
disagreements allowed are the worker's deliberate refusals of calls PyTorch
runs, listed in DELIBERATE by op and message. Anything else fails, and so
does a DELIBERATE entry that no longer matches a refusal on the CPU, so the
list only shrinks. A kernel that crashes or returns other values is never
allowed: it read or wrote out of bounds."""

from __future__ import annotations

import collections

import pytest
import torch
from helpers import make_session
from parity import FLOAT8, SPEC, STRICT, all_cases, check, kind_of
from reflop_worker import prechecks
from reflop_worker.ops_generated import FLAG_FP8
from test_corpus import ok_after

# Closed enums: PyTorch reads any other reduction as "mean" or "sum".
_REDUCTION = ("reduction must be 0, 1 or 2",)
_NLL_REDUCTION = ("reduction must be 0 (none), 1 (mean) or 2 (sum)",)
# A class weight is 1-D (PyTorch takes any tensor with C elements).
_NLL_WEIGHT = (" entries",)
# Kernel parameters the CPU kernels read as int64 and the CUDA (and the CPU
# transposed) kernels truncate to int32.
_INT32 = ("does not fit in int32",)

# The meta kernels refuse these dtypes at any size; the CPU kernels return
# an empty output first (parity_grid.py).
_EMPTY = ("expected a floating-point or complex dtype",)
# random_'s float bounds are from + 1 and to - 1 rounded to the dtype, cast
# to int64 in C++: undefined outside int64, so the worker refuses them.
_INT64 = ("outside int64",)

# op -> substrings of the worker's message for calls it refuses and PyTorch
# runs (false rejections).
DELIBERATE = {
    "aten.exponential.default": ("dtype must be a floating point",),
    # The CPU kernel also runs complex input; the meta kernel refuses it.
    "aten.native_dropout.default": ("can't be cast to the desired output type",),
    "aten.normal.Tensor_float": _EMPTY,
    "aten.normal.float_float": _EMPTY,
    "aten.randint.default": _INT64,
    "aten.randint.low": _INT64,
    "aten.randint_like.default": _INT64,
    "aten.randint_like.low_dtype": _INT64,
    "aten.randn.default": _EMPTY,
    "aten.randn_like.default": _EMPTY,
    # Tensor lengths: the client sends int lists (ADR-032); no meta kernel.
    "aten._ctc_loss.Tensor": ("with Meta tensors",),
    # A tensor of scalars must be on the CPU in PyTorch; V4 sees it on meta.
    "aten._foreach_addcdiv.Tensor": ("Expected scalars to be on CPU",),
    "aten._foreach_addcmul.Tensor": ("Expected scalars to be on CPU",),
    "aten.binary_cross_entropy.default": _REDUCTION,
    # The CPU kernel gives self's shape, CUDA the broadcast shape or an error.
    "aten.binary_cross_entropy_backward.default": _REDUCTION + ("must broadcast to self",),
    "aten.binary_cross_entropy_with_logits.default": _REDUCTION,
    "aten.convolution.default": _INT32,
    # The CPU kernels run a grad_output of another batch size, CUDA's refuse
    # it; the kernels take dilation 0, which convolution refuses.
    "aten.convolution_backward.default": _INT32 + (
        "must have the output's shape", "stride and dilation positive"),
    # sparse=True: the worker supports dense gradients only.
    "aten.embedding.default": ("sparse gradients are not supported",),
    # Dense, non-overlapping layouts only (PLAN §8.3, ADR-044).
    "aten.empty_strided.default": ("strides must describe a dense, non-overlapping layout",),
    "aten.new_empty_strided.default": ("strides must describe a dense, non-overlapping layout",),
    # torch.pdist refuses p < 0 and NaN; the aten op does not check.
    "aten._pdist_forward.default": ("p must be non-negative",),
    # The CPU composite takes any normalized_shape of the right length and
    # mixed dtypes; the worker holds it to the CUDA kernel's rules.
    "aten._fused_rms_norm.default": ("normalized_shape", "weight is"),
    # The CPU oneDNN path refuses broadcasting and mixed dtypes, the other
    # paths take them: the worker takes grad_output in self's shape and dtype.
    "aten.gelu_backward.default": ("grad_output is",),
    # The CPU kernel broadcasts grad_output, CUDA's needs the output's shape.
    "aten.glu_backward.default": (" must be [",),
    "aten.huber_loss.default": _REDUCTION,
    "aten.huber_loss_backward.default": _REDUCTION,
    "aten.mse_loss_backward.default": _REDUCTION,
    "aten.nll_loss2d_backward.default": _NLL_REDUCTION + _NLL_WEIGHT,
    "aten.nll_loss2d_forward.default": _NLL_REDUCTION + _NLL_WEIGHT,
    # total_weight in another dtype runs only with reduction="none".
    "aten.nll_loss_backward.default": _NLL_REDUCTION + _NLL_WEIGHT + ("expected one dtype",),
    "aten.nll_loss_forward.default": _NLL_REDUCTION,
    "aten.smooth_l1_loss_backward.default": _REDUCTION,
    "aten.soft_margin_loss.default": _REDUCTION,
    "aten.soft_margin_loss_backward.default": _REDUCTION,
}
# Ops whose CUDA kernels take float8 only for some arguments (abs: e5m2;
# pow: exponents 0 and 1; native_dropout: train=False), so they are not FP8
# ops (ADR-050). A float8 cast of a seed is checked on CUDA only.
FP8 = ("does not take float8",)
FP8_DELIBERATE = {"aten.abs.default", "aten.pow.Tensor_Scalar", "aten.native_dropout.default"}
# Refused on every op that takes it. PyTorch's CPU kernels pin host memory
# only while CUDA is available, so this is a false rejection only then.
ANY_OP = ("pin_memory=True is not supported",)

# prechecks.py rules for one device's kernels only.
ONE_DEVICE = {
    "aten._int_mm.default": "cuda",
    "aten.channel_shuffle.default": "cuda",
    "aten.native_dropout_backward.default": "cuda",
    "aten.linalg_lu_factor_ex.default": "cpu",
    "aten._embedding_bag_backward.default": "cpu",
    "aten._embedding_bag_dense_backward.default": "cpu",
    "aten.nansum.default": "cpu",
}
# Their seeds exist only where CUDA does (CUDA_PROGRAMS).
ONE_DEVICE.update(dict.fromkeys([f"aten.{n}.default" for n in (
    "_fused_rms_norm", "_fused_rms_norm_backward", "_thnn_fused_lstm_cell", "_thnn_fused_lstm_cell_backward_impl",
    "_thnn_fused_gru_cell", "_thnn_fused_gru_cell_backward")], "cuda"))
# Float8 rules: a CPU session has no float8.
ONE_DEVICE.update(dict.fromkeys(["aten.cat.default", "aten.stack.default", "aten.where.self", "aten.eq.Tensor",
                                 "aten.ne.Tensor", "aten._scaled_mm.default"], "cuda"))

CASES = all_cases(cuda=torch.cuda.is_available())
BY_OP = collections.defaultdict(list)
for c in CASES:
    BY_OP[c.op].append(c)

# device -> ops whose cases all passed, and prechecks.py rules that refused
# a case.
PASSED = collections.defaultdict(set)
REFUSED = collections.defaultdict(set)
# device -> ops the worker ran a float8 case of.
RAN_FP8 = collections.defaultdict(set)


def _has_fp8(v):
    if isinstance(v, (list, tuple)):
        return any(_has_fp8(x) for x in v)
    return v in FLOAT8 or (isinstance(v, torch.Tensor) and v.dtype in FLOAT8)


@pytest.fixture(scope="module", autouse=True)
def _record_refusals():
    saved = dict(prechecks.CHECKS)

    def wrap(name, rule):
        def recorded(a, cuda):
            try:
                rule(a, cuda)
            except RuntimeError:
                REFUSED["cuda" if cuda else "cpu"].add(name)
                raise
        return recorded

    prechecks.CHECKS.update({name: wrap(name, rule) for name, rule in saved.items()})
    yield
    prechecks.CHECKS.update(saved)


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return request.param


@pytest.fixture(scope="module")
def sess(device):
    return make_session(device)


@pytest.mark.parametrize("op", sorted(BY_OP))
def test_worker_decides_like_pytorch(sess, device, op):
    deliberate = DELIBERATE.get(op, ()) + (FP8 if op in FP8_DELIBERATE else ())
    found, new = [], []
    for c in BY_OP[op]:
        marks = []
        why = check(sess, c, device, marks.append)
        if not why:
            if not marks and _has_fp8(c.args):
                RAN_FP8[device].add(op)
            continue
        if kind_of(why) == "false-rejection" and any(m in why for m in deliberate + ANY_OP):
            found.append(why)
        else:
            new.append(f"{kind_of(why)}: {c.label}: {why}")
    assert not new, f"{len(new)} disagreements:\n" + "\n".join(new[:10])
    if device == "cpu":
        # False rejections are checked on the CPU only (parity.check).
        stale = [m for m in DELIBERATE.get(op, ()) if not any(m in why for why in found)]
        assert not stale, f"no longer refused, remove from DELIBERATE: {stale}"
    elif op in FP8_DELIBERATE:
        assert any(FP8[0] in why for why in found), "the CUDA kernel no longer takes float8: remove from FP8_DELIBERATE"
    ok_after(sess)
    PASSED[device].add(op)


def test_every_precheck_refuses_a_case(device):
    # Runs after the cases on its device (pytest groups tests by the
    # module-scoped param). A rule no case reaches is untested.
    if PASSED[device] != set(BY_OP):
        pytest.skip("needs every case on this device to pass")
    rules = {name for name in prechecks.CHECKS if ONE_DEVICE.get(name, device) == device}
    assert not rules - REFUSED[device], f"never refused a case on {device}: {sorted(rules - REFUSED[device])}"


def test_every_fp8_op_runs_float8(device):
    # The other half of the float8 allowlist: each FP8 op's kernel runs a
    # float8 case the worker accepted, and gives the same values.
    if device != "cuda":
        pytest.skip("a CPU session has no float8")
    if PASSED[device] != set(BY_OP):
        pytest.skip("needs every case on this device to pass")
    fp8 = {name for name, s in SPEC.items() if s.flags & FLAG_FP8} - {"reflop.dataset_gather.default"}
    assert not fp8 - RAN_FP8[device], f"never ran float8: {sorted(fp8 - RAN_FP8[device])}"


@pytest.mark.parametrize("dt", FLOAT8)
def test_float8_only_on_fp8_ops(dt):
    t = torch.empty(2, 3, dtype=dt, device="meta")
    f = torch.empty(2, 3, device="meta")
    for name, a in (("aten.sum.default", {"self": t, "dtype": None}),
                    ("aten.sum.default", {"self": f, "dtype": dt}),
                    ("aten._foreach_neg.default", {"self": [f, t]}),
                    ("aten.arange.default", {"end": 3, "dtype": dt})):
        with pytest.raises(RuntimeError, match="aten.* does not take float8"):
            prechecks.float8(name, SPEC[name].flags, a)
    for name, a in (("aten.cat.default", {"tensors": [t, t], "dim": 0}),
                    ("aten._to_copy.default", {"self": f, "dtype": dt}),
                    ("aten.zeros.default", {"size": [2], "dtype": dt}),
                    ("aten.sum.default", {"self": f, "dtype": torch.float16})):
        prechecks.float8(name, SPEC[name].flags, a)


def test_deliberate_lists_real_ops():
    assert set(DELIBERATE) <= set(SPEC)
    assert STRICT <= set(SPEC) and FP8_DELIBERATE <= set(SPEC)
    assert not any(SPEC[n].flags & FLAG_FP8 for n in FP8_DELIBERATE)


def test_cases_cover_the_ops_this_fix_checks():
    # Every op with a new V4 check (ADR-042) has seeds and mutations.
    for op in STRICT:
        assert len(BY_OP[op]) > 10, op
    assert len(CASES) > 10_000
