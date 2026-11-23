# SPDX-License-Identifier: Apache-2.0
"""Every op call the worker accepts, PyTorch's kernel accepts too, with the
same values; every call PyTorch rejects, the worker rejects before a kernel
runs (#21, ADR-042).

The cases are seeds recorded from small PyTorch programs and their one-
argument mutations (parity.py), about 13,000 per device. Disagreements that
are known and safe (the kernel raises after V4 passed, or the worker is
stricter than PyTorch) are listed in KNOWN per op and kind; anything else
fails, and so does a KNOWN entry that no longer occurs, so the list only
shrinks. A kernel that crashes or returns other values is never known: it
read or wrote out of bounds."""

from __future__ import annotations

import collections

import pytest
import torch
from helpers import make_session
from parity import SPEC, STRICT, all_cases, check, kind_of
from test_corpus import ok_after

# op -> kinds of disagreement left to fix. "I3": V4 passes and the kernel
# raises after the output is allocated. "false-rejection": the worker
# refuses a call PyTorch runs. "worker-failed": the worker's kernel output
# does not match what V4 validated (fails closed at E3).
KNOWN = {
    "aten._adaptive_avg_pool2d.default": {"I3"},
    "aten._adaptive_avg_pool2d_backward.default": {"I3"},
    "aten._adaptive_avg_pool3d.default": {"I3"},
    "aten._adaptive_avg_pool3d_backward.default": {"I3"},
    "aten._embedding_bag.default": {"worker-failed"},
    "aten._embedding_bag_backward.default": {"I3"},
    "aten._embedding_bag_dense_backward.default": {"I3"},
    "aten._unsafe_index_put.default": {"I3"},
    "aten.affine_grid_generator.default": {"I3", "false-rejection"},
    "aten.avg_pool2d.default": {"I3"},
    "aten.avg_pool2d_backward.default": {"I3"},
    "aten.bitwise_not.default": {"I3"},
    "aten.convolution.default": {"I3"},
    "aten.convolution_backward.default": {"I3"},
    # sparse=True: the worker supports dense gradients only (deliberate).
    "aten.embedding.default": {"false-rejection"},
    # The worker allocates dense, non-overlapping layouts only (deliberate).
    "aten.empty_strided.default": {"false-rejection"},
    "aten.glu_backward.default": {"I3"},
    "aten.im2col.default": {"I3"},
    "aten.index.Tensor": {"I3"},
    "aten.index_add.default": {"I3"},
    "aten.index_copy.default": {"I3"},
    "aten.index_fill.int_Scalar": {"I3"},
    "aten.index_fill.int_Tensor": {"I3"},
    "aten.index_put.default": {"I3"},
    "aten.index_select.default": {"I3"},
    "aten.logsumexp.default": {"I3"},
    "aten.max.dim": {"I3"},
    "aten.mm.default": {"I3"},
    "aten.new_empty_strided.default": {"I3"},
    "aten.replication_pad2d.default": {"I3"},
    "aten.replication_pad3d.default": {"I3"},
    "aten.upsample_bicubic2d.default": {"I3"},
    "aten.upsample_bilinear2d.default": {"I3"},
    "aten.upsample_linear1d.default": {"I3"},
    "aten.upsample_trilinear3d.default": {"I3"},
    "aten.view_copy.default": {"false-rejection"},
}
# pin_memory=True raises after V4 passed ("No CUDA GPUs are available"
# without CUDA, "Only dense CPU tensors can be pinned" on a CUDA session);
# a CPU session on a machine with CUDA runs it. Not checked for staleness.
KNOWN_ANYWHERE = {op: {"I3"} for op in ("aten.ones.default", "aten.ones_like.default", "aten.zeros.default")}
# On CUDA, convolution_backward with an input of another shape returns a
# grad_input of the shape grad_output implies; the worker fails closed (E3).
KNOWN_CUDA = {"aten.convolution_backward.default": {"worker-failed"}}
NEVER_KNOWN = {"crash", "values", "ran", "meta-shapes"}

CASES = all_cases()
BY_OP = collections.defaultdict(list)
for c in CASES:
    BY_OP[c.op].append(c)


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
    found = collections.defaultdict(list)
    for c in BY_OP[op]:
        why = check(sess, c, device)
        if why:
            found[kind_of(why)].append(f"{c.label}: {why}")
    known = KNOWN.get(op, set())
    allowed = known | KNOWN_ANYWHERE.get(op, set()) | (KNOWN_CUDA.get(op, set()) if device == "cuda" else set())
    new = {k: v for k, v in found.items() if k not in allowed or k in NEVER_KNOWN}
    assert not new, "\n".join(f"{k}: {x}" for k, v in new.items() for x in v[:5])
    if device == "cpu":
        assert known <= set(found), f"fixed, remove from KNOWN: {sorted(known - set(found))}"
    ok_after(sess)


def test_known_lists_real_ops():
    for table in (KNOWN, KNOWN_CUDA, KNOWN_ANYWHERE):
        assert set(table) <= set(SPEC) and not any(k & NEVER_KNOWN for k in table.values())
    assert STRICT <= set(SPEC)


def test_cases_cover_the_ops_this_fix_checks():
    # Every op with a new V4 check (ADR-042) has seeds and mutations.
    for op in STRICT:
        assert len(BY_OP[op]) > 10, op
    assert len(CASES) > 10_000
