# SPDX-License-Identifier: Apache-2.0
"""Stage V4: meta inference (PLAN.md §9.2).

Runs every node on PyTorch `meta` tensors to obtain all intermediate shapes
and dtypes without touching device memory. An op that has no meta kernel
cannot be validated and is rejected (fail closed, I11).
"""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from . import natives, validators
from ._pb import compute_pb2 as cpb
from ._pb import graph_pb2 as gpb
from .errors import ReflopError, invalid, limit
from .graph import Program, Ref, ScalarRef, placeholder


class DatasetMeta:
    """Stand-in for a dataset input during meta inference."""

    def __init__(self, dtype: torch.dtype, shape: tuple) -> None:
        self.dtype = dtype
        self.shape = torch.Size(shape)


def call_args(node, resolve):
    """Split resolved arguments into positional and keyword arguments by the
    static schema (kwarg-only names come from the generated table)."""
    pos, kw = [], {}
    for spec, a in zip(node.op.args, node.args):
        v = resolve(a)
        if spec.kwarg_only:
            kw[spec.name] = v
        else:
            pos.append(v)
    return pos, kw


def _ctc_loss_backward(node, pos, kw):
    # The gradient for log_probs: its shape and dtype, contiguous. The
    # inputs' shapes are checked first, here as well as in the ctc validator:
    # the kernels read them unchecked.
    a = validators._named(node, pos, kw)
    validators.ctc_backward_static(node, a)
    lp = a["log_probs"]
    return torch.empty(tuple(lp.shape), dtype=lp.dtype, device="meta")


def _view_backward(view):
    """select_backward, diagonal_backward and slice_backward.

    PyTorch's meta kernels for these return empty(input_sizes) and check
    nothing; the device kernels allocate zeros(input_sizes) and only then
    find a bad index, identical diagonal dims or a grad_output that does not
    fit. Run the forward view on a meta tensor of input_sizes instead (it
    checks dim, index and offset) and require grad_output to have the view's
    exact shape, as autograd gives it."""

    def infer(node, pos, kw):
        a = validators._named(node, pos, kw)
        grad = a["grad_output"]
        base = torch.empty(a["input_sizes"], dtype=grad.dtype, device="meta")
        want = tuple(view(base, a).shape)
        if tuple(grad.shape) != want:
            raise RuntimeError(f"grad_output has shape {list(grad.shape)}; this view of input_sizes "
                               f"{list(a['input_sizes'])} has shape {list(want)}")
        return torch.empty(a["input_sizes"], dtype=grad.dtype, device="meta")

    return infer


def _as_strided_view(out, a):
    """as_strided_scatter's view of the clone, with the bounds the kernel's
    setStorage would check (the meta as_strided does not check them)."""
    off = a.get("storage_offset")
    off = out.storage_offset() if off is None else off
    try:
        top = validators._reach(a["size"], a["stride"], off)
    except (ValueError, OverflowError) as e:
        raise RuntimeError(str(e)) from None
    # A view without elements touches no memory; PyTorch accepts any offset.
    if top >= validators._storage_numel(out) and all(n != 0 for n in a["size"]):
        raise RuntimeError("the view reaches past the end of self's storage")
    return out.as_strided(a["size"], a["stride"], off)


def _scatter(view):
    """select_scatter, slice_scatter, diagonal_scatter and as_strided_scatter.

    PyTorch's meta kernels for these check nothing: a bad index, dims, offset
    or src shape passes, and the device kernel allocates its output before it
    raises. The kernels clone self, take the view of the clone and copy src
    into it; do the same on meta tensors. The view checks dim, index and
    offset; src must have the view's exact shape (any dtype, as copy_ casts);
    the view must not write one element twice."""

    def infer(node, pos, kw):
        a = validators._named(node, pos, kw)
        out = a["self"].clone()
        v = view(out, a)
        src = a["src"]
        if tuple(src.shape) != tuple(v.shape):
            raise RuntimeError(f"src has shape {list(src.shape)}; the view of self has shape {list(v.shape)}")
        if any(n > 1 and st == 0 for n, st in zip(v.shape, v.stride())):
            raise RuntimeError("more than one element of the view refers to a single memory location")
        return out

    return infer


# aten ops that PyTorch ships without a meta kernel, or with one that checks
# nothing; each one checks its inputs itself.
OVERRIDES = {
    "aten.select_scatter.default": _scatter(lambda o, a: o.select(a["dim"], a["index"])),
    "aten.slice_scatter.default": _scatter(
        lambda o, a: torch.ops.aten.slice.Tensor(o, a["dim"], a["start"], a["end"], a["step"])),
    "aten.diagonal_scatter.default": _scatter(lambda o, a: o.diagonal(a["offset"], a["dim1"], a["dim2"])),
    "aten.as_strided_scatter.default": _scatter(_as_strided_view),
    "aten._ctc_loss_backward.default": _ctc_loss_backward,
    "aten.select_backward.default": _view_backward(lambda b, a: b.select(a["dim"], a["index"])),
    "aten.diagonal_backward.default": _view_backward(lambda b, a: b.diagonal(a["offset"], a["dim1"], a["dim2"])),
    "aten.slice_backward.default": _view_backward(
        lambda b, a: torch.ops.aten.slice.Tensor(b, a["dim"], a["start"], a["end"], a["step"])),
}


def infer(prog: Program, limits: cpb.Limits) -> None:
    S = cpb.STAGE_V4_META
    metas: dict = {}
    for i, inp in enumerate(prog.inputs):
        if inp.kind == gpb.GraphInput.KIND_TENSOR:
            metas[Ref(-1, i)] = torch.empty(inp.shape, dtype=inp.dtype, device="meta")
        elif inp.kind == gpb.GraphInput.KIND_DATASET_REF:
            metas[Ref(-1, i)] = DatasetMeta(inp.dtype, inp.shape)
    last_use: dict = {}

    for node in prog.nodes:
        def resolve(a, node=node):
            if isinstance(a, Ref):
                if a.node >= 0 and a.index >= prog.nodes[a.node].outputs:
                    raise invalid(f"reference to missing output {a.index} of node {a.node}", S, node.index, node.op.id)
                last_use[a] = node.index
                return metas[a]
            if isinstance(a, ScalarRef):
                return placeholder(prog.inputs[a.input].scalar_kind)
            if isinstance(a, list):
                return [resolve(x) for x in a]
            if a == "session-device":
                return torch.device("meta")
            return a

        pos, kw = call_args(node, resolve)
        validators.check_static(node, pos, kw)
        try:
            with torch.inference_mode():
                if natives.is_native(node.op):
                    out = natives.meta(node.op.name, pos, kw)
                elif node.op.name in OVERRIDES:
                    out = OVERRIDES[node.op.name](node, pos, kw)
                else:
                    out = node.op.fn(*pos, **kw)
        except ReflopError:
            raise
        except NotImplementedError as e:
            raise invalid(f"{node.op.name} cannot be validated (no meta kernel): {str(e)[:200]}", S,
                          node.index, node.op.id) from None
        except Exception as e:  # noqa: BLE001 - every kernel error becomes a structured rejection
            raise ReflopError(cpb.ERROR_SHAPE_MISMATCH, f"{node.op.name}: {str(e)[:500]}", S, node.index,
                              node.op.id) from None
        flat = [x for x in tree_flatten(out)[0] if isinstance(x, torch.Tensor)]
        for k, m in enumerate(flat):
            if m.dim() > limits.max_dims:
                raise limit(f"{node.op.name} output has {m.dim()} dims", S, node.index, node.op.id)
            if m.numel() > limits.max_tensor_numel:
                raise limit(f"{node.op.name} output has {m.numel()} elements; the limit is "
                            f"{limits.max_tensor_numel}", S, node.index, node.op.id)
            metas[Ref(node.index, k)] = m
        node.outputs = len(flat)

    for r, (dt, shape) in zip(prog.outputs, prog.output_meta):
        if r.node >= 0 and r.index >= prog.nodes[r.node].outputs:
            raise invalid(f"graph output refers to missing output {r.index} of node {r.node}", S)
        m = metas.get(r)
        if m is None or isinstance(m, DatasetMeta):
            raise invalid("graph output is not a tensor", S)
        if m.dtype != dt or tuple(m.shape) != tuple(shape):
            raise ReflopError(cpb.ERROR_SHAPE_MISMATCH,
                              f"declared output {dt}{list(shape)} but inferred {m.dtype}{list(m.shape)}", S)
        last_use[r] = len(prog.nodes)
    prog.metas = metas
    prog.last_use = last_use
