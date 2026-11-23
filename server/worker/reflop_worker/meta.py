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


def _same(what, t, shape, dtype):
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype:
        raise RuntimeError(f"{what} is {t.dtype}{list(t.shape)}; the forward gives {dtype}{list(shape)}")


def _nonempty(t, what="self"):
    # The CPU and CUDA kernels index every non-batch dim; PyTorch's meta
    # kernels for the adaptive pools do not check it.
    if any(n == 0 for n in t.shape[1:]):
        raise RuntimeError(f"{what} {list(t.shape)} has an empty non-batch dimension")


def _adaptive_forward(node, pos, kw):
    _nonempty(validators._named(node, pos, kw)["self"])
    return node.op.fn(*pos, **kw)


def _pool_backward(forward, spatial, indices=False):
    """Pooling backward ops (ADR-042).

    PyTorch's meta kernels for these check grad_output's rank or nothing at
    all, and the CPU kernels index grad_output, indices and self with each
    other's sizes: a grad_output with another batch size, or a self with an
    empty dim, reads and writes out of bounds. Run the forward on a meta
    tensor of self and require grad_output (and indices) to have the exact
    shape and dtype the forward gives, as autograd passes them."""

    def infer(node, pos, kw):
        a = validators._named(node, pos, kw)
        self_, grad = a["self"], a["grad_output"]
        if self_.dim() not in (spatial + 1, spatial + 2) or grad.dim() != self_.dim():
            raise RuntimeError(f"grad_output is {grad.dim()}-d and self {self_.dim()}-d; both must be "
                               f"{spatial + 1}-d or {spatial + 2}-d")
        _nonempty(self_)
        out = forward(self_, a, tuple(grad.shape[-spatial:]))
        out, idx = out if isinstance(out, tuple) else (out, None)
        _same("grad_output", grad, out.shape, self_.dtype)
        if indices:
            _same("indices", a["indices"], idx.shape, torch.int64)
        return torch.empty(tuple(self_.shape), dtype=self_.dtype, device="meta")

    return infer


def _softmax_backward(node, pos, kw):
    # The CPU log_softmax backward kernel reads output with grad_output's
    # sizes and accepts any pair (ADR-042); autograd passes equal ones.
    a = validators._named(node, pos, kw)
    grad, out = a["grad_output"], a["output"]
    _same("grad_output", grad, out.shape, out.dtype)
    return node.op.fn(*pos, **kw)


def _weight_norm_shape(v, dim):
    """The shape of g and of the norms for weight_norm of v over dim, as the
    kernels require: dim is 0 or v's last dim, not wrapped."""
    if v.dim() == 0 or dim not in (0, v.dim() - 1):
        raise RuntimeError(f"dim must be 0 or {v.dim() - 1} for a {v.dim()}-d v, got {dim}")
    return tuple(n if d == dim else 1 for d, n in enumerate(v.shape))


def _norms_dtype(v):
    return torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype


def _weight_norm(node, pos, kw):
    # The kernels read v.size(dim) elements of g; PyTorch checks neither its
    # shape nor (in meta) the norms' shape (ADR-042).
    a = validators._named(node, pos, kw)
    v, g = a["v"], a["g"]
    _same("g", g, _weight_norm_shape(v, a["dim"]), v.dtype)
    return (torch.empty(tuple(v.shape), dtype=v.dtype, device="meta"),
            torch.empty(tuple(g.shape), dtype=_norms_dtype(v), device="meta"))


def _weight_norm_backward(node, pos, kw):
    a = validators._named(node, pos, kw)
    v = a["saved_v"]
    shape = _weight_norm_shape(v, a["dim"])
    _same("grad_w", a["grad_w"], v.shape, v.dtype)
    _same("saved_g", a["saved_g"], shape, v.dtype)
    _same("saved_norms", a["saved_norms"], shape, _norms_dtype(v))
    return (torch.empty(tuple(v.shape), dtype=v.dtype, device="meta"),
            torch.empty(shape, dtype=v.dtype, device="meta"))


def _embedding_bag_backward(node, pos, kw):
    """_embedding_bag_backward and _embedding_bag_dense_backward (ADR-042).

    The kernels read bag_size, maximum_indices and per_sample_weights with
    grad's and indices' sizes; PyTorch checks none of them. Require the
    shapes the forward gives: grad (bags, D) with D > 0, bag_size (bags,),
    offset2bag (indices,) (or empty, with offsets), maximum_indices (bags, D) in max mode,
    per_sample_weights (indices,) with grad's dtype."""
    a = validators._named(node, pos, kw)
    grad, idx = a["grad"], a["indices"]
    if grad.dim() != 2 or idx.dim() != 1:
        raise RuntimeError(f"grad must be 2-d and indices 1-d, got {grad.dim()}-d and {idx.dim()}-d")
    bags, n = grad.shape[0], idx.shape[0]
    if grad.shape[1] == 0:
        # The CUDA kernel reads out of bounds; the CPU forward rejects it.
        raise RuntimeError("grad has an empty embedding dimension")
    for name in ("offset2bag", "bag_size", "maximum_indices") + (("offsets",) if "offsets" in a else ()):
        if a[name].dtype not in validators.INDEX_DTYPES:
            raise RuntimeError(f"{name} must be int32 or int64, got {a[name].dtype}")
    offsets = a.get("offsets")
    if offsets is not None and (offsets.dim() != 1 or offsets.shape[0] not in (bags, bags + 1)):
        raise RuntimeError(f"offsets has shape {list(offsets.shape)}; grad has {bags} bags")
    # Only _embedding_bag_backward can rebuild an empty offset2bag (from
    # offsets); the dense backward reads it as is.
    if tuple(a["offset2bag"].shape) not in ((n,), (0,) if offsets is not None else (n,)):
        raise RuntimeError(f"offset2bag has shape {list(a['offset2bag'].shape)}; indices has {n} entries")
    if tuple(a["bag_size"].shape) != (bags,):
        raise RuntimeError(f"bag_size has shape {list(a['bag_size'].shape)}; grad has {bags} bags")
    if a["mode"] == 2 and tuple(a["maximum_indices"].shape) != tuple(grad.shape):
        raise RuntimeError(f"maximum_indices has shape {list(a['maximum_indices'].shape)}; "
                           f"grad has shape {list(grad.shape)}")
    psw = a.get("per_sample_weights")
    if psw is not None:
        _same("per_sample_weights", psw, (n,), grad.dtype)
    return node.op.fn(*pos, **kw)


def _embedding_dense_backward(node, pos, kw):
    # grad_output is indices.shape + (D,), as autograd passes it; PyTorch's
    # meta kernel does not check it, and the CUDA kernel reads out of
    # bounds when D is 0 (ADR-042).
    a = validators._named(node, pos, kw)
    grad, idx = a["grad_output"], a["indices"]
    if grad.dim() != idx.dim() + 1 or tuple(grad.shape[:-1]) != tuple(idx.shape) or grad.shape[-1] == 0:
        raise RuntimeError(f"grad_output has shape {list(grad.shape)}; indices {list(idx.shape)} needs "
                           f"{list(idx.shape)} + [D] with D > 0")
    return node.op.fn(*pos, **kw)


A = torch.ops.aten

# aten ops that PyTorch ships without a meta kernel, or with one that checks
# nothing; each one checks its inputs itself.
OVERRIDES = {
    "aten._adaptive_avg_pool2d.default": _adaptive_forward,
    "aten._adaptive_avg_pool3d.default": _adaptive_forward,
    "aten._adaptive_avg_pool2d_backward.default": _pool_backward(
        lambda s, a, size: A._adaptive_avg_pool2d.default(s, size), 2),
    "aten._adaptive_avg_pool3d_backward.default": _pool_backward(
        lambda s, a, size: A._adaptive_avg_pool3d.default(s, size), 3),
    "aten.adaptive_max_pool2d_backward.default": _pool_backward(
        lambda s, a, size: A.adaptive_max_pool2d.default(s, size), 2, indices=True),
    "aten.adaptive_max_pool3d_backward.default": _pool_backward(
        lambda s, a, size: A.adaptive_max_pool3d.default(s, size), 3, indices=True),
    "aten.max_pool2d_with_indices_backward.default": _pool_backward(
        lambda s, a, _: A.max_pool2d_with_indices.default(s, a["kernel_size"], a["stride"], a["padding"],
                                                          a["dilation"], a["ceil_mode"]), 2, indices=True),
    "aten.max_pool3d_with_indices_backward.default": _pool_backward(
        lambda s, a, _: A.max_pool3d_with_indices.default(s, a["kernel_size"], a["stride"], a["padding"],
                                                          a["dilation"], a["ceil_mode"]), 3, indices=True),
    "aten.avg_pool2d_backward.default": _pool_backward(
        lambda s, a, _: A.avg_pool2d.default(s, a["kernel_size"], a["stride"], a["padding"], a["ceil_mode"],
                                             a["count_include_pad"], a["divisor_override"]), 2),
    "aten.avg_pool3d_backward.default": _pool_backward(
        lambda s, a, _: A.avg_pool3d.default(s, a["kernel_size"], a["stride"], a["padding"], a["ceil_mode"],
                                             a["count_include_pad"], a["divisor_override"]), 3),
    "aten._softmax_backward_data.default": _softmax_backward,
    "aten._log_softmax_backward_data.default": _softmax_backward,
    "aten._weight_norm_interface.default": _weight_norm,
    "aten._weight_norm_interface_backward.default": _weight_norm_backward,
    "aten._embedding_bag_backward.default": _embedding_bag_backward,
    "aten._embedding_bag_dense_backward.default": _embedding_bag_backward,
    "aten.embedding_dense_backward.default": _embedding_dense_backward,
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
