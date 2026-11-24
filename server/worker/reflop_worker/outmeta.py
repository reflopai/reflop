# SPDX-License-Identifier: Apache-2.0
"""Output shapes and dtypes the device kernels give where PyTorch's meta
kernels say otherwise (#25 E).

A meta kernel gives one answer for every device; for the ops below the CPU or
the CUDA kernel returns another shape or dtype. fixup() turns the meta output
into what the kernel on the session's device returns. The worker applies it
in V4, so E3 compares each kernel output with the right meta, and the client
applies it to the metas it records, so a later graph that binds the output
as a handle declares the same shape and dtype the worker holds (E1).

This file is the only copy that is edited: server/worker/reflop_worker/
outmeta.py. codegen/gen.py copies it to client/python/reflop/
_outmeta_generated.py, and `gen.py --check` keeps the two equal. It imports
only torch.

Arguments are passed by schema name. A Tensor argument may be a Python
number (a wrapped number, or a scalar graph input); every rule here reads
only tensors that the ops require to be tensors."""

from __future__ import annotations

import torch

_FAST_EMBEDDING_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _fix(t, shape=None, dtype=None):
    """t when it already has `shape` and `dtype`, else a meta tensor that
    has them (t's strides when only the dtype changes)."""
    shape = tuple(t.shape) if shape is None else tuple(shape)
    dtype = t.dtype if dtype is None else dtype
    if tuple(t.shape) == shape and t.dtype == dtype:
        return t
    if tuple(t.shape) == shape:
        return torch.empty_like(t, dtype=dtype, device="meta")
    return torch.empty(shape, dtype=dtype, device="meta")


def _self_dtype(out, a, cuda):
    # The kernels compute in self's dtype; the meta kernels promote self with
    # target (and weight).
    return _fix(out, dtype=a["self"].dtype)


def _threshold_backward(out, a, cuda):
    # A binary op of grad_output and self; the meta kernel keeps grad_output's dtype.
    return _fix(out, dtype=torch.result_type(a["grad_output"], a["self"]))


def _log_sigmoid_forward(out, a, cuda):
    # The CUDA kernel computes no buffer and returns it empty.
    output, buffer = out
    return (output, _fix(buffer, (0,))) if cuda else out


def _prelu_kernel_backward(out, a, cuda):
    # Both gradients have the broadcast shape of grad_output, self and weight
    # (autograd reduces grad_weight afterwards); the meta kernel gives
    # weight's shape when weight has more dims than self.
    shape = torch.broadcast_shapes(a["grad_output"].shape, a["self"].shape, a["weight"].shape)
    return (_fix(out[0], shape, a["self"].dtype), _fix(out[1], shape, a["weight"].dtype))


def _embedding_bag(out, a, cuda):
    """offset2bag, bag_size and max_indices are in the promoted dtype of
    indices and offsets on both devices (the meta kernel keeps indices').
    Their shapes differ on the CPU: bag_size and (outside max mode)
    max_indices have one entry per bag, and offset2bag is empty in sum mode
    on the fast path, which the worker makes depend on dtype and
    padding_idx alone (it passes contiguous weight and per_sample_weights)."""
    output, offset2bag, bag_size, max_indices = out
    dt = torch.promote_types(a["indices"].dtype, a["offsets"].dtype)
    if cuda:
        return output, _fix(offset2bag, dtype=dt), _fix(bag_size, dtype=dt), _fix(max_indices, dtype=dt)
    weight, mode = a["weight"], a["mode"]
    bags = output.shape[0]
    fast = weight.dtype in _FAST_EMBEDDING_DTYPES and a["padding_idx"] < 0
    o2b = (0,) if mode == 0 and fast else (a["indices"].shape[0],)
    mi = (bags, weight.shape[1]) if mode == 2 else (bags,)
    return output, _fix(offset2bag, o2b, dt), _fix(bag_size, (bags,), dt), _fix(max_indices, mi, dt)


def _bn_cpu_dtype(a, names):
    """The dtype the CPU batch norm kernels save statistics and compute
    parameter gradients in: float32 when the first defined parameter's
    dtype is not input's ("mixed", mixed_data_type.h), else input's."""
    x = a["input"].dtype
    p = next((a[n].dtype for n in names if isinstance(a.get(n), torch.Tensor)), x)
    return torch.float32 if p != x else x


def _batch_norm_stats(training_arg):
    """On the CPU the saved statistics are in _bn_cpu_dtype (the meta
    kernels say the accumulate type, as on CUDA) and empty in eval mode
    (the meta kernels and CUDA give them per channel)."""

    def fix(out, a, cuda):
        out = tuple(out)
        if len(out) == 5:
            # _native_batch_norm_legit_functional's updated running statistics
            # keep their dtypes (the meta kernel says the accumulate type).
            out = out[:3] + (_fix(out[3], dtype=a["running_mean"].dtype), _fix(out[4], dtype=a["running_var"].dtype))
        if cuda:
            return out
        training = a[training_arg] if training_arg else False
        shape = None if training else (0,)
        dt = _bn_cpu_dtype(a, ("weight", "bias", "running_mean", "running_var"))
        return out[0], _fix(out[1], shape, dt), _fix(out[2], shape, dt), *out[3:]

    return fix


def _batch_norm_backward(out, a, cuda):
    """Both kernels skip grad_input unless output_mask asks for it. On the
    CPU the parameter gradients are in _bn_cpu_dtype (the meta kernel says
    weight's dtype, or input's without weight)."""
    out = tuple(out)
    if not cuda:
        dt = _bn_cpu_dtype(a, ("weight", "running_mean", "running_var", "save_mean", "save_invstd"))
        out = (out[0],) + tuple(None if t is None else _fix(t, dtype=dt) for t in out[1:])
    return out if a["output_mask"][0] else (None,) + out[1:]


def _log_sigmoid_backward(out, a, cuda):
    # The CUDA kernel ignores buffer, which its forward returns empty.
    if not cuda:
        return out
    shape = torch.broadcast_shapes(a["grad_output"].shape, a["self"].shape)
    return _fix(out, shape, a["self"].dtype)


_LOW = (torch.float16, torch.bfloat16)


def _mixed(a):
    """The CPU norm kernels take float32 weight or bias for a half or
    bfloat16 input, and then compute the statistics in float32."""
    return a["input"].dtype in _LOW and any(
        isinstance(a.get(n), torch.Tensor) and a[n].dtype == torch.float32 for n in ("weight", "bias"))


def _norm_stats(out, a, cuda):
    """native_layer_norm and native_group_norm: on the CPU, mean and rstd
    are float32 only when the parameters are (the meta kernels say float32
    for every half or bfloat16 input to layer_norm, input's dtype for every
    input to group_norm)."""
    x = a["input"]
    if cuda or x.dtype not in _LOW:
        return out
    dt = torch.float32 if _mixed(a) else x.dtype
    return out[0], _fix(out[1], dtype=dt), _fix(out[2], dtype=dt)


def _group_norm_backward(out, a, cuda):
    # The CPU kernel gives grad_input in input's dtype and the parameter
    # gradients in weight's; the meta kernel follows mean's.
    if cuda:
        return out
    x, w = a["input"], a["weight"]
    pd = x.dtype if w is None else w.dtype
    return tuple(None if t is None else _fix(t, dtype=x.dtype if i == 0 else pd) for i, t in enumerate(out))


def _pixel_unshuffle(out, a, cuda):
    # The CPU kernel returns a clone of an empty input as it is.
    x = a["self"]
    return _fix(out, x.shape) if not cuda and x.numel() == 0 else out


def _softmax_backward(out, a, cuda):
    # Gradients in grad_output's dtype, half for float32 gradients of a half
    # input; the meta kernel says input_dtype.
    g = a["grad_output"].dtype
    half = g == torch.float32 and a["input_dtype"] == torch.float16
    return _fix(out, dtype=torch.float16 if half else g)


def _native_dropout(out, a, cuda):
    # The CPU kernel returns an empty input with a mask in input's dtype.
    x = a["input"]
    return (out[0], _fix(out[1], dtype=x.dtype)) if not cuda and x.numel() == 0 else out


def _native_dropout_backward(out, a, cuda):
    # The CPU kernel computes grad_output * mask * scale (a double), so an
    # integer or bool product becomes float32; the meta kernel keeps
    # grad_output's dtype, as CUDA does.
    if cuda:
        return out
    dt = torch.result_type(a["grad_output"], a["mask"])
    return _fix(out, dtype=dt if dt.is_floating_point or dt.is_complex else torch.float32)


def _randperm(out, a, cuda):
    # The kernels default to int64, the meta kernel to float32.
    return _fix(out, dtype=torch.int64) if a["dtype"] is None else out


FIXUPS = {
    "aten.huber_loss.default": _self_dtype,
    "aten.soft_margin_loss.default": _self_dtype,
    "aten.soft_margin_loss_backward.default": _self_dtype,
    "aten.mse_loss_backward.default": _self_dtype,
    "aten.smooth_l1_loss_backward.default": _self_dtype,
    "aten.binary_cross_entropy.default": _self_dtype,
    "aten.binary_cross_entropy_backward.default": _self_dtype,
    "aten.mish_backward.default": _self_dtype,
    "aten.threshold_backward.default": _threshold_backward,
    "aten.log_sigmoid_forward.default": _log_sigmoid_forward,
    "aten._prelu_kernel_backward.default": _prelu_kernel_backward,
    "aten._embedding_bag.default": _embedding_bag,
    "aten.native_batch_norm.default": _batch_norm_stats("training"),
    "aten._native_batch_norm_legit.default": _batch_norm_stats("training"),
    "aten._native_batch_norm_legit.no_stats": _batch_norm_stats("training"),
    "aten._native_batch_norm_legit_functional.default": _batch_norm_stats("training"),
    "aten._native_batch_norm_legit_no_training.default": _batch_norm_stats(None),
    "aten.pixel_unshuffle.default": _pixel_unshuffle,
    "aten.native_batch_norm_backward.default": _batch_norm_backward,
    "aten.log_sigmoid_backward.default": _log_sigmoid_backward,
    "aten.native_layer_norm.default": _norm_stats,
    "aten.native_group_norm.default": _norm_stats,
    "aten.native_group_norm_backward.default": _group_norm_backward,
    "aten._softmax_backward_data.default": _softmax_backward,
    "aten._log_softmax_backward_data.default": _softmax_backward,
    "aten.native_dropout.default": _native_dropout,
    "aten.native_dropout_backward.default": _native_dropout_backward,
    "aten.randperm.default": _randperm,
}


def _affine_grid_generator(a):
    # The meta kernel refuses an empty grid, which the kernels return, and
    # gives a negative size the shape of another grid.
    theta, size = a["theta"], list(a["size"])
    # The kernels ignore size[1] (channels).
    if len(size) not in (4, 5) or any(n < 0 for n in size[:1] + size[2:]):
        raise RuntimeError(f"size {size} must have 4 or 5 entries, the batch and spatial ones non-negative")
    k = len(size) - 2
    if theta.dim() != 3 or tuple(theta.shape) != (size[0], k, k + 1):
        raise RuntimeError(f"theta has shape {list(theta.shape)}; size {size} needs {[size[0], k, k + 1]}")
    if all(size[2:]):
        return torch.ops.aten.affine_grid_generator.default(theta, size, a["align_corners"])
    return torch.empty(size[:1] + size[2:] + [k], dtype=theta.dtype, device="meta")


def _view_copy(a):
    # The kernels copy with reshape semantics; the meta kernel takes a view
    # and refuses a size that needs a copy.
    x = a["self"]
    return torch.empty(tuple(x.reshape(a["size"]).shape), dtype=x.dtype, device="meta")


# Ops whose meta kernel refuses calls the kernels run: the meta output,
# from arguments by schema name (meta tensors).
META = {
    "aten.affine_grid_generator.default": _affine_grid_generator,
    "aten.view_copy.default": _view_copy,
}


def fixup(name: str, out, a: dict, cuda: bool):
    """`out`, the meta output of op `name` with arguments `a` (by schema
    name), as the kernel on the session's device returns it."""
    f = FIXUPS.get(name)
    return out if f is None else f(out, a, cuda)
