# SPDX-License-Identifier: Apache-2.0
"""Kernels that write to their arguments run on copies (ADR-058, GH#49).

The worker never changes a server tensor in place. Checkpoint cuts
(checkpoint.py) keep references to the current tensors as a consistent
view while a background thread copies them, and a handle changes only when
an execute commits, by pointing it at another tensor.

A few whitelisted kernels write to arguments anyway: native_batch_norm in
training mode updates running_mean and running_var (its schema does not say
so, and clients up to 0.2.3 rely on it to keep their statistics), and the
schemas of _native_batch_norm_legit and atan2.out mark arguments as written.
For these, every tensor the kernel would write is copied first and the
kernel writes the copy. Later nodes of the graph read the copy where they
would have read the original, and when the execute commits, every handle
that held the original holds the copy. A write that would resize its
tensor is refused. The original is never changed, so a checkpoint cut that
holds it stays consistent, and a failed execute changes no handle.
"""

from __future__ import annotations

import functools

import torch

from ._pb import compute_pb2 as cpb
from .errors import ReflopError
from .ops_generated import OPS

# Writes the schema does not declare.
HIDDEN = {"aten.native_batch_norm.default": ("running_mean", "running_var")}

# Kernels that write only when this argument is true.
WHEN = {"aten.native_batch_norm.default": "training", "aten._native_batch_norm_legit.default": "training"}


@functools.cache
def written(op_id: int):
    """((index into op.args, ...), index of the gating argument or None) for
    the arguments the op's kernel may write, or None if it writes none."""
    op = OPS[op_id]
    names = set(HIDDEN.get(op.name, ()))
    schema = getattr(op.fn, "_schema", None)
    if schema is not None:
        names.update(a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write)
    if not names:
        return None
    targets = tuple(i for i, a in enumerate(op.args) if a.name in names)
    if len(targets) != len(names):
        raise AssertionError(f"{op.name}: written arguments {sorted(names)} are not all in the opset schema")
    gate = WHEN.get(op.name)
    g = next(i for i, a in enumerate(op.args) if a.name == gate) if gate else None
    return targets, g


def plan(op):
    """written() as places in a node's call: ("p", position) or ("k", name)."""
    w = written(op.id)
    if w is None:
        return None
    pos_of, n = {}, 0
    for i, a in enumerate(op.args):
        if not a.kwarg_only:
            pos_of[i] = n
            n += 1

    def place(i):
        return ("k", op.args[i].name) if op.args[i].kwarg_only else ("p", pos_of[i])

    targets, gate = w
    return tuple(place(i) for i in targets), (place(gate) if gate is not None else None)


def _get(where, pos, kw):
    return pos[where[1]] if where[0] == "p" else kw.get(where[1])


def _set(where, pos, kw, v) -> None:
    if where[0] == "p":
        pos[where[1]] = v
    else:
        kw[where[1]] = v


def shapes(p, pos: list, kw: dict) -> list:
    """Shapes of the tensors the kernel may write, whatever the gate says."""
    out = []
    for where in p[0]:
        v = kw.get(where[1]) if where[0] == "k" else pos[where[1]]
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            out.append(tuple(x.shape) if isinstance(x, torch.Tensor) else None)
    return out


def copy_written(p, pos: list, kw: dict) -> list:
    """Replace each tensor the kernel would write with a copy, in pos and kw.
    Returns [(original, copy)]. A tensor passed twice gets one copy, so the
    kernel sees the aliasing it would have seen."""
    targets, gate = p
    if gate is not None and not _get(gate, pos, kw):
        return []
    copies: dict[int, tuple] = {}

    def copy(t):
        if not isinstance(t, torch.Tensor):
            return t
        return copies.setdefault(id(t), (t, t.clone()))[1]

    for where in targets:
        if where[0] == "k" and where[1] not in kw:
            continue
        v = _get(where, pos, kw)
        _set(where, pos, kw, [copy(x) for x in v] if isinstance(v, (list, tuple)) else copy(v))
    return list(copies.values())


def adopt(node, pairs: list, env, moved: dict, stage) -> None:
    """After the kernel: later nodes read each copy instead of its original,
    and moved records which tensor, as bound, each copy stands for.

    env is the executor's list of value slots (or any dict of values).
    moved maps id(current copy) -> [tensor as bound, current copy]."""
    keys = range(len(env)) if isinstance(env, list) else list(env)
    for orig, c in pairs:
        if c.shape != orig.shape or c.dtype != orig.dtype:
            raise ReflopError(cpb.ERROR_EXECUTION, f"{node.op.name}: the kernel would resize an argument it writes "
                              f"from {list(orig.shape)} to {list(c.shape)}", stage, node.index, node.op.id)
        entry = moved.pop(id(orig), None)
        if entry is None:
            entry = [orig, c]
        else:
            entry[1] = c
        moved[id(c)] = entry
        for k in keys:
            if env[k] is orig:
                env[k] = c


def commit(moved: dict, handles) -> None:
    """Point every handle that held a written tensor at its final copy."""
    if not moved:
        return
    final = {id(root): c for root, c in moved.values()}
    for h, t in list(handles.t.items()):
        c = final.get(id(t))
        if c is not None:
            handles.replace(h, c)
