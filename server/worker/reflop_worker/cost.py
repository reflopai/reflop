# SPDX-License-Identifier: Apache-2.0
"""Stage V5: cost and peak-memory estimates (PLAN.md §9.2, §9.6).

The FLOP estimate feeds the optional per-execute FLOP quota and the audit
log. The peak-memory estimate (liveness analysis over meta sizes, plus the
WORKSPACE a few kernels allocate beyond their outputs) is checked against
the device's total memory: a graph that cannot fit fails fast, before any
allocation.
"""

from __future__ import annotations

import math

import torch

from ._pb import graph_pb2 as gpb
from .graph import Program, Ref


def _numel(t) -> int:
    return int(t.numel()) if isinstance(t, torch.Tensor) else 0


def _bytes(t) -> int:
    if not isinstance(t, torch.Tensor):
        return 0
    return int(t.numel()) * t.element_size()


def node_flops(node, ins: list, outs: list) -> int:
    kind = node.op.cost
    name = node.op.name
    out0 = outs[0] if outs else None
    if kind == "matmul":
        tens = [t for t in ins if isinstance(t, torch.Tensor)]
        if name.startswith(("aten.mm", "aten._int_mm", "aten.addmm", "aten._addmm_activation", "aten._scaled_mm")):
            a, b = [t for t in tens if t.dim() == 2][-2:]
            return 2 * a.shape[0] * a.shape[1] * b.shape[1]
        if name.startswith(("aten.bmm", "aten.baddbmm", "aten.addbmm")):
            a, b = [t for t in tens if t.dim() == 3][-2:]
            return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        if name.startswith(("aten.dot", "aten.vdot")):
            return 2 * _numel(tens[0])
        if name.startswith(("aten.mv", "aten.addmv")):
            m = [t for t in tens if t.dim() == 2][0]
            return 2 * m.numel()
        return _numel(out0)
    if kind == "conv":
        if name == "aten.convolution.default" and out0 is not None:
            w = ins[1]  # (out, in/groups, *kernel)
            k = math.prod(w.shape[2:]) if w.dim() > 2 else 1
            return 2 * out0.numel() * (w.shape[1] if w.dim() > 1 else 1) * k
        return 2 * sum(_numel(t) for t in outs) * 9
    if kind == "attention":
        q, k = ins[0], ins[1]
        if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and q.dim() >= 2:
            f = 4 * q.shape[:-1].numel() * k.shape[-2] * q.shape[-1]
            return f * (2 if "backward" in name else 1)
        return _numel(out0)
    if kind == "reduce":
        return sum(_numel(t) for t in ins if isinstance(t, torch.Tensor))
    if kind == "sort":
        n = _numel(ins[0])
        return int(n * max(1.0, math.log2(max(n, 2))))
    if kind == "cubic":
        a = ins[0]
        if isinstance(a, torch.Tensor) and a.dim() >= 2:
            n = max(a.shape[-1], a.shape[-2])
            return a.shape[:-2].numel() * n ** 3
        return 0
    return sum(_numel(t) for t in outs)


def _affine_grid_workspace(node, outs) -> int:
    # The kernels build a base grid with one more coordinate than the output
    # from one linspace per spatial size, and an empty grid still builds its
    # linspaces: [2, 1, 0, 2**31] asked for 8 GB past the quota.
    out, spatial = outs[0], node.args[1][2:]
    k = len(spatial)
    n = math.prod(out.shape[:-1]) * (k + 1) + sum(s for s in spatial if s > 1)
    return n * out.element_size()


# Bytes a kernel allocates for the call's duration besides its outputs,
# where they are not bounded by the outputs' size.
WORKSPACE = {"aten.affine_grid_generator.default": _affine_grid_workspace}


def analyze(prog: Program, input_bytes: int = 0) -> None:
    """Fill prog.flops and prog.peak_bytes."""
    metas = prog.metas
    flops = 0
    live = input_bytes
    peak = live
    # Bytes for every node output; views alias their input, and keep it
    # alive as long as they live.
    out_bytes: dict = {}
    last_use = alias_last_use(prog)
    expiring: dict[int, list] = {}
    for r, last in last_use.items():
        expiring.setdefault(last, []).append(r)
    for node in prog.nodes:
        ins = []
        for a in node.args:
            if isinstance(a, Ref):
                ins.append(metas.get(a))
            elif isinstance(a, list):
                ins.extend(metas.get(x) for x in a if isinstance(x, Ref))
            else:
                ins.append(a)
        outs = [metas[Ref(node.index, k)] for k in range(node.outputs)]
        flops += node_flops(node, ins, outs)
        is_view = getattr(node.op.fn, "is_view", False)
        for k, m in enumerate(outs):
            b = 0 if is_view else _bytes(m)
            out_bytes[Ref(node.index, k)] = b
            live += b
        w = WORKSPACE.get(node.op.name)
        peak = max(peak, live + (w(node, outs) if w else 0))
        for r in expiring.get(node.index, []):
            if r.node >= 0:
                live -= out_bytes.get(r, 0)
        # Outputs nobody reads are dropped right away.
        for k in range(node.outputs):
            r = Ref(node.index, k)
            if r not in last_use:
                live -= out_bytes[r]
    # A graph output that is a non-contiguous view (x.t(), x.expand(...)) is
    # copied whole when it is returned (.contiguous() before the host copy)
    # or later fetched; until then it costs nothing. Count that copy at the
    # end of the graph, when every output is alive (GH#4: a 0-d tensor
    # expanded to [2**31, 2] took 17 GB past every rail).
    prog.view_output_bytes = view_output_bytes(prog, out_bytes)
    prog.flops = flops
    prog.peak_bytes = max(peak, live + prog.view_output_bytes)


def alias_last_use(prog: Program) -> dict:
    """prog.last_use with each view's base kept until the view's own last
    use: the view shares the base's memory (GH#4). Walk backwards so chains
    of views extend their first base."""
    last = dict(prog.last_use)
    for node in reversed(prog.nodes):
        if not getattr(node.op.fn, "is_view", False):
            continue
        ends = [last[r] for k in range(node.outputs) if (r := Ref(node.index, k)) in last]
        if not ends:
            continue
        end = max(ends)
        for a in node.args:
            for b in (a if isinstance(a, list) else [a]):
                if isinstance(b, Ref) and last.get(b, -1) < end:
                    last[b] = end
    return last


def view_output_bytes(prog: Program, out_bytes: dict) -> int:
    total = 0
    for r in dict.fromkeys(prog.outputs):
        m = prog.metas.get(r)
        if r.node >= 0 and out_bytes.get(r, 0) == 0 and isinstance(m, torch.Tensor) and not m.is_contiguous():
            total += _bytes(m)
    return total


def input_bytes(prog: Program) -> int:
    return sum(_bytes(prog.metas[Ref(-1, i)]) for i, s in enumerate(prog.inputs)
               if s.kind == gpb.GraphInput.KIND_TENSOR)
