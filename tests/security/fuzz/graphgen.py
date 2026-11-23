# SPDX-License-Identifier: Apache-2.0
"""Structure-aware generators for worker graph fuzzing (GH#4, ADR-044).

Three kinds of case, each a worker request (and its execute, if any):

- `random_case`: a Graph built from the opset by schema type, with unknown
  and out-of-range op ids, wrong arities, wrong argument kinds, dangling and
  forward references, odd dtypes, shapes and enum values. Mostly exercises
  decode, structure and schema (V1-V3).
- `composed_case`: one to four parity seeds (recorded PyTorch calls and
  their mutations) chained into one graph, a later node reading an earlier
  node's output where dtype and shape allow. Passes V4 often enough to
  exercise V5, bindings, runtime checks and execution.
- `bytes_case`: a serialized worker envelope with bytes flipped, inserted,
  dropped or truncated (V1 decode).

Every case comes from (seed, index) alone, so a failure replays exactly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
from parity import SPEC, Call, encode, infer_outputs
from reflop_worker._pb import capabilities_pb2 as kpb
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.core import Session
from reflop_worker.dtypes import PB_TO_TORCH, TORCH_TO_PB
from reflop_worker.graph import _base
from reflop_worker.ops_generated import OPS

# Every case's memory stays under this: V5 rejects bigger graphs with a
# quota error, so a fuzzed size never reaches the allocator.
DEVICE_QUOTA = 256 << 20
OP_IDS = sorted(OPS)
DTYPES = [d for d in PB_TO_TORCH if PB_TO_TORCH[d] in (torch.float32, torch.float64, torch.float16,
                                                      torch.bfloat16, torch.int64, torch.int32, torch.bool,
                                                      torch.uint8, torch.complex64)]
DIMS = (0, 1, 1, 2, 3, 4, 5)
INTS = (-(2**63), -(2**31), -5, -2, -1, 0, 0, 1, 1, 2, 3, 4, 7, 2**31, 2**62)
FLOATS = (0.0, 1.0, -1.0, 0.5, 1e-12, 1e30, math.inf, -math.inf, math.nan)


def make_session(device: str = "cpu") -> Session:
    from reflop.connection import default_limits

    init = wpb.WorkerInit(seed=0, device_memory_quota_bytes=DEVICE_QUOTA, host_memory_budget_bytes=DEVICE_QUOTA)
    init.device.kind = gpb.DEVICE_CPU if device == "cpu" else gpb.DEVICE_CUDA
    init.limits.CopyFrom(default_limits(kpb))
    # Returned outputs are copied to host bytes whole (V5 counts the device
    # side only); keep that host copy small too.
    init.limits.max_fetch_bytes = 64 << 20
    return Session(init)


@dataclass
class Case:
    kind: str                       # "random" | "composed" | "bytes"
    graph: gpb.Graph | None = None
    inputs: list = field(default_factory=list)  # one binding source per graph input (tensor, scalar or None)
    outputs: list = field(default_factory=list)  # cpb.OutputSpec kinds
    raw: bytes | None = None        # for "bytes": a serialized WorkerEnvelope


def rng_for(seed: int, index: int) -> random.Random:
    return random.Random(seed * 1_000_003 + index)


# ------------------------------------------------------------------ random


def _meta(r: random.Random) -> gpb.TensorMeta:
    dt = r.choice(DTYPES) if r.random() < 0.97 else r.choice((0, 99, 2**20))
    rank = r.choice((0, 1, 1, 2, 2, 3, 4))
    shape = [r.choice(DIMS) for _ in range(rank)]
    if r.random() < 0.03:
        shape.append(r.choice((-1, 2**40, 2**63 - 1)))
    if r.random() < 0.02:
        shape = [1] * r.choice((9, 40))
    return gpb.TensorMeta(dtype=dt, shape=shape)


def _ref(r: random.Random, n_inputs: int, n_nodes: int) -> gpb.ValueRef:
    roll = r.random()
    if roll < 0.04:
        return gpb.ValueRef()  # empty
    if roll < 0.08:
        return gpb.ValueRef(graph_input=n_inputs + r.choice((0, 1, 2**31)))  # dangling
    if roll < 0.12:
        return gpb.ValueRef(node_output=gpb.NodeOutput(node=n_nodes + r.choice((0, 1)), output=0))  # forward
    if n_nodes and roll < 0.6:
        return gpb.ValueRef(node_output=gpb.NodeOutput(node=r.randrange(n_nodes),
                                                       output=r.choice((0, 0, 0, 1, 2, 7))))
    if n_inputs:
        return gpb.ValueRef(graph_input=r.randrange(n_inputs))
    return gpb.ValueRef(graph_input=0)


def _arg(r: random.Random, spec_type: str, n_inputs: int, n_nodes: int, sess) -> gpb.Arg:
    t = _base(spec_type)
    opt = t.endswith("?")
    core = t.rstrip("?")
    if r.random() < 0.05:  # a value of the wrong kind
        return r.choice((gpb.Arg(i=1), gpb.Arg(f=0.5), gpb.Arg(b=True), gpb.Arg(none=True),
                         gpb.Arg(tensor=_ref(r, n_inputs, n_nodes)), gpb.Arg(ints=gpb.Int64List(values=[1, 2])),
                         gpb.Arg(dtype=gpb.DTYPE_FLOAT32), gpb.Arg(enum_code=3), gpb.Arg()))
    if opt and r.random() < 0.3:
        return gpb.Arg(none=True)
    if core == "Tensor":
        if r.random() < 0.1:
            return r.choice((gpb.Arg(i=r.choice(INTS)), gpb.Arg(f=r.choice(FLOATS)), gpb.Arg(b=True)))
        return gpb.Arg(tensor=_ref(r, n_inputs, n_nodes))
    if core in ("Tensor[]", "Tensor?[]"):
        items = [gpb.ValueRef() if core == "Tensor?[]" and r.random() < 0.2 else _ref(r, n_inputs, n_nodes)
                 for _ in range(r.choice((0, 1, 2, 3)))]
        return gpb.Arg(tensors=gpb.TensorList(items=items))
    if core in ("int", "SymInt"):
        return gpb.Arg(i=r.choice(INTS))
    if core.startswith(("int[", "SymInt[")):
        return gpb.Arg(ints=gpb.Int64List(values=[r.choice(INTS[2:-2]) for _ in range(r.choice((0, 1, 2, 3, 4)))]))
    if core == "float":
        return gpb.Arg(f=r.choice(FLOATS))
    if core.startswith("float["):
        return gpb.Arg(floats=gpb.DoubleList(values=[r.choice(FLOATS) for _ in range(r.choice((0, 1, 2)))]))
    if core == "bool":
        return gpb.Arg(b=r.random() < 0.5)
    if core.startswith("bool["):
        return gpb.Arg(bools=gpb.BoolList(values=[r.random() < 0.5 for _ in range(r.choice((1, 2, 3, 4)))]))
    if core == "Scalar":
        return r.choice((gpb.Arg(i=r.choice(INTS)), gpb.Arg(f=r.choice(FLOATS)), gpb.Arg(b=False),
                         gpb.Arg(complex=gpb.ComplexValue(real=1.0, imag=-1.0)),
                         gpb.Arg(tensor=_ref(r, n_inputs, n_nodes))))
    if core == "Scalar[]":
        return gpb.Arg(floats=gpb.DoubleList(values=[r.choice(FLOATS) for _ in range(r.choice((0, 1, 2)))]))
    if core == "ScalarType":
        return gpb.Arg(dtype=r.choice(DTYPES) if r.random() < 0.9 else 77)
    if core == "Layout":
        return gpb.Arg(layout=r.choice((gpb.LAYOUT_STRIDED, gpb.LAYOUT_STRIDED, 5)))
    if core == "Device":
        if r.random() < 0.85:
            return gpb.Arg(device=sess.gdevice)
        return gpb.Arg(device=gpb.Device(kind=r.choice((gpb.DEVICE_CPU, gpb.DEVICE_CUDA)), index=r.choice((0, 3))))
    if core == "MemoryFormat":
        return gpb.Arg(memory_format=r.choice((1, 2, 3, 4, 9)))
    if core == "str":
        return gpb.Arg(enum_code=r.choice((0, 1, 2, 3, 99)))
    return gpb.Arg(none=True)


def random_case(r: random.Random, sess) -> Case:
    g = gpb.Graph(opset_version=1 if r.random() < 0.97 else r.choice((0, 2)), device=sess.gdevice)
    if r.random() < 0.02:
        g.device.index = 5
    K = gpb.GraphInput
    for _ in range(r.choice((0, 1, 2, 2, 3, 4))):
        roll = r.random()
        if roll < 0.8:
            g.inputs.append(K(kind=K.KIND_TENSOR, meta=_meta(r)))
        elif roll < 0.95:
            g.inputs.append(K(kind=K.KIND_SCALAR, meta=gpb.TensorMeta(
                dtype=r.choice((gpb.DTYPE_INT64, gpb.DTYPE_FLOAT64, gpb.DTYPE_BOOL, gpb.DTYPE_COMPLEX128,
                                gpb.DTYPE_FLOAT32)))))
        else:
            g.inputs.append(K(kind=r.choice((K.KIND_DATASET_REF, 9)), meta=_meta(r)))
    for i in range(r.choice((1, 1, 2, 3, 5))):
        roll = r.random()
        op_id = r.choice(OP_IDS) if roll < 0.95 else r.choice((0, max(OP_IDS) + 1, 2**31))
        op = OPS.get(op_id)
        specs = list(op.args) if op is not None else []
        n = len(specs)
        if r.random() < 0.05:
            n += r.choice((-1, 1))
        args = [_arg(r, specs[k].type if k < len(specs) else "Tensor", len(g.inputs), i, sess)
                for k in range(max(n, 0))]
        g.nodes.append(gpb.Node(op_id=op_id, args=args))
    last = len(g.nodes) - 1
    for k in range(r.choice((1, 1, 1, 2))):
        g.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=last, output=k)) if r.random() < 0.95
                         else _ref(r, len(g.inputs), len(g.nodes)))
    _set_output_meta(r, sess, g)
    return Case("random", g, _sources_for(r, g), _output_kinds(r, len(g.outputs)))


def _set_output_meta(r, sess, g):
    """The inferred output metadata most of the time, so a graph that is
    otherwise valid gets past V4; random metadata otherwise."""
    from reflop_worker import meta
    from reflop_worker.errors import ReflopError
    from reflop_worker.graph import Ref

    del g.output_meta[:]
    inferred = None
    if r.random() < 0.85:
        probe = gpb.Graph()
        probe.CopyFrom(g)
        del probe.outputs[:]
        probe.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=max(len(g.nodes) - 1, 0), output=0)))
        probe.output_meta.append(gpb.TensorMeta(dtype=gpb.DTYPE_FLOAT32, shape=[1]))
        try:
            prog = sess.builder.build(probe)
            prog.outputs, prog.output_meta = [], []
            meta.infer(prog, sess.limits)
            inferred = prog.metas
        except (ReflopError, Exception):  # noqa: BLE001 - any failure: fall back to random metadata
            inferred = None
    for o in g.outputs:
        m = None
        if inferred is not None and o.WhichOneof("ref") == "node_output":
            m = inferred.get(Ref(o.node_output.node, o.node_output.output))
        elif inferred is not None and o.WhichOneof("ref") == "graph_input":
            m = inferred.get(Ref(-1, o.graph_input))
        if isinstance(m, torch.Tensor) and m.dtype in TORCH_TO_PB:
            g.output_meta.append(gpb.TensorMeta(dtype=TORCH_TO_PB[m.dtype], shape=list(m.shape)))
        else:
            g.output_meta.append(_meta(r))
    if r.random() < 0.02 and g.output_meta:
        del g.output_meta[-1]


def _sources_for(r, g):
    """A binding source per input: a tensor that matches its metadata (with
    integer values around index bounds), a scalar, or None (no binding)."""
    out = []
    K = gpb.GraphInput
    for gi in g.inputs:
        if gi.kind == K.KIND_TENSOR and gi.meta.dtype in PB_TO_TORCH:
            shape = tuple(gi.meta.shape)
            if all(0 <= n <= 64 for n in shape) and len(shape) <= 8 and math.prod(shape) <= 4096:
                out.append(_tensor(r, PB_TO_TORCH[gi.meta.dtype], shape))
                continue
        if gi.kind == K.KIND_SCALAR:
            out.append(r.choice((1, -1, 0, 2.5, math.nan, True, complex(1, 2), 2**40)))
            continue
        out.append(None)
    return out


def _tensor(r, dt, shape):
    g = torch.Generator().manual_seed(r.randrange(2**31))
    if dt == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dt.is_floating_point:
        t = torch.randn(shape, generator=g)
        if t.numel() and r.random() < 0.1:
            t.view(-1)[0] = r.choice((math.nan, math.inf, -math.inf))
        return t.to(dt)
    if dt.is_complex:
        return torch.randn(shape, generator=g, dtype=dt)
    lo, hi = r.choice(((0, 3), (-3, 6), (0, 1), (-2**31, 2**31 - 1)))
    if dt == torch.uint8:
        lo, hi = 0, 255
    return torch.randint(lo, hi + 1, shape, generator=g).to(dt)


def _output_kinds(r, n):
    K = cpb.OutputSpec
    return [r.choice((K.KIND_RETURN, K.KIND_RETURN, K.KIND_RETURN, K.KIND_DISCARD, K.KIND_STORE)) for _ in range(n)]


# ------------------------------------------------------------------ composed


class _Pool:
    def __init__(self) -> None:
        from parity import all_cases

        self.calls = [c for c in all_cases() if not SPEC[c.op].name.startswith("reflop.")]
        self.seeds = [c for c in self.calls if c.label == "seed"]


_POOL: _Pool | None = None


def _pool() -> _Pool:
    global _POOL
    if _POOL is None:
        _POOL = _Pool()
    return _POOL


def composed_case(r: random.Random, sess) -> Case:
    """Chain parity seeds: node j's tensor arguments come from graph inputs
    (the seed's own tensors) or, where an earlier node output has the same
    dtype and shape, from that output."""
    # Mostly unmutated seeds (valid calls), so chains get past V4 and run.
    pool = _pool()
    calls = [r.choice(pool.seeds if r.random() < 0.7 else pool.calls) for _ in range(r.choice((1, 2, 2, 3, 4)))]
    g = gpb.Graph(opset_version=1, device=sess.gdevice)
    sources: list = []
    produced: list = []  # (node index, output index, dtype, shape)
    for call in calls:
        try:
            inputs, nodes = encode(sess, call)
        except Exception:  # noqa: BLE001 - a seed the encoder cannot express
            continue
        base = len(g.inputs)
        node_base = len(g.nodes)
        remap_inputs = {}
        for k, t in enumerate(inputs):
            match = [p for p in produced if p[2] == t.dtype and p[3] == tuple(t.shape)]
            if match and r.random() < 0.6:
                remap_inputs[k] = r.choice(match)
            else:
                remap_inputs[k] = len(g.inputs)
                g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_TENSOR,
                                               meta=gpb.TensorMeta(dtype=TORCH_TO_PB[t.dtype], shape=list(t.shape))))
                sources.append(t)
        for n in nodes:
            n = _relink(n, remap_inputs, node_base)
            g.nodes.append(n)
        last = len(g.nodes) - 1
        outs, err = infer_outputs(sess, call)
        if err is None:
            produced += [(last, k, dt, tuple(shape)) for k, (dt, shape) in enumerate(outs)]
        del base
    if not g.nodes:
        return random_case(r, sess)
    last = len(g.nodes) - 1
    picks = [(last, 0)] + [(p[0], p[1]) for p in r.sample(produced, min(len(produced), r.choice((0, 1, 2))))]
    for n, k in dict.fromkeys(picks):
        g.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=n, output=k)))
    _set_output_meta(r, sess, g)
    return Case("composed", g, sources, _output_kinds(r, len(g.outputs)))


def _relink(n: gpb.Node, remap_inputs: dict, node_base: int) -> gpb.Node:
    out = gpb.Node()
    out.CopyFrom(n)

    def fix(ref: gpb.ValueRef):
        which = ref.WhichOneof("ref")
        if which == "graph_input":
            tgt = remap_inputs[ref.graph_input]
            if isinstance(tgt, tuple):
                ref.Clear()
                ref.node_output.node, ref.node_output.output = tgt[0], tgt[1]
            else:
                ref.graph_input = tgt
        elif which == "node_output":
            ref.node_output.node += node_base

    for a in out.args:
        if a.WhichOneof("value") == "tensor":
            fix(a.tensor)
        elif a.WhichOneof("value") == "tensors":
            for it in a.tensors.items:
                if it.WhichOneof("ref"):
                    fix(it)
    return out


# ------------------------------------------------------------------ bytes


def bytes_case(r: random.Random, sess) -> Case:
    base = (random_case if r.random() < 0.5 else composed_case)(r, sess)
    env = wpb.WorkerEnvelope(client=cpb.Envelope(request_id=1, register_graph=cpb.RegisterGraph(graph=base.graph)))
    data = bytearray(env.SerializeToString())
    for _ in range(r.choice((1, 1, 2, 4, 8))):
        op = r.random()
        if not data:
            break
        i = r.randrange(len(data))
        if op < 0.4:
            data[i] ^= 1 << r.randrange(8)
        elif op < 0.6:
            data[i] = r.choice((0x00, 0xFF, 0x7F, 0x80))
        elif op < 0.75:
            data[i:i] = bytes(r.randrange(256) for _ in range(r.choice((1, 2, 5))))
        elif op < 0.9:
            del data[i:i + r.choice((1, 2, 8))]
        else:
            del data[i:]
    return Case("bytes", raw=bytes(data))


def case(seed: int, index: int, sess) -> Case:
    r = rng_for(seed, index)
    roll = r.random()
    if roll < 0.45:
        return random_case(r, sess)
    if roll < 0.85:
        return composed_case(r, sess)
    return bytes_case(r, sess)


_ = Call  # re-exported for corpus tools
