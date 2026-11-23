# SPDX-License-Identifier: Apache-2.0
"""Meta/kernel parity cases (#21): the worker must decide every op call the
way PyTorch's own kernel does.

Seed calls come from running small PyTorch programs on CPU tensors and
recording every aten call in the opset (`record`). Each seed is then
mutated one argument at a time (`mutations`): ints around each dim's
bounds, int lists of the wrong length or with negative, zero and huge
entries, optionals as None, tensors of another rank, size or dtype, and a
non-contiguous self. `check` runs the worker and the native kernel on the
same one-node graph and compares the outcome.

Run as a script for discovery (every case in a forked child, so a kernel
that crashes is reported instead of ending the run); run it inside a
memory-capped scope, a mutated size can ask for any amount of memory:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- \
        env PYTHONPATH=client/python:server/worker:tests/security/corpus \
        .venv/bin/python tests/security/corpus/parity.py cpu"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from helpers import NAME, I, decode, make_session, tensor_binding, tnode
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker.dtypes import TORCH_TO_PB
from reflop_worker.graph import MEMFMT, _base
from reflop_worker.ops_generated import FLAG_BANNED, FLAG_CLIENT, FLAG_DDS, FLAG_RNG, FLAG_UNINIT, OPS
from torch.utils._python_dispatch import TorchDispatchMode

A = torch.ops.aten
SPEC = {o.name: o for o in OPS.values()}
SKIP_FLAGS = FLAG_RNG | FLAG_DDS | FLAG_CLIENT | FLAG_BANNED
HUGE = 2**31
# A mutated call whose output would have more elements than this is not run
# natively (the worker's own limits decide it; nothing to compare).
MAX_NATIVE_NUMEL = 1 << 21


@dataclass
class Call:
    op: str
    args: list                     # one value per schema argument, in schema order
    label: str = "seed"
    transposed: bool = False       # args[0] goes through a transpose node first


def _name(func) -> str:
    ns, base = func._schema.name.split("::")
    return f"{ns}.{base}.{func._overloadname}"


class _Recorder(TorchDispatchMode):
    def __init__(self, out: list) -> None:
        super().__init__()
        self.out = out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = _name(func)
        spec = SPEC.get(name)
        if spec is not None and spec.fn is not None and not spec.flags & SKIP_FLAGS:
            vals = []
            for i, sa in enumerate(func._schema.arguments):
                if i < len(args):
                    v = args[i]
                elif sa.name in kwargs:
                    v = kwargs[sa.name]
                else:
                    v = sa.default_value
                vals.append(_snap(v))
            self.out.append(Call(name, vals))
        return func(*args, **kwargs)


def _snap(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone().contiguous()
    if isinstance(v, (list, tuple)):
        return [_snap(x) for x in v]
    return v


def record(programs) -> list[Call]:
    # The autograd engine sizes its per-device queues at the first backward
    # of the process; register the reflop device first, or backward on
    # reflop tensors fails later in the same test run.
    from reflop.device import register

    register()
    out: list[Call] = []
    for p in programs:
        with _Recorder(out):
            p()
    seen, uniq = set(), []
    for c in out:
        key = (c.op, _sig(c.args))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _sig(v):
    if isinstance(v, torch.Tensor):
        return ("T", tuple(v.shape), str(v.dtype))
    if isinstance(v, list):
        return tuple(_sig(x) for x in v)
    if isinstance(v, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return str(v)
    return v


# ------------------------------------------------------------------ encoding


class Unencodable(Exception):
    pass


def encode(sess, call: Call):
    """The one-node graph for `call`: (inputs metas, input tensors, node args)."""
    spec = SPEC[call.op]
    inputs: list[torch.Tensor] = []

    def tensor_ref(t):
        inputs.append(t)
        return gpb.ValueRef(graph_input=len(inputs) - 1)

    gargs = []
    for k, (sa, v) in enumerate(zip(spec.args, call.args)):
        t = _base(sa.type)
        core = t.rstrip("?")
        if v is None:
            gargs.append(gpb.Arg(none=True))
        elif core == "Tensor":
            if isinstance(v, torch.Tensor):
                if k == 0 and call.transposed:
                    inputs.append(v.transpose(0, -1).contiguous())
                    gargs.append(None)  # patched to the transpose node's output below
                else:
                    gargs.append(gpb.Arg(tensor=tensor_ref(v)))
            elif isinstance(v, bool):
                gargs.append(gpb.Arg(b=v))
            elif isinstance(v, int):
                gargs.append(gpb.Arg(i=v))
            elif isinstance(v, float):
                gargs.append(gpb.Arg(f=v))
            else:
                raise Unencodable(type(v))
        elif core in ("Tensor[]", "Tensor?[]"):
            gargs.append(gpb.Arg(tensors=gpb.TensorList(
                items=[gpb.ValueRef() if x is None else tensor_ref(x) for x in v])))
        elif core in ("int", "SymInt"):
            gargs.append(gpb.Arg(i=int(v)))
        elif core.startswith(("int[", "SymInt[")):
            gargs.append(gpb.Arg(ints=gpb.Int64List(values=[int(x) for x in (v if isinstance(v, list) else [v])])))
        elif core == "float":
            gargs.append(gpb.Arg(f=float(v)))
        elif core.startswith("float["):
            gargs.append(gpb.Arg(floats=gpb.DoubleList(values=[float(x) for x in v])))
        elif core == "bool":
            gargs.append(gpb.Arg(b=bool(v)))
        elif core.startswith("bool["):
            gargs.append(gpb.Arg(bools=gpb.BoolList(values=[bool(x) for x in v])))
        elif core == "Scalar":
            if isinstance(v, bool):
                gargs.append(gpb.Arg(b=v))
            elif isinstance(v, int):
                gargs.append(gpb.Arg(i=v))
            elif isinstance(v, float):
                gargs.append(gpb.Arg(f=v))
            else:
                raise Unencodable(type(v))
        elif core == "Scalar[]":
            gargs.append(gpb.Arg(floats=gpb.DoubleList(values=[float(x) for x in v])))
        elif core == "ScalarType":
            gargs.append(gpb.Arg(dtype=TORCH_TO_PB[v]))
        elif core == "Layout":
            gargs.append(gpb.Arg(layout=gpb.LAYOUT_STRIDED))
        elif core == "Device":
            gargs.append(gpb.Arg(device=sess.gdevice))
        elif core == "MemoryFormat":
            gargs.append(gpb.Arg(memory_format={f: k for k, f in MEMFMT.items()}[v]))
        elif core == "str":
            gargs.append(gpb.Arg(enum_code=list(spec.enums[sa.name]).index(v)))
        else:
            raise Unencodable(sa.type)
    nodes = []
    if call.transposed:
        nodes.append(gpb.Node(op_id=NAME["aten.transpose.int"],
                              args=[gpb.Arg(tensor=gpb.ValueRef(graph_input=0)), I(0), I(-1)]))
        gargs[0] = tnode(0)
    nodes.append(gpb.Node(op_id=spec.id, args=gargs))
    return inputs, nodes


def graph_of(sess, inputs, nodes, outs):
    g = gpb.Graph(opset_version=1, device=sess.gdevice)
    for t in inputs:
        g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_TENSOR,
                                       meta=gpb.TensorMeta(dtype=TORCH_TO_PB[t.dtype], shape=list(t.shape))))
    g.nodes.extend(nodes)
    last = len(nodes) - 1
    for k, (dt, shape) in enumerate(outs):
        g.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=last, output=k)))
        g.output_meta.append(gpb.TensorMeta(dtype=TORCH_TO_PB[dt], shape=list(shape)))
    return g


# ------------------------------------------------------------------ native side


def native_args(call: Call, device="cpu"):
    out = []
    for k, (sa, v) in enumerate(zip(SPEC[call.op].args, call.args)):
        if _base(sa.type).rstrip("?") == "Device":
            # Factories record device=None: place them explicitly, or the
            # meta probe below would allocate on the CPU.
            v = torch.device(device)
        elif isinstance(v, torch.Tensor):
            v = v.to(device)
            if k == 0 and call.transposed:
                v = v.transpose(0, -1).contiguous().transpose(0, -1)
        elif isinstance(v, list):
            v = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        out.append(v)
    return out


def _kw(call: Call, vals):
    spec = SPEC[call.op]
    pos, kw = [], {}
    for sa, v in zip(spec.args, vals):
        if sa.kwarg_only:
            kw[sa.name] = v
        else:
            pos.append(v)
    return pos, kw


def flat_tensors(out):
    return [x for x in torch.utils._pytree.tree_flatten(out)[0] if isinstance(x, torch.Tensor)]


NATIVE_REJECTS = object()
TOO_BIG = object()


def native(call: Call, device="cpu"):
    """The native result as a list of CPU tensors, NATIVE_REJECTS, or TOO_BIG."""
    spec = SPEC[call.op]
    # Meta first: a mutated size can ask for terabytes.
    margs = native_args(call, "meta")
    pos, kw = _kw(call, margs)
    try:
        m = flat_tensors(spec.fn(*pos, **kw))
        if any(t.numel() > MAX_NATIVE_NUMEL for t in m):
            return TOO_BIG
    except Exception:  # noqa: BLE001
        if _max_int(call.args) > 4096:
            return TOO_BIG
    pos, kw = _kw(call, native_args(call, device))
    try:
        return [t.cpu() for t in flat_tensors(spec.fn(*pos, **kw))]
    except Exception:  # noqa: BLE001 - PyTorch rejects this call
        return NATIVE_REJECTS


def _max_int(v):
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return abs(v)
    if isinstance(v, list):
        return max((_max_int(x) for x in v), default=0)
    return 0


# ------------------------------------------------------------------ worker side


def _call(sess, env):
    out = []
    sess.handle(env, out.append)
    return out[0]


def infer_outputs(sess, call: Call):
    """The worker's decision at registration: (output metas, None) when
    stages V2-V4 accept the graph, else (None, the ReflopError)."""
    from reflop_worker import meta
    from reflop_worker.errors import ReflopError

    inputs, nodes = encode(sess, call)
    g = graph_of(sess, inputs, nodes, DUMMY)
    try:
        prog = sess.builder.build(g)
        prog.outputs, prog.output_meta = [], []
        meta.infer(prog, sess.limits)
    except ReflopError as e:
        return None, e
    last = len(nodes) - 1
    return [(m.dtype, tuple(m.shape)) for k in range(prog.nodes[last].outputs)
            for m in [prog.metas[_ref(last, k)]]], None


def _ref(n, k):
    from reflop_worker.graph import Ref
    return Ref(n, k)


def worker(sess, call: Call, outs):
    """('rejected'|'failed'|'ran', the error or the returned tensors)."""
    inputs, nodes = encode(sess, call)
    g = graph_of(sess, inputs, nodes, outs)
    r = _call(sess, cpb.Envelope(request_id=1, register_graph=cpb.RegisterGraph(graph=g)))
    if r.WhichOneof("body") == "error":
        return "rejected", r.error
    ex = cpb.Execute(graph_hash=r.graph_registered.graph_hash)
    for t in inputs:
        ex.bindings.append(tensor_binding(t))
    for _ in g.outputs:
        ex.outputs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_RETURN))
    e = _call(sess, cpb.Envelope(request_id=2, execute=ex))
    if e.WhichOneof("body") == "error":
        return "failed", e.error
    return "ran", [decode(t) for t in e.execute_result.returned]


DUMMY = [(torch.float32, (7, 11, 13))]
# Ops the worker checks more strictly than PyTorch does, on purpose: their
# CPU kernels accept inconsistent shapes and then read or write out of
# bounds, so the worker requires the shapes autograd passes (ADR-040/042).
STRICT = {
    "aten._adaptive_avg_pool2d_backward.default", "aten._adaptive_avg_pool3d_backward.default",
    "aten.adaptive_max_pool2d_backward.default", "aten.adaptive_max_pool3d_backward.default",
    "aten.max_pool2d_with_indices_backward.default", "aten.max_pool3d_with_indices_backward.default",
    "aten.avg_pool2d_backward.default", "aten.avg_pool3d_backward.default",
    "aten._log_softmax_backward_data.default", "aten._softmax_backward_data.default",
    "aten._weight_norm_interface.default", "aten._weight_norm_interface_backward.default",
    "aten._embedding_bag.default", "aten._embedding_bag_backward.default",
    "aten._embedding_bag_dense_backward.default", "aten.embedding_dense_backward.default",
    "aten.select_backward.default", "aten.diagonal_backward.default", "aten.slice_backward.default",
}
# Runtime checks that run before any kernel launch (I4): a call PyTorch
# rejects may pass V4 when the reason depends on the data.
BEFORE_KERNEL = (cpb.STAGE_E1_BINDINGS, cpb.STAGE_E2_RUNTIME)


def check(sess, call: Call, device="cpu", mark=lambda s: None):
    """None when the worker decides `call` the way PyTorch does, else why not.

    The worker decides first; `mark("rejected")` runs once it has refused
    the call without launching a kernel, before the native kernel runs. A
    rejection by a runtime check (E1/E2: index bounds, I4) is always
    acceptable: PyTorch's CPU kernels skip some of those checks and read out
    of bounds instead. A STRICT op's native kernel is not run on a call the
    worker refuses: it may crash on it."""
    outs, err = infer_outputs(sess, call)
    if err is not None:
        mark("rejected")
        if call.op in STRICT:
            return None
        want = native(call, device)
        if isinstance(want, list):
            return f"false rejection: {cpb.ValidationStage.Name(err.stage)} {err.message[:300]}"
        return None
    if any(math.prod(shape) > MAX_NATIVE_NUMEL for _, shape in outs):
        return None  # too big to run on either side; the Owner's rails decide it
    kind, got = worker(sess, call, outs)
    if kind == "failed" and got.stage in BEFORE_KERNEL:
        mark("rejected")
        return None
    want = native(call, device)
    if want is TOO_BIG:
        return None
    if want is NATIVE_REJECTS:
        if kind == "failed":
            return f"validation passed; the kernel raised (I3): {got.message[:300]}"
        return "PyTorch rejects; the worker ran it"
    if kind != "ran":
        return f"PyTorch accepts; the worker failed: {cpb.ValidationStage.Name(got.stage)} {got.message[:300]}"
    if [(t.dtype, tuple(t.shape)) for t in want] != outs:
        return f"meta shapes {outs} differ from the kernel's {[(t.dtype, tuple(t.shape)) for t in want]}"
    if SPEC[call.op].flags & FLAG_UNINIT:
        return None
    tol = 1e-5 if device == "cpu" else 1e-3
    for w, g in zip(want, got):
        if w.dtype.is_floating_point or w.dtype.is_complex:
            if not torch.allclose(g, w, rtol=tol, atol=tol / 10, equal_nan=True):
                return "values differ"
        elif not torch.equal(g, w):
            return "values differ"
    return None


KINDS = {"I3": "the kernel raised (I3)", "false-rejection": "false rejection", "worker-failed": "PyTorch accepts",
         "meta-shapes": "meta shapes", "values": "values differ", "ran": "the worker ran it",
         "crash": "CRASHED"}


def kind_of(why: str) -> str:
    return next(k for k, marker in KINDS.items() if marker in why)


def check_isolated(sess, call: Call, timeout: float = 60.0):
    """check() in a forked child, so a kernel that crashes the process is
    reported instead of ending the run. A crash after the worker rejected
    the call is PyTorch's kernel crashing on input the worker refuses."""
    import os
    import select
    import signal

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            torch.set_num_threads(1)
            why = check(sess, call, mark=lambda s: os.write(w, b"R"))
            os.write(w, b"." + ("" if why is None else why).encode())
        finally:
            os._exit(0)
    os.close(w)
    chunks = []
    while True:
        ready, _, _ = select.select([r], [], [], timeout)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            break
        b = os.read(r, 1 << 16)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    _, status = os.waitpid(pid, 0)
    out = b"".join(chunks).decode()
    rejected = out.startswith("R")
    out = out[1:] if rejected else out
    if os.WIFSIGNALED(status) or not out.startswith("."):
        return None if rejected else f"validation passed; the kernel CRASHED (status {status})"
    return out[1:] or None


# ------------------------------------------------------------------ mutations


def _tensors(call):
    return [(k, v) for k, v in enumerate(call.args) if isinstance(v, torch.Tensor)]


def _ints_for(call):
    ts = [v for _, v in _tensors(call)]
    rank = max((t.dim() for t in ts), default=1)
    size = max((max(t.shape, default=1) for t in ts), default=1)
    return rank, size


INT64 = (-(2**63), 2**63 - 1)


def _int_values(v, rank, size):
    vals = {v - 1, v + 1, -rank - 1, -rank, -1, 0, rank - 1, rank, size, size + 1, -size - 1, HUGE} - {v}
    return sorted(x for x in vals if INT64[0] <= x <= INT64[1])


def _retensor(t: torch.Tensor, shape, dtype=None, lo_hi=None):
    dtype = dtype or t.dtype
    g = torch.Generator().manual_seed(hash((tuple(shape), str(dtype))) & 0xFFFF)
    if dtype == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g).to(dtype)
    lo, hi = lo_hi if lo_hi else ((int(t.min()), int(t.max())) if t.numel() else (0, 1))
    return torch.randint(lo, hi + 1, shape, generator=g).to(dtype)


def mutations(seed: Call):
    spec = SPEC[seed.op]
    rank, size = _ints_for(seed)
    out = []

    def mut(k, v, label):
        args = list(seed.args)
        args[k] = v
        out.append(Call(seed.op, args, label))

    for k, (sa, v) in enumerate(zip(spec.args, seed.args)):
        core = _base(sa.type).rstrip("?")
        opt = _base(sa.type).endswith("?")
        if opt and v is not None:
            mut(k, None, f"{sa.name}=None")
        if core in ("int", "SymInt") and isinstance(v, int) and not isinstance(v, bool):
            for x in _int_values(v, rank, size):
                mut(k, x, f"{sa.name}={x}")
        elif core.startswith(("int[", "SymInt[")) and v is not None:
            v = v if isinstance(v, list) else [v]
            for i, e in enumerate(v):
                for x in sorted(x for x in {-1, 0, e + 1, e - 1, -e - 1, HUGE} - {e} if INT64[0] <= x <= INT64[1]):
                    mut(k, v[:i] + [x] + v[i + 1:], f"{sa.name}[{i}]={x}")
            if v:
                mut(k, v[:-1], f"{sa.name} shorter")
                mut(k, v + [v[-1]], f"{sa.name} longer")
            mut(k, v + [1, 1], f"{sa.name} +2")
            mut(k, [], f"{sa.name}=[]")
        elif core == "bool" and isinstance(v, bool):
            mut(k, not v, f"{sa.name}={not v}")
        elif core == "Tensor" and isinstance(v, torch.Tensor):
            out.extend(_tensor_mutations(seed, k, v, sa.name))
        elif core in ("Tensor[]", "Tensor?[]") and v:
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    for c in _tensor_mutations(Call(seed.op, [t]), 0, t, f"{sa.name}[{i}]"):
                        mut(k, v[:i] + [c.args[0]] + v[i + 1:], c.label)
            mut(k, v[:-1], f"{sa.name} shorter")
    ts = _tensors(seed)
    if ts and ts[0][0] == 0 and seed.args[0].dim() >= 2:
        out.append(Call(seed.op, list(seed.args), "self non-contiguous", transposed=True))
    return out


def _tensor_mutations(seed, k, t, name):
    res = []

    def mk(new, label):
        args = list(seed.args)
        args[k] = new
        res.append(Call(seed.op, args, f"{name} {label}"))

    s = list(t.shape)
    mk(_retensor(t, [1] + s), "unsqueezed")
    if s:
        mk(_retensor(t, s[1:]), "rank-1")
        mk(_retensor(t, s[:-1] + [s[-1] + 1]), "last+1")
        if s[-1] > 1:
            mk(_retensor(t, s[:-1] + [s[-1] - 1]), "last-1")
        mk(_retensor(t, s[:-1] + [0]), "last=0")
        if len(s) > 1:
            mk(_retensor(t, [s[0] + 1] + s[1:]), "first+1")
    other = {torch.float32: torch.float64, torch.float64: torch.float32, torch.int64: torch.int32,
             torch.int32: torch.int64, torch.float16: torch.float32, torch.bfloat16: torch.float32,
             torch.bool: torch.float32}.get(t.dtype)
    if other is not None:
        mk(t.to(other), f"as {other}")
    return res


# ------------------------------------------------------------------ seed programs


def _g(*shape, seed=0, dtype=torch.float32, grad=False):
    t = torch.randn(shape, generator=torch.Generator().manual_seed(seed + 7 * len(shape))).to(dtype)
    return t.requires_grad_() if grad else t


def _bk(f, *shapes):
    """f(*inputs) forward and backward on fresh leaf tensors."""
    def p():
        xs = [_g(*s, seed=i, grad=True) for i, s in enumerate(shapes)]
        y = f(*xs)
        ys = flat_tensors(y)
        sum(z.float().sum() for z in ys if z.requires_grad).backward()
    return p


def _fw(f, *shapes):
    return lambda: f(*[_g(*s, seed=i) for i, s in enumerate(shapes)])


IDX = torch.tensor([[0, 2], [1, 1], [3, 0]])


def _bag_dense_backward(mode):
    w, idx, off = _g(6, 4), torch.tensor([1, 2, 4, 5, 4, 3]), torch.tensor([0, 2, 4])
    out, o2b, bag_size, maxi = A._embedding_bag.default(w, idx, off, False, mode)
    A._embedding_bag_dense_backward.default(torch.ones_like(out), idx, o2b, bag_size, maxi, 6, False, mode, None, -1)

PROGRAMS = [
    # reductions over dims
    _bk(lambda x: (x.sum(1), x.mean((0, 2), keepdim=True), x.prod(1), x.amax(-1), x.amin(0),
                   x.var(1), x.std(0, correction=0), x.logsumexp(2), x.norm(dim=1),
                   torch.linalg.vector_norm(x, 3, dim=(0, 1)), x.nansum(1), x.cumsum(1), x.cumprod(2),
                   x.logcumsumexp(0), torch.var_mean(x, 1)[0], torch.std_mean(x, 2)[1]), (3, 4, 5)),
    _fw(lambda x: (x.argmax(1), x.argmin(), x.max(1), x.min(0), x.sort(1), x.topk(2, 1), x.kthvalue(2, 2),
                   x.median(1), x.nanmedian(0), x.mode(1), x.cummax(1), x.cummin(0), x.aminmax(dim=1),
                   (x > 0).any(1), (x > 0).all(0), (x > 0).any((0, 1)), (x > 0).all((1,)),
                   torch.nonzero_static(x > 0, size=5)), (3, 4, 5)),
    _bk(lambda x: (x.softmax(1), x.log_softmax(-1), F.glu(x, 1)), (3, 4, 5)),
    # conv, pool, padding, upsample
    _bk(lambda x, w: F.conv2d(x, w, padding=1, stride=2), (2, 3, 6, 6), (4, 3, 3, 3)),
    _bk(lambda x, w: F.conv1d(x, w, groups=2, dilation=2), (2, 4, 9), (6, 2, 3)),
    _bk(lambda x, w: F.conv_transpose2d(x, w, stride=2), (1, 2, 4, 4), (2, 3, 3, 3)),
    _bk(lambda x: (F.max_pool2d(x, 2), F.avg_pool2d(x, 3, 1, 1), F.adaptive_avg_pool2d(x, 2),
                   F.adaptive_max_pool2d(x, 3), F.avg_pool1d(x[0], 2), F.adaptive_avg_pool1d(x[0], 3)),
        (2, 3, 6, 6)),
    _bk(lambda x: (F.max_pool3d(x, 2), F.avg_pool3d(x, 2), F.adaptive_avg_pool3d(x, 2),
                   F.adaptive_max_pool3d(x, 2)), (1, 2, 4, 4, 4)),
    _fw(lambda x: F.max_unpool2d(*F.max_pool2d(x, 2, return_indices=True), 2), (1, 2, 4, 4)),
    _fw(lambda x: F.max_unpool3d(*F.max_pool3d(x, 2, return_indices=True), 2), (1, 1, 4, 4, 4)),
    _bk(lambda x: (F.pad(x, (1, 2, 0, 1)), F.pad(x, (1, 1, 1, 1), mode="reflect"),
                   F.pad(x, (2, 2, 0, 1), mode="replicate"), F.pad(x[0], (1, 2), mode="reflect")), (1, 2, 4, 4)),
    _bk(lambda x: (F.pad(x, (1, 1, 1, 1, 1, 1), mode="reflect"), F.pad(x, (1, 0, 1, 0, 0, 1), mode="replicate")),
        (1, 1, 3, 3, 3)),
    _bk(lambda x: (F.interpolate(x, scale_factor=2), F.interpolate(x, size=(5, 7), mode="bilinear"),
                   F.interpolate(x, size=(3, 5), mode="bicubic", align_corners=True),
                   F.interpolate(x, size=(5, 5), mode="nearest-exact"),
                   F.interpolate(x, size=(6, 3), mode="bilinear", antialias=True)), (1, 2, 3, 4)),
    _bk(lambda x: (F.interpolate(x, size=5), F.interpolate(x, size=7, mode="linear"),
                   F.interpolate(x, size=6, mode="nearest-exact")), (1, 2, 4)),
    _bk(lambda x: (F.interpolate(x, size=(3, 4, 5)), F.interpolate(x, size=(3, 3, 3), mode="trilinear"),
                   F.interpolate(x, size=(2, 3, 4), mode="nearest-exact")), (1, 1, 2, 2, 2)),
    _fw(lambda x: (A.upsample_nearest2d.default(x, [4, 6]), A.upsample_bilinear2d.default(x, [4, 6], False),
                   A.upsample_nearest1d.default(x[0], [5]), A.upsample_linear1d.default(x[0], [5], True),
                   A.upsample_bicubic2d.default(x, [5, 5], False),
                   A._upsample_nearest_exact2d.default(x, [5, 5]),
                   A._upsample_nearest_exact1d.default(x[0], [6]),
                   A._upsample_bilinear2d_aa.default(x, [2, 2], False)), (1, 2, 3, 4)),
    _fw(lambda x: (A.upsample_nearest3d.default(x, [3, 4, 5]), A.upsample_trilinear3d.default(x, [3, 3, 3], False),
                   A._upsample_nearest_exact3d.default(x, [3, 3, 3])), (1, 1, 2, 2, 2)),
    _bk(lambda x: F.affine_grid(x, [2, 3, 4, 5], align_corners=False), (2, 2, 3)),
    _bk(lambda x: F.fold(F.unfold(x, 2, stride=1, padding=1), (5, 5), 2, stride=1, padding=1), (1, 2, 5, 5)),
    _bk(lambda x, w, b: torch._weight_norm(w, b, 0) @ x, (3, 2), (4, 3), (4, 1)),
    # shape and views
    _bk(lambda x: (x.select(1, 2), x[:, 1:3], x[..., ::2], x.view(12, 5), x.expand(2, 3, 4, 5), x.permute(2, 0, 1),
                   x.unsqueeze(1), x[:, :1].squeeze(1), x[:1, :1].squeeze((0, 1)), x.diagonal(1, 0, 2),
                   x.split([1, 3], 1), x.split(2, 2), x.chunk(2, 0), x.flip(0, 2), x.repeat(1, 2, 1),
                   x.transpose(0, 2), x.unfold(2, 2, 1), x.narrow(1, 1, 2), x.unbind(1), x.tile(2, 1),
                   x.roll(1, 2), x.roll((1, -1), (0, 1)), x.rot90(1, (1, 2)), x.tril(1), x.triu(-1),
                   x.as_strided((2, 2), (5, 1), 3), torch.diag_embed(x, 1, 0, 2),
                   x.repeat_interleave(2, 1)), (3, 4, 5)),
    _bk(lambda a, b: (torch.cat([a, b], 1), torch.stack([a, b], 2)), (2, 3), (2, 3)),
    _fw(lambda x: (A.unsqueeze_copy.default(x, 1), A.view_copy.default(x, [4, 15]),
                   A.permute_copy.default(x, [1, 2, 0]), A.expand_copy.default(x[:1], [2, 4, 5]),
                   A.transpose_copy.int(x, 0, 1), A.squeeze_copy.dims(x[:1], [0]),
                   A.slice_copy.Tensor(x, 1, 1, 3, 1), A.select_copy.int(x, 2, 1),
                   A.split_with_sizes_copy.default(x, [2, 2], 1), A.unbind_copy.int(x, 0),
                   A.diagonal_copy.default(x, 0, 1, 2), A.unfold_copy.default(x, 1, 2, 2),
                   A._unsafe_view.default(x, [60])), (3, 4, 5)),
    _fw(lambda x: (A.full.default([2, 3], 1.5), A.zeros.default([3]), A.ones.default([2, 2]),
                   A.new_zeros.default(x, [2]), A.new_ones.default(x, [3, 1]), A.new_full.default(x, [2], 2.0),
                   A.arange.default(5), A.arange.start(1, 6), A.arange.start_step(0, 10, 3),
                   A.linspace.default(0, 1, 5), A.logspace.default(0, 2, 4, 10.0),
                   A.empty.memory_format([2, 3]), A.empty_strided.default([2, 3], [3, 1]),
                   A.new_empty.default(x, [4]), A.new_empty_strided.default(x, [2, 2], [2, 1])), (3, 4)),
    # indexing
    _bk(lambda x: (x.gather(1, IDX), x.index_select(1, IDX[0]), x[IDX[:, 0]], x[:, IDX[0]],
                   x.take(IDX), torch.zeros(3, 5).scatter_add(1, IDX, x[:3, :2]),
                   torch.zeros(3, 5).scatter(1, IDX, x[:3, :2]), x.scatter(1, IDX, 1.5),
                   x[:3].scatter_reduce(1, IDX, x[:3, :2], "amax"), x.index_add(1, IDX[0], x[:, :2]),
                   x.index_copy(0, IDX[:2, 0], x[:2]), x.index_fill(1, IDX[0], -1.0),
                   x.index_fill(1, IDX[0], torch.tensor(2.0)),
                   x.index_put((IDX[:, 0], IDX[:, 1]), torch.ones(3), accumulate=True)), (4, 5)),
    _bk(lambda w: F.embedding(IDX, w, padding_idx=0), (6, 4)),
    _bk(lambda w: F.embedding_bag(IDX.reshape(-1), w, torch.tensor([0, 2, 4]), mode="mean"), (6, 4)),
    _bk(lambda w: F.embedding_bag(IDX.reshape(-1), w, torch.tensor([0, 2, 4]), mode="sum"), (6, 4)),
    _fw(lambda x: (A._unsafe_index.Tensor(x, [IDX[:, 0]]),
                   A._unsafe_index_put.default(x, [IDX[:, 0]], torch.ones(3, 5), False)), (4, 5)),
    lambda: _bag_dense_backward(1),
    lambda: _bag_dense_backward(2),
    # misc
    _bk(lambda a, b, w: (torch.lerp(a, b, 0.3), torch.lerp(a, b, w.sigmoid())), (3, 4), (3, 4), (3, 4)),
]


def seeds():
    return record(PROGRAMS)


def all_cases():
    out = []
    for s in seeds():
        out.append(s)
        out.extend(mutations(s))
    return out


if __name__ == "__main__":  # discovery: print every disagreement
    import collections
    import sys

    sess = make_session(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    bad = collections.defaultdict(list)
    for c in all_cases():
        why = check_isolated(sess, c)
        if why:
            bad[c.op].append(f"{c.label}: {why}")
    for op, items in sorted(bad.items()):
        print(f"== {op} ({len(items)}): {sorted({kind_of(w) for w in items})}")
        for it in items[:12]:
            print("   ", it)
