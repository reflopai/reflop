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
import types
import zlib
from dataclasses import dataclass

import torch
from helpers import NAME, I, decode, make_session, tensor_binding, tnode
from reflop_worker import execute, natives
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker.dtypes import TORCH_TO_PB
from reflop_worker.graph import MEMFMT, _base
from reflop_worker.ops_generated import FLAG_ALLOC, FLAG_BANNED, FLAG_CLIENT, FLAG_DDS, FLAG_RNG, FLAG_UNINIT, OPS
from torch.utils._python_dispatch import TorchDispatchMode

A = torch.ops.aten
SPEC = {o.name: o for o in OPS.values()}
SKIP_FLAGS = FLAG_DDS | FLAG_CLIENT | FLAG_BANNED
HUGE = 2**31
# A mutated call whose output would have more elements than this is not run
# natively (the worker's own limits decide it; nothing to compare).
MAX_NATIVE_NUMEL = 1 << 21


@dataclass
class Call:
    op: str
    args: list                     # one value per schema argument, in schema order
    label: str = "seed"
    transposed: int | None = None  # this tensor argument goes through a transpose node first
    lifted: int | None = None      # this float argument is bound as a scalar graph input
    value: bool = False            # a float-value mutation of an op that does not allocate from it
    fp8: bool = False              # a seed with only dtypes changed to float8: as safe to run as the seed


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
                    if _base(spec.args[i].type).rstrip("?") == "ScalarType" and type(v) is int:
                        v = SCALAR_TYPES[v]  # the schema default is c10's enum value
                vals.append(_snap(v))
            self.out.append(Call(name, vals))
        return func(*args, **kwargs)


# c10::ScalarType's first values, for schema defaults such as dtype=4.
SCALAR_TYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32,
                torch.float64, torch.complex32, torch.complex64, torch.complex128, torch.bool]


def _snap(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().clone().contiguous()  # native_args places it
    if isinstance(v, (list, tuple)):
        return [_snap(x) for x in v]
    return v


def record(programs, register_device=True) -> list[Call]:
    # The autograd engine sizes its per-device queues at the first backward
    # of the process; register the reflop device first, or backward on
    # reflop tensors fails later in the same test run.
    if register_device:
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


def record_cuda(programs: str = "CUDA_PROGRAMS") -> list[Call]:
    """record() of parity_programs.<programs> in a child process: once the
    reflop device is registered, a local CUDA backward fails (BUGS.md #28),
    and a backward before it breaks reflop backward (record)."""
    import os
    import subprocess
    import sys
    import tempfile

    code = ("import sys, torch, parity, parity_programs as p; "
            "calls = parity.record(getattr(p, sys.argv[1]), register_device=False); "
            "torch.save([(c.op, parity._tag(c.args)) for c in calls], sys.argv[2])")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "calls.pt")
        subprocess.run([sys.executable, "-W", "ignore", "-c", code, programs, path], env=env, check=True,
                       timeout=600)
        return [Call(op, _untag(args)) for op, args in torch.load(path, weights_only=True)]


_TAG = "torch."


def _tag(v):
    # weights_only loads no memory_format or layout; name them instead.
    if isinstance(v, list):
        return [_tag(x) for x in v]
    if isinstance(v, (torch.dtype, torch.memory_format, torch.layout)):
        return _TAG + str(v).removeprefix("torch.")
    if isinstance(v, torch.device):
        return _TAG + "device:" + str(v)
    return v


def _untag(v):
    if isinstance(v, list):
        return [_untag(x) for x in v]
    if isinstance(v, str) and v.startswith(_TAG):
        name = v.removeprefix(_TAG)
        return torch.device(name.removeprefix("device:")) if name.startswith("device:") else getattr(torch, name)
    return v


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
        if k == call.lifted:
            # The client's path for a finite Python float (ADR-011): a
            # scalar graph input, a placeholder until the execute binds it.
            inputs.append(float(v))
            gargs.append(gpb.Arg(tensor=gpb.ValueRef(graph_input=len(inputs) - 1)))
        elif v is None:
            gargs.append(gpb.Arg(none=True))
        elif core == "Tensor":
            if isinstance(v, torch.Tensor):
                if k == call.transposed:
                    tin = len(inputs)
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
            elif isinstance(v, complex):
                gargs.append(gpb.Arg(complex=gpb.ComplexValue(real=v.real, imag=v.imag)))
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
    if call.transposed is not None:
        nodes.append(gpb.Node(op_id=NAME["aten.transpose.int"],
                              args=[gpb.Arg(tensor=gpb.ValueRef(graph_input=tin)), I(0), I(-1)]))
        gargs[call.transposed] = tnode(0)
    nodes.append(gpb.Node(op_id=spec.id, args=gargs))
    return inputs, nodes


def graph_of(sess, inputs, nodes, outs):
    g = gpb.Graph(opset_version=1, device=sess.gdevice)
    for t in inputs:
        if isinstance(t, float):
            g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_SCALAR,
                                           meta=gpb.TensorMeta(dtype=gpb.DTYPE_FLOAT64)))
            continue
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
            v = v.to(device, copy=True)  # in-place ops must not change the seed
            if k == call.transposed:
                v = v.transpose(0, -1).contiguous().transpose(0, -1)
        elif isinstance(v, list):
            v = [x.to(device, copy=True) if isinstance(x, torch.Tensor) else x for x in v]
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


def _extent(shape) -> int:
    # An empty output's kernel can still allocate by its other sizes (cost.WORKSPACE).
    return math.prod(d or 1 for d in shape)


NATIVE_REJECTS = object()
TOO_BIG = object()


def native(call: Call, device="cpu"):
    """The native result as a list of CPU tensors, NATIVE_REJECTS, or TOO_BIG."""
    spec = SPEC[call.op]
    # Meta first: a mutated size can ask for terabytes.
    margs = native_args(call, "meta")
    pos, kw = _kw(call, margs)
    try:
        m = flat_tensors(_run(spec, pos, kw, "meta"))
        if any(_extent(t.shape) > MAX_NATIVE_NUMEL for t in m):
            return TOO_BIG
    except Exception:  # noqa: BLE001
        if _max_int(call.args) > 4096:
            return TOO_BIG
    pos, kw = _kw(call, native_args(call, device))
    try:
        return [t.cpu() for t in flat_tensors(_run(spec, pos, kw, device))]
    except Exception:  # noqa: BLE001 - PyTorch rejects this call
        return NATIVE_REJECTS


def _run(spec, pos, kw, device):
    """The reference: PyTorch's kernel, or for a reflop-native op the
    worker's own implementation (natives.py), which runs PyTorch kernels."""
    if natives.is_native(spec):
        if device == "meta":
            return natives.meta(spec.name, pos, kw)
        return natives.call(spec.name, pos, kw, torch.device(device))
    if device != "meta" and spec.name in execute.PRE:
        # The worker adjusts these arguments before the kernel (execute.PRE).
        pos = list(pos)
        execute.PRE[spec.name](types.SimpleNamespace(op=spec), pos, kw, device == "cuda")
    return spec.fn(*pos, **kw)


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
        if isinstance(t, float):
            b = cpb.Binding()
            b.scalar.f = t
            ex.bindings.append(b)
        else:
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
# bounds, so the worker requires the shapes autograd passes (ADR-040/042/043).
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
    # ADR-032: the CTC backward kernels read their inputs unchecked
    "aten._ctc_loss_backward.default", "aten._ctc_loss_backward.Tensor",
    # ADR-043
    "aten._cdist_backward.default", "aten._pdist_backward.default", "aten._fft_r2c.default",
    "aten._fft_c2r.default", "aten._fft_c2c.default", "aten.native_batch_norm.default",
    "aten._native_batch_norm_legit.default", "aten._native_batch_norm_legit.no_stats",
    "aten._native_batch_norm_legit_functional.default", "aten._native_batch_norm_legit_no_training.default",
    "aten.native_batch_norm_backward.default", "aten.native_layer_norm.default",
    "aten.native_layer_norm_backward.default", "aten.native_group_norm.default",
    "aten.native_group_norm_backward.default", "aten.grid_sampler_2d.default",
    "aten.grid_sampler_2d_backward.default", "aten.grid_sampler_3d.default", "aten.grid_sampler_3d_backward.default",
    "reflop.fused_adam.default", "reflop.fused_adamw.default", "reflop.fused_sgd.default",
    "reflop.fused_adagrad.default", "reflop.amp_unscale_and_check.default", "reflop.amp_update_scale.default",
    # ADR-049: given a scale far below output/input, the kernels index (or
    # size a host buffer) past the input; the worker takes the scales
    # F.interpolate passes.
    *(f"aten.{n}{b}.default" for b in ("", "_backward") for n in (
        "upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d", "_upsample_nearest_exact1d",
        "_upsample_nearest_exact2d", "_upsample_nearest_exact3d", "upsample_linear1d", "upsample_bilinear2d",
        "upsample_bicubic2d", "upsample_trilinear3d", "_upsample_bilinear2d_aa")),
}
def _ctc_written(call: Call, want):
    """The reference log_alpha with its never-written entries set to -inf,
    as the worker sets them (#25 D, I13)."""
    a = dict(zip((s.name for s in SPEC[call.op].args), call.args))
    la = want[1].clone()
    _, t, s = la.shape
    for b, (il, tl) in enumerate(zip(torch.as_tensor(a["input_lengths"]).tolist(),
                                     torch.as_tensor(a["target_lengths"]).tolist())):
        la[b, il:] = float("-inf")
        la[b, :, 2 * tl + 1:] = float("-inf")
    return [want[0], la]


# Ops whose kernels leave part of an output unwritten (whatever the allocator
# held); the worker fills it, the reference is filled the same way.
UNWRITTEN = {"aten._ctc_loss.default": _ctc_written, "aten._ctc_loss.Tensor": _ctc_written}
# Runtime checks that run before any kernel launch (I4): a call PyTorch
# rejects may pass V4 when the reason depends on the data.
BEFORE_KERNEL = (cpb.STAGE_E1_BINDINGS, cpb.STAGE_E2_RUNTIME)
# A session offers fp8 only on a CUDA device that computes in it
# (core.detect_capabilities), so a CPU session refuses fp8 tensors and
# dtypes that PyTorch's CPU kernels take.
CAPABILITY = "requires capability"


def check(sess, call: Call, device="cpu", mark=lambda s: None):
    """None when the worker decides `call` the way PyTorch does, else why not.

    The worker decides first; `mark("rejected")` runs once it has refused
    the call without launching a kernel, before the native kernel runs. A
    rejection by a runtime check (E1/E2: index bounds, I4) is always
    acceptable: PyTorch's CPU kernels skip some of those checks and read out
    of bounds instead. A STRICT op's native kernel, and any kernel on CUDA,
    is not run on a call the worker refuses: it may crash on it."""
    outs, err = infer_outputs(sess, call)
    if err is not None:
        mark("rejected")
        # On CUDA a kernel run on input the worker refused can trip a device
        # assert and end the process's CUDA context; false rejections are
        # checked on the CPU.
        # A float8 cast of a seed is as safe to run as the seed: a refusal of
        # one the CUDA kernel runs is a false rejection (ADR-050).
        if call.op in STRICT or (device != "cpu" and not call.fp8) or CAPABILITY in err.message:
            return None
        want = native(call, device)
        if isinstance(want, list):
            return f"false rejection: {cpb.ValidationStage.Name(err.stage)} {err.message[:300]}"
        return None
    if not outs:
        # Every output is undefined (the backward of a cell given no
        # gradient): no graph can return one, so none runs the kernel.
        want = native(call, device)
        return f"the meta has no outputs; the kernel returned {len(want)}" if isinstance(want, list) and want else None
    if any(_extent(shape) > MAX_NATIVE_NUMEL for _, shape in outs):
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
            if call.value:
                # A float an op does not allocate from may reach the worker
                # as a scalar input, which V4 sees only as a placeholder
                # (ADR-011); its kernel's host check is the rule (ADR-049).
                return None
            return f"validation passed; the kernel raised (I3): {got.message[:300]}"
        return "PyTorch rejects; the worker ran it"
    if kind != "ran":
        return f"PyTorch accepts; the worker failed: {cpb.ValidationStage.Name(got.stage)} {got.message[:300]}"
    if [(t.dtype, tuple(t.shape)) for t in want] != outs:
        return f"meta shapes {outs} differ from the kernel's {[(t.dtype, tuple(t.shape)) for t in want]}"
    if SPEC[call.op].flags & (FLAG_UNINIT | FLAG_RNG):
        return None  # the values are not the kernel's to compare
    tol = 1e-5 if device == "cpu" else 1e-3
    if call.op in UNWRITTEN:
        want = UNWRITTEN[call.op](call, want)
    for w, g in zip(want, got, strict=True):
        if w.dtype.is_floating_point and w.dtype.itemsize == 1:
            w, g = w.float(), g.float()  # float8 has no allclose kernel
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
DTYPES = sorted(TORCH_TO_PB, key=str)
FLOATS = (0.0, -0.5, -1.0, 0.5, 2.0, 1e-10, 1e-30, 5e-324, 1e30, -1e30, 3e39, math.nan, math.inf, -math.inf)


def _int_values(v, rank, size):
    vals = {v - 1, v + 1, -rank - 1, -rank, -1, 0, rank - 1, rank, size, size + 1, -size - 1, HUGE} - {v}
    return sorted(x for x in vals if INT64[0] <= x <= INT64[1])


def _retensor(t: torch.Tensor, shape, dtype=None, lo_hi=None):
    dtype = dtype or t.dtype
    # zlib.crc32, not hash(): Python salts str hashes per process, so the
    # values (and which cases a run checks) used to differ between runs.
    g = torch.Generator().manual_seed(zlib.crc32(repr((tuple(shape), str(dtype))).encode()) & 0xFFFF)
    if dtype == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g).to(dtype)
    if dtype.is_complex:
        return torch.randn(shape, generator=g, dtype=dtype)
    lo, hi = lo_hi if lo_hi else ((int(t.min()), int(t.max())) if t.numel() else (0, 1))
    return torch.randint(lo, hi + 1, shape, generator=g).to(dtype)


def mutations(seed: Call):
    spec = SPEC[seed.op]
    rank, size = _ints_for(seed)
    out = []

    def mut(k, v, label):
        args = list(seed.args)
        args[k] = v
        out.append(_like(seed, args, label))

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
        elif core == "ScalarType" and (isinstance(v, torch.dtype) or (v is None and opt)):
            for dt in DTYPES:
                if dt != v:
                    mut(k, dt, f"{sa.name}={str(dt).removeprefix('torch.')}")
                    out[-1].fp8 = dt in FLOAT8 and _nonempty(seed)
        elif core in ("float", "Scalar") and (type(v) is float or (v is None and opt)):
            out.extend(_float_mutations(seed, k, v, sa.name))
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
        out.append(Call(seed.op, list(seed.args), "self non-contiguous", transposed=0))
    if _nonempty(seed) and any(_castable(t) for t in _flat(seed.args)):
        for dt in FLOAT8:
            args = [_to_fp8(v, dt) for v in seed.args]
            out.append(_like(seed, args, f"floats as {str(dt).removeprefix('torch.')}", fp8=True))
    return out


def _like(seed, args, label, **kw):
    # A mutation of a seed that transposes an argument transposes it too,
    # while it is still a matrix.
    t = seed.transposed
    if t is not None and not (isinstance(args[t], torch.Tensor) and args[t].dim() >= 2):
        t = None
    return Call(seed.op, args, label, transposed=t, **kw)


FLOAT8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def _flat(v):
    return [y for x in v for y in (_flat(x) if isinstance(x, (list, tuple)) else [x])]


def _castable(t):
    return isinstance(t, torch.Tensor) and t.dtype.is_floating_point and t.dtype.itemsize > 1


def _nonempty(seed):
    # An empty call's kernel may return before it looks at the dtype.
    return all(t.numel() for t in _flat(seed.args) if isinstance(t, torch.Tensor))


def _to_fp8(v, dt):
    if isinstance(v, list):
        return [_to_fp8(x, dt) for x in v]
    return v.to(dt) if _castable(v) else v


def _float_mutations(seed, k, v, name):
    # Literal values, and the finite ones of an op that does not allocate
    # from them also as a scalar input, the way the client sends a Python
    # float (ADR-011). An allocating op takes literals only (ADR-049).
    value = not SPEC[seed.op].flags & FLAG_ALLOC
    res = []
    for x in FLOATS:
        if v is not None and (x == v or math.isnan(x) and math.isnan(v)):
            continue
        args = list(seed.args)
        args[k] = x
        res.append(_like(seed, args, f"{name}={x}", value=value))
        if value and math.isfinite(x):
            res.append(_like(seed, list(args), f"{name}={x} lifted", lifted=k, value=True))
    return res


def _tensor_mutations(seed, k, t, name):
    res = []

    def mk(new, label):
        args = list(seed.args)
        args[k] = new
        res.append(_like(seed, args, f"{name} {label}"))

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
             torch.bool: torch.float32, torch.float8_e4m3fn: torch.float8_e5m2,
             torch.float8_e5m2: torch.float8_e4m3fn}.get(t.dtype)
    if other is not None:
        mk(t.to(other), f"as {other}")
    if _castable(t) and t.numel():
        # Not marked fp8: refusing float8 mixed into a non-FP8 op's other
        # operands is the allowlist (ADR-050), so it is not run natively.
        mk(t.to(torch.float8_e4m3fn), "as float8_e4m3fn")
    return res


def seeds(refused=True, cuda=False):
    """The seeds; with refused, also the calls PyTorch refuses (REFUSED);
    with cuda, also the seeds of CUDA-only kernels (CUDA_PROGRAMS)."""
    from parity_programs import DIRECT, PROGRAMS, REFUSED

    out = record(PROGRAMS) + DIRECT() + (REFUSED() if refused else [])
    return out + (record_cuda() if cuda else [])


def all_cases(cuda=False):
    """Seeds and their mutations. cuda adds the CUDA-only seeds; it changes
    the case list, so it is off unless the run is on CUDA."""
    from parity_grid import GRID

    out = []
    for s in seeds(cuda=cuda):
        out.append(s)
        out.extend(mutations(s))
    return out + GRID()


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
        for it in items:
            print("   ", it)
