# SPDX-License-Identifier: Apache-2.0
"""Run fuzz cases against an in-process worker Session and check the
properties of GH#4 (ADR-044):

1. Decode (V1) of a corrupted frame fails only with a decode or protocol
   error, as the worker's control loop expects.
2. No exception escapes `Session.handle`: every request ends in exactly one
   final response, a result or a structured error with a stage set. An
   `INTERNAL` error is a bug, except the known E3 fail-closed check (a
   kernel's output shape differs from what V4 validated).
3. A result matches what registration promised: the returned outputs have
   the registered output metadata.
4. Execution holds no more memory than V5 computed (`peak_memory_bytes`),
   within a small allowance for transient copies (see `allowance`).
5. The worker keeps serving after every case.

Cases are written as replay files (framed WorkerEnvelopes, the worker's
own wire format) so a failure can be added to `corpus/`."""

from __future__ import annotations

import math
import os
import struct
import weakref
from pathlib import Path

import torch
from google.protobuf.message import DecodeError
from graphgen import Case
from reflop_worker import wire
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.dtypes import PB_TO_TORCH, TORCH_TO_PB
from reflop_worker.errors import FatalError
from reflop_worker.safetensors import tensor_bytes
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

CORPUS = Path(__file__).resolve().parent / "corpus"
KNOWN_INTERNAL = ("output is ",)  # E3: "<op>: output is <meta>, validated <meta>" (fails closed, ADR-042)


class LiveBytes(TorchDispatchMode):
    """Bytes of tensor storage alive at once, for storages created by aten
    calls made while the mode is active (all of a worker execute: bindings,
    node outputs, runtime checks, output copies)."""

    def __init__(self) -> None:
        super().__init__()
        self.live = 0
        self.peak = 0
        self._refs: dict[int, list] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t.device.type != "meta":
                st = t.untyped_storage()
                ptr, nb = st.data_ptr(), st.nbytes()
                if nb == 0:
                    continue
                e = self._refs.get(ptr)
                if e is None:
                    e = self._refs[ptr] = [0, nb]
                    self.live += nb
                    self.peak = max(self.peak, self.live)
                e[0] += 1
                weakref.finalize(t, self._drop, ptr)
        return out

    def _drop(self, ptr: int) -> None:
        e = self._refs.get(ptr)
        if e is None:
            return
        e[0] -= 1
        if e[0] == 0:
            del self._refs[ptr]
            self.live -= e[1]


def allowance(registered: cpb.GraphRegistered, case: Case) -> int:
    """Transient memory V5 does not count, by design: a contiguous copy of
    each returned output while it is serialized, and a runtime check's
    temporary over its input (both at most one tensor of the graph)."""
    largest = max([_nbytes(m) for m in registered.output_meta] +
                  [t.numel() * t.element_size() for t in case.inputs if isinstance(t, torch.Tensor)] + [0])
    return 2 * largest + 4096


def _nbytes(m: gpb.TensorMeta) -> int:
    return math.prod(m.shape) * PB_TO_TORCH[m.dtype].itemsize if m.dtype in PB_TO_TORCH else 0


def _handle(i: int) -> bytes:
    return bytes([0x10]) + i.to_bytes(15, "big")


def execute_request(case: Case) -> cpb.Execute:
    ex = cpb.Execute()
    K = gpb.GraphInput
    for gi, src in zip(case.graph.inputs, case.inputs):
        b = cpb.Binding()
        if isinstance(src, torch.Tensor):
            b.inline.meta.CopyFrom(gpb.TensorMeta(dtype=TORCH_TO_PB[src.dtype], shape=list(src.shape)))
            b.inline.data = tensor_bytes(src)
        elif gi.kind == K.KIND_SCALAR and src is not None:
            if isinstance(src, bool):
                b.scalar.b = src
            elif isinstance(src, int) and -(2**63) <= src < 2**63:
                b.scalar.i = src
            elif isinstance(src, complex):
                b.scalar.c.real, b.scalar.c.imag = src.real, src.imag
            elif isinstance(src, float):
                b.scalar.f = src
        ex.bindings.append(b)
    for k, kind in enumerate(case.outputs):
        spec = cpb.OutputSpec(kind=kind)
        if kind == cpb.OutputSpec.KIND_STORE:
            spec.handle = _handle(k + 1)
        ex.outputs.append(spec)
    return ex


def frames(case: Case) -> list[bytes]:
    """The case as the frames the worker would read (register, then execute
    with an empty graph_hash that replay fills in)."""
    if case.kind == "bytes":
        return [case.raw]
    reg = wpb.WorkerEnvelope(client=cpb.Envelope(request_id=1, register_graph=cpb.RegisterGraph(graph=case.graph)))
    out = [reg.SerializeToString()]
    if case.graph.outputs:
        out.append(wpb.WorkerEnvelope(client=cpb.Envelope(request_id=2, execute=execute_request(case)))
                   .SerializeToString())
    return out


def save(case: Case, path: Path) -> None:
    with open(path, "wb") as f:
        for fr in frames(case):
            f.write(struct.pack(">I", len(fr)) + fr)


def load(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out, i = [], 0
    while i < len(data):
        (n,) = struct.unpack(">I", data[i:i + 4])
        out.append(data[i + 4:i + 4 + n])
        i += 4 + n
    return out


def decode(frame: bytes):
    """What the worker's control loop does with a frame (wire.Channel.recv)."""
    msg = wpb.WorkerEnvelope()
    msg.ParseFromString(frame)
    wire.check_no_unknown(msg)
    wire.check_enums(msg)
    return msg


def run_frames(sess, frs: list[bytes], case: Case | None = None) -> list[str]:
    """Run decoded frames through the session; the findings, if any."""
    findings: list[str] = []
    registered = None
    for fr in frs:
        try:
            env = decode(fr)
        except (DecodeError, wire.ProtocolError, RecursionError):
            return findings
        except Exception as e:  # noqa: BLE001 - property 1
            return [f"decode raised {type(e).__name__}: {e}"]
        if env.WhichOneof("body") != "client":
            return findings
        req = env.client
        which = req.WhichOneof("body")
        if which == "execute":
            if registered is None:
                return findings
            req.execute.graph_hash = registered.graph_hash
        meter = LiveBytes() if which == "execute" else None
        out: list = []
        try:
            if meter is None:
                sess.handle(req, out.append)
            else:
                with meter:
                    sess.handle(req, out.append)
        except FatalError as e:
            return [f"{which}: worker lost its device: {e}"]
        except Exception as e:  # noqa: BLE001 - property 2
            return [f"{which}: {type(e).__name__} escaped Session.handle: {str(e)[:300]}"]
        finals = [o for o in out if o.end]
        if len(finals) != 1:
            return [f"{which}: {len(finals)} final responses"]
        r = finals[0]
        body = r.WhichOneof("body")
        if body == "error":
            e = r.error
            if e.code == cpb.ERROR_INTERNAL and not any(k in e.message for k in KNOWN_INTERNAL):
                findings.append(f"{which}: INTERNAL error: {e.message[:300]}")
            if e.stage == cpb.STAGE_NONE and e.code not in (cpb.ERROR_PROTOCOL, cpb.ERROR_UNKNOWN_GRAPH):
                findings.append(f"{which}: error without a stage: {cpb.ErrorCode.Name(e.code)} {e.message[:200]}")
            return findings
        if which == "register_graph":
            if body != "graph_registered":
                return [f"register answered {body}"]
            registered = r.graph_registered
        elif which == "execute":
            if body != "execute_result":
                return [f"execute answered {body}"]
            res = r.execute_result
            returned = [m for m, s in zip(registered.output_meta, req.execute.outputs)
                        if s.kind == cpb.OutputSpec.KIND_RETURN]
            got = [t.meta for t in res.returned]
            if [(m.dtype, list(m.shape)) for m in got] != [(m.dtype, list(m.shape)) for m in returned]:
                findings.append(f"returned {[(m.dtype, list(m.shape)) for m in got]}, registered "
                                f"{[(m.dtype, list(m.shape)) for m in returned]}")
            for t in res.returned:
                if len(t.data) != _nbytes(t.meta):
                    findings.append(f"returned {len(t.data)} bytes for {t.meta.dtype}{list(t.meta.shape)}")
            if case is not None and meter.peak > registered.peak_memory_bytes + allowance(registered, case):
                findings.append(f"execute held {meter.peak} bytes; V5 computed {registered.peak_memory_bytes}")
            _free_stored(sess, req.execute)
    return findings


def _free_stored(sess, ex: cpb.Execute) -> None:
    hs = [o.handle for o in ex.outputs if o.kind == cpb.OutputSpec.KIND_STORE]
    if hs:
        sess.handle(cpb.Envelope(request_id=3, free=cpb.Free(handles=hs)), lambda _e: None)


def still_serving(sess) -> str | None:
    g = gpb.Graph(opset_version=1, device=sess.gdevice)
    g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_TENSOR,
                                   meta=gpb.TensorMeta(dtype=gpb.DTYPE_FLOAT32, shape=[2])))
    from helpers import NAME

    g.nodes.append(gpb.Node(op_id=NAME["aten.neg.default"],
                            args=[gpb.Arg(tensor=gpb.ValueRef(graph_input=0))]))
    g.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=0, output=0)))
    g.output_meta.append(gpb.TensorMeta(dtype=gpb.DTYPE_FLOAT32, shape=[2]))
    case = Case("random", g, [torch.ones(2)], [cpb.OutputSpec.KIND_RETURN])
    out: list = []
    sess.handle(cpb.Envelope(request_id=8, register_graph=cpb.RegisterGraph(graph=g)), out.append)
    if out[-1].WhichOneof("body") != "graph_registered":
        return f"worker stopped serving: {out[-1]}"
    ex = execute_request(case)
    ex.graph_hash = out[-1].graph_registered.graph_hash
    out = []
    sess.handle(cpb.Envelope(request_id=9, execute=ex), out.append)
    if out[-1].WhichOneof("body") != "execute_result":
        return f"worker stopped serving: {out[-1]}"
    return None


def artifact_dir() -> Path | None:
    d = os.environ.get("REFLOP_FUZZ_OUT")
    if not d:
        return None
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p
