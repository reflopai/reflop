# SPDX-License-Identifier: Apache-2.0
"""Request sequences for the worker's other entry points (#25 A, ADR-049).

graphgen.py sends one graph and its execute. A sequence case is a run of
requests on a fresh session, chosen from a model of what the session holds
so that most requests reach real state:

- upload_tensor streams, two at once at times and left unfinished at times,
  with damaged offsets, chunk order, final flags, checksums, handles and
  metas, and sizes around the host budget;
- graphs that bind uploaded handles (one handle twice, at times) and store,
  rebind, return or discard their outputs, views included;
- fetch, free, reseed, ping, stats and execute of an unknown graph;
- dataset_open over shard files written here, damaged or not, then
  reflop.dataset_gather graphs and dataset_close;
- snapshot_write and checkpoints, given the go or refused, then
  snapshot_read of the file just written or of a generated one.

A replay file carries the files next to the envelopes that use them: a
frame b"FILE" + bytes is the content of the next descriptor, and b"PREV" +
JSON lists damage to the last file a snapshot_write produced. Neither "F"
(field 8, wire type 6) nor "P" (field 10) starts an envelope the worker
accepts. Every case comes from (seed, index) alone."""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import dataclass, field

import torch
from parity import SPEC, Call, encode, graph_of, infer_outputs
from reflop_worker._pb import capabilities_pb2 as kpb
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.core import Session
from reflop_worker.dtypes import TORCH_TO_PB
from reflop_worker.safetensors import MAX_HEADER, NAMES

# Small limits put every boundary within a few KiB of data.
HOST_BUDGET = 1 << 20
DEVICE_QUOTA = 16 << 20
LIMITS = {"max_fetch_bytes": 256 << 10, "upload_chunk_bytes": 4096, "max_handles_per_session": 24,
          "max_tensor_numel": 1 << 18, "max_dims": 6}
DTYPES = (torch.float32, torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64,
          torch.int32, torch.int8, torch.uint8, torch.bool, torch.complex64)
DIMS = (0, 1, 1, 2, 3, 4, 7, 16)
NAMES_OK = ("x", "y", "label", "img_0", "a" * 64, "t9")
K = cpb.OutputSpec


def make_session(device: str = "cpu") -> Session:
    from reflop.connection import default_limits

    init = wpb.WorkerInit(seed=0, device_memory_quota_bytes=DEVICE_QUOTA, host_memory_budget_bytes=HOST_BUDGET)
    init.device.kind = gpb.DEVICE_CPU if device == "cpu" else gpb.DEVICE_CUDA
    lim = default_limits(kpb)
    for k, v in LIMITS.items():
        setattr(lim, k, v)
    init.limits.CopyFrom(lim)
    return Session(init)


@dataclass
class Step:
    env: wpb.WorkerEnvelope
    files: list = field(default_factory=list)  # bytes: a file; list: damage to the last snapshot written


@dataclass
class SeqCase:
    steps: list = field(default_factory=list)


def frames(case: SeqCase) -> list[bytes]:
    out = []
    for s in case.steps:
        for f in s.files:
            out.append(b"FILE" + f if isinstance(f, bytes) else b"PREV" + json.dumps(f).encode())
        out.append(s.env.SerializeToString())
    return out


def nbytes(dt: torch.dtype, shape) -> int:
    return math.prod(shape) * torch.empty((), dtype=dt).element_size()


def meta(dt, shape) -> gpb.TensorMeta:
    return gpb.TensorMeta(dtype=TORCH_TO_PB[dt], shape=list(shape))


_GEN: dict[str, Session] = {}


def _gen_session(device: str) -> Session:
    """For output metas only (the worker's own V2-V4); never runs a case."""
    if device not in _GEN:
        _GEN[device] = make_session(device)
    return _GEN[device]


# ------------------------------------------------------------------ safetensors files


def st_items(r: random.Random, n: int, rows: int | None = None) -> list:
    """[(name, dtype, shape, data)] with random content."""
    names = r.sample(NAMES_OK, n)
    out = []
    for name in names:
        dt = r.choice(DTYPES)
        row = [r.choice((1, 2, 3, 4)) for _ in range(r.choice((0, 1, 1, 2)))]
        shape = ([rows] + row) if rows is not None else [r.choice(DIMS) for _ in range(r.choice((0, 1, 2, 3)))]
        out.append((name, dt, shape, r.randbytes(nbytes(dt, shape))))
    return out


def st_file(r: random.Random, items: list, damage: bool) -> bytes:
    """A safetensors file of items; if damage, broken in one of many ways."""
    hdr, off, data = {}, 0, b""
    for name, dt, shape, raw in items:
        hdr[name] = {"dtype": NAMES[dt], "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data += raw
    text = None
    if damage:
        kind = r.randrange(16)
        keys = list(hdr)
        k = r.choice(keys) if keys else None
        if kind == 0 and k:
            hdr[k]["shape"] = hdr[k]["shape"] + [r.choice((2**64, 2**63, -1, 1.5, True, "3", None))]
        elif kind == 1 and k:
            # Zero elements, but a dim no tensor can have.
            hdr[k]["shape"] = [0, r.choice((2**63, 2**64, 2**80))]
            hdr[k]["data_offsets"] = [0, 0]
        elif kind == 2 and k:
            hdr[k]["shape"] = [1] * r.choice((7, 16, 17))
        elif kind == 3 and k:
            b, e = hdr[k]["data_offsets"]
            hdr[k]["data_offsets"] = r.choice(([e, b], [b - 1, e], [b, e + 1], [b, e, e], [b + 0.0, e], [b], None))
        elif kind == 4 and k:
            hdr[k]["dtype"] = r.choice(("F128", "", "f32", 5, "I64 ", "U16"))
        elif kind == 5 and k:
            hdr[k]["extra"] = 1
        elif kind == 6:
            hdr["__metadata__"] = r.choice(({"a": 1}, "x", [1], {"a": "b"}))
        elif kind == 7 and k:
            text = json.dumps(hdr)[:-1] + ", " + json.dumps(k) + ": " + json.dumps(hdr[k]) + "}"  # duplicate key
        elif kind == 8:
            text = r.choice(("[]", "null", "{", "{}" + "\x00" * 3, '{"a": 1}', "\udcff"))
        elif kind == 9 and k:
            hdr[r.choice(("", "k" * 257, "__reflop__/x", "__metadata__x"))] = hdr.pop(k)
        elif kind == 10 and len(keys) >= 2:
            hdr[keys[1]]["data_offsets"] = list(hdr[keys[0]]["data_offsets"])  # overlap
        elif kind == 11:
            data += r.randbytes(r.choice((1, 8)))  # unreferenced bytes
        elif kind == 12 and k:
            hdr[k] = r.choice((1, [], "x"))
    h = (text if text is not None else json.dumps(hdr)).encode("utf-8", "surrogatepass")
    h += b" " * ((8 - len(h) % 8) % 8)
    out = struct.pack("<Q", len(h)) + h + data
    if damage and kind >= 13:
        raw = bytearray(out)
        if kind == 13:
            raw[:8] = struct.pack("<Q", r.choice((0, len(h) + 1, 2**64 - 1, MAX_HEADER + 1, 7)))
        elif kind == 14 and raw:
            del raw[r.randrange(len(raw)):]  # truncated
        elif raw:
            i = r.randrange(len(raw))
            raw[i] ^= 1 << r.randrange(8)
        out = bytes(raw)
    return out


# ------------------------------------------------------------------ the generator


class Gen:
    def __init__(self, r: random.Random, device: str) -> None:
        self.r = r
        self.device = device
        self.gsess = _gen_session(device)
        self.case = SeqCase()
        self.rid = 0
        self.n = 0
        self.live: dict[bytes, tuple] = {}     # handle -> (dtype, shape), as the generator expects it
        self.gone: list[bytes] = []            # freed or never created
        self.streams: list[dict] = []          # uploads started and not finished
        self.datasets: dict[bytes, dict] = {}  # id -> {name: (dtype, full shape)}
        self.snap: dict[str, tuple] | None = None  # keys of the last snapshot written

    # -- helpers

    def next_rid(self) -> int:
        self.rid += 1
        return self.rid

    def new_handle(self) -> bytes:
        self.n += 1
        return bytes([0x20 + self.r.randrange(8)]) + self.n.to_bytes(15, "big")

    def bad_handle(self) -> bytes:
        r = self.r
        opts = [b"", bytes(16), b"\xff" + bytes(range(1, 16)), b"\x01" * 15, b"\x01" * 17, r.randbytes(16)]
        if self.gone:
            opts.append(r.choice(self.gone))
        return r.choice(opts)

    def some_handle(self) -> bytes:
        if self.live and self.r.random() < 0.85:
            return self.r.choice(list(self.live))
        return self.bad_handle()

    def shape(self) -> list[int]:
        return [self.r.choice(DIMS) for _ in range(self.r.choice((0, 1, 1, 2, 2, 3)))]

    def client(self, files=None, **body) -> int:
        rid = self.next_rid()
        self.case.steps.append(Step(wpb.WorkerEnvelope(client=cpb.Envelope(request_id=rid, **body)), files or []))
        return rid

    def internal(self, files=None, **body) -> None:
        self.case.steps.append(Step(wpb.WorkerEnvelope(**body), files or []))

    # -- uploads

    def upload_start(self, second: bool = False) -> None:
        r = self.r
        dt, shape = r.choice(DTYPES), self.shape()
        m = meta(dt, shape)
        valid, big = True, False
        roll = 0.2 if second else r.random()
        if roll < 0.04:
            m.dtype, valid = r.choice((0, 99)), False
        elif roll < 0.08:
            m.shape[:], valid = [1] * 7, False
        elif roll < 0.12:
            m.shape[:], valid = [r.choice(((1 << 18) + 1, 1 << 40, 2**62)), r.choice((0, 2))], False
        elif roll < 0.14:
            m.shape[:], valid = [r.choice((-1, -(2**63))), 3], False
        elif roll < 0.16:
            # No elements, but strides that overflow int64.
            m.shape[:], valid = [0, r.choice((2**62, 2**40)), r.choice((2**62, 2**40))], False
        elif roll < 0.22:
            # Near half the budget: two at once fit or not.
            dt, shape = torch.float64, [HOST_BUDGET // 16 + r.randint(-2, 2)]  # within max_numel
            m = meta(dt, shape)
            big = True
        elif roll < 0.30:
            # More than one fetch chunk; the largest is past the fetch limit.
            dt, shape = torch.float32, [r.choice((1100, 2048, 70000))]
            m = meta(dt, shape)
        size = nbytes(dt, shape) if valid else r.choice((0, 4, 64))
        data = r.randbytes(size)
        h = self.new_handle()
        roll = r.random()
        if roll < 0.04 and self.live:
            h = r.choice(list(self.live))  # HANDLE_EXISTS
        elif roll < 0.07:
            h = self.bad_handle()
        elif roll < 0.10 and self.streams:
            h = self.streams[0]["handle"]  # the same handle in two streams
        cuts = {0, size} | {r.randrange(size + 1) for _ in range(r.choice((0, 1, 2, 4)))}
        cuts = sorted(cuts | {size // 2} if big else cuts)
        chunks = [[a, data[a:b]] for a, b in zip(cuts, cuts[1:])] or [[0, b""]]
        st = {"rid": self.next_rid(), "handle": h, "meta": m, "chunks": chunks, "data": data, "valid": valid,
              "sha": hashlib.sha256(data).digest(), "dt": dt, "shape": shape}
        self._damage_stream(st)
        self.streams.append(st)
        self.upload_continue(st, 1 if big else 0)
        if big and not second and st["chunks"] and r.random() < 0.6:
            self.upload_start(second=True)  # while the first stream holds its buffer

    def _damage_stream(self, st) -> None:
        r = self.r
        if r.random() > 0.3:
            return
        st["valid"] = False
        ch = st["chunks"]
        kind = r.randrange(9)
        i = r.randrange(len(ch))
        if kind == 0:
            ch[i][0] += r.choice((1, -1 if ch[i][0] else 2**63))
        elif kind == 1 and len(ch) >= 2:
            ch[0], ch[1] = ch[1], ch[0]
        elif kind == 2 and len(ch) >= 2:
            del ch[i]
        elif kind == 3:
            st["early"] = i  # final before the last chunk
        elif kind == 4:
            st["sha"] = r.choice((hashlib.sha256(b"x").digest(), st["sha"][:16], b"\x00" * 32))
        elif kind == 5:
            st["sha"] = b""
            st["valid"] = True  # no checksum is allowed
        elif kind == 6:
            ch[i][1] = ch[i][1] + r.randbytes(r.choice((1, 9)))  # past the end
        elif kind == 7 and len(ch) >= 2:
            st["switch"] = i  # the handle changes mid-stream
        elif kind == 8:
            st["never"] = True  # never finished
            st["valid"] = False

    def fill(self) -> None:
        """Small uploads until the handle table is nearly full."""
        r = self.r
        for _ in range(LIMITS["max_handles_per_session"] - len(self.live) - r.choice((0, 1, 2))):
            h, data = self.new_handle(), r.randbytes(4)
            self.client(upload_tensor=cpb.UploadTensor(handle=h, meta=meta(torch.uint8, [4]), data=data, final=True))
            self.live[h] = (torch.uint8, (4,))

    def upload_continue(self, st=None, n: int = 0) -> None:
        r = self.r
        if st is None:
            if not self.streams:
                return self.upload_start()
            st = r.choice(self.streams)
        n = n or r.choice((1, 1, 2, len(st["chunks"])))
        for _ in range(n):
            if not st["chunks"]:
                break
            off, data = st["chunks"].pop(0)
            last = not st["chunks"] or st.get("early") == len(st["chunks"])
            if st.get("never") and last:
                st["chunks"].insert(0, [off, data])
                break
            u = cpb.UploadTensor(handle=st["handle"], offset=off, data=data, final=last)
            if off == 0 or r.random() < 0.2:
                u.meta.CopyFrom(st["meta"])
            if st.get("switch") is not None and len(st["chunks"]) <= st["switch"]:
                u.handle = self.new_handle()
            if last:
                u.sha256 = st["sha"]
            env = wpb.WorkerEnvelope(client=cpb.Envelope(request_id=st["rid"], upload_tensor=u))
            self.case.steps.append(Step(env))
            if last:
                self.streams.remove(st)
                if st["valid"] and len(self.live) < LIMITS["max_handles_per_session"]:
                    self.live[st["handle"]] = (st["dt"], tuple(st["shape"]))
                if r.random() < 0.05:
                    # A chunk after the end: a new stream that has no meta.
                    self.case.steps.append(Step(wpb.WorkerEnvelope(client=cpb.Envelope(
                        request_id=st["rid"], upload_tensor=cpb.UploadTensor(handle=st["handle"], data=b"x")))))
                break

    # -- graphs over handles

    def _template(self, x: tuple, y: tuple | None):
        """A one-node Call over placeholders of x (and y)."""
        r = self.r
        dt, shape = x
        X = torch.empty(shape, dtype=dt, device="meta")
        rank = len(shape)
        opts = [("aten.neg.default", [X]), ("aten.clone.default", [X, None]), ("aten.alias.default", [X]),
                ("aten.sum.default", [X, None]),
                ("aten.expand.default", [X, [r.choice((2, 3, 8))] + [d if d != 1 else r.choice((1, 5)) for d in shape],
                                         False])]
        if dt.is_complex:
            opts.append(("aten._conj.default", [X]))
        if rank >= 1:
            d = r.randrange(rank)
            opts.append(("aten.slice.Tensor", [X, d, r.choice((0, 1, None)), r.choice((None, -1, 5)), 2]))
            if shape[d]:
                opts.append(("aten.select.int", [X, d, r.randrange(-shape[d], shape[d])]))
        if rank == 2:
            opts.append(("aten.t.default", [X]))
        if rank >= 2:
            opts += [("aten.transpose.int", [X, 0, rank - 1]), ("aten.diagonal.default", [X, 0, 0, 1]),
                     ("aten.permute.default", [X, list(range(rank))[::-1]])]
        if y is not None:
            Y = X if y is x else torch.empty(y[1], dtype=y[0], device="meta")
            opts += [("aten.add.Tensor", [X, Y, 1]), ("aten.mul.Tensor", [X, Y])]
        op, args = r.choice(opts)
        assert len(args) == len(SPEC[op].args), op
        return Call(op, args)

    def graph(self) -> None:
        r = self.r
        if not self.live:
            return self.upload_start()
        hx = r.choice(list(self.live))
        x = self.live[hx]
        hy, y = None, None
        if r.random() < 0.4:
            hy = hx if r.random() < 0.5 else r.choice(list(self.live))
            y = x if hy == hx else self.live[hy]
        call = self._template(x, y)
        inputs, nodes = encode(self.gsess, call)
        outs, err = infer_outputs(self.gsess, call)
        if err is not None:
            outs = [x]
        twice = r.random() < (0.5 if len(self.live) >= LIMITS["max_handles_per_session"] - 2 else 0.08)
        g = graph_of(self.gsess, inputs, nodes, outs * (2 if twice else 1))
        if twice:
            g.outputs[1].CopyFrom(g.outputs[0])
        self.client(register_graph=cpb.RegisterGraph(graph=g))
        ex = cpb.Execute()
        handles = [hx] + ([hy] if len(inputs) > 1 else [])
        for t, h in zip(inputs, handles):
            ex.bindings.append(self._binding(h, t.dtype, tuple(t.shape)))
        for dt, shape in [tuple(o) for o in outs] * (2 if twice else 1):
            ex.outputs.append(self._output(dt, tuple(shape), handles))
        if r.random() < 0.05:
            del ex.bindings[-1]
        self.client(execute=ex)

    def _binding(self, h: bytes, dt, shape) -> cpb.Binding:
        r = self.r
        b = cpb.Binding()
        roll = r.random()
        if roll < 0.85:
            b.handle = h
        elif roll < 0.90 and nbytes(dt, shape) <= 4096:
            b.inline.meta.CopyFrom(meta(dt, shape))
            b.inline.data = r.randbytes(nbytes(dt, shape))
        elif roll < 0.96:
            b.handle = self.bad_handle()
        elif roll < 0.98:
            b.dataset.dataset_id, b.dataset.tensor = r.randbytes(16), "x"
        else:
            other = [k for k, v in self.live.items() if v != (dt, shape)]
            b.handle = r.choice(other) if other else h
        return b

    def _output(self, dt, shape, inputs) -> cpb.OutputSpec:
        r = self.r
        roll = r.random()
        if roll < 0.35:
            h = self.new_handle()
            self.live[h] = (dt, shape)
            return K(kind=K.KIND_STORE, handle=h)
        if roll < 0.65:
            return K(kind=K.KIND_RETURN)
        if roll < 0.75:
            return K(kind=K.KIND_DISCARD)
        if roll < 0.90:
            same = [h for h, v in self.live.items() if v == (dt, shape)]
            if same:
                return K(kind=K.KIND_REBIND, handle=r.choice(same))
            return K(kind=K.KIND_RETURN)
        if roll < 0.94:
            return K(kind=K.KIND_STORE, handle=r.choice(list(self.live) + inputs))
        if roll < 0.97:
            return K(kind=K.KIND_STORE, handle=self.bad_handle())
        return K(kind=K.KIND_REBIND, handle=self.some_handle())

    def unknown_graph(self) -> None:
        self.client(execute=cpb.Execute(graph_hash=self.r.randbytes(self.r.choice((1, 32)))))

    # -- fetch, free, reseed, ping, stats

    def fetch(self) -> None:
        r = self.r
        hs = r.sample(list(self.live), min(len(self.live), r.choice((0, 1, 1, 2, 3))))
        if r.random() < 0.15 and hs:
            hs.append(hs[0])
        if r.random() < 0.15:
            hs.insert(r.randrange(len(hs) + 1), self.bad_handle())
        if r.random() < 0.05:
            hs = list(self.live)
        self.client(fetch=cpb.Fetch(handles=hs))

    def free(self) -> None:
        r = self.r
        hs = r.sample(list(self.live), min(len(self.live), r.choice((1, 1, 2))))
        for h in hs:
            del self.live[h]
            self.gone.append(h)
        if r.random() < 0.2:
            hs.append(self.bad_handle())
        if r.random() < 0.1 and hs:
            hs.append(hs[0])
        self.client(free=cpb.Free(handles=hs))

    def reseed(self) -> None:
        self.client(reseed=cpb.Reseed(seed=self.r.getrandbits(64)))

    def ping(self) -> None:
        self.client(ping=cpb.Ping(nonce=self.r.getrandbits(64)))

    def stats(self) -> None:
        self.internal(stats_request=wpb.StatsRequest())

    # -- datasets

    def dataset_open(self) -> None:
        r = self.r
        ds_id = r.choice(list(self.datasets)) if self.datasets and r.random() < 0.2 else r.randbytes(16)
        shard_rows = [r.choice((0, 1, 2, 3, 6)) for _ in range(r.choice((1, 1, 2, 3)))]
        protos = st_items(r, r.choice((1, 1, 2, 3)), rows=0)
        files, valid = [], True
        for rows in shard_rows:
            items = [(n, dt, [rows] + s[1:], r.randbytes(nbytes(dt, [rows] + s[1:]))) for n, dt, s, _ in protos]
            damage = r.random() < 0.12
            valid &= not damage
            files.append(st_file(r, items, damage))
        msg = wpb.DatasetOpen(dataset_id=ds_id)
        total = sum(shard_rows)
        for n, dt, s, _ in protos:
            msg.tensors.append(wpb.DatasetTensor(name=n, meta=meta(dt, [total] + s[1:])))
        for i, rows in enumerate(shard_rows):
            msg.shards.append(wpb.DatasetShard(index=i, num_samples=rows))
        if r.random() < 0.25:
            valid = False
            kind = r.randrange(8)
            t = msg.tensors[r.randrange(len(msg.tensors))]
            if kind == 0:
                t.meta.shape[0] += 1
            elif kind == 1:
                t.meta.shape.append(2)
            elif kind == 2:
                t.meta.dtype = r.choice((0, 99, gpb.DTYPE_FLOAT64 if t.meta.dtype != gpb.DTYPE_FLOAT64
                                         else gpb.DTYPE_INT64))
            elif kind == 3:
                del t.meta.shape[:]  # a 0-d dataset tensor
            elif kind == 4:
                msg.tensors.append(wpb.DatasetTensor(name="extra", meta=meta(torch.float32, [total])))
            elif kind == 5:
                msg.shards[0].num_samples += 1
            elif kind == 6:
                files.append(files[0])  # one descriptor too many
            else:
                files.pop()  # one too few
        self.internal(files=files, dataset_open=msg)
        if valid:
            self.datasets[ds_id] = {t.name: (_dt(t.meta.dtype), tuple(t.meta.shape)) for t in msg.tensors}
        else:
            self.datasets.pop(ds_id, None)

    def dataset_close(self) -> None:
        r = self.r
        if self.datasets and r.random() < 0.8:
            ds_id = r.choice(list(self.datasets))
            del self.datasets[ds_id]
        else:
            ds_id = r.randbytes(16)
        self.internal(dataset_close=wpb.DatasetClose(dataset_id=ds_id))

    def gather(self) -> None:
        r = self.r
        if not self.datasets:
            return self.dataset_open()
        ds_id = r.choice(list(self.datasets))
        name = r.choice(list(self.datasets[ds_id]))
        dt, full = self.datasets[ds_id][name]
        n = full[0]
        k = r.choice((0, 1, 3, 8))
        lo, hi = (-2, n + 2) if r.random() < 0.15 else (0, n)
        idx = [r.randrange(lo, hi) for _ in range(k)] if hi > lo else [0] * k
        idt = r.choice((torch.int64, torch.int64, torch.int32, torch.float32 if r.random() < 0.1 else torch.int64))
        declared = list(full)
        if r.random() < 0.05:
            declared[0] += 1
        g = gpb.Graph(opset_version=1, device=self.gsess.gdevice)
        g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_DATASET_REF, meta=meta(dt, declared)))
        g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_TENSOR, meta=meta(idt, [k])))
        g.nodes.append(gpb.Node(op_id=SPEC["reflop.dataset_gather.default"].id,
                                args=[gpb.Arg(tensor=gpb.ValueRef(graph_input=0)),
                                      gpb.Arg(tensor=gpb.ValueRef(graph_input=1))]))
        g.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=0, output=0)))
        g.output_meta.append(meta(dt, [k] + list(full[1:])))
        self.client(register_graph=cpb.RegisterGraph(graph=g))
        ex = cpb.Execute()
        b = ex.bindings.add()
        b.dataset.dataset_id = ds_id if r.random() < 0.93 else r.randbytes(16)
        b.dataset.tensor = name if r.random() < 0.95 else "nope"
        ib = ex.bindings.add()
        it = torch.tensor(idx, dtype=idt)
        ib.inline.meta.CopyFrom(meta(idt, [k]))
        ib.inline.data = it.numpy().tobytes()
        if r.random() < 0.3:
            h = self.new_handle()
            self.live[h] = (dt, (k, *full[1:]))
            ex.outputs.append(K(kind=K.KIND_STORE, handle=h))
        else:
            ex.outputs.append(K(kind=K.KIND_RETURN))
        self.client(execute=ex)

    # -- snapshots and checkpoints

    def snapshot_write(self) -> None:
        r = self.r
        rid = self.next_rid()
        msg = wpb.SnapshotWrite(request_id=rid)
        keys: dict[str, tuple] = {}
        if r.random() < 0.3:
            msg.checkpoint = True
            msg.tag = r.randbytes(r.choice((0, 5, 4096, 4097)))
            msg.generation = r.getrandbits(64)
            keys = {h.hex(): v for h, v in self.live.items()}
            if r.random() < 0.05:
                msg.keys.append("x")
        else:
            hs = r.sample(list(self.live), min(len(self.live), r.choice((1, 2, 3))))
            names = r.sample(NAMES_OK, len(hs))
            for h, k in zip(hs, names):
                msg.handles.append(h)
                msg.keys.append(k)
                keys[k] = self.live[h]
            roll = r.random()
            if roll < 0.05:
                msg.keys[0:1] = [r.choice(("", "__reflop__/meta", "__metadata__", "k" * 257))]
            elif roll < 0.08 and len(msg.keys) >= 2:
                msg.keys[1] = msg.keys[0]
            elif roll < 0.11:
                msg.handles.append(self.bad_handle())
            elif roll < 0.13:
                msg.keys.append("z")
        self.internal(snapshot_write=msg)
        go = wpb.SnapshotGo(request_id=rid if r.random() < 0.95 else rid + 1000, proceed=r.random() < 0.85)
        if not go.proceed:
            go.code, go.reason = r.choice((cpb.ERROR_QUOTA, cpb.ERROR_SAFETY_RAIL, 0)), "no"
        self.internal(snapshot_go=go)
        if go.proceed:
            self.snap = keys

    def snapshot_read(self) -> None:
        r = self.r
        files, keys = [], {}
        if self.snap is not None and r.random() < 0.6:
            damage = []
            if r.random() < 0.3:
                damage = [r.choice((["flip", r.random(), 1 << r.randrange(8)], ["cut", r.random()],
                                    ["add", r.randbytes(4).hex()],
                                    ["rng", r.choice((8, 9, 11, 12, 16, 17, 19)), 1 << r.randrange(8)]))]
            files.append(damage)
            keys = dict(self.snap)
        else:
            items = st_items(r, r.choice((1, 2, 3)))
            files.append(st_file(r, items, r.random() < 0.4))
            keys = {n: (dt, tuple(s)) for n, dt, s, _ in items}
        msg = wpb.SnapshotRead(request_id=self.next_rid())
        names = [k for k in keys if not k.startswith("__reflop__/")]
        pick = r.sample(names, min(len(names), r.choice((1, 1, 2, 3))))
        for k in pick:
            h = self.new_handle()
            msg.keys.append(k)
            msg.handles.append(h)
        roll = r.random()
        if roll < 0.05 and self.live:
            msg.handles[0:1] = [r.choice(list(self.live))]
        elif roll < 0.08:
            msg.keys.append(r.choice(("nope", "__reflop__/meta")))
            msg.handles.append(self.new_handle())
        elif roll < 0.10 and msg.keys:
            msg.handles.append(self.new_handle())
        elif roll < 0.12 and len(msg.handles) >= 2:
            msg.handles[1] = msg.handles[0]
        elif roll < 0.14 and msg.handles:
            msg.handles[0] = self.bad_handle()
        msg.restore_rng = r.random() < 0.2
        self.internal(files=files, snapshot_read=msg)
        if roll >= 0.14:
            for h, k in zip(msg.handles, msg.keys):
                if k in keys:
                    self.live[h] = keys[k]


def _dt(pb: int):
    from reflop_worker.dtypes import PB_TO_TORCH

    return PB_TO_TORCH.get(pb, torch.float32)


ACTIONS = (("upload_start", 16), ("upload_continue", 10), ("graph", 18), ("fetch", 10), ("free", 5),
           ("reseed", 1), ("ping", 1), ("stats", 1), ("unknown_graph", 1), ("dataset_open", 7), ("gather", 7),
           ("dataset_close", 2), ("snapshot_write", 8), ("snapshot_read", 8), ("fill", 3))


def sequence(seed: int, index: int, device: str = "cpu") -> SeqCase:
    r = random.Random(seed * 1_000_003 + index + 7_777_777)
    g = Gen(r, device)
    names, weights = zip(*ACTIONS)
    for _ in range(r.randint(4, 16)):
        getattr(g, r.choices(names, weights)[0])()
    return g.case
