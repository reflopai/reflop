# SPDX-License-Identifier: Apache-2.0
"""Run request sequences (seqgen.py) against an in-process worker Session
and check what each entry point promises (#25 A, ADR-049).

The runner does what reflopd does for the reflopd-internal messages: it
writes the case's files to a temporary directory and passes descriptors,
collects snapshot cuts and gives the go. After every request it compares
the session with what the request should have done:

- every client request ends in exactly one final response, the last one,
  with no exception escaping and no INTERNAL error; a non-final upload
  chunk gets no response or one error;
- uploads: a stored handle holds exactly the bytes sent; a failed stream
  is dropped; streams in progress never reserve more than the host budget;
- fetch: chunks are well formed and carry the handle's bytes, and a fetch
  holds at most one tensor copy at a time;
- execute: results have the registered metas, memory stays within V5, a
  failed execute leaves the handle table as it was, and a gather returns
  the rows of the dataset (an out-of-range index never succeeds);
- dataset_open raises only ReflopError, closes every descriptor, and a
  dataset it accepts holds the shard files' data;
- snapshot_write raises only ReflopError, answers exactly once, and writes
  the cut's bytes, or nothing if refused; snapshot_read raises only
  ReflopError, closes its descriptor, and restores the file's tensors or
  changes nothing;
- the session's handles stay within its limits, and no descriptor or
  mapping of a case file is left once the case ends.

Each run returns the findings and the set of events it reached (the
coverage test asserts that every path was taken)."""

from __future__ import annotations

import gc
import json
import math
import os
import threading
from pathlib import Path

import torch
from fuzzrun import LiveBytes, artifact_dir, decode, device_fault, save_frames, still_serving
from google.protobuf.message import DecodeError
from reflop_worker import wire
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.checkpoint import META_KEY, RNG_CPU_KEY, RNG_CUDA_KEY
from reflop_worker.dtypes import PB_TO_TORCH, TORCH_TO_PB
from reflop_worker.errors import FatalError, ReflopError
from reflop_worker.handles import check_new
from reflop_worker.safetensors import NAMES, tensor_bytes
from seqgen import HOST_BUDGET, frames, make_session, sequence

FILE, PREV = b"FILE", b"PREV"
GATHER = "reflop.dataset_gather.default"
K = cpb.OutputSpec
BY_NAME = {v: k for k, v in NAMES.items()}


class Finding(Exception):
    pass


def ref_parse(data: bytes):
    """A reference safetensors reader: {name: (dtype, shape, bytes)}, or
    None where the file is malformed. Kept apart from the worker's parser
    so that a disagreement is visible."""
    if len(data) < 8:
        return None
    n = int.from_bytes(data[:8], "little")
    if n == 0 or 8 + n > len(data):
        return None

    def no_dupes(pairs):
        d = dict(pairs)
        if len(d) != len(pairs):
            raise ValueError("duplicate")
        return d

    try:
        hdr = json.loads(data[8:8 + n].decode("utf-8"), object_pairs_hook=no_dupes)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(hdr, dict):
        return None
    body = data[8 + n:]
    out, spans = {}, []
    for name, e in hdr.items():
        if name == "__metadata__":
            continue
        if not isinstance(e, dict) or set(e) != {"dtype", "shape", "data_offsets"}:
            return None
        dt = BY_NAME.get(e["dtype"]) if isinstance(e["dtype"], str) else None
        shape, offs = e["shape"], e["data_offsets"]
        if dt is None or not isinstance(shape, list) or not isinstance(offs, list) or len(offs) != 2:
            return None
        if not all(type(d) is int and d >= 0 for d in shape) or not all(type(o) is int for o in offs):
            return None
        b, f = offs
        if not 0 <= b <= f <= len(body) or f - b != math.prod(shape) * dt.itemsize:
            return None
        spans.append((b, f))
        out[name] = (dt, tuple(shape), body[b:f])
    spans.sort()
    end = 0
    for b, f in spans:
        if b < end:
            return None
        end = f
    return out if end == len(body) else None


_BOOL = bytes([0] + [1] * 255)


def _canon(dt, raw: bytes) -> bytes:
    """raw as the worker holds it: bool bytes other than 0 read as 1."""
    return raw.translate(_BOOL) if dt == torch.bool else raw


def _rows(dt, shape, raw: bytes) -> torch.Tensor:
    if not raw:
        return torch.empty(shape, dtype=dt)
    return torch.frombuffer(bytearray(_canon(dt, raw)), dtype=torch.uint8).view(dt).reshape(shape)


def _storages(sess) -> set[int]:
    """Data pointers of the storage the session already holds."""
    out = {t.untyped_storage().data_ptr() for t in sess.handles.t.values()}
    for ds in sess.datasets.values():
        for t in ds.tensors.values():
            for p in t.parts + [t._on_device]:
                if p is not None:
                    out.add(p.untyped_storage().data_ptr())
    return out


def _nbytes(m) -> int:
    return math.prod(m.shape) * PB_TO_TORCH[m.dtype].itemsize if m.dtype in PB_TO_TORCH else 0


def _same(a: dict, b: dict) -> bool:
    """The same handles bound to the same tensor objects."""
    return a.keys() == b.keys() and all(a[k] is b[k] for k in a)


def _meta(t: torch.Tensor) -> tuple:
    return TORCH_TO_PB[t.dtype], tuple(t.shape)


class Runner:
    def __init__(self, sess, tmp: Path) -> None:
        self.sess = sess
        self.tmp = tmp
        self.events: set[str] = set()
        self.queued: list = []
        self.registered = None        # (GraphRegistered, Graph) of the last successful registration
        self.uploads: dict[int, dict] = {}  # rid -> {"handle", "meta", "data", "chunks"}
        self.last_snap: bytes | None = None
        self.pending: dict[int, dict] = {}  # snapshot writes waiting for the go
        self.done: list = []
        self.lock = threading.Lock()
        self.datasets: dict[bytes, dict] = {}  # id -> {name: reference tensor}
        self.nfile = 0

    # -------------------------------------------------------------- driver

    def run(self, frames: list[bytes]) -> list[str]:
        try:
            for fr in frames:
                if fr[:4] in (FILE, PREV):
                    self.queued.append(fr[4:] if fr[:4] == FILE else json.loads(fr[4:]))
                    continue
                try:
                    env = decode(fr)
                except (DecodeError, wire.ProtocolError, RecursionError):
                    continue  # the worker treats these as fatal before any handler runs
                which = env.WhichOneof("body")
                if which == "client":
                    self.client(env.client)
                elif which is not None:
                    getattr(self, which)(getattr(env, which))
                self.queued = []
                self.check_state()
            self.finish()
        except Finding as f:
            self.abandon()
            return [str(f)]
        return []

    def file(self, data: bytes) -> tuple[int, tuple]:
        self.nfile += 1
        p = self.tmp / f"f{self.nfile}"
        p.write_bytes(data)
        fd = os.open(p, os.O_RDONLY)
        st = os.fstat(fd)
        return fd, (st.st_dev, st.st_ino)

    def closed(self, fd: int, ident: tuple, what: str) -> None:
        try:
            st = os.fstat(fd)
        except OSError:
            return
        if (st.st_dev, st.st_ino) == ident:
            os.close(fd)
            raise Finding(f"{what} left its descriptor open")

    def check_state(self) -> None:
        s, lim = self.sess, self.sess.limits
        fault = device_fault(s)
        if fault:
            raise Finding(fault)
        if len(s.handles) > s.handles.max:
            raise Finding(f"{len(s.handles)} live handles; the limit is {s.handles.max}")
        for h, t in s.handles.t.items():
            try:
                check_new(h)
            except ReflopError:
                raise Finding(f"invalid handle {h.hex()} in the table") from None
            if t.device.type != s.device.type or t.dtype not in TORCH_TO_PB:
                raise Finding(f"handle {h.hex()[:8]} holds a {t.dtype} tensor on {t.device}")
            if t.dim() > lim.max_dims or t.numel() > lim.max_tensor_numel:
                raise Finding(f"handle {h.hex()[:8]} holds {t.dtype}{list(t.shape)}, beyond the session's limits")
        reserved = sum(len(o["buf"]) for o in s.uploads.values())
        if reserved > HOST_BUDGET:
            raise Finding(f"uploads in progress reserve {reserved} bytes; the budget is {HOST_BUDGET}")
        if set(s.uploads) != set(self.uploads):
            raise Finding(f"upload streams {sorted(s.uploads)}, expected {sorted(self.uploads)}")

    # -------------------------------------------------------------- client requests

    def client(self, req: cpb.Envelope) -> None:
        s = self.sess
        which = req.WhichOneof("body")
        rid = req.request_id
        if which == "execute" and not req.execute.graph_hash and self.registered is not None:
            req.execute.graph_hash = self.registered[0].graph_hash
        before = dict(s.handles.t)
        ctx = {}
        if which == "fetch":
            ctx["want"] = [(_meta(s.handles.t[h]), tensor_bytes(s.handles.t[h])) if h in s.handles.t else None
                           for h in req.fetch.handles]
        elif which == "upload_tensor":
            ctx["new"] = rid not in s.uploads
        meter = LiveBytes(exclude=_storages(s)) if which in ("execute", "fetch") else None
        out: list = []
        try:
            if meter is None:
                s.handle(req, out.append)
            else:
                with meter:
                    s.handle(req, out.append)
        except FatalError as e:
            raise Finding(f"{which}: worker lost its device: {e}") from None
        except Exception as e:  # noqa: BLE001
            raise Finding(f"{which}: {type(e).__name__} escaped Session.handle: {str(e)[:300]}") from None
        if any(o.request_id != rid for o in out):
            raise Finding(f"{which}: a response carries another request id")
        if which == "upload_tensor" and not req.upload_tensor.final and not (out and out[-1].error.code):
            if out:
                raise Finding(f"upload: a non-final chunk was answered with {out[-1].WhichOneof('body')}")
            return self.upload(req, None, ctx)
        finals = [k for k, o in enumerate(out) if o.end]
        if finals != [len(out) - 1]:
            raise Finding(f"{which}: final responses at {finals} of {len(out)}")
        r = out[-1]
        body = r.WhichOneof("body")
        if body == "error":
            e = r.error
            self.events.add(f"{which}:error:{cpb.ErrorCode.Name(e.code)}")
            if e.code == cpb.ERROR_INTERNAL:
                raise Finding(f"{which}: INTERNAL error: {e.message[:300]}")
            if (which in ("register_graph", "execute") and e.stage == cpb.STAGE_NONE
                    and e.code not in (cpb.ERROR_PROTOCOL, cpb.ERROR_UNKNOWN_GRAPH)):
                raise Finding(f"{which}: error without a stage: {cpb.ErrorCode.Name(e.code)} {e.message[:200]}")
            if which in ("execute", "upload_tensor", "fetch") and not _same(s.handles.t, before):
                raise Finding(f"{which}: failed with {cpb.ErrorCode.Name(e.code)} but changed the handle table")
            if which == "register_graph":
                self.registered = None
            elif which == "upload_tensor":
                self.uploads.pop(rid, None)
                if rid in s.uploads:
                    raise Finding("upload: a failed stream is still open")
            elif which == "fetch":
                self.fetch_failed(req.fetch, e, ctx)
            elif which == "execute":
                self.execute_failed(req.execute, e)
            return
        self.events.add(f"{which}:ok")
        getattr(self, which)(req, r, ctx, out=out, meter=meter, before=before)

    def register_graph(self, req, r, ctx, **_):
        if r.WhichOneof("body") != "graph_registered":
            raise Finding(f"register answered {r.WhichOneof('body')}")
        self.registered = (r.graph_registered, req.register_graph.graph)

    def upload_tensor(self, req, r, ctx, **_):
        if r.WhichOneof("body") != "uploaded":
            raise Finding(f"upload answered {r.WhichOneof('body')}")
        self.upload(req, r, ctx)

    def upload(self, req, r, ctx) -> None:
        u, rid = req.upload_tensor, req.request_id
        if ctx["new"]:
            self.uploads[rid] = {"handle": u.handle, "meta": (u.meta.dtype, tuple(u.meta.shape)), "data": bytearray(),
                                 "chunks": 0}
        st = self.uploads[rid]
        st["data"] += u.data
        st["chunks"] += 1
        if r is None:
            return
        del self.uploads[rid]
        t = self.sess.handles.t.get(st["handle"])
        if t is None or r.uploaded.handle != st["handle"]:
            raise Finding("upload: answered uploaded, but the handle is not in the table")
        if _meta(t) != st["meta"]:
            raise Finding(f"upload: stored {_meta(t)}, the stream declared {st['meta']}")
        if tensor_bytes(t) != _canon(t.dtype, bytes(st["data"])):
            raise Finding("upload: the stored tensor differs from the bytes sent")
        self.events.add("upload:multi-chunk" if st["chunks"] > 1 else "upload:one-chunk")

    def fetch(self, req, r, ctx, out, meter, **_):
        want = ctx["want"]
        if None in want:
            raise Finding("fetch of an unknown handle succeeded")
        total = sum(len(b) for _m, b in want)
        if total > self.sess.limits.max_fetch_bytes:
            raise Finding(f"fetch of {total} bytes succeeded over the limit")
        chunk = self.sess.limits.upload_chunk_bytes
        if not want:
            if len(out) != 1 or out[0].fetch_chunk != cpb.FetchChunk():
                raise Finding("an empty fetch is answered with one empty chunk")
            return
        got: list[bytearray] = [bytearray() for _ in want]
        k = 0
        for o in out:
            c = o.fetch_chunk
            if o.WhichOneof("body") != "fetch_chunk" or c.index != k or c.offset != len(got[k]):
                raise Finding(f"fetch: chunk out of order (index {c.index}, offset {c.offset})")
            if c.HasField("meta") != (c.offset == 0):
                raise Finding("fetch: meta is sent with the first chunk of a tensor only")
            if c.offset == 0 and (c.meta.dtype, tuple(c.meta.shape)) != want[k][0]:
                raise Finding(f"fetch: meta {c.meta} for a {want[k][0]} tensor")
            if len(c.data) > chunk:
                raise Finding(f"fetch: a chunk of {len(c.data)} bytes")
            got[k] += c.data
            done = len(got[k]) == len(want[k][1])
            if c.last_of_tensor != done:
                raise Finding("fetch: last_of_tensor is wrong")
            if done:
                k += 1
        if k != len(want) or [bytes(g) for g in got] != [b for _m, b in want]:
            raise Finding("fetch: the bytes differ from the handles'")
        self.events.add("fetch:multi-chunk" if len(out) > len(want) else "fetch:one-chunk")
        largest = max(len(b) for _m, b in want)
        if meter.peak > largest + 4096:
            raise Finding(f"fetch held {meter.peak} bytes of tensors; the largest is {largest}")

    def fetch_failed(self, f, e, ctx) -> None:
        want = ctx["want"]
        if None not in want and sum(len(b) for _m, b in want) <= self.sess.limits.max_fetch_bytes:
            raise Finding(f"fetch of known handles failed: {e.message[:200]}")

    def free(self, req, r, ctx, before, **_):
        left = set(before) - set(req.free.handles)
        if set(self.sess.handles.t) != left:
            raise Finding("free: the table is not the old one minus the freed handles")

    def reseed(self, req, r, ctx, **_):
        if r.WhichOneof("body") != "reseeded":
            raise Finding(f"reseed answered {r.WhichOneof('body')}")

    def ping(self, req, r, ctx, **_):
        if r.pong.nonce != req.ping.nonce:
            raise Finding("ping: the nonce is not echoed")

    def execute(self, req, r, ctx, meter, before, **_):
        if r.WhichOneof("body") != "execute_result":
            raise Finding(f"execute answered {r.WhichOneof('body')}")
        reg, g = self.registered
        ex, res = req.execute, r.execute_result
        metas = [(m.dtype, tuple(m.shape)) for m in reg.output_meta]
        ret = [m for m, o in zip(metas, ex.outputs) if o.kind == K.KIND_RETURN]
        if [(t.meta.dtype, tuple(t.meta.shape)) for t in res.returned] != ret:
            raise Finding(f"execute returned {[t.meta for t in res.returned]}, registered {ret}")
        for t in res.returned:
            if len(t.data) != _nbytes(t.meta):
                raise Finding(f"execute returned {len(t.data)} bytes for {t.meta}")
        for o, m in zip(ex.outputs, metas):
            self.events.add(f"execute:{K.Kind.Name(o.kind)}")
            if o.kind in (K.KIND_STORE, K.KIND_REBIND):
                t = self.sess.handles.t.get(o.handle)
                if t is None or _meta(t) != m:
                    raise Finding(f"execute: {K.Kind.Name(o.kind)} output is {t if t is None else _meta(t)}, "
                                  f"registered {m}")
                if not t.is_contiguous() or t.untyped_storage().nbytes() != t.numel() * t.element_size():
                    self.events.add("execute:stored-view")
        largest = max([_nbytes(m) for m in reg.output_meta] +
                      [_nbytes(i.meta) for i in g.inputs if i.kind == gpb.GraphInput.KIND_TENSOR] + [0])
        if meter.peak > reg.peak_memory_bytes + 2 * largest + 4096:
            raise Finding(f"execute held {meter.peak} bytes; V5 computed {reg.peak_memory_bytes}")
        gathered = self.gather_rows(ex, g)
        if gathered is not None:
            idx, ref = gathered
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= ref.shape[0]):
                raise Finding(f"gather of rows {idx.tolist()} from {ref.shape[0]} succeeded")
            self.events.add(f"gather:ok:{K.Kind.Name(ex.outputs[0].kind)}")
            want = tensor_bytes(ref.index_select(0, idx.long()))
            o = ex.outputs[0]
            got = (res.returned[0].data if o.kind == K.KIND_RETURN
                   else tensor_bytes(self.sess.handles.t[o.handle]) if o.kind == K.KIND_STORE else want)
            if got != want:
                raise Finding("gather returned rows other than the dataset's")

    def execute_failed(self, ex, e) -> None:
        if self.registered is None:
            return
        gathered = self.gather_rows(ex, self.registered[1])
        if gathered is not None and e.code in (cpb.ERROR_INDEX_OUT_OF_BOUNDS, cpb.ERROR_EXECUTION):
            self.events.add(f"gather:error:{cpb.ErrorCode.Name(e.code)}")

    def gather_rows(self, ex, g):
        """(indices, reference rows) when this execute is a gather of a dataset the runner knows."""
        from parity import SPEC

        if (len(g.nodes) != 1 or g.nodes[0].op_id != SPEC[GATHER].id or len(ex.bindings) != 2
                or ex.bindings[0].WhichOneof("value") != "dataset" or ex.bindings[1].WhichOneof("value") != "inline"):
            return None
        ref = self.datasets.get(ex.bindings[0].dataset.dataset_id, {}).get(ex.bindings[0].dataset.tensor)
        m = ex.bindings[1].inline.meta
        if (ref is None or m.dtype not in (gpb.DTYPE_INT64, gpb.DTYPE_INT32)
                or len(ex.bindings[1].inline.data) != _nbytes(m)):
            return None
        return _rows(PB_TO_TORCH[m.dtype], tuple(m.shape), ex.bindings[1].inline.data), ref

    # -------------------------------------------------------------- reflopd-internal messages

    def stats_request(self, _msg) -> None:
        st = self.sess.stats()
        if st.handles != len(self.sess.handles):
            raise Finding(f"stats reports {st.handles} handles, the table has {len(self.sess.handles)}")

    def dataset_open(self, msg: wpb.DatasetOpen) -> None:
        s = self.sess
        files = [f for f in self.queued if isinstance(f, bytes)]
        opened = [self.file(f) for f in files]
        try:
            s.open_dataset(msg, [fd for fd, _ in opened])
            ok = True
        except ReflopError as e:
            ok = False
            self.events.add(f"dataset_open:error:{cpb.ErrorCode.Name(e.code)}")
        except Exception as e:  # noqa: BLE001
            for fd, ident in opened:
                try:
                    self.closed(fd, ident, "dataset_open")
                except Finding:
                    pass
            raise Finding(f"dataset_open raised {type(e).__name__}: {str(e)[:300]}") from None
        for fd, ident in opened:
            self.closed(fd, ident, "dataset_open")
        self.datasets.pop(msg.dataset_id, None)
        if not ok:
            if msg.dataset_id in s.datasets:
                raise Finding("dataset_open failed but the dataset is open")
            return
        self.events.add("dataset_open:ok")
        ds = s.datasets.get(msg.dataset_id)
        parsed = [ref_parse(f) for f in files]
        if ds is None or None in parsed or len(parsed) != len(msg.shards):
            raise Finding("dataset_open accepted shards the reference reader rejects")
        model = {}
        for t in msg.tensors:
            got = ds.tensors.get(t.name)
            parts = [_rows(*p[t.name]) for p in parsed if t.name in p]
            want = torch.cat(parts, 0) if len(parts) == len(parsed) else None
            if got is None or want is None or (got.dtype, tuple(got.shape)) != (want.dtype, tuple(want.shape)):
                raise Finding(f"dataset tensor {t.name!r} does not match its shards")
            have = torch.cat(got.parts, 0) if got.parts else torch.empty(0)
            if _canon(have.dtype, tensor_bytes(have)) != tensor_bytes(want):
                raise Finding(f"dataset tensor {t.name!r} holds other data than its shards")
            model[t.name] = want
        self.datasets[msg.dataset_id] = model

    def dataset_close(self, msg) -> None:
        try:
            self.sess.close_dataset(msg.dataset_id)
        except Exception as e:  # noqa: BLE001
            raise Finding(f"dataset_close raised {type(e).__name__}: {e}") from None
        self.datasets.pop(msg.dataset_id, None)
        if msg.dataset_id in self.sess.datasets:
            raise Finding("dataset_close left the dataset open")

    def snapshot_write(self, msg: wpb.SnapshotWrite) -> None:
        s, rid = self.sess, msg.request_id
        self.nfile += 1
        path = self.tmp / f"snap{self.nfile}"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        st = os.fstat(fd)
        ident = (st.st_dev, st.st_ino)
        if msg.checkpoint:
            want = {h.hex(): (_meta(t), tensor_bytes(t)) for h, t in s.handles.t.items()}
            want[RNG_CPU_KEY] = ((gpb.DTYPE_UINT8, (torch.get_rng_state().numel(),)), tensor_bytes(torch.get_rng_state()))
        else:
            want = {k: (_meta(s.handles.t[h]), tensor_bytes(s.handles.t[h]))
                    for h, k in zip(msg.handles, msg.keys) if h in s.handles.t}
        cuts: list = []
        n_done = len(self.done)
        try:
            s.snapshot_write(msg, fd, lambda r, n: cuts.append((r, n)), self._done)
        except ReflopError as e:
            self.events.add(f"snapshot_write:error:{cpb.ErrorCode.Name(e.code)}")
            self.closed(fd, ident, "snapshot_write")
            if cuts or len(self.done) != n_done:
                raise Finding("snapshot_write failed but also reported a cut or an answer") from None
            return
        except Exception as e:  # noqa: BLE001
            raise Finding(f"snapshot_write raised {type(e).__name__}: {str(e)[:300]}") from None
        if rid not in s.writer.go:
            # Skipped (not enough GPU memory): answered at once.
            self.closed(fd, ident, "snapshot_write")
            self.answered(rid, n_done)
            return
        if len(cuts) != 1 or cuts[0][0] != rid:
            raise Finding(f"snapshot_write reported cuts {cuts}")
        self.events.add("snapshot_write:cut" + (":checkpoint" if msg.checkpoint else ""))
        self.pending[rid] = {"fd": fd, "ident": ident, "path": path, "want": want, "bytes": cuts[0][1],
                             "n_done": n_done, "checkpoint": msg.checkpoint}

    def _done(self, env) -> None:
        with self.lock:
            self.done.append(env)

    def answered(self, rid: int, n_done: int):
        th = self.sess.writer.thread
        if th is not None:
            th.join(60)
            if th.is_alive():
                raise Finding("the snapshot writer did not finish within 60 s")
        with self.lock:
            got = [e for e in self.done[n_done:] if e.request_id == rid]
        if len(got) != 1 or not got[0].end:
            raise Finding(f"snapshot {rid}: {len(got)} answers")
        e = got[0]
        if e.WhichOneof("body") == "error" and e.error.code == cpb.ERROR_INTERNAL:
            raise Finding(f"snapshot {rid}: INTERNAL error: {e.error.message[:300]}")
        return e

    def snapshot_go(self, g: wpb.SnapshotGo) -> None:
        p = self.pending.get(g.request_id)
        self.sess.writer.on_go(g)
        if p is None:
            self.events.add("snapshot_go:unknown")
            return
        del self.pending[g.request_id]
        e = self.answered(g.request_id, p["n_done"])
        self.closed(p["fd"], p["ident"], "the snapshot writer")
        data = p["path"].read_bytes()
        if not g.proceed:
            if e.WhichOneof("body") != "error" or e.error.code != (g.code or cpb.ERROR_SAFETY_RAIL):
                raise Finding(f"a refused snapshot was answered with {e}")
            if data:
                raise Finding(f"a refused snapshot wrote {len(data)} bytes")
            self.events.add("snapshot_write:refused")
            return
        body = e.WhichOneof("body")
        n = e.checkpoint_saved.bytes if body == "checkpoint_saved" else e.snapshot_saved.bytes
        if body not in ("checkpoint_saved", "snapshot_saved") or n != p["bytes"] or len(data) != n:
            raise Finding(f"snapshot: answered {body} of {n} bytes, cut {p['bytes']}, wrote {len(data)}")
        parsed = ref_parse(data)
        if parsed is None:
            raise Finding("snapshot_write wrote a file the reference reader rejects")
        want = p["want"]
        keys = set(want) | ({META_KEY} if p["checkpoint"] else set())
        keys |= {RNG_CUDA_KEY} if p["checkpoint"] and self.sess.device.type == "cuda" else set()
        if set(parsed) != keys:
            raise Finding(f"snapshot holds keys {sorted(parsed)[:8]}, expected {sorted(keys)[:8]}")
        for k, (m, raw) in want.items():
            dt, shape, got = parsed[k]
            if (TORCH_TO_PB[dt], shape) != m or got != raw:
                raise Finding(f"snapshot key {k!r} holds other data than the handle had at the cut")
        self.last_snap = data
        self.events.add("snapshot_write:ok" + (":checkpoint" if p["checkpoint"] else ""))

    def snapshot_read(self, msg: wpb.SnapshotRead) -> None:
        s = self.sess
        src = self.queued[0] if self.queued else b""
        roundtrip = isinstance(src, list) and not src and self.last_snap is not None
        if isinstance(src, list):
            data = bytearray(self.last_snap or b"")
            rng = (ref_parse(bytes(data)) or {}).get(RNG_CPU_KEY)
            for d in src:
                if d[0] == "rng" and rng is not None:
                    # A field of the mt19937 state: seeded, left or next.
                    at = bytes(data).index(rng[2]) + d[1]
                    data[at] ^= d[2]
                elif d[0] == "flip" and data:
                    data[int(d[1] * len(data))] ^= d[2]
                elif d[0] == "cut":
                    del data[int(d[1] * len(data)):]
                elif d[0] == "add":
                    data += bytes.fromhex(d[1])
            src = bytes(data)
        fd, ident = self.file(src)
        before = dict(s.handles.t)
        rng = torch.get_rng_state()
        try:
            s.snapshot_read(msg, fd)
            ok = True
        except ReflopError as e:
            ok = False
            self.events.add(f"snapshot_read:error:{cpb.ErrorCode.Name(e.code)}")
        except Exception as e:  # noqa: BLE001
            try:
                self.closed(fd, ident, "snapshot_read")
            except Finding:
                pass
            raise Finding(f"snapshot_read raised {type(e).__name__}: {str(e)[:300]}") from None
        self.closed(fd, ident, "snapshot_read")
        if not ok:
            if not _same(s.handles.t, before) or not torch.equal(torch.get_rng_state(), rng):
                raise Finding("a failed snapshot_read changed the session")
            return
        parsed = ref_parse(src)
        if parsed is None:
            raise Finding("snapshot_read accepted a file the reference reader rejects")
        if set(s.handles.t) != set(before) | set(msg.handles) or any(s.handles.t[h] is not t for h, t in before.items()):
            raise Finding("snapshot_read changed handles it did not restore")
        for h, k in zip(msg.handles, msg.keys):
            if k not in parsed:
                raise Finding(f"snapshot_read restored {k!r}, which the reference reader does not see")
            dt, shape, raw = parsed[k]
            t = s.handles.t[h]
            if (t.dtype, tuple(t.shape)) != (dt, shape) or tensor_bytes(t) != _canon(dt, raw):
                raise Finding(f"snapshot_read restored {k!r} as other data than the file's")
        if msg.restore_rng:
            # set_state normalizes some fields, so compare with a generator given the same bytes.
            g = torch.Generator()
            g.set_state(_rows(torch.uint8, (-1,), parsed[RNG_CPU_KEY][2]).clone())
            if not torch.equal(torch.get_rng_state(), g.get_state()):
                raise Finding("snapshot_read did not restore the RNG state")
        self.events.add("snapshot_read:ok" + (":roundtrip" if roundtrip else "") + (":rng" if msg.restore_rng else ""))

    # -------------------------------------------------------------- end of a case

    def finish(self) -> None:
        s = self.sess
        for rid in list(s.writer.go):
            p = self.pending.pop(rid, None)
            s.writer.on_go(wpb.SnapshotGo(request_id=rid, proceed=False, code=cpb.ERROR_SAFETY_RAIL))
            if p is not None:
                self.answered(rid, p["n_done"])
                self.closed(p["fd"], p["ident"], "the snapshot writer")
        err = still_serving(s)
        if err:
            raise Finding(err)
        s.writer.close()
        for ds_id in list(s.datasets):
            s.close_dataset(ds_id)
        s.handles.t.clear()
        s.uploads.clear()
        gc.collect()
        self.check_leaks()

    def abandon(self) -> None:
        """After a finding: release what the case holds so the next case starts clean."""
        s = self.sess
        for rid in list(s.writer.go):
            s.writer.on_go(wpb.SnapshotGo(request_id=rid, proceed=False))
        s.writer.close()
        for ds_id in list(s.datasets):
            s.close_dataset(ds_id)
        for p in self.pending.values():
            try:
                os.close(p["fd"])
            except OSError:
                pass

    def check_leaks(self) -> None:
        tmp = str(self.tmp)
        for fd in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
            if target.startswith(tmp):
                os.close(int(fd))
                raise Finding(f"descriptor {fd} for {target[len(tmp):]} is still open after the case")
        with open("/proc/self/maps") as f:
            for line in f:
                if tmp in line:
                    raise Finding(f"a case file is still mapped: {line.split()[-1][len(tmp):]}")


def run_sequence(sess, frames: list[bytes], tmp: Path) -> tuple[list[str], set[str]]:
    r = Runner(sess, tmp)
    findings = r.run(frames)
    return findings, r.events


def run_fresh(device: str, frs: list[bytes], tmp: Path) -> tuple[list[str], set[str]]:
    """A sequence on a new session, as each case gets."""
    sess = make_session(device)
    try:
        return run_sequence(sess, frs, tmp)
    finally:
        sess.writer.close()


def fuzz_sequences(device: str, seed: int, start: int, end: int, tmp: Path, *, first: bool = False,
                   progress=None) -> list[str]:
    """Sequences start..end of a seed, each on a new session; the failures.
    first and progress as in fuzzrun.fuzz_graphs."""
    out = artifact_dir()
    tag = "seq-" if device == "cpu" else f"seq-{device}-"
    failures = []
    for i in range(start, end):
        if progress is not None:
            progress(i)
        frs = frames(sequence(seed, i, device))
        if out is not None:
            save_frames(frs, out / f"{tag}current.bin")
        d = tmp / str(i)
        d.mkdir()
        findings, _ = run_fresh(device, frs, d)
        if findings:
            failures.append(f"seed {seed} case {i}: " + "; ".join(findings))
            if out is not None:
                save_frames(frs, out / f"{tag}seed{seed}-case{i}.bin")
            if first:
                break
    return failures
