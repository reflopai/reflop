# SPDX-License-Identifier: Apache-2.0
"""Snapshot and checkpoint files, written in the background (ADR-029).

Server tensors are never modified in place (a rebind replaces the table
entry, and kernels that write to arguments write copies, writes.py), so
holding references to the current tensors gives a consistent
point-in-time view. The cut is taken on the request thread between two
requests: requests before it are in the file, later ones are not. A
background thread then copies the tensors to the host through a small
pinned ring buffer on a separate CUDA stream and writes them to the file
descriptor reflopd passed. Execution continues meanwhile.

Costs, measured on an RTX 5070 (docs/fault_tolerance.md):
- GPU memory: a tensor rebound during the copy keeps its old version
  allocated until it has been copied, so the session's state can need up to
  twice its size. Each reference is dropped as soon as its tensor has been
  copied. When the GPU does not have that much free memory, the cut is
  skipped, or, when reflopd allows it (SnapshotWrite.allow_stall), the state
  is copied on the request thread before any later request runs: no tensor
  is rebound meanwhile, so no second copy is needed, and the session's
  requests wait for the copy (ADR-061).
- Targets: reflopd may pass two descriptors, a memfd for a RAM checkpoint and
  a disk file, and picks one when it answers the cut (SnapshotGo.target).
- Host memory: two RING-byte buffers per session, pinned once.
- Nothing is cached between checks; the file is plain safetensors and is
  validated again in full when it is read back (I6, I9).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from contextlib import ExitStack

import torch

from ._pb import compute_pb2 as cpb
from ._pb import worker_pb2 as wpb
from .errors import ReflopError
from .safetensors import RESERVED, SafetensorsError, check_key, flat, header

RING = 64 << 20
META_KEY = RESERVED + "meta"
RNG_CPU_KEY = RESERVED + "rng/cpu"
RNG_CUDA_KEY = RESERVED + "rng/cuda"
MAX_TAG = 4096
# Headroom the GPU keeps beyond the state's size before a cut is allowed.
GPU_MARGIN = 256 << 20
# Keep backing storage alive until CUDA confirms it has unmapped the pointer,
# including when teardown raises and the session is collected.
_registered_buffers: dict[int, bytearray] = {}


class Skip(Exception):
    """The checkpoint is not taken now; the previous one stays."""


def free_device_bytes(device: torch.device) -> int:
    """Memory a new allocation can use: free on the device plus what the
    caching allocator holds but does not use."""
    free, _total = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


class Ring:
    """Two host buffers for staging chunks; pinned when the device is CUDA.

    They are bytearrays so that os.write can take them directly (the runtime
    has no numpy and the worker uses no ctypes); CUDA pins them with
    cudaHostRegister."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.bufs = [bytearray(RING) for _ in range(2)]
        self.views = [torch.frombuffer(b, dtype=torch.uint8) for b in self.bufs]
        self._registered: list[int] = []
        if device.type == "cuda":
            with torch.cuda.device(device):
                rt = torch.cuda.cudart()
                for buf, view in zip(self.bufs, self.views):
                    ptr = view.data_ptr()
                    if int(rt.cudaHostRegister(ptr, RING, 0)) != 0:
                        break
                    _registered_buffers[ptr] = buf
                    self._registered.append(ptr)
        self.pinned = len(self._registered) == len(self.bufs)

    def close(self) -> None:
        if not self._registered:
            return
        with torch.cuda.device(self.device):
            rt = torch.cuda.cudart()
            error = 0
            for ptr in self._registered[:]:
                status = int(rt.cudaHostUnregister(ptr))
                if status == 0:
                    self._registered.remove(ptr)
                    del _registered_buffers[ptr]
                elif error == 0:
                    error = status
            self.pinned = False
            if error:
                torch.cuda.check_error(error)


class Writer:
    """One snapshot or checkpoint write at a time per session."""

    def __init__(self, sess) -> None:
        self.sess = sess
        self.ring: Ring | None = None
        self.thread: threading.Thread | None = None
        self.go: dict[int, tuple[threading.Event, list]] = {}
        self.lock = threading.Lock()

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def close(self) -> None:
        """Release staging buffers after the in-process writer has finished."""
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.ring is not None:
            self.ring.close()
            self.ring = None

    # -------------------------------------------------------------- the cut

    def plan(self, msg: wpb.SnapshotWrite) -> tuple[list, bytes, int, bool]:
        """Take the cut. Returns (items, header bytes, total file bytes, stall).

        items = [(key, tensor)] in file order; stall: copy now, on the
        request thread (the GPU has no room for a background copy). Runs on
        the request thread."""
        sess = self.sess
        if self.busy():
            raise ReflopError(cpb.ERROR_BUSY, "a snapshot or checkpoint of this session is still being written")
        items: list = []
        if msg.checkpoint:
            if msg.handles or msg.keys:
                raise ReflopError(cpb.ERROR_PROTOCOL, "a checkpoint names no handles")
            if len(msg.tag) > MAX_TAG:
                raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, f"checkpoint tags are at most {MAX_TAG} bytes")
            for h, t in sorted(sess.handles.t.items()):
                items.append((h.hex(), t))
            meta = json.dumps({"generation": int(msg.generation), "created_unix": int(time.time()),
                               "tag": base64.b64encode(bytes(msg.tag)).decode()}).encode()
            items.append((META_KEY, torch.frombuffer(bytearray(meta), dtype=torch.uint8)))
            items.append((RNG_CPU_KEY, torch.get_rng_state()))
            if sess.device.type == "cuda":
                items.append((RNG_CUDA_KEY, torch.cuda.get_rng_state(sess.device)))
        else:
            if len(msg.handles) != len(msg.keys) or not msg.handles:
                raise ReflopError(cpb.ERROR_PROTOCOL, "handles and keys must be non-empty and equal in length")
            seen = set()
            for h, k in zip(msg.handles, msg.keys):
                try:
                    check_key(k)
                except SafetensorsError as e:
                    raise ReflopError(cpb.ERROR_PROTOCOL, f"snapshot key {k[:64]!r}: {e}") from None
                if k in seen:
                    raise ReflopError(cpb.ERROR_PROTOCOL, f"duplicate key {k[:64]!r}")
                seen.add(k)
                items.append((k, sess.handles.get(h)))
            items.sort(key=lambda kv: kv[0])
        try:
            hb, data = header([(k, t.dtype, tuple(t.shape)) for k, t in items])
        except SafetensorsError as e:
            raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, str(e)) from None
        stall = False
        short = self.copy_room(items)
        if short:
            if not msg.allow_stall:
                raise Skip(f"{short}. The next checkpoint that goes to the Owner's disk copies it while your "
                           "requests wait instead")
            stall = True
        return items, hb, len(hb) + data, stall

    def copy_room(self, items) -> str:
        """Why the GPU has no room for a background copy of items, or ""."""
        sess = self.sess
        if sess.device.type != "cuda":
            return ""
        on_device = sum(t.numel() * t.element_size() for _k, t in items if t.device.type == "cuda")
        free = free_device_bytes(sess.device)
        if on_device + GPU_MARGIN > free:
            return (f"the GPU has {free >> 20} MiB free; a background copy of {on_device >> 20} MiB of state needs "
                    f"that much again (plus {GPU_MARGIN >> 20} MiB)")
        return ""

    # -------------------------------------------------------------- start / go

    def start(self, msg: wpb.SnapshotWrite, fds, cut, done, wait_go=None) -> None:
        """Cut now, report the size with cut(rid, bytes, stall), wait for go,
        write to the descriptor go names.

        fds: one descriptor, or the list SnapshotWrite.targets announced.
        done(envelope) is called once, from the writer thread or from here.
        For a stalled cut, wait_go(gate) must deliver SnapshotGo (by reading
        the control channel) when cut() did not already."""
        rid = msg.request_id
        fds = [fds] if isinstance(fds, int) else list(fds)
        try:
            items, hb, total, stall = self.plan(msg)
            if stall and wait_go is None:
                raise Skip("the GPU has no room for a background copy of the state")
        except Skip as e:
            _close_all(fds)
            done(cpb.Envelope(request_id=rid, end=True, checkpoint_saved=cpb.CheckpointSaved(
                skipped=True, reason=str(e), generation=msg.generation)))
            return
        except BaseException:
            _close_all(fds)
            raise
        ev = None
        if self.sess.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()  # all work queued so far on the request stream
        gate = threading.Event()
        verdict: list = []
        with self.lock:
            self.go[rid] = (gate, verdict)
        if self.ring is None:
            self.ring = Ring(self.sess.device)
        if stall:
            # No request runs until the state is copied, so nothing is
            # rebound and the GPU needs no second copy.
            t0 = time.monotonic()
            cut(rid, total, True)
            if not gate.is_set():
                wait_go(gate)
            self._run(msg, fds, items, hb, total, ev, gate, verdict, done, stalled_since=t0)
            return
        self.thread = threading.Thread(target=self._run, args=(msg, fds, items, hb, total, ev, gate, verdict, done),
                                       name="reflop-snapshot", daemon=True)
        self.thread.start()
        cut(rid, total, False)

    def on_go(self, g: wpb.SnapshotGo) -> None:
        with self.lock:
            entry = self.go.pop(g.request_id, None)
        if entry is not None:
            entry[1].append(g)
            entry[0].set()

    # -------------------------------------------------------------- the thread

    def _run(self, msg, fds, items, hb, total, ev, gate, verdict, done, stalled_since=None) -> None:
        rid = msg.request_id
        try:
            gate.wait()
            g = verdict[0]
            if not g.proceed:
                items.clear()
                done(cpb.Envelope(request_id=rid, end=True, error=cpb.Error(
                    code=g.code or cpb.ERROR_SAFETY_RAIL, message=g.reason[:500])))
                return
            if g.target >= len(fds):
                raise ValueError(f"write target {g.target} of {len(fds)}")
            fd = fds[g.target]
            _close_all(fds[:g.target] + fds[g.target + 1:])
            fds = [fd]
            self._write(fd, items, hb, ev)
            if msg.checkpoint:
                paused = 0 if stalled_since is None else int((time.monotonic() - stalled_since) * 1000)
                done(cpb.Envelope(request_id=rid, end=True, checkpoint_saved=cpb.CheckpointSaved(
                    bytes=total, generation=msg.generation, paused_ms=max(paused, 1) if stalled_since else 0)))
            else:
                done(cpb.Envelope(request_id=rid, end=True, snapshot_saved=cpb.SnapshotSaved(bytes=total)))
        except BaseException as e:  # noqa: BLE001 - the thread must always answer
            items.clear()
            what = "checkpoint" if msg.checkpoint else "snapshot"
            done(cpb.Envelope(request_id=rid, end=True, error=cpb.Error(
                code=cpb.ERROR_INTERNAL, message=f"{what} write failed: {type(e).__name__}: {str(e)[:300]}")))
        finally:
            _close_all(fds)

    def _write(self, fd: int, items: list, hb: bytes, ev) -> None:
        _write_all(fd, memoryview(hb))
        ring = self.ring
        cuda = ev is not None
        stream = torch.cuda.Stream(self.sess.device) if cuda else None
        evs = [torch.cuda.Event() for _ in range(2)] if cuda else None
        if cuda:
            stream.wait_event(ev)
        k = 0
        pending = None  # (slot, nbytes, item index if this chunk ends the item, else -1)

        def flush_pending():
            ps, pm, last = pending
            if cuda:
                evs[ps].synchronize()
            _write_all(fd, memoryview(ring.bufs[ps])[:pm])
            if last >= 0:
                items[last] = None  # copied: the GPU memory of an old version can go

        with ExitStack() as stack:
            if cuda:
                stack.enter_context(torch.cuda.stream(stream))
                # A disk error can leave a copy in flight to the other slot.
                stack.callback(stream.synchronize)
            for i in range(len(items)):
                t = items[i][1]
                n = t.numel() * t.element_size()
                if n == 0:
                    items[i] = None
                    continue
                src = t.detach()
                if not src.is_contiguous():
                    src = src.contiguous()
                flat_src = flat(src).view(torch.uint8)
                for off in range(0, n, RING):
                    m = min(RING, n - off)
                    slot = k % 2
                    if pending is not None and pending[0] == slot:
                        flush_pending()
                        pending = None
                    ring.views[slot][:m].copy_(flat_src[off:off + m], non_blocking=cuda and ring.pinned)
                    if cuda:
                        evs[slot].record(stream)
                    if pending is not None:
                        flush_pending()
                    pending = (slot, m, i if off + m >= n else -1)
                    k += 1
                del src, flat_src, t
            if pending is not None:
                flush_pending()


def _close_all(fds) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _write_all(fd: int, mv: memoryview) -> None:
    while mv:
        n = os.write(fd, mv)
        if n <= 0:
            raise OSError("short write")
        mv = mv[n:]
