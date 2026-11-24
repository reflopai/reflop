# SPDX-License-Identifier: Apache-2.0
"""What the sequence fuzzer found (#25), as plain regression cases. Replays
of the original fuzz cases are in tests/security/fuzz/corpus/seq."""

from __future__ import annotations

import os

import pytest
import torch
from helpers import call, graph, make_session, meta, tin
from reflop_worker import safetensors as st
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.errors import ReflopError
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

K = cpb.OutputSpec
H = [bytes([0x20 + i]) * 16 for i in range(8)]


def _upload(sess, h, t=None, m=None, data=None):
    u = cpb.UploadTensor(handle=h, final=True)
    u.meta.CopyFrom(m if m is not None else meta(t.dtype, t.shape))
    u.data = data if data is not None else st.tensor_bytes(t)
    return call(sess, cpb.Envelope(request_id=9, upload_tensor=u))


# ---------------------------------------------------------------- uploads


def test_an_upload_with_no_elements_whose_strides_overflow():
    # numel is 0, so the element limit passed, and torch.empty raised a
    # RuntimeError that the session answered as INTERNAL.
    sess = make_session("cpu")
    r = _upload(sess, H[0], m=gpb.TensorMeta(dtype=gpb.DTYPE_FLOAT32, shape=[0, 2**62, 2**62]), data=b"")
    assert r[-1].error.code == cpb.ERROR_RESOURCE_LIMIT, r
    assert len(sess.handles) == 0


def test_bool_bytes_are_canonical_at_every_way_in(tmp_path):
    # Kernels disagree on a bool byte that is neither 0 nor 1 (CPU clone gives
    # 1, CUDA clone and index_select keep it), and C++ assumes 0 or 1.
    raw = bytes([0, 1, 2, 255, 0x80])
    sess = make_session("cpu")
    _upload(sess, H[0], m=meta(torch.bool, [5]), data=raw)
    assert sess.handles.t[H[0]].view(torch.uint8).tolist() == [0, 1, 1, 1, 1]
    # An inline binding, returned as is by clone on CPU: stays canonical.
    g = graph(sess, [(torch.bool, (5,))], [("aten.alias.default", [tin(0)])], [(torch.bool, (5,))])
    reg = call(sess, cpb.Envelope(request_id=1, register_graph=cpb.RegisterGraph(graph=g)))[-1]
    ex = cpb.Execute(graph_hash=reg.graph_registered.graph_hash, outputs=[K(kind=K.KIND_RETURN)])
    b = ex.bindings.add()
    b.inline.meta.CopyFrom(meta(torch.bool, [5]))
    b.inline.data = raw
    r = call(sess, cpb.Envelope(request_id=2, execute=ex))[-1]
    assert r.execute_result.returned[0].data == bytes([0, 1, 1, 1, 1]), r
    # A snapshot read.
    p = tmp_path / "snap"
    hb, _ = st.header([("m", torch.bool, (5,))])
    p.write_bytes(hb + raw)
    fd = os.open(p, os.O_RDONLY)
    sess.snapshot_read(wpb.SnapshotRead(request_id=3, keys=["m"], handles=[H[1]]), fd)
    assert sess.handles.t[H[1]].view(torch.uint8).tolist() == [0, 1, 1, 1, 1]


# ---------------------------------------------------------------- execute outputs


def _two_outputs(sess):
    g = graph(sess, [(torch.float32, (3,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (3,))] * 2,
              outputs=[(0, 0), (0, 0)])
    return call(sess, cpb.Envelope(request_id=1, register_graph=cpb.RegisterGraph(graph=g)))[-1]


@pytest.mark.parametrize("first", ["store", "rebind"])
def test_execute_commits_every_output_or_none(first):
    # E1 did not count new handles against the limit: the commit stored (or
    # rebound) the first output, then failed on the second.
    sess = make_session("cpu", max_handles_per_session=3)
    _upload(sess, H[0], torch.ones(3))
    _upload(sess, H[1], torch.ones(3))
    reg = _two_outputs(sess)
    ex = cpb.Execute(graph_hash=reg.graph_registered.graph_hash)
    ex.bindings.add().handle = H[0]
    if first == "store":
        ex.outputs.extend([K(kind=K.KIND_STORE, handle=H[2]), K(kind=K.KIND_STORE, handle=H[3])])
    else:
        _upload(sess, H[2], torch.ones(3))
        ex.outputs.extend([K(kind=K.KIND_REBIND, handle=H[1]), K(kind=K.KIND_STORE, handle=H[3])])
    before = dict(sess.handles.t)
    r = call(sess, cpb.Envelope(request_id=2, execute=ex))[-1]
    assert r.error.code == cpb.ERROR_RESOURCE_LIMIT, r
    assert sess.handles.t == before
    # With room for both, both are stored.
    sess.handles.free(H[1])
    if first == "rebind":
        sess.handles.free(H[2])
        ex.outputs[0].CopyFrom(K(kind=K.KIND_STORE, handle=H[2]))
    r = call(sess, cpb.Envelope(request_id=3, execute=ex))[-1]
    assert r.WhichOneof("body") == "execute_result", r
    assert torch.equal(sess.handles.t[H[2]], -torch.ones(3)) and torch.equal(sess.handles.t[H[3]], -torch.ones(3))


# ---------------------------------------------------------------- datasets


def _shard(path, tensors: dict) -> str:
    items = [(n, t.dtype, tuple(t.shape)) for n, t in tensors.items()]
    hb, _ = st.header(items)
    with open(path, "wb") as f:
        f.write(hb)
        for t in tensors.values():
            f.write(st.tensor_bytes(t))
    return str(path)


def _manifest(tensors: dict, rows: list[int]) -> wpb.DatasetOpen:
    msg = wpb.DatasetOpen(dataset_id=b"d" * 16)
    for n, (dt, shape) in tensors.items():
        msg.tensors.append(wpb.DatasetTensor(name=n, meta=meta(dt, shape)))
    for i, k in enumerate(rows):
        msg.shards.append(wpb.DatasetShard(index=i, num_samples=k))
    return msg


def _open(paths):
    fds = [os.open(p, os.O_RDONLY) for p in paths]
    return fds, [os.fstat(fd).st_ino for fd in fds]


def _all_closed(fds, inodes) -> bool:
    for fd, ino in zip(fds, inodes):
        try:
            if os.fstat(fd).st_ino == ino:
                return False
        except OSError:
            pass
    return True


def _good(tmp_path, rows=(2, 3, 1)):
    paths = [_shard(tmp_path / f"s{i}", {"x": torch.full((k, 2), float(i))}) for i, k in enumerate(rows)]
    return paths, {"x": (torch.float32, (sum(rows), 2))}, list(rows)


def _refusals(tmp_path):
    paths, want, rows = _good(tmp_path)
    bad_header = tmp_path / "bad"
    bad_header.write_bytes(b"\x08\0\0\0\0\0\0\0{\"x\": 1}")
    huge = tmp_path / "huge"
    hb = b'{"x":{"dtype":"F32","shape":[0,1099511627776,1099511627776],"data_offsets":[0,0]}}'
    huge.write_bytes(len(hb).to_bytes(8, "little") + hb)
    return {
        "unknown dtype": (paths, {"x": (None, (6, 2))}, rows),
        "0-d tensor": (paths, {"x": (torch.float32, ())}, rows),
        "rank differs": (paths, {"x": (torch.float32, (6, 2, 1))}, rows),
        "rank differs, 1-d": (paths, {"x": (torch.float32, (6,))}, rows),
        "bad shard header": ([paths[0], str(bad_header), paths[2]], want, rows),
        "huge dims": ([paths[0], str(huge), paths[2]], want, rows),
        "sizes do not add up": (paths, {"x": (torch.float32, (7, 2))}, rows),
        "one descriptor short": (paths[:2], want, rows),
    }


@pytest.mark.parametrize("case", ["unknown dtype", "0-d tensor", "rank differs", "rank differs, 1-d",
                                  "bad shard header", "huge dims", "sizes do not add up", "one descriptor short"])
def test_a_refused_dataset_is_error_dataset_and_closes_every_descriptor(tmp_path, case):
    # A KeyError (dtype), an IndexError (0-d) and a SafetensorsError (shard
    # header) escaped as non-ReflopErrors, and an error at shard i left the
    # descriptors of shards i+1… open.
    paths, want, rows = _refusals(tmp_path)[case]
    msg = _manifest({n: (torch.float32 if dt is None else dt, s) for n, (dt, s) in want.items()}, rows)
    if case == "unknown dtype":
        msg.tensors[0].meta.dtype = 99
    sess = make_session("cpu")
    fds, inodes = _open(paths)
    with pytest.raises(ReflopError) as e:
        sess.open_dataset(msg, fds)
    assert e.value.code == cpb.ERROR_DATASET
    assert _all_closed(fds, inodes)
    assert msg.dataset_id not in sess.datasets


def test_a_dataset_names_each_tensor_once(tmp_path):
    paths, want, rows = _good(tmp_path)
    msg = _manifest(want, rows)
    msg.tensors.append(msg.tensors[0])
    sess = make_session("cpu")
    fds, inodes = _open(paths)
    with pytest.raises(ReflopError) as e:
        sess.open_dataset(msg, fds)
    assert e.value.code == cpb.ERROR_DATASET and _all_closed(fds, inodes)


def test_an_open_dataset_closes_its_descriptors(tmp_path):
    paths, want, rows = _good(tmp_path)
    sess = make_session("cpu")
    fds, inodes = _open(paths)
    sess.open_dataset(_manifest(want, rows), fds)
    assert _all_closed(fds, inodes)  # the maps keep the files open


class _Largest(TorchDispatchMode):
    """Size of the largest storage aten calls create, by device type."""

    def __init__(self) -> None:
        super().__init__()
        self.largest = {"cpu": 0, "cuda": 0}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t.device.type in self.largest:
                d = t.device.type
                self.largest[d] = max(self.largest[d], t.untyped_storage().nbytes())
        return out


def test_a_gather_reads_rows_shard_by_shard(tmp_path, devname):
    # The first gather concatenated every shard in host memory: the whole
    # dataset at once, outside the V5 host rail.
    rows = [3, 0, 4, 1, 0, 2]
    parts = [torch.arange(sum(rows[:i]), sum(rows[:i + 1]), dtype=torch.float32).repeat_interleave(64).view(-1, 64)
             for i in range(len(rows))]
    raw = torch.tensor([0, 1, 2, 255, 7, 0, 3, 1, 9, 0], dtype=torch.uint8)  # bool bytes, not all 0 or 1
    masks = [raw[sum(rows[:i]):sum(rows[:i + 1])].view(torch.bool) for i in range(len(rows))]
    paths = [_shard(tmp_path / f"s{i}", {"x": p, "m": m}) for i, (p, m) in enumerate(zip(parts, masks))]
    msg = _manifest({"x": (torch.float32, (10, 64)), "m": (torch.bool, (10,))}, rows)
    sess = make_session(devname)
    sess.open_dataset(msg, _open(paths)[0])
    ds = sess.datasets[msg.dataset_id]
    whole = torch.cat(parts)
    idx = torch.tensor([9, 0, 3, 3, 7, 2, 8, 0], device=sess.device)
    for name, ref in (("x", whole), ("m", torch.cat(masks).view(torch.uint8).clamp(max=1).view(torch.bool))):
        t = ds.tensors[name]
        with _Largest() as m:
            got = t.gather(idx)
        assert torch.equal(got.cpu().view(torch.uint8), ref.index_select(0, idx.cpu()).view(torch.uint8)), name
        if t._on_device is None:
            assert m.largest["cpu"] <= 8 * 64 * 4, (name, m.largest)  # the 8 rows, never the 10
    assert torch.equal(ds.tensors["x"].gather(idx[:0]).cpu(), whole[:0])
    ds.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_resident_dataset_is_copied_shard_by_shard(tmp_path):
    rows = [2, 0, 3]
    parts = [torch.randn(k, 5) for k in rows]
    paths = [_shard(tmp_path / f"s{i}", {"x": p}) for i, p in enumerate(parts)]
    sess = make_session("cuda")
    with _Largest() as m:
        sess.open_dataset(_manifest({"x": (torch.float32, (5, 5))}, rows), _open(paths)[0])
    t = sess.datasets[b"d" * 16].tensors["x"]
    assert t._on_device is not None
    assert torch.equal(t._on_device.cpu(), torch.cat(parts))
    assert m.largest["cpu"] < 5 * 5 * 4, m.largest  # no host copy of the whole dataset
