# SPDX-License-Identifier: Apache-2.0
"""Request sequences against the worker (#25, ADR-049): uploads, fetches,
executes, datasets, snapshots and checkpoints in random order on one
in-process CPU session, with files and descriptors handled as reflopd
does. The generator is seqgen.py; the properties are in seqrun.py.

CI runs a fixed budget from fixed seeds; the nightly workflow runs a larger
one from a fresh seed:

    REFLOP_FUZZ_SEED=<n> REFLOP_FUZZ_SEQ_CASES=<n> REFLOP_FUZZ_OUT=<dir> pytest tests/security/fuzz

REFLOP_FUZZ_DEVICE=cuda runs them on CUDA, as test_graph_fuzz.py does.

Before each case its replay file is written to REFLOP_FUZZ_OUT (if set),
so a case that crashes the process is on disk. Add a failing case to
corpus/seq/ as a regression; every file there replays on every run."""

from __future__ import annotations

import os

import pytest
from fuzzrun import CORPUS, fuzz_device, load, retarget
from helpers import NAME
from isolated import run_isolated
from reflop_worker._pb import graph_pb2 as gpb
from seqgen import frames, sequence
from seqrun import fuzz_sequences, run_fresh

SEEDS = [int(s) for s in os.environ.get("REFLOP_FUZZ_SEED", "1,2,3,4").split(",")]
CASES = int(os.environ.get("REFLOP_FUZZ_SEQ_CASES", "100"))
DEVICE = fuzz_device()


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz_sequences(seed, tmp_path):
    if DEVICE != "cpu":
        failures = run_isolated("seq", DEVICE, seed, CASES)
    else:
        failures = fuzz_sequences(DEVICE, seed, 0, CASES, tmp_path)
    assert not failures, "\n".join(failures[:20])


SEQ_CORPUS = sorted((CORPUS / "seq").glob("*.bin"))


@pytest.mark.parametrize("path", SEQ_CORPUS, ids=[p.stem for p in SEQ_CORPUS])
def test_seq_corpus(path, tmp_path):
    frs = load(path)
    if DEVICE != "cpu":
        frs = retarget(frs, gpb.Device(kind=gpb.DEVICE_CUDA))
    assert run_fresh(DEVICE, frs, tmp_path)[0] == []


def test_an_output_dtype_the_wire_cannot_carry(tmp_path):
    # Seed 101, case 260: add(<0-dim complex64>, <float16 [3, 0]>) infers
    # complex32, which has no wire dtype, and building the graph raised
    # KeyError before any request reached the worker. The generator now
    # declares the input's meta and the worker refuses the graph in V4.
    # corpus/seq/complex32-promoted-output.bin replays the same sequence
    # whatever the generator does later.
    case = sequence(101, 260, DEVICE)
    C64, F16 = gpb.DTYPE_COMPLEX64, gpb.DTYPE_FLOAT16
    promoted = [g for g in (s.env.client.register_graph.graph for s in case.steps
                            if s.env.client.WhichOneof("body") == "register_graph")
                if [i.meta.dtype for i in g.inputs] == [C64, F16]
                and [n.op_id for n in g.nodes] == [NAME["aten.add.Tensor"]]]
    assert promoted, "the generator no longer makes this graph here; find a seed that does"
    assert [m.dtype for m in promoted[0].output_meta] == [C64]
    failures, seen = run_fresh(DEVICE, frames(case), tmp_path)
    assert failures == []
    assert "register_graph:error:ERROR_SHAPE_MISMATCH" in seen


# Paths the generator must reach, so a change to it cannot quietly stop
# testing one: each entry point's success, its refusals that leave state
# behind, and the limits.
REQUIRED = {
    "upload_tensor:ok", "upload:multi-chunk", "upload_tensor:error:ERROR_PROTOCOL",
    "upload_tensor:error:ERROR_RESOURCE_LIMIT", "upload_tensor:error:ERROR_SAFETY_RAIL",
    "fetch:one-chunk", "fetch:multi-chunk", "fetch:error:ERROR_RESOURCE_LIMIT", "fetch:error:ERROR_UNKNOWN_HANDLE",
    "register_graph:ok", "execute:KIND_STORE", "execute:KIND_REBIND", "execute:KIND_RETURN", "execute:KIND_DISCARD",
    "execute:stored-view", "execute:error:ERROR_BINDING", "execute:error:ERROR_HANDLE_EXISTS",
    "execute:error:ERROR_RESOURCE_LIMIT", "execute:error:ERROR_UNKNOWN_HANDLE", "execute:error:ERROR_UNKNOWN_DATASET",
    "free:ok", "dataset_open:ok", "dataset_open:error:ERROR_DATASET",
    "gather:ok:KIND_RETURN", "gather:ok:KIND_STORE", "gather:error:ERROR_INDEX_OUT_OF_BOUNDS",
    "snapshot_write:ok", "snapshot_write:ok:checkpoint", "snapshot_write:refused", "snapshot_write:error:ERROR_BUSY",
    "snapshot_read:ok:roundtrip", "snapshot_read:ok:roundtrip:rng", "snapshot_read:error:ERROR_PROTOCOL",
    "snapshot_read:error:ERROR_RESOURCE_LIMIT", "snapshot_read:error:ERROR_HANDLE_EXISTS",
}


def test_sequences_reach_every_path(tmp_path):
    seen: set[str] = set()
    for i in range(400):
        d = tmp_path / str(i)
        d.mkdir()
        seen |= run_fresh(DEVICE, frames(sequence(99, i, DEVICE)), d)[1]
        if REQUIRED <= seen:
            return
    pytest.fail(f"never reached: {sorted(REQUIRED - seen)}")
