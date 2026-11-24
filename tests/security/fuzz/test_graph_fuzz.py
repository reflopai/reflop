# SPDX-License-Identifier: Apache-2.0
"""Structure-aware fuzzing of the worker's graph pipeline (GH#4, ADR-044):
decode, structure, schema, meta inference, memory analysis, bindings,
runtime checks and execution, on an in-process CPU worker. The properties
are in fuzzrun.py.

CI runs a fixed budget from fixed seeds; the nightly workflow runs a larger
one from a fresh seed:

    REFLOP_FUZZ_SEED=<n> REFLOP_FUZZ_CASES=<n> REFLOP_FUZZ_OUT=<dir> pytest tests/security/fuzz

REFLOP_FUZZ_DEVICE=cuda runs the same cases on CUDA, each seed in child
processes (isolated.py), and replays the corpus with its graphs moved to
CUDA. CI has no GPU; run it on an Owner machine.

Before each case its replay file is written to REFLOP_FUZZ_OUT (if set),
so a case that crashes the process is on disk. Add a failing case to
corpus/ as a regression; every file there replays on every run."""

from __future__ import annotations

import os

import pytest
from fuzzrun import CORPUS, decode, frames, fuzz_device, fuzz_graphs, load, retarget, run_frames, still_serving
from graphgen import case, make_session
from isolated import run_isolated

SEEDS = [int(s) for s in os.environ.get("REFLOP_FUZZ_SEED", "1,2,3,4").split(",")]
CASES = int(os.environ.get("REFLOP_FUZZ_CASES", "1250"))
DEVICE = fuzz_device()


@pytest.fixture(scope="module")
def sess():
    return make_session(DEVICE)


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz(request, seed):
    if DEVICE != "cpu":
        failures = run_isolated("graph", DEVICE, seed, CASES)
        assert not failures, "\n".join(failures[:20])
        return
    sess = request.getfixturevalue("sess")
    failures = fuzz_graphs(sess, seed, 0, CASES)
    assert not failures, "\n".join(failures[:20])
    assert still_serving(sess) is None


CORPUS_FILES = sorted(CORPUS.glob("*.bin"))


@pytest.mark.parametrize("path", CORPUS_FILES, ids=[p.stem for p in CORPUS_FILES])
def test_corpus(sess, path):
    assert run_frames(sess, retarget(load(path), sess.gdevice)) == []
    assert still_serving(sess) is None


def test_generators_reach_every_stage(sess):
    """The cases are not all rejected at the first stage: some register,
    some fail in each validation stage."""
    from reflop_worker._pb import compute_pb2 as cpb

    stages, registered = set(), 0
    for i in range(400):
        c = case(99, i, sess)
        if c.kind == "bytes":
            continue
        out: list = []
        try:
            env = decode(frames(c)[0]).client
        except Exception:  # noqa: BLE001 - rejected at V1 (unknown enum value)
            stages.add(cpb.STAGE_V1_DECODE)
            continue
        sess.handle(env, out.append)
        r = out[-1]
        if r.WhichOneof("body") == "error":
            stages.add(r.error.stage)
        else:
            registered += 1
    assert {cpb.STAGE_V1_DECODE, cpb.STAGE_V2_STRUCTURE, cpb.STAGE_V3_SCHEMA, cpb.STAGE_V4_META} <= stages, stages
    assert registered > 40, registered
