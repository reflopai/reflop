# SPDX-License-Identifier: Apache-2.0
"""Structure-aware fuzzing of the worker's graph pipeline (GH#4, ADR-044):
decode, structure, schema, meta inference, memory analysis, bindings,
runtime checks and execution, on an in-process CPU worker. The properties
are in fuzzrun.py.

CI runs a fixed budget from fixed seeds; the nightly workflow runs a larger
one from a fresh seed:

    REFLOP_FUZZ_SEED=<n> REFLOP_FUZZ_CASES=<n> REFLOP_FUZZ_OUT=<dir> pytest tests/security/fuzz

Before each case its replay file is written to REFLOP_FUZZ_OUT (if set),
so a case that crashes the process is on disk. Add a failing case to
corpus/ as a regression; every file there replays on every run."""

from __future__ import annotations

import os

import pytest
from fuzzrun import CORPUS, artifact_dir, decode, frames, load, run_frames, save, still_serving
from graphgen import case, make_session

SEEDS = [int(s) for s in os.environ.get("REFLOP_FUZZ_SEED", "1,2,3,4").split(",")]
CASES = int(os.environ.get("REFLOP_FUZZ_CASES", "1250"))


@pytest.fixture(scope="module")
def sess():
    return make_session("cpu")


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz(sess, seed):
    out = artifact_dir()
    failures = []
    for i in range(CASES):
        c = case(seed, i, sess)
        if out is not None:
            save(c, out / "current.bin")
        findings = run_frames(sess, frames(c), c)
        if not findings and i % 50 == 0:
            gone = still_serving(sess)
            findings = [gone] if gone else []
        if findings:
            failures.append(f"seed {seed} case {i} ({c.kind}): " + "; ".join(findings))
            if out is not None:
                save(c, out / f"seed{seed}-case{i}.bin")
    assert not failures, "\n".join(failures[:20])
    assert still_serving(sess) is None


CORPUS_FILES = sorted(CORPUS.glob("*.bin"))


@pytest.mark.parametrize("path", CORPUS_FILES, ids=[p.stem for p in CORPUS_FILES])
def test_corpus(sess, path):
    assert run_frames(sess, load(path)) == []
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
