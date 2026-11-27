# SPDX-License-Identifier: Apache-2.0
"""NativeBaseline in the worker (ADR-064): the request picks one of reflopd's own test models by name and a
batch from a fixed list. Anything else is refused before a model is built or memory is allocated, and a run
leaves the session's RNG streams as they were."""

import pytest
import torch
from reflop.connection import default_limits
from reflop_worker import nativebench, testzoo
from reflop_worker._pb import capabilities_pb2 as kpb
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.core import Session


def session() -> Session:
    init = wpb.WorkerInit(seed=0)
    init.device.kind = gpb.DEVICE_CPU
    init.limits.CopyFrom(default_limits(kpb))
    return Session(init)


@pytest.mark.parametrize("workload,batch", [
    ("", 32), ("evil", 32), ("__import__('os')", 32), ("../testzoo", 32), ("x" * 10000, 32), ("MLP", 32),
    ("mlp", 0), ("mlp", 7), ("mlp", 2**31), ("mlp", 129),
])
def test_refused_before_anything_runs(workload, batch, monkeypatch):
    monkeypatch.setattr(testzoo, "build", lambda *_: pytest.fail("a model was built for a refused request"))
    out = []
    session().handle(cpb.Envelope(request_id=3, native_baseline=cpb.NativeBaseline(workload=workload, batch=batch)),
                     out.append)
    assert len(out) == 1 and out[0].WhichOneof("body") == "error", out
    e = out[0].error
    assert e.code == cpb.ERROR_PROTOCOL and len(e.message) < 300


def test_runs_reflopds_model_and_keeps_the_session_rng():
    s = session()
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    out = []
    s.handle(cpb.Envelope(request_id=4, native_baseline=cpb.NativeBaseline(workload="mlp", batch=8)), out.append)
    r = out[0].native_baseline_result
    assert r.workload == "mlp" and r.batch == 8 and r.ms_per_step > 0 and r.zoo_version == testzoo.ZOO_VERSION
    assert len(r.ms_reps) == testzoo.REPEATS and r.steps >= 2
    assert torch.equal(torch.get_rng_state(), before)


def test_every_listed_workload_is_known():
    for name in testzoo.WORKLOADS:
        assert nativebench.check(cpb.NativeBaseline(workload=name, batch=testzoo.DEFAULT_BATCH)) is testzoo.WORKLOADS[name]
