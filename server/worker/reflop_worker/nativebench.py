# SPDX-License-Identifier: Apache-2.0
"""Native baselines for `reflop test` (ADR-064): one of reflopd's own testzoo workloads, trained with plain
PyTorch on the session's device, timed the way the client times it on reFLOP.

Nothing the Guest sent runs here: the request names a workload and a batch size, both checked against fixed
lists before anything is allocated. The session's state is left as it was: its RNG streams are restored, and
the run's memory is released before the next request.
"""

from __future__ import annotations

import time

import torch

from . import testzoo
from ._pb import compute_pb2 as cpb
from .errors import ReflopError


def check(req: cpb.NativeBaseline) -> testzoo.Workload:
    w = testzoo.WORKLOADS.get(req.workload)
    if w is None:
        raise ReflopError(cpb.ERROR_PROTOCOL, f"unknown test workload {req.workload[:40]!r}; "
                          f"known: {', '.join(testzoo.WORKLOADS)}")
    if req.batch not in testzoo.BATCHES:
        raise ReflopError(cpb.ERROR_PROTOCOL, f"batch {req.batch} is not one of {list(testzoo.BATCHES)}")
    return w


def run(device: torch.device, req: cpb.NativeBaseline) -> cpb.NativeBaselineResult:
    w = check(req)
    cuda = device.type == "cuda"
    with torch.random.fork_rng(devices=[device.index or 0] if cuda else []), torch.enable_grad():
        r = _measure(w, req.batch, device)
    if cuda:
        torch.cuda.empty_cache()   # the run's tensors are gone with _measure's frame
    return cpb.NativeBaselineResult(workload=w.name, batch=req.batch, ms_per_step=r["ms_per_step"],
                                    ms_reps=r["ms_reps"], steps=r["steps"], zoo_version=testzoo.ZOO_VERSION,
                                    measured_unix=int(time.time()))


def _measure(w: testzoo.Workload, batch: int, device: torch.device) -> dict:
    model = testzoo.build(w.name).to(device)
    opt = testzoo.optimizer(model)
    x, y = w.batch(batch)
    x, y = x.to(device), y.to(device)
    step = testzoo.make_step(model, opt, w.loss)
    if device.type == "cuda":
        return testzoo.time_steps(lambda: step(x, y), lambda _loss: torch.cuda.synchronize(device))
    return testzoo.time_steps(lambda: step(x, y), lambda _loss: None)
