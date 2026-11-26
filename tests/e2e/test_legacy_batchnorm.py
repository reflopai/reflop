# SPDX-License-Identifier: Apache-2.0
"""Clients up to 0.2.3 still train BatchNorm's running statistics (GH#49).

Those clients send native_batch_norm with training=True for every BatchNorm
forward and get the new running statistics only through the kernel's write
to the bound tensors. The worker now runs that kernel on copies and points
the statistics' handles at them (ADR-058); this pins that the statistics
still come out as in plain PyTorch. The legacy client is simulated by
dropping the client's batch norm special, so native_batch_norm is recorded
as is, as 0.2.3 recorded it. (A forward whose output is dropped never ran
on 0.2.3, so its statistics stayed: BUGS.md #3, fixed on the client in
0.2.4. Nothing on the server can change that.)
"""

from __future__ import annotations

import copy
import os

import pytest
import reflop
import torch
import torch.nn as nn
from reflop import session
from reflop import tensor as rtensor

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
TOL = 1e-4 if DEVICE == "cuda" else 1e-5


@pytest.fixture(scope="module", params=["local", "server"])
def dev(request):
    if request.param == "server" and os.environ.get("REFLOP_E2E") != "1":
        pytest.skip("needs a running reflopd (REFLOP_E2E=1)")
    if request.param == "local":
        index, d = 43, reflop.device(index=43, local=DEVICE)
    else:
        index, d = 44, reflop.device(index=44, kind=DEVICE)
    yield d
    session.by_index(index).close()


@pytest.fixture
def legacy(monkeypatch):
    from reflop.session import Session

    sent = []
    monkeypatch.delitem(rtensor._SPECIALS, "aten.native_batch_norm.default")
    record = Session.record_values

    def spy(self, name, func, vals, metas, self_is_value=False):
        if name == "aten.native_batch_norm.default":
            sent.append(bool(vals[5]))
        return record(self, name, func, vals, metas, self_is_value)

    monkeypatch.setattr(Session, "record_values", spy)
    return sent


def test_legacy_training_updates_the_running_statistics(dev, legacy):
    torch.manual_seed(0)
    local = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 4, 3), nn.BatchNorm2d(4))
    remote = copy.deepcopy(local).to(dev)
    g = torch.Generator().manual_seed(1)
    for _ in range(3):
        x = torch.randn(4, 3, 10, 10, generator=g) * 2 + 1
        want = local(x)
        torch.testing.assert_close(remote(x.to(dev)).cpu(), want, atol=TOL, rtol=TOL)
    reflop.synchronize()
    assert legacy and all(legacy), "native_batch_norm was not sent in training mode"
    for r, lo in zip(remote.modules(), local.modules()):
        if isinstance(lo, nn.BatchNorm2d):
            torch.testing.assert_close(r.running_mean.cpu(), lo.running_mean, atol=TOL, rtol=TOL)
            torch.testing.assert_close(r.running_var.cpu(), lo.running_var, atol=TOL, rtol=TOL)
