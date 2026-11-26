# SPDX-License-Identifier: Apache-2.0
"""Session numerics: requested with reflop.device(numerics=...), reported
by reflop.numerics(), and kept out of the caller's process by in-process
workers (BUGS.md #18, GH#19).

Regression: an in-process worker (reflop.device(local=...)) applied its
session's TF32, cuDNN benchmark and deterministic switches to the whole
process when it started, and left them there. Native reference code that
ran afterwards, and later in-process sessions, then ran with another
session's switches, so CUDA results in a long test run depended on test
order, and a check against a native run did not compare like with like.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import reflop
import torch
from reflop import session

ROOT = Path(__file__).resolve().parents[2]
CHILD = Path(__file__).with_name("numerics_child.py")
sys.path.insert(0, str(Path(__file__).parent))
from numerics_child import STYLES  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"


def _child(style):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "client/python"), str(ROOT / "server/worker")]),
               CUDA_VISIBLE_DEVICES="", REFLOP_NO_KEYCHAIN="1")
    env.pop("REFLOP_LOCAL", None)
    r = subprocess.run([sys.executable, str(CHILD), style], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


GETTERS = ("float32_matmul_precision", "matmul.allow_tf32", "cudnn.allow_tf32")


def _same_for_the_caller(now, before):
    # The effective precision, cuDNN benchmark and deterministic mode are
    # the caller's again, and every getter that answered still answers the
    # same. A getter that raised may answer now: PyTorch raises when the
    # legacy and new TF32 APIs were mixed, and which API last wrote a
    # value is its own record, not a setting.
    assert {k: v for k, v in now.items() if k not in GETTERS} == {k: v for k, v in before.items() if k not in GETTERS}
    for k in GETTERS:
        if before[k] != "raises":
            assert now[k] == before[k], k


@pytest.mark.parametrize("style", list(STYLES))
def test_in_process_sessions_leave_the_callers_numerics(style):
    out = _child(style)
    _same_for_the_caller(out["after"], out["before"])
    for s in out["sessions"]:
        _same_for_the_caller(s["caller_now"], out["before"])
        # The worker ran its requests with its own switches.
        w = s["want"]
        assert s["during"] == [w["tf32_matmul"], w["tf32_cudnn"], w["cudnn_benchmark"], w["deterministic"]]
        assert s["reported"] == w
        assert s["value"] == [0.0, 2.0, 4.0, 6.0]


@pytest.fixture
def opened():
    made = []
    yield made
    for i in made:
        try:
            session.by_index(i).close()
        except KeyError:
            pass


def test_numerics_left_out_keep_the_defaults(opened):
    d = reflop.device(53, local=DEVICE, numerics={"deterministic": True})
    opened.append(53)
    n = reflop.numerics(d)
    assert n["deterministic"] is True and n["tf32_matmul"] is False and n["cudnn_benchmark"] is False
    assert reflop.numerics(53) == n


def test_reopening_with_other_numerics_raises(opened):
    reflop.device(54, local=DEVICE, numerics={"deterministic": True})
    opened.append(54)
    assert reflop.device(54, numerics={"deterministic": True}).index == 54  # the same: fine
    with pytest.raises(ValueError, match="already open"):
        reflop.device(54, numerics={"deterministic": False})


@pytest.mark.parametrize("bad, err", [({"tf32": True}, ValueError), ({"deterministic": 1}, TypeError),
                                      ([("deterministic", True)], TypeError)])
def test_bad_numerics_are_refused(bad, err):
    with pytest.raises(err):
        reflop.device(55, local=DEVICE, numerics=bad)
    with pytest.raises(KeyError):
        session.by_index(55)


def test_numerics_of_a_device_that_is_not_open():
    with pytest.raises(ValueError, match="not open"):
        reflop.numerics(56)


@pytest.mark.skipif(os.environ.get("REFLOP_E2E") != "1", reason="needs a running reflopd (REFLOP_E2E=1)")
def test_the_server_applies_requested_numerics(opened):
    d = reflop.device(57, kind=DEVICE, numerics={"deterministic": True, "cudnn_benchmark": False})
    opened.append(57)
    n = reflop.numerics(d)
    assert n["deterministic"] is True and n["cudnn_benchmark"] is False
    x = torch.arange(6.0).reshape(2, 3).to(d)
    assert (x @ x.t()).cpu().tolist() == [[5.0, 14.0], [14.0, 50.0]]
