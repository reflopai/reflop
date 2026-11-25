# SPDX-License-Identifier: Apache-2.0
"""torch.linalg calls that check `info` after the _ex kernel (#15, bench
B11). aten._linalg_check_errors is not in the opset: the client fetches
`info` and runs the check, so the call runs on the server under every
placement and raises PyTorch's own errors."""

from __future__ import annotations

import os

import pytest
import reflop
import torch

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
G = torch.Generator().manual_seed(0)
A = torch.randn(4, 4, generator=G, dtype=torch.float64)
SPD = A @ A.T + 4 * torch.eye(4, dtype=torch.float64)
B = torch.randn(4, 2, generator=G, dtype=torch.float64)


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=24, local=DEVICE)


CASES = {
    "inv": lambda a, b: torch.linalg.inv(a),
    "solve": lambda a, b: torch.linalg.solve(a, b),
    "cholesky": lambda a, b: torch.linalg.cholesky(a),
    "inv_batched": lambda a, b: torch.linalg.inv(torch.stack([a, 2 * a])),
    "cholesky_upper": lambda a, b: torch.linalg.cholesky(a, upper=True),
    "tensor_inverse": lambda a, b: a.inverse(),
}


@pytest.mark.parametrize("mode", ["remote", "auto"])
@pytest.mark.parametrize("name", list(CASES))
def test_results_match_pytorch(dev, name, mode, monkeypatch):
    from reflop import tensor

    def no_local_run(*a, **k):
        raise AssertionError("ran on the client")

    f = CASES[name]
    want = f(SPD, B)
    a, b = SPD.to(dev), B.to(dev)
    monkeypatch.setattr(tensor, "_run_locally", no_local_run)
    with reflop.placement(mode):
        got = f(a, b)
    monkeypatch.undo()
    assert got.device == dev
    assert torch.allclose(got.cpu(), want, rtol=1e-10, atol=1e-10)


def test_singular_matrix_raises_pytorchs_error(dev):
    z = torch.zeros(3, 3, dtype=torch.float64)
    with pytest.raises(torch.linalg.LinAlgError) as want:
        torch.linalg.inv(z)
    with reflop.placement("remote"), pytest.raises(torch.linalg.LinAlgError, match="singular") as got:
        torch.linalg.inv(z.to(dev))
    if DEVICE == "cpu":
        assert str(got.value) == str(want.value)
    with reflop.placement("remote"), pytest.raises(torch.linalg.LinAlgError, match="singular"):
        torch.linalg.solve(z.to(dev), B[:3].to(dev))


def test_not_positive_definite_raises_pytorchs_error(dev):
    m = -SPD
    with pytest.raises(torch.linalg.LinAlgError) as want:
        torch.linalg.cholesky(m)
    with reflop.placement("remote"), pytest.raises(torch.linalg.LinAlgError, match="positive-definite") as got:
        torch.linalg.cholesky(m.to(dev))
    assert str(got.value) == str(want.value)


def test_batch_errors_name_the_batch_element(dev):
    batch = torch.stack([SPD, torch.zeros(4, 4, dtype=torch.float64)])
    with reflop.placement("remote"), pytest.raises(torch.linalg.LinAlgError, match=r"\(Batch element 1\)"):
        torch.linalg.inv(batch.to(dev))


def test_ex_variants_return_info_without_raising(dev):
    m = torch.stack([SPD, -SPD])
    with reflop.placement("remote"):
        L, info = torch.linalg.cholesky_ex(m.to(dev))
    assert info.cpu().tolist()[0] == 0 and info.cpu().tolist()[1] != 0
    assert torch.allclose(L.cpu()[0], torch.linalg.cholesky(SPD), rtol=1e-10, atol=1e-10)
