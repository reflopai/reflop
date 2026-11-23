# SPDX-License-Identifier: Apache-2.0
"""Backward through select, diagonal and slice matches plain PyTorch on every
dim. select_backward and diagonal_backward take a grad_output with fewer dims
than the input; the server used to check their dims against grad_output's
rank and reject y.select(1, 0).sum().backward()."""

from __future__ import annotations

import itertools
import os

import pytest
import reflop
import torch

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
X = torch.randn(3, 4, 5, generator=torch.Generator().manual_seed(11))
Z = torch.complex(torch.randn(4, 3, generator=torch.Generator().manual_seed(12)),
                  torch.randn(4, 3, generator=torch.Generator().manual_seed(13)))


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=13, local=DEVICE)


CASES = {}
for d in range(-3, 3):
    CASES[f"select_dim{d}"] = lambda x, d=d: x.select(d, -1)
    CASES[f"slice_dim{d}"] = lambda x, d=d: x.narrow(d, 1, 2)
    CASES[f"unbind_dim{d}"] = lambda x, d=d: x.unbind(d)[0] * x.unbind(d)[-1]
for a, b in itertools.permutations(range(-3, 3), 2):
    if a % 3 != b % 3:
        for o in (-1, 0, 1):
            CASES[f"diagonal_{a}_{b}_off{o}"] = lambda x, a=a, b=b, o=o: x.diagonal(o, a, b)
CASES.update({
    "index_last": lambda x: x[:, 2, -1],
    "select_chain": lambda x: x.select(2, 3).select(1, 0),
    "ellipsis": lambda x: x[..., 0] + x[..., -1],
    "diagonal_then_select": lambda x: x.diagonal(0, 1, 2).select(1, 2),
    "slice_step": lambda x: x[:, ::2, 1::2],
})


@pytest.mark.parametrize("case", sorted(CASES))
def test_view_backward_matches_torch(dev, case):
    f = CASES[case]
    x = X.clone().requires_grad_(True)
    r = X.to(dev).requires_grad_(True)
    for t in (x, r):
        f(t).pow(2).sum().backward()
    assert torch.allclose(r.grad.cpu(), x.grad, atol=1e-5)


@pytest.mark.parametrize("part", ["real", "imag"])
def test_complex_part_backward(dev, part):
    z = Z.clone().requires_grad_(True)
    r = Z.to(dev).requires_grad_(True)
    for t in (z, r):
        getattr(t, part).pow(2).sum().backward()
    assert torch.allclose(r.grad.cpu(), z.grad, atol=1e-5)


def test_training_step_with_last_token_select(dev):
    """A sequence model that reads its last time step, as LSTM classifiers do."""
    torch.manual_seed(0)
    lin, rlin = torch.nn.Linear(5, 2), torch.nn.Linear(5, 2)
    rlin.load_state_dict(lin.state_dict())
    rlin = rlin.to(dev)
    for m, x in ((lin, X), (rlin, X.to(dev))):
        m(x)[:, -1].sum().backward()
    for p, rp in zip(lin.parameters(), rlin.parameters()):
        assert torch.allclose(rp.grad.cpu(), p.grad, atol=1e-5)
