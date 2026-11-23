# SPDX-License-Identifier: Apache-2.0
"""Tensors whose size-1 dims have a stride other than 1 (x[::12] of ten
elements) count as contiguous in PyTorch, and reshape(-1) keeps the stride;
turning them into bytes failed with "self.stride(-1) must be 1 to view
Float as Byte". Uploads, tensorpacks and the worker's returned values all
went through that path (ADR-043)."""

from __future__ import annotations

import pytest
import torch

CASES = [torch.arange(10.0)[::12], torch.arange(12.0).reshape(3, 4)[:, ::7][1:2], torch.arange(5)[3:4][::9],
         torch.arange(6.0).reshape(2, 3)[:1, 1:2]]


@pytest.mark.parametrize("t", CASES, ids=[str(tuple(t.stride())) for t in CASES])
def test_bytes_of_strided_single_elements(t):
    from reflop.data.tensorpack import _tensor_bytes
    from reflop.session import tensor_bytes
    from reflop_worker.safetensors import tensor_bytes as worker_bytes

    want = t.clone().numpy().tobytes()
    assert tensor_bytes(t) == want
    assert _tensor_bytes(t) == want
    assert worker_bytes(t) == want


def test_reads_and_uploads_round_trip():
    import reflop

    d = reflop.device(index=7, local="cpu")
    x = torch.arange(10.0).to(d)
    assert x[::12].cpu().tolist() == [0.0]
    assert torch.arange(10.0)[::12].to(d).cpu().tolist() == [0.0]
