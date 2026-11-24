# SPDX-License-Identifier: Apache-2.0
"""The client's metas follow the session's device where its kernels return
other shapes than PyTorch's meta kernel (#25 E), cached or not."""

import torch
from reflop.tensor import _META_CACHE, meta_cached

A = torch.ops.aten


def test_meta_cache_keeps_devices_apart():
    x = torch.empty(4, 5, device="meta")
    f = A.log_sigmoid_forward.default
    _META_CACHE.clear()
    for _ in range(2):  # the second round reads the cache
        _, cpu_buffer = meta_cached(f, (x,), {}, cuda=False)
        _, cuda_buffer = meta_cached(f, (x,), {}, cuda=True)
        assert tuple(cpu_buffer.shape) == (4, 5)
        assert tuple(cuda_buffer.shape) == (0,)
    assert len([k for k in _META_CACHE if k[0] is f]) == 2
