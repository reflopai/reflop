# SPDX-License-Identifier: Apache-2.0
"""Regression tests for view metadata caching, storage offsets, in-place views, and eviction."""

import pytest
import reflop
import torch
from reflop import tensor
from reflop.trace import UnsupportedOperatorError


@pytest.fixture
def cpu_device():
    dev = reflop.device(local="cpu")
    try:
        yield dev
    finally:
        reflop.close()


def test_write_through_sliced_and_selected_view(cpu_device):
    """Writing through sliced or selected views preserves storage offsets and writes to correct elements."""
    dev = cpu_device

    # Repeat twice: iteration 0 warms the cache, iteration 1 tests cache hits
    for _ in range(2):
        # Sliced view
        base = torch.zeros(4, 5, device=dev)
        sl = base[1:3, 2:4]
        assert sl.storage_offset() == torch.zeros(4, 5)[1:3, 2:4].storage_offset()
        sl.add_(10.0)
        reflop.step_boundary()

        expected = torch.zeros(4, 5)
        expected[1:3, 2:4] += 10.0
        assert torch.equal(base.cpu(), expected)

        # Selected view
        base2 = torch.zeros(5, 5, device=dev)
        sel = base2.select(0, 2)
        assert sel.storage_offset() == torch.zeros(5, 5).select(0, 2).storage_offset()
        sel.add_(3.0)
        reflop.step_boundary()

        expected2 = torch.zeros(5, 5)
        expected2.select(0, 2).add_(3.0)
        assert torch.equal(base2.cpu(), expected2)


def test_write_through_as_strided_offsets(cpu_device):
    """Writing through as_strided views relies on view_meta.storage_offset() in as_strided_scatter."""
    dev = cpu_device

    # 1. 1D as_strided with non-zero offset
    # Iteration 0: cache miss / warmup
    b1 = torch.zeros(10, device=dev)
    v1 = b1.as_strided((4,), (1,), storage_offset=3)
    assert v1.storage_offset() == 3

    # Iteration 1: cache hit; offset must reach as_strided_scatter and write to correct indices
    b2 = torch.zeros(10, device=dev)
    v2 = b2.as_strided((4,), (1,), storage_offset=3)
    assert v2.storage_offset() == 3
    v2.add_(5.0)
    reflop.step_boundary()

    expected_1d = torch.zeros(10)
    expected_1d.as_strided((4,), (1,), storage_offset=3).add_(5.0)
    assert torch.equal(b2.cpu(), expected_1d)

    # 2. 2D as_strided with non-zero offset and chained transpose
    # The transpose is inverted first, and the intermediate as_strided view scatters into base
    # via as_strided_scatter using view_meta.storage_offset().
    # Iteration 0: warmup
    b_c1 = torch.zeros(12, device=dev)
    v_c1 = b_c1.as_strided((2, 3), (3, 1), storage_offset=4).transpose(0, 1)
    assert v_c1.storage_offset() == 4

    # Iteration 1: cache hit
    b_c2 = torch.zeros(12, device=dev)
    v_c2 = b_c2.as_strided((2, 3), (3, 1), storage_offset=4).transpose(0, 1)
    assert v_c2.storage_offset() == 4
    v_c2.add_(7.0)
    reflop.step_boundary()

    expected_2d = torch.zeros(12)
    expected_2d.as_strided((2, 3), (3, 1), storage_offset=4).transpose(0, 1).add_(7.0)
    assert torch.equal(b_c2.cpu(), expected_2d)


def test_chained_views_write_through(cpu_device):
    """Writing through chained views correctly handles compound view inverses and offsets."""
    dev = cpu_device

    for _ in range(2):
        base = torch.zeros(6, 6, device=dev)
        v = base[1:5, 1:5].transpose(0, 1)[1:3, :]
        v.add_(7.0)
        reflop.step_boundary()

        expected = torch.zeros(6, 6)
        expected[1:5, 1:5].transpose(0, 1)[1:3, :].add_(7.0)
        assert torch.equal(base.cpu(), expected)


def test_inplace_view_after_hit_does_not_mutate_cache(cpu_device):
    """In-place views must not corrupt or mutate cached meta tensors on subsequent hits."""
    dev = cpu_device

    # 1. Warm cache for transpose
    x1 = torch.zeros(3, 4, device=dev)
    t1 = x1.transpose(0, 1)
    assert t1.shape == (4, 3)

    # 2. In-place view mutation on t1
    t1.transpose_(0, 1)  # mutate t1 metadata in place back to (3, 4)
    assert t1.shape == (3, 4)

    # 3. Create a second tensor and perform the exact same transpose (cache hit)
    x2 = torch.zeros(3, 4, device=dev)
    t2 = x2.transpose(0, 1)
    # t2 must have the correct shape (4, 3), completely unaffected by t1's in-place mutation
    assert t2.shape == (4, 3)
    assert t2._rf_meta.shape == torch.Size([4, 3])
    assert t2._rf_meta is not t1._rf_meta


def test_view_cache_lru_eviction_under_lowered_max(cpu_device):
    """View cache strictly respects lowered _VIEW_CACHE_MAX and evicts via LRU during view ops."""
    dev = cpu_device
    orig_view_max = tensor._VIEW_CACHE_MAX
    try:
        tensor._VIEW_CACHE_MAX = 3
        tensor._VIEW_CACHE.clear()

        # Run 3 distinct view operations to fill the cache to max capacity (3 entries)
        b = torch.zeros(10, 10, device=dev)
        _ = b.transpose(0, 1)    # entry 0
        _ = b.squeeze()          # entry 1
        _ = b.unsqueeze(0)        # entry 2
        assert len(tensor._VIEW_CACHE) == 3
        keys_initial = list(tensor._VIEW_CACHE.keys())

        # Touch entry 0 again (cache hit -> moves entry 0 to end of OrderedDict)
        _ = b.transpose(0, 1)
        keys_after_touch = list(tensor._VIEW_CACHE.keys())
        assert keys_after_touch[-1] == keys_initial[0]  # entry 0 moved to end

        # Introduce a 4th distinct view op: entry 1 (oldest unaccessed) must be evicted
        _ = b[1:5, 1:5]
        assert len(tensor._VIEW_CACHE) == 3
        assert keys_initial[1] not in tensor._VIEW_CACHE  # entry 1 was evicted
        assert keys_initial[0] in tensor._VIEW_CACHE      # entry 0 was retained
    finally:
        tensor._VIEW_CACHE_MAX = orig_view_max
        tensor._VIEW_CACHE.clear()


def test_full_meta_cache_functional_correctness(cpu_device):
    """When _META_CACHE reaches max capacity, functional ops still compute correct shapes and outputs."""
    dev = cpu_device
    orig_meta_max = tensor._META_CACHE_MAX
    try:
        tensor._META_CACHE_MAX = 2
        tensor._META_CACHE.clear()

        # Fill _META_CACHE with 2 functional ops
        a = torch.ones(3, 4, device=dev)
        b = torch.ones(3, 4, device=dev)
        c = a + b  # op 1: add
        _ = a * b  # op 2: mul
        assert len(tensor._META_CACHE) == 2

        # Perform a 3rd functional op when cache is completely full
        # It must safely bypass cache insertion and return correct results
        e = torch.relu(c)  # op 3: relu
        reflop.step_boundary()

        assert len(tensor._META_CACHE) == 2  # did not grow past limit
        assert torch.equal(e.cpu(), torch.full((3, 4), 2.0))

        # Existing cached ops still execute correctly
        f = a + b
        reflop.step_boundary()
        assert torch.equal(f.cpu(), torch.full((3, 4), 2.0))
    finally:
        tensor._META_CACHE_MAX = orig_meta_max
        tensor._META_CACHE.clear()


def test_conj_and_neg_bits_preserved_on_cache_hit(cpu_device):
    """View cache preserves conjugate and negative bits across cache hits, failing closed properly."""
    dev = cpu_device

    # 1. Direct view_meta_cached calls preserve is_conj() and is_neg()
    meta = torch.empty(2, 2, dtype=torch.complex64, device="meta")

    # aten._conj.default: miss and hit
    c1 = tensor.view_meta_cached(torch.ops.aten._conj.default, (meta,), {})
    c2 = tensor.view_meta_cached(torch.ops.aten._conj.default, (meta,), {})
    assert c1.is_conj()
    assert c2.is_conj()

    # aten._neg_view.default: miss and hit
    n1 = tensor.view_meta_cached(torch.ops.aten._neg_view.default, (meta,), {})
    n2 = tensor.view_meta_cached(torch.ops.aten._neg_view.default, (meta,), {})
    assert n1.is_neg()
    assert n2.is_neg()

    # 2. Reflop device views fail closed with UnsupportedOperatorError on both miss and hit
    z = torch.randn(4, dtype=torch.complex64, device=dev)
    for _ in range(2):
        with pytest.raises(UnsupportedOperatorError, match="negative bit"):
            torch._neg_view(z)

    # 3. Conj view and resolve_conj / view_as_real on reflop device
    w = z.conj()
    assert not w.is_conj() and not w._rf_meta.is_conj()
    r = torch.view_as_real(z.conj().resolve_conj())
    assert r.shape == (4, 2)
