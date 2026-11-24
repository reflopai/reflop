# SPDX-License-Identifier: Apache-2.0
"""Wire dtype ↔ torch dtype, with checked size arithmetic."""

from __future__ import annotations

import torch

from ._pb import graph_pb2 as gpb

PB_TO_TORCH = {
    gpb.DTYPE_BOOL: torch.bool, gpb.DTYPE_UINT8: torch.uint8, gpb.DTYPE_INT8: torch.int8,
    gpb.DTYPE_INT16: torch.int16, gpb.DTYPE_INT32: torch.int32, gpb.DTYPE_INT64: torch.int64,
    gpb.DTYPE_FLOAT16: torch.float16, gpb.DTYPE_BFLOAT16: torch.bfloat16, gpb.DTYPE_FLOAT32: torch.float32,
    gpb.DTYPE_FLOAT64: torch.float64, gpb.DTYPE_COMPLEX64: torch.complex64,
    gpb.DTYPE_COMPLEX128: torch.complex128, gpb.DTYPE_FLOAT8_E4M3FN: torch.float8_e4m3fn,
    gpb.DTYPE_FLOAT8_E5M2: torch.float8_e5m2,
}
TORCH_TO_PB = {v: k for k, v in PB_TO_TORCH.items()}
CAPABILITY = {
    gpb.DTYPE_COMPLEX64: "complex", gpb.DTYPE_COMPLEX128: "complex",
    gpb.DTYPE_FLOAT8_E4M3FN: "fp8", gpb.DTYPE_FLOAT8_E5M2: "fp8",
}
INT64_MAX = (1 << 63) - 1


def canonical_bool_(t: torch.Tensor) -> torch.Tensor:
    """In place, on a tensor built from guest bytes: a bool byte other than
    0 or 1 becomes 1. C++ kernels assume bools hold 0 or 1, and kernels
    disagree on what other bytes mean (#25)."""
    if t.dtype == torch.bool and t.numel():
        t.view(torch.uint8).clamp_(max=1)
    return t


def itemsize(dt: torch.dtype) -> int:
    return torch.empty((), dtype=dt, device="meta").element_size()


def numel_checked(shape, max_numel: int) -> int:
    """Product of dims with overflow and limit checks (§2.3 integer overflow).

    The non-zero dims are bounded too: a zero dim makes the element count 0
    whatever the others are, but their strides must still fit in int64
    (GH#4, #25: [0, 2**62, 2**62] makes torch.empty raise)."""
    n = nz = 1
    for d in shape:
        if d < 0:
            raise ValueError(f"negative dimension {d}")
        n *= d
        nz *= d or 1
        if nz > max_numel:
            raise OverflowError(f"tensor of shape {list(shape)} exceeds {max_numel} elements")
    return n


def nbytes_checked(shape, dt: torch.dtype, max_numel: int) -> int:
    n = numel_checked(shape, max_numel) * itemsize(dt)
    if n > INT64_MAX:
        raise OverflowError("tensor size overflows int64")
    return n
