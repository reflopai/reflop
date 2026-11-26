# SPDX-License-Identifier: Apache-2.0
"""PyTorch's process-wide numerics switches, as a reFLOP worker sets them.

A worker applies its session's numerics (TF32 for matmuls and cuDNN, cuDNN
benchmark, deterministic algorithms) to the whole process. A real worker is
a process of its own; an in-process worker (LocalConnection) shares the
caller's, so LocalConnection swaps these switches in for each request and
gives the caller's back afterwards (BUGS.md #18, GH#19).

Recent PyTorch (2.15 here) keeps TF32 as a tree of `fp32_precision`
settings, and the legacy `allow_tf32` flags write into it. It remembers
which API set a value and refuses to read one API's view after the other
changed it, so a switch is read and written back through the API the
caller's state can be read with, and only when its value changed.
"""

from __future__ import annotations

import torch


def _leaf(path: str):
    o = torch.backends
    for part in path.split("."):
        o = getattr(o, part)
    return o


_CUDNN_LEAVES = ("cudnn", "cudnn.conv", "cudnn.rnn")


def _read_matmul():
    # set_float32_matmul_precision() also sets the oneDNN matmul leaf, which
    # the legacy flag never touches: give such a caller its setting back
    # through the same call, or a later get_float32_matmul_precision() raises.
    if torch.backends.mkldnn.matmul.fp32_precision != "none":
        try:
            return "generic", torch.get_float32_matmul_precision()
        except RuntimeError:
            pass
    try:
        return "legacy", bool(torch.backends.cuda.matmul.allow_tf32)
    except RuntimeError:  # set through fp32_precision
        return "leaf", torch.backends.cuda.matmul.fp32_precision


def _read_cudnn():
    try:
        return "legacy", bool(torch.backends.cudnn.allow_tf32)
    except RuntimeError:
        return "leaf", tuple(_leaf(p).fp32_precision for p in _CUDNN_LEAVES)


def snapshot() -> tuple:
    """The switches a worker sets, in a form restore() gives back."""
    return (_read_matmul(), _read_cudnn(), bool(torch.backends.cudnn.benchmark),
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


def restore(snap: tuple) -> None:
    """Set the switches to snap, writing only those whose value differs."""
    matmul, cudnn, benchmark, det, warn = snap
    if _read_matmul() != matmul:
        kind, v = matmul
        if kind == "generic":
            torch.set_float32_matmul_precision(v)
        elif kind == "legacy":
            torch.backends.cuda.matmul.allow_tf32 = v
        else:
            torch.backends.cuda.matmul.fp32_precision = v
    if _read_cudnn() != cudnn:
        kind, v = cudnn
        if kind == "legacy":
            torch.backends.cudnn.allow_tf32 = v
        else:
            for p, x in zip(_CUDNN_LEAVES, v):
                _leaf(p).fp32_precision = x
    if bool(torch.backends.cudnn.benchmark) != benchmark:
        torch.backends.cudnn.benchmark = benchmark
    if (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()) != (det, warn):
        torch.use_deterministic_algorithms(det, warn_only=warn)


FIELDS = ("tf32_matmul", "tf32_cudnn", "deterministic", "cudnn_benchmark")


def request(defaults, overrides: dict | None):
    """A full Numerics message: overrides (a dict of FIELDS) on defaults.
    The server takes all four fields from a request, so none may be left
    to chance."""
    from ._pb import capabilities_pb2 as kpb

    n = kpb.Numerics()
    if defaults is not None:
        n.CopyFrom(defaults)
    for k, v in (overrides or {}).items():
        setattr(n, k, v)
    return n


def check(overrides: dict | None) -> dict | None:
    """Validate a numerics= argument."""
    if overrides is None:
        return None
    if not isinstance(overrides, dict):
        raise TypeError("numerics must be a dict, e.g. {'deterministic': True}")
    for k, v in overrides.items():
        if k not in FIELDS:
            raise ValueError(f"unknown numerics setting {k!r}; expected one of {', '.join(FIELDS)}")
        if not isinstance(v, bool):
            raise TypeError(f"numerics[{k!r}] must be True or False")
    return dict(overrides)


def as_dict(n) -> dict:
    return {k: bool(getattr(n, k)) for k in FIELDS}
