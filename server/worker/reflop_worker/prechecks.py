# SPDX-License-Identifier: Apache-2.0
"""Input checks that PyTorch's meta kernels leave to the device kernels (#25 E).

For the ops below the meta kernel accepts inputs that the CPU or the CUDA
kernel rejects, usually after it has allocated its output (I3; V5 counted
it), sometimes with an internal assert, a host read out of bounds or a
crash. V4 runs CHECKS[op](a, cuda) before the meta kernel, with the
arguments by schema name and whether the session's device is CUDA, and
rejects what that device's kernel rejects. Each rule comes from running the
meta, CPU and CUDA kernels on the same inputs; test_meta_parity.py keeps
them honest, and its DELIBERATE table lists the few calls rejected here that
a kernel would run.

A Tensor argument may be a Python number (a wrapped number, or a scalar
graph input); _dtype and _shape read it the way PyTorch wraps it. A check
raises RuntimeError, which V4 reports as ERROR_SHAPE_MISMATCH."""

from __future__ import annotations

import math

import torch

from .ops_generated import FLAG_FP8

A = torch.ops.aten
INT32_MAX = 2**31 - 1
_LOW = (torch.float16, torch.bfloat16)
_INDEX = (torch.int32, torch.int64)


def _fail(msg):
    raise RuntimeError(msg)


def _dtype(x):
    if isinstance(x, torch.Tensor):
        return x.dtype
    if isinstance(x, bool):
        return torch.bool
    if isinstance(x, int):
        return torch.int64
    if isinstance(x, float):
        return torch.float64
    if isinstance(x, complex):
        return torch.complex128
    _fail(f"expected a tensor, got {type(x).__name__}")


def _shape(x):
    return tuple(x.shape) if isinstance(x, torch.Tensor) else ()


def _dim(x):
    return len(_shape(x))


def _same_dtype(a, *names):
    got = [(n, _dtype(a[n])) for n in names if a.get(n) is not None]
    if len({d for _, d in got}) > 1:
        _fail("expected one dtype, got " + ", ".join(f"{n} {d}" for n, d in got))


def _wrap(dim, rank):
    rank = max(rank, 1)
    if not -rank <= dim < rank:
        _fail(f"dim {dim} is out of range for {rank} dims")
    return dim % rank


def _same(*names):
    def check(a, cuda):
        _same_dtype(a, *names)

    return check


def _all(*checks):
    def check(a, cuda):
        for c in checks:
            c(a, cuda)

    return check


# Pooling, padding and upsampling.

def _output_size_len(k):
    # The CUDA 3d kernel read a 1-entry output_size past its end on the host.
    def check(a, cuda):
        if len(a["output_size"]) != k:
            _fail(f"output_size must have {k} entries, got {len(a['output_size'])}")

    return check


def _adaptive_backward(a, cuda):
    g = a["grad_output"]
    if any(n == 0 for n in _shape(g)[1:]):
        _fail(f"grad_output {list(_shape(g))} has an empty non-batch dimension")


def _int32(*names):
    # The kernels store these in int; a larger value was truncated.
    def check(a, cuda):
        for n in names:
            v = a.get(n)
            for x in v if isinstance(v, (list, tuple)) else [v]:
                if isinstance(x, int) and abs(x) > INT32_MAX:
                    _fail(f"{n} {v} does not fit in int32")

    return check


def _rank(name, *ranks):
    def check(a, cuda):
        if _dim(a[name]) not in ranks:
            _fail(f"{name} must have {' or '.join(map(str, ranks))} dims, got {_dim(a[name])}")

    return check


def _upsample(k, vec):
    def check(a, cuda):
        x = a["input" if vec else "self"]
        if _dim(x) != k + 2:
            _fail(f"input must have {k + 2} dims, got {_dim(x)}")
        if any(n == 0 for n in x.shape[2:]):
            _fail(f"input {list(x.shape)} has an empty spatial dimension")
        size = a["output_size"]
        if size is None and vec and a["scale_factors"] is not None:
            sf = a["scale_factors"]
            if len(sf) != k:
                _fail(f"scale_factors must have {k} entries, got {len(sf)}")
            size = [math.floor(n * s) for n, s in zip(x.shape[2:], sf)]
        if size is not None and (len(size) != k or any(n <= 0 for n in size)):
            _fail(f"output_size must be {k} positive sizes, got {list(size)}")

    return check


_SCALES = {1: ("scales",), 2: ("scales_h", "scales_w"), 3: ("scales_d", "scales_h", "scales_w")}


def _scales(k, backward):
    # A kernel given scale s > 0 maps output index o to input index o / s.
    # The nearest and linear kernels clamp that index only from above, and
    # the antialiased one sizes a host buffer by 1 / s: on the CPU, scales of
    # 1e-10 overflowed it (SIGSEGV). F.interpolate passes s with
    # out = floor(in * s), so out / s <= in; this refuses out / s > in + out.
    # NaN, 0 and negative scales make the kernels use in / out instead.
    def check(a, cuda):
        size = list(a["input_size"])[2:] if backward else _shape(a["self"])[2:]
        for name, n, m in zip(_SCALES[k], size, a["output_size"]):
            s = a.get(name)
            if s is not None and s > 0 and m > s * (n + m):
                _fail(f"{name} {s} is out of range for input size {n} and output size {m}")

    return check


def _upsampling(name, k):
    """The CHECKS entries for upsampling op `name` with k spatial dims."""
    fwd, bwd = _scales(k, False), _scales(k, True)
    out = {f"aten.{name}.default": fwd, f"aten.{name}_backward.default": bwd}
    if name in ("upsample_linear1d", "upsample_bilinear2d", "upsample_bicubic2d", "upsample_trilinear3d"):
        out[f"aten.{name}.default"] = _all(_upsample(k, False), fwd)
        out[f"aten.{name}.vec"] = _upsample(k, True)
    return out


_UPSAMPLING = {
    **_upsampling("upsample_nearest1d", 1),
    **_upsampling("upsample_nearest2d", 2),
    **_upsampling("upsample_nearest3d", 3),
    **_upsampling("_upsample_nearest_exact1d", 1),
    **_upsampling("_upsample_nearest_exact2d", 2),
    **_upsampling("_upsample_nearest_exact3d", 3),
    **_upsampling("upsample_linear1d", 1),
    **_upsampling("upsample_bilinear2d", 2),
    **_upsampling("upsample_bicubic2d", 2),
    **_upsampling("upsample_trilinear3d", 3),
    **_upsampling("_upsample_bilinear2d_aa", 2),
}


# Fill values.

_NO_INF = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
_LOOSE = (torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz,
          torch.float8_e5m2fnuz)


def _overflows(x, dt):
    # c10's overflows<dt>(x) for a bool, int64 or double x.
    if isinstance(x, bool):
        return False
    if dt.is_floating_point:
        if isinstance(x, float) and math.isnan(x):
            return False
        if isinstance(x, float) and math.isinf(x):
            return dt in _NO_INF
        return not -torch.finfo(dt).max <= x <= torch.finfo(dt).max
    info = torch.iinfo(dt)
    if isinstance(x, float):
        return not (math.isfinite(x) and info.min <= x <= info.max)
    if info.min == 0:
        # An int converts to an unsigned type modulo 2**bits, down to -max.
        return x > info.max or -x > info.max
    return not info.min <= x <= info.max


def _converts(v, dt, half_complex32):
    """Whether Scalar v converts to dtype dt (c10::checked_convert)."""
    if dt == torch.bool:
        return True
    if dt.is_complex:
        dt = {torch.complex32: torch.float16 if half_complex32 else torch.float32,
              torch.complex64: torch.float32, torch.complex128: torch.float64}[dt]
        parts = (v.real, v.imag) if isinstance(v, complex) else (v,)
    elif isinstance(v, complex):
        if v.imag != 0:
            return False
        parts = (v.real,)
    else:
        parts = (v,)
    return not any(_overflows(x, dt) for x in parts)


def _fill(value, like=None, default=None, scalar_tensor=False):
    # The kernels convert the fill value to the output dtype after the
    # output is allocated (I3). From each device's kernel: full, the _like
    # ops, fill and constant_pad_nd check complex32 as complex<float> on the
    # CPU and as Half on CUDA; scalar_tensor's CPU kernel checks complex32
    # as Half and the 16- and 8-bit floats only as a double.
    def check(a, cuda):
        v = a[value]
        dt = a.get("dtype") or (_dtype(a[like]) if like else default)
        if dt is None:  # full: the default dtype for the value's kind
            dt = {torch.float64: torch.float32, torch.complex128: torch.complex64}.get(_dtype(v), _dtype(v))
        if scalar_tensor and not cuda and dt in _LOOSE:
            dt = torch.float64  # checked as a double, then cast
        if not _converts(v, dt, cuda or scalar_tensor):
            _fail(f"{value} {v!r} cannot be converted to {dt} without overflow")

    return check


def _constant_pad(a, cuda):
    if any(p > 0 for p in a["pad"]):  # otherwise nothing is filled
        _fill("value", "self")(a, cuda)


# Indexing and search.

def _index_add(a, cuda):
    # PyTorch's C++ meta function for index_add, which the Python meta
    # kernel does not implement.
    x, idx, src = a["self"], a["index"], a["source"]
    if _dim(idx) > 1 or _dtype(idx) not in _INDEX:
        _fail(f"index must be an int32 or int64 vector, got {_dtype(idx)} with {_dim(idx)} dims")
    _same_dtype(a, "self", "source")
    dim = _wrap(a["dim"], _dim(x))
    sd = _dim(src)
    if not (dim == 0 or dim < sd):
        _fail(f"dim {dim} is out of range for a {sd}-d source")
    if idx.numel() != (1 if sd == 0 else src.shape[dim]):
        _fail(f"index has {idx.numel()} entries; source has {1 if sd == 0 else src.shape[dim]} along dim {dim}")
    xs, ss = list(_shape(x)), list(_shape(src))
    if sd and _dim(x):
        del xs[dim], ss[dim]
    if xs != ss:
        _fail(f"self {list(_shape(x))} and source {list(_shape(src))} differ outside dim {dim}")


def _index_int64(a, cuda):
    if _dtype(a["index"]) != torch.int64:
        _fail(f"index must be int64, got {_dtype(a['index'])}")


def _index_vector(a, cuda):
    if _dim(a["index"]) > 1:
        _fail(f"index must have at most 1 dim, got {_dim(a['index'])}")


def _searchsorted(a, cuda):
    s = a.get("sorter")
    if s is not None and _dtype(s) != torch.int64:
        _fail(f"sorter must be int64, got {_dtype(s)}")


def _bucketize(a, cuda):
    if _dim(a["boundaries"]) != 1:
        _fail(f"boundaries must be 1-d, got {_dim(a['boundaries'])} dims")


def _some_index(a, cuda):
    # With no index at all the kernels fail an internal assert.
    if not any(i is not None for i in a["indices"]):
        _fail("at least one index must be given")


def _index_put(a, cuda):
    _some_index(a, cuda)
    x, idx, v = a["self"], a["indices"], a["values"]
    _same_dtype(a, "self", "values")
    if any(i is not None and _dtype(i) in (torch.bool, torch.uint8) for i in idx):
        return  # the indexed shape depends on the mask's values
    res = tuple(A.index.Tensor(x, idx).shape)
    if torch.broadcast_shapes(_shape(v), res) != res:
        _fail(f"values {list(_shape(v))} do not broadcast to the indexed shape {list(res)}")


# Other single ops.

def _pdist(a, cuda):
    if _dim(a["self"]) < 2:
        _fail(f"self must have at least 2 dims, got {_dim(a['self'])}")


def _log_sigmoid_backward(a, cuda):
    # The CUDA kernel ignores buffer: its forward returns it empty.
    if cuda:
        _same_dtype(a, "grad_output", "self")
        return
    _same_dtype(a, "grad_output", "self", "buffer")
    shapes = [_shape(a[n]) for n in ("grad_output", "self", "buffer")]
    try:
        torch.broadcast_shapes(*shapes)
    except RuntimeError:
        _fail(f"grad_output, self and buffer {[list(s) for s in shapes]} do not broadcast")


def _grid_sampler(a, cuda):
    _same_dtype(a, "input", "grid")
    x = _shape(a["input"])
    if 0 in x[2:]:
        _fail(f"input {list(x)} has an empty spatial dimension")


def _embedding_bag_backward(a, cuda):
    # The CPU max-mode kernels index maximum_indices by bag_size, which the
    # CPU forward gives one entry per bag (CUDA's has one per offset).
    g = _shape(a["grad"])
    if not cuda and a["mode"] == 2 and g and _shape(a["bag_size"]) != g[:1]:
        _fail(f"bag_size {list(_shape(a['bag_size']))} must have one entry per bag ({g[0]})")


def _im2col(a, cuda):
    if any(p < 0 for p in a["padding"]):
        _fail(f"padding {list(a['padding'])} must be non-negative")


def _fft_c2r(a, cuda):
    n, dims = a["last_dim_size"], list(a["dim"])
    if n <= 0 or not dims:
        _fail(f"last_dim_size {n} must be positive and dim {dims} non-empty")
    m = a["self"].shape[dims[-1]]
    if m not in (n, n // 2 + 1):
        _fail(f"self has {m} entries along dim {dims[-1]}; last_dim_size {n} needs {n // 2 + 1}")


def _bitwise_not(a, cuda):
    dt = _dtype(a["self"])
    if dt.is_floating_point or dt.is_complex:
        _fail(f"bitwise_not takes bool or integer tensors, got {dt}")


def _threshold_backward(a, cuda):
    if torch.result_type(a["grad_output"], a["self"]) == torch.bool:
        _fail("threshold_backward is not implemented for bool")


def _affine_grid_generator(a, cuda):
    # Each spatial axis is a linspace; one of length 0 or 1 is a 0-d tensor,
    # which bool, f16 and bf16 have no kernel for. CUDA multiplies a
    # non-empty grid by theta with baddbmm, which takes no integer type.
    dt, spatial = _dtype(a["theta"]), list(a["size"])[2:]
    if dt == torch.bool or (dt in _LOW and any(n <= 1 for n in spatial)):
        _fail(f"affine_grid_generator has no {dt} kernel for size {list(a['size'])}")
    if cuda and not (dt.is_floating_point or dt.is_complex) and math.prod(list(a["size"])[:1] + spatial) > 0:
        _fail(f"affine_grid_generator has no CUDA kernel for {dt}")


def _logsumexp(a, cuda):
    if not list(a["dim"]) and _dim(a["self"]) > 0:
        _fail("dim must not be empty")


def _nonempty_reduce(a, cuda):
    if a["self"].numel() == 0:
        _fail("self is empty; the reduction has no identity")


def _nonempty_dim(a, cuda):
    x = a["self"]
    if _dim(x) and x.shape[_wrap(a["dim"], _dim(x))] == 0:
        _fail(f"self {list(x.shape)} is empty along dim {a['dim']}")


def _lu_factor(a, cuda):
    if not a["pivot"] and not cuda:
        _fail("LU without pivoting is only implemented on CUDA")


def _channel_shuffle(a, cuda):
    # The CUDA kernel views self as (N, groups, C / groups, -1).
    x, g = a["self"], a["groups"]
    if cuda:
        x.view(x.shape[0], g, x.shape[1] // g, -1)


def _int_mm(a, cuda):
    x, y = a["self"], a["mat2"]
    if cuda and (x.shape[0] <= 16 or x.shape[1] <= 0 or x.shape[1] % 8 or y.shape[1] <= 0 or y.shape[1] % 8):
        _fail(f"_int_mm on CUDA needs self.size(0) > 16 and the other sizes positive multiples of 8, got "
              f"{list(x.shape)} and {list(y.shape)}")


def pin_memory(a):
    """pin_memory=True asks for page-locked host memory, which reFLOP does
    not hand out (it can be pinned even in a CPU session on a CUDA host)."""
    if a.get("pin_memory"):
        _fail("pin_memory=True is not supported")


def _is_float8(v):
    if isinstance(v, (list, tuple)):
        return any(_is_float8(x) for x in v)
    dt = v if isinstance(v, torch.dtype) else getattr(v, "dtype", None)
    return isinstance(dt, torch.dtype) and dt.is_floating_point and dt.itemsize == 1


def float8(name, flags, a):
    """Float8 tensors and dtypes only on ops flagged FP8 (ADR-050). The CUDA
    kernels take float8 only to move, compare and matrix-multiply it; any
    other op has V4 accept it and the kernel refuse it after allocating its
    output. A CPU session never offers float8 (core.detect_capabilities)."""
    if not flags & FLAG_FP8 and any(_is_float8(v) for v in a.values()):
        _fail(f"{name} does not take float8 tensors or dtypes (only the FP8 ops in docs/OPSET.md do)")


def _promotes(*names):
    # The meta kernels promote float8 with other dtypes; CUDA's result_type
    # refuses to. It promotes the dimensioned operands, then the 0-dim ones,
    # and a 0-dim operand matters only when the dimensioned ones are not
    # floating point. Python numbers take part in neither.
    def check(a, cuda):
        dims, zeros = set(), set()
        for n in names:
            v = a[n]
            for t in v if isinstance(v, (list, tuple)) else [v]:
                if isinstance(t, torch.Tensor):
                    (dims if t.dim() else zeros).add(t.dtype)
        floating = any(d.is_floating_point or d.is_complex for d in dims)
        for group in (dims, zeros, dims | zeros if dims and not floating else ()):
            if len(group) > 1 and _is_float8(list(group)):
                _fail(f"float8 does not promote with another dtype: {', '.join(sorted(map(str, group)))}")

    return check


_E4M3, _E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
_HIGH = (torch.bfloat16, torch.float16, torch.float32)


def _scaled_mm(a, cuda):
    # The CUDA kernel's and cuBLASLt's rules for one scale per matrix
    # (tensor-wise) and one per row of self and column of mat2 (row-wise);
    # the meta kernel checks the dtypes, layouts and multiples of 16. Any
    # other scaling is refused (ADR-050).
    x, w, sa, sb = a["self"], a["mat2"], a["scale_a"], a["scale_b"]
    if x.dim() != 2 or w.dim() != 2:
        return  # the meta kernel refuses it
    (m, k), n = x.shape, w.shape[1]
    if w.shape[0] != k:
        _fail(f"self {list(x.shape)} and mat2 {list(w.shape)} cannot be multiplied")
    if x.dtype == _E5M2 and w.dtype == _E5M2:
        _fail("two float8_e5m2 matrices do not multiply")
    out = a["out_dtype"] or x.dtype
    if sa.numel() == 1 and sb.numel() == 1:
        if out not in (*_HIGH, _E4M3, _E5M2):
            _fail(f"out_dtype {out} is not supported")
        if out == _E5M2 and _E5M2 not in (x.dtype, w.dtype):
            _fail("a float8_e5m2 output needs a float8_e5m2 input")
    elif tuple(sa.shape) == (m, 1) and tuple(sb.shape) == (1, n):
        # PyTorch's row-wise kernels are for compute capability 9.0 and
        # later; fp8 is offered from 8.9.
        if cuda and torch.cuda.get_device_capability() < (9, 0):
            _fail("row-wise scaling needs compute capability 9.0")
        if w.dtype != _E4M3:
            _fail(f"row-wise scaling needs mat2 in float8_e4m3fn, got {w.dtype}")
        if out not in _HIGH:
            _fail(f"row-wise scaling gives bfloat16, float16 or float32, not {out}")
    else:
        _fail(f"scale_a and scale_b must have one element each, or shapes [{m}, 1] and [1, {n}]")
    bias, sr = a["bias"], a["scale_result"]
    if bias is not None:
        if tuple(bias.shape) != (n,):
            _fail(f"bias must have shape [{n}], got {list(bias.shape)}")
        if out == torch.float32 or bias.dtype not in _LOW or (out in _LOW and bias.dtype != out):
            _fail(f"a bias in {bias.dtype} does not go with out_dtype {out}")
    if sr is not None and (sr.dtype != torch.float32 or sr.numel() != 1):
        _fail("scale_result must be one float32 value")


# Losses.

def _bce(a, cuda):
    x = _shape(a["self"])
    names = ("grad_output", "self", "target") if "grad_output" in a else ("self", "target")
    _same_dtype(a, *names)
    shapes = [_shape(a[n]) for n in names] if "grad_output" in a else [x]
    if a.get("weight") is not None:
        shapes.append(_shape(a["weight"]))
    if torch.broadcast_shapes(*shapes) != x:
        _fail(f"{', '.join(names)} and weight must broadcast to self {list(x)}")


def _bce_logits(a, cuda):
    x = _shape(a["self"])
    if _shape(a["target"]) != x:
        _fail(f"target {list(_shape(a['target']))} must have self's shape {list(x)}")
    for n in ("weight", "pos_weight"):
        w = a.get(n)
        if w is not None and torch.broadcast_shapes(_shape(w), x) != x:
            _fail(f"{n} {list(_shape(w))} does not broadcast to {list(x)}")


def _soft_margin(a, cuda):
    xt = torch.broadcast_shapes(_shape(a["self"]), _shape(a["target"]))
    if "grad_output" in a:
        if torch.broadcast_shapes(_shape(a["grad_output"]), xt) != xt:
            _fail(f"grad_output {list(_shape(a['grad_output']))} does not broadcast to {list(xt)}")
    elif xt != _shape(a["self"]):
        _fail(f"target {list(_shape(a['target']))} does not broadcast to self {list(_shape(a['self']))}")


# gelu_backward and glu_backward take grad_output in the shape autograd
# passes. PyTorch broadcasts it on some paths only (the CPU's oneDNN path,
# picked by dtype, layout and CPU features, and CUDA's glu reject it), so
# the worker requires it on every path (deliberate).

def _gelu_backward(a, cuda):
    g, x = a["grad_output"], a["self"]
    if _shape(g) != _shape(x) or _dtype(g) != _dtype(x):
        _fail(f"grad_output is {_dtype(g)}{list(_shape(g))}; self is {_dtype(x)}{list(_shape(x))}")


def _glu_backward(a, cuda):
    _same_dtype(a, "grad_output", "self")
    x = list(_shape(a["self"]))
    d = _wrap(a["dim"], len(x))
    if not x or x[d] % 2:
        _fail(f"self {x} must have an even size along dim {a['dim']}")
    x[d] //= 2
    if list(_shape(a["grad_output"])) != x:
        _fail(f"grad_output {list(_shape(a['grad_output']))} must be {x}")


def _nll(two_d, backward):
    def check(a, cuda):
        x, t, w = a["self"], a["target"], a.get("weight")
        _same_dtype(a, "self", "weight", *(("grad_output", "total_weight") if backward else ()))
        if _dtype(t) != torch.int64:
            _fail(f"target must be int64, got {_dtype(t)}")
        xs, ts = _shape(x), _shape(t)
        if two_d:
            if len(xs) != 4 or ts != (xs[0],) + xs[2:]:
                _fail(f"self {list(xs)} must be (N, C, H, W) and target (N, H, W), got {list(ts)}")
            classes = xs[1]
        else:
            if len(xs) not in (1, 2) or len(ts) != len(xs) - 1 or ts != xs[:1] * (len(xs) - 1):
                _fail(f"self {list(xs)} must be (C,) or (N, C) and target () or (N,), got {list(ts)}")
            classes = xs[-1]
        if w is not None and (_dim(w) != 1 or w.numel() != classes):
            _fail(f"weight {list(_shape(w))} must have {classes} entries")
        if backward:
            if _shape(a["total_weight"]) not in ((), (1,)):
                _fail(f"total_weight must have one element, got {list(_shape(a['total_weight']))}")
            g = _shape(a["grad_output"])
            if a["reduction"] == 0 and ts:
                if g != ts:
                    _fail(f"grad_output {list(g)} must have target's shape {list(ts)}")
            elif g not in ((), (1,)):
                _fail(f"grad_output must have one element, got {list(g)}")

    return check


# Convolution: PyTorch's check_shape_forward, which the meta kernel skips.

def _conv(a, bias_shape):
    x, w = a["input"], a["weight"]
    k, groups, transposed = _dim(w), a["groups"], a["transposed"]
    if k < 3:
        _fail(f"weight must have at least 3 dims, got {k}")
    if not 0 < groups <= INT32_MAX:
        _fail(f"groups {groups} must be in [1, 2**31)")
    p = {}
    for n in ("stride", "padding", "dilation", "output_padding"):
        v = list(a[n])
        if len(v) not in (1, k - 2):
            _fail(f"{n} {v} must have 1 or {k - 2} entries")
        # The CUDA kernels and the CPU transposed ones wrap these to int32;
        # the worker rejects them on every path (deliberate).
        if any(abs(i) > INT32_MAX for i in v):
            _fail(f"{n} {v} does not fit in int32")
        p[n] = v * (k - 2) if len(v) == 1 else v
    if any(i < 0 for i in p["padding"] + p["output_padding"]) or any(i <= 0 for i in p["stride"] + p["dilation"]):
        _fail("padding and output_padding must be non-negative, stride and dilation positive")
    xs, ws = _shape(x), _shape(w)
    if len(xs) != k:
        _fail(f"input {list(xs)} must have {k} dims, as weight {list(ws)}")
    if ws[0] < groups or ws[0] % groups:
        _fail(f"weight {list(ws)} must have a multiple of groups={groups} at dim 0")
    if any(n == 0 for n in ws[2:]):
        _fail(f"weight {list(ws)} has an empty kernel dimension")
    if x.numel() == 0 and xs[0] and xs[1]:
        _fail(f"input {list(xs)} is empty outside the batch and channel dims")
    if transposed:
        cin, bias = ws[0], (ws[1] * groups,)
    else:
        cin, bias = ws[1] * groups, (ws[0],)
        for i in range(k - 2):
            if xs[i + 2] + 2 * p["padding"][i] < p["dilation"][i] * (ws[i + 2] - 1) + 1:
                _fail(f"padded input {list(xs)} is smaller than the dilated kernel {list(ws)}")
    if xs[1] != cin:
        _fail(f"input {list(xs)} must have {cin} channels for weight {list(ws)} and groups={groups}")
    if bias_shape is not None and tuple(bias_shape) != bias:
        _fail(f"bias has shape {list(bias_shape)}; weight {list(ws)} needs {list(bias)}")
    return p


def _convolution(a, cuda):
    _same_dtype(a, "input", "weight", "bias")
    b = a.get("bias")
    _conv(a, None if b is None else _shape(b))


def _convolution_backward(a, cuda):
    _same_dtype(a, "grad_output", "input", "weight")
    # Autograd passes bias_sizes [0] when there is no bias; the kernels
    # read it only for the bias gradient.
    p = _conv(a, a["bias_sizes"] if a["output_mask"][2] else None)
    if not a["transposed"] and any(p["output_padding"]):
        _fail("output_padding must be 0 for a convolution that is not transposed")
    out = A.convolution.default(a["input"], a["weight"], None, a["stride"], a["padding"], a["dilation"],
                                a["transposed"], a["output_padding"], a["groups"])
    if _shape(a["grad_output"]) != tuple(out.shape):
        _fail(f"grad_output {list(_shape(a['grad_output']))} must have the output's shape {list(out.shape)}")


# The batch norm family.

def _batch_norm(training_arg):
    def check(a, cuda):
        x = a["input"]
        training = a[training_arg] if training_arg else False
        if x.numel() == 0 and (training or cuda or "grad_out" in a):
            # CUDA rejects every empty input, the CPU training kernels too;
            # the CPU backward crashed on one.
            _fail(f"input {list(_shape(x))} is empty")
        if "running_mean" in a and (a["running_mean"] is None) != (a["running_var"] is None):
            _fail("running_mean and running_var must be given together")
        if "grad_out" not in a:
            # The CUDA kernels read the running statistics in any dtype.
            _bn_params(a, ("weight", "bias") + (() if cuda else ("running_mean", "running_var")))
            return
        if not cuda:
            _bn_params(a, ("weight", "running_mean", "running_var", "save_mean", "save_invstd"))
            return
        if a["weight"] is None and (a["output_mask"][1] or a["output_mask"][2]):
            _fail("the CUDA kernel needs weight for the weight and bias gradients")
        # The CUDA kernels take the statistics their forward saves, in the
        # accumulate type, and running statistics in weight's dtype (some
        # paths cast other dtypes, the fused one rejects them).
        _bn_params(a, ("weight", "running_mean", "running_var"))
        for n in ("save_mean", "save_invstd"):
            if a.get(n) is not None and _dtype(a[n]) != _acc(x.dtype):
                _fail(f"{n} is {_dtype(a[n])}; the CUDA kernel takes {_acc(x.dtype)} for a {x.dtype} input")

    return check


def _bn_params(a, names):
    """The defined tensors in `names` share one dtype: input's, or float32
    for a half or bfloat16 input (mixed_data_type.h on the CPU; the CUDA
    kernels read them through accessors of one parameter type)."""
    x = a["input"].dtype
    got = {n: _dtype(a[n]) for n in names if a.get(n) is not None}
    ok = {x} | ({torch.float32} if x in _LOW else set())
    if len(set(got.values())) > 1 or not set(got.values()) <= ok:
        _fail(f"{', '.join(f'{n} {d}' for n, d in got.items())} must share one dtype, "
              f"{' or '.join(str(d) for d in sorted(ok, key=str))} for a {x} input")


# Layer norm and group norm. The CPU kernels take float32 parameters for a
# half or bfloat16 input ("mixed"); the CUDA kernels take input's dtype.

def _acc(dt):
    return torch.float64 if dt == torch.float64 else torch.float32


def _norm_params(a, cuda):
    """The dtype weight and bias share (None if neither is given)."""
    x = a["input"]
    _same_dtype(a, "weight", "bias")
    p = next((_dtype(a[n]) for n in ("weight", "bias") if a.get(n) is not None), None)
    ok = {x.dtype} | ({torch.float32} if x.dtype in _LOW and not cuda else set())
    if p is not None and p not in ok:
        _fail(f"weight and bias are {p}; input is {x.dtype}")
    return p


def _stats(a, want):
    for n in ("mean", "rstd"):
        if _dtype(a[n]) != want:
            _fail(f"{n} is {_dtype(a[n])}; expected {want}")


def _layer_norm(a, cuda):
    p = _norm_params(a, cuda)
    if "grad_out" not in a:
        return
    m = a["output_mask"]
    if (m[1] and a["weight"] is None) or (m[2] and a["bias"] is None):
        _fail("output_mask asks for the gradient of a weight or bias that is not given")
    x, w = a["input"], a["weight"]
    if cuda:
        _stats(a, _acc(x.dtype))
    elif p is not None and p != x.dtype and (w is None or _dtype(w) != p):
        _fail("the CPU kernel takes float32 parameters only with a float32 weight")
    else:
        _stats(a, torch.float32 if p is not None and p != x.dtype else x.dtype)


def _group_norm(a, cuda):
    if "grad_out" not in a:
        _norm_params(a, cuda)
        return
    x, w = a["input"], a["weight"]
    st = _dtype(a["mean"])
    _stats(a, st)
    m = a["output_mask"]
    if w is not None and _dtype(w) != x.dtype:
        if cuda or x.dtype not in _LOW or _dtype(w) != torch.float32:
            _fail(f"weight is {_dtype(w)}; input is {x.dtype}")
        want = torch.float32
    elif w is None and not cuda and x.dtype in _LOW and st == torch.float32 and not (m[1] or m[2]):
        want = torch.float32
    else:
        want = x.dtype
    if st != want:
        _fail(f"mean and rstd are {st}; input is {x.dtype} and weight {None if w is None else _dtype(w)}")


def _rms_norm(a, cuda):
    # _check_rms_norm_inputs and the CUDA kernels' dtypes. On the CPU the
    # op is a composite that takes any normalized_shape of the right length
    # and promotes dtypes; the worker holds both devices to the CUDA rules.
    # The backward (CUDA only) takes what the forward returns, as ADR-032:
    # its kernel checks only numel, and its meta gives other shapes.
    x, ns, w = a["input"], list(a["normalized_shape"]), a["weight"]
    k = len(ns)
    if not k:
        _fail("normalized_shape must have at least one entry")
    if x.dim() < k or list(x.shape[-k:]) != ns:
        _fail(f"input {list(x.shape)} must end in normalized_shape {ns}")
    if w is not None and list(w.shape) != ns:
        _fail(f"weight is {list(w.shape)}; normalized_shape is {ns}")
    if w is not None and w.dtype != x.dtype:
        _fail(f"weight is {w.dtype}; input is {x.dtype}")
    if "grad_out" not in a:
        return
    g, r = a["grad_out"], a["rstd"]
    if g.shape != x.shape or g.dtype != x.dtype:
        _fail(f"grad_out must be {x.dtype}{list(x.shape)}, got {g.dtype}{list(g.shape)}")
    want = list(x.shape[:-k]) + [1] * k
    if list(r.shape) != want or r.dtype != _acc(x.dtype):
        _fail(f"rstd must be {_acc(x.dtype)}{want}, got {r.dtype}{list(r.shape)}")
    if a["output_mask"][1] and w is None:
        _fail("output_mask asks for the gradient of a weight that is not given")


# Factories, random numbers and reductions. The meta kernels take any dtype;
# the kernels dispatch on the output dtype, mostly after allocating the
# output. _kind sorts a dtype into b (bool), i (integer), f (16- to 64-bit
# float), c (complex) or 8 (8-bit float). None of these ops take float8
# (float8 above), so no rule here mentions it.

def _kind(dt):
    if dt == torch.bool:
        return "b"
    if dt.is_complex:
        return "c"
    if dt.is_floating_point:
        return "8" if dt.itemsize == 1 else "f"
    return "i"


def _numel(x):
    return x.numel() if isinstance(x, torch.Tensor) else 1


def _out_dtype(a, like):
    dt = a.get("dtype")
    return _dtype(a[like]) if dt is None and like is not None else dt


def _refuse(what, dt, kinds):
    if dt is not None and _kind(dt) in kinds:
        _fail(f"{what} {dt} is not supported")


def _no(kinds, like=None, numel=None):
    """Refuse an output dtype (the dtype argument, else `like`'s dtype) of a
    kind in `kinds`. With `numel`, only when numel(a) > 0: those kernels
    return an empty output before dispatching."""

    def check(a, cuda):
        if numel is None or numel(a) > 0:
            _refuse("dtype", _out_dtype(a, like), kinds)

    return check


def _arange(a, cuda):
    # The kernels convert start and step to the output's accumulate type:
    # float on CUDA for a float32 output, a double on the CPU (the 16-bit
    # floats' float, which the meta kernel checks).
    _no("bc")(a, cuda)
    dt = a.get("dtype")
    if dt is None:
        dt = torch.float32 if any(isinstance(a.get(n), float) for n in ("start", "end", "step")) else None
    if cuda and dt == torch.float32:
        for name in ("start", "step"):
            v = a.get(name)
            if v is not None and not _converts(v, torch.float32, True):
                _fail(f"{name} {v!r} cannot be converted to {dt} without overflow")


def _randperm(a, cuda):
    _refuse("dtype", a["dtype"], "bc")
    if cuda and a["dtype"] == torch.bfloat16:
        _fail("dtype torch.bfloat16 is not supported")


_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}
_DIGITS = {torch.float16: 11, torch.bfloat16: 8, torch.float32: 24, torch.float64: 53}
INT64 = (-2**63, 2**63 - 1)


def _round_int(x, dt):
    # static_cast<int64_t>(static_cast<dt>(x)); undefined behavior when the
    # float is outside int64, which this refuses.
    y = torch.tensor(x, dtype=torch.int64).to(dt).item()
    if not (math.isfinite(y) and INT64[0] <= y < 2**63):
        _fail(f"{x} rounds to {y} in {dt}, outside int64")
    return int(y)


def _shifted(x, r, dt, sign):
    # update_from and update_to: move x to the next value dt holds.
    shift = max(abs(x).bit_length() - 1, 0) - _DIGITS[dt] + 1
    v = r + sign * (1 << shift) if shift >= 0 else None
    if v is None or not INT64[0] <= v <= INT64[1]:
        _fail(f"{x} has no neighbor in int64 for {dt}")
    return v


def _random_range(low, high, like=None, default=torch.float32):
    """random_(from, to), as randint and randint_like run it on their output
    (DistributionTemplates.h: random_from_to_impl). Every check comes
    before the kernel returns an empty output."""

    def check(a, cuda):
        dt = _out_dtype(a, like) or default
        lo = a[low] if low is not None else 0
        hi = a[high]
        if not lo < hi:
            _fail(f"{low or 'low'} {lo} must be less than {high} {hi}")
        if dt in _DIGITS:
            r = _round_int(lo + 1, dt)
            lo = _shifted(lo + 1, r, dt, 1) if r < lo else lo
            r = _round_int(hi - 1, dt)
            hi = _shifted(hi - 1, r, dt, -1) if r >= hi else hi
            if not lo < hi:
                _fail(f"{low or 'low'} and {high} round to {lo} >= {hi} in {dt}")
            m = torch.finfo(dt).max
            bounds = (-m, m)
        elif dt == torch.bool:
            bounds = (0, 1)
        else:
            bounds = (torch.iinfo(dt).min, torch.iinfo(dt).max)
        for what, v in ((low or "low", lo), (f"{high} - 1", hi - 1)):
            if not bounds[0] <= v <= bounds[1]:
                _fail(f"{what} {v} is out of bounds for {dt}")

    return check


def _uniform(a, cuda):
    # uniform_impl_ checks the range in the real dtype (an empty complex
    # output returns first).
    xd = _dtype(a["self"])
    _refuse("self", xd, "bi")
    if xd.is_complex and not _numel(a["self"]):
        return
    dt = _REAL.get(xd, xd)
    lo, hi, m = a["from"], a["to"], torch.finfo(dt).max
    if not (-m <= lo <= m and -m <= hi <= m and lo <= hi and hi - lo <= m):
        _fail(f"from {lo} and to {hi} are not a range in {dt}")


def _bernoulli_p(a, cuda):
    p = a["p"]
    if not 0 <= p <= 1:
        _fail(f"p {p} must be in [0, 1]")
    if _numel(a["self"]):
        _refuse("self", _dtype(a["self"]), "c")


def _bernoulli(a, cuda):
    # The probabilities are self (default) or p (Tensor); E2 checks their
    # values (validators.probability_runtime).
    x, p = a["self"], a.get("p")
    if not _numel(x):
        return
    if p is None:
        _refuse("self", _dtype(x), "bic")
        return
    _refuse("self", _dtype(x), "c")
    _refuse("p", _dtype(p), "bic")
    if torch.broadcast_shapes(_shape(p), _shape(x)) != _shape(x):
        _fail(f"p {list(_shape(p))} does not expand to self {list(_shape(x))}")


def _normal_std(a, cuda):
    # CHECK_NORMAL_TENSOR_STD; E2 checks std's values (validators.
    # normal_std_runtime).
    _refuse("std", _dtype(a["std"]), "c")


def _pow(base, x):
    # std::pow on doubles: inf on overflow, NaN for a negative base and a
    # fractional exponent. Complex operands are left to Python.
    if isinstance(base, complex) or isinstance(x, complex):
        try:
            return complex(base) ** complex(x)
        except (OverflowError, ZeroDivisionError):
            return complex(math.inf, math.inf)
    try:
        return math.pow(base, x)
    except OverflowError:
        odd = float(x).is_integer() and int(x) % 2 == 1
        return -math.inf if base < 0 and odd else math.inf
    except ValueError:
        return math.nan


def _space(log):
    # The kernels convert start and end to the output dtype after
    # allocating it (I3). For one step they fill start (logspace: base **
    # start) instead, which the CPU checks for a one-element output as
    # scalar_tensor does (_fill). No step converts nothing.
    def check(a, cuda):
        start, end, steps = a["start"], a["end"], a["steps"]
        dt = a["dtype"]
        if dt is None:
            dt = torch.complex64 if isinstance(start, complex) or isinstance(end, complex) else torch.float32
        if steps == 1:
            v = _pow(a["base"], start) if log else start
            if not _converts(v, torch.float64 if not cuda and dt in _LOOSE else dt, True):
                _fail(f"{'base ** start' if log else 'start'} {v!r} cannot be converted to {dt} without overflow")
        elif steps > 1:
            for name in ("start", "end"):
                if not _converts(a[name], dt, cuda):
                    _fail(f"{name} {a[name]!r} cannot be converted to {dt} without overflow")

    return check


def _multinomial(a, cuda):
    p = a["self"]
    if _kind(_dtype(p)) != "f":
        _fail(f"probabilities must be a floating-point tensor, got {_dtype(p)}")
    if _dim(p) and _shape(p)[-1] == 0:
        _fail("probabilities have no categories")


def _native_dropout(a, cuda):
    # train=None trains. The CPU kernel draws the mask with bernoulli_(1 - p)
    # in input's dtype and scales it by a double; the CUDA kernel computes
    # in floats only (it skips that for p == 1, which this refuses anyway).
    x, train = a["input"], a["train"]
    if train is False or not _numel(x):
        return
    _refuse("input", _dtype(x), "bic" if cuda else "bi")
    if not cuda and not 0 <= a["p"] <= 1:
        _fail(f"p {a['p']} must be in [0, 1]")


def _native_dropout_backward(a, cuda):
    # The CUDA kernel only; the CPU kernel multiplies any dtypes.
    g, m = _dtype(a["grad_output"]), _dtype(a["mask"])
    if cuda:
        _refuse("grad_output", g, "bic")
        if m != torch.bool:
            _fail(f"mask must be bool, got {m}")


def _softmax_backward(log):
    # meta.py makes grad_output's shape and dtype output's. The kernels
    # take input_dtype half with float32 gradients (CUDA only), else
    # ignore it on the CPU and need it to match on CUDA.
    def check(a, cuda):
        g, d = _dtype(a["grad_output"]), a["input_dtype"]
        half = g == torch.float32 and d == torch.float16
        if cuda and g != d and not half:
            _fail(f"input_dtype {d} must be grad_output's dtype {g}")
        if _numel(a["output"]):
            _refuse("grad_output", g, "bic")
            if half and not cuda:
                _fail("the CPU kernel does not take float32 gradients for a half input")

    return check


def _nansum(a, cuda):
    # The CPU kernel only; an integer or bool input runs as sum.
    x, dt = _dtype(a["self"]), a["dtype"]
    if cuda or _kind(x) in "bi":
        return
    if x.is_complex:
        _fail("nansum on the CPU does not take complex inputs")
    if _numel(a["self"]) and dt is not None and dt.is_complex:
        _fail(f"nansum of {x} into {dt} is not supported")


def _scan(a, cuda):
    # cumsum and cumprod; an empty input returns before dispatch.
    if _numel(a["self"]):
        _refuse("dtype", a["dtype"], "b")


def _foreach_norm(a, cuda):
    # linalg_vector_norm's checks per tensor, which the meta kernel skips.
    o, dt = a["ord"], a["dtype"]
    if isinstance(o, (bool, complex)):
        _fail(f"ord must be an int or a float, got {o!r}")
    for t in a["self"]:
        sd = t.dtype
        if not (sd.is_floating_point or sd.is_complex):
            _fail(f"expected floating-point or complex tensors, got {sd}")
        if dt is not None:
            if not (dt.is_floating_point or dt.is_complex) or dt.is_complex != sd.is_complex:
                _fail(f"dtype {dt} does not fit a {sd} tensor")
            if torch.promote_types(sd, dt) != dt:
                _fail(f"{sd} does not convert to {dt} without narrowing")
        if not t.numel() and (o < 0 or o == math.inf):
            _fail(f"the {o} norm of an empty tensor is undefined")


_RANDOM = {
    "aten.arange.default": _arange,
    "aten.arange.start": _arange,
    "aten.arange.start_step": _arange,
    "aten.linspace.default": _all(_no("b", numel=lambda a: a["steps"] > 1), _space(False)),
    "aten.logspace.default": _all(_no("b", numel=lambda a: a["steps"] > 1), _space(True)),
    "aten.rand.default": _no("bi"),
    "aten.rand_like.default": _no("bi", "self"),
    "aten.randint.default": _all(_no("c"), _random_range(None, "high")),
    "aten.randint.low": _all(_no("c"), _random_range("low", "high")),
    "aten.randint_like.default": _all(_no("c", "self"), _random_range(None, "high", "self")),
    "aten.randint_like.low_dtype": _all(_no("c", "self"), _random_range("low", "high", "self")),
    "aten.randperm.default": _randperm,
    "aten.uniform.default": _uniform,
    "aten.normal.float_Tensor": _all(_normal_std, _fill("mean", "std", scalar_tensor=True)),
    "aten.normal.Tensor_Tensor": _normal_std,
    "aten.bernoulli.p": _bernoulli_p,
    "aten.bernoulli.default": _bernoulli,
    "aten.bernoulli.Tensor": _bernoulli,
    "aten.multinomial.default": _multinomial,
    "aten.native_dropout.default": _native_dropout,
    "aten.native_dropout_backward.default": _native_dropout_backward,
    "aten._softmax_backward_data.default": _softmax_backward(False),
    "aten._log_softmax_backward_data.default": _softmax_backward(True),
    "aten.nansum.default": _nansum,
    "aten.cumsum.default": _scan,
    "aten.cumprod.default": _scan,
    "aten._foreach_norm.Scalar": _foreach_norm,
}

_POOL3 = ("kernel_size", "stride", "padding", "divisor_override")

CHECKS = {
    "aten.cat.default": _promotes("tensors"),
    "aten.stack.default": _promotes("tensors"),
    "aten.where.self": _promotes("self", "other"),
    "aten.eq.Tensor": _promotes("self", "other"),
    "aten.ne.Tensor": _promotes("self", "other"),
    "aten._scaled_mm.default": _scaled_mm,
    "aten._adaptive_avg_pool2d.default": _output_size_len(2),
    "aten._adaptive_avg_pool3d.default": _output_size_len(3),
    "aten._adaptive_avg_pool2d_backward.default": _adaptive_backward,
    "aten._adaptive_avg_pool3d_backward.default": _adaptive_backward,
    "aten.avg_pool2d.default": _int32(*_POOL3),
    "aten.avg_pool2d_backward.default": _int32(*_POOL3),
    "aten.replication_pad2d.default": _rank("self", 3, 4),
    "aten.replication_pad3d.default": _rank("self", 4, 5),
    **_UPSAMPLING,
    "aten.full.default": _fill("fill_value"),
    "aten.full_like.default": _fill("fill_value", "self"),
    "aten.new_full.default": _fill("fill_value", "self"),
    "aten.fill.Scalar": _fill("value", "self"),
    "aten.scalar_tensor.default": _fill("s", default=torch.float32, scalar_tensor=True),
    "aten.constant_pad_nd.default": _constant_pad,
    "aten.affine_grid_generator.default": _affine_grid_generator,
    "aten.addmm.default": _same("self", "mat1", "mat2"),
    "aten._addmm_activation.default": _same("self", "mat1", "mat2"),
    "aten.addbmm.default": _same("self", "batch1", "batch2"),
    "aten.mm.default": _same("self", "mat2"),
    "aten.mv.default": _same("self", "vec"),
    "aten._cdist_forward.default": _same("x1", "x2"),
    "aten.complex.default": _same("real", "imag"),
    "aten._prelu_kernel.default": _same("self", "weight"),
    "aten._prelu_kernel_backward.default": _same("grad_output", "self", "weight"),
    "aten.huber_loss_backward.default": _same("grad_output", "self", "target"),
    "aten.grid_sampler_2d.default": _grid_sampler,
    "aten.grid_sampler_3d.default": _grid_sampler,
    "aten.grid_sampler_2d_backward.default": _grid_sampler,
    "aten.grid_sampler_3d_backward.default": _grid_sampler,
    "aten.log_sigmoid_backward.default": _log_sigmoid_backward,
    "aten._embedding_bag_backward.default": _embedding_bag_backward,
    "aten._embedding_bag_dense_backward.default": _embedding_bag_backward,
    "aten.index_add.default": _index_add,
    "aten.index_copy.default": _all(_same("self", "source"), _index_int64),
    "aten.index_fill.int_Scalar": _index_int64,
    "aten.index_fill.int_Tensor": _index_int64,
    "aten.index_select.default": _index_vector,
    "aten.searchsorted.Tensor": _searchsorted,
    "aten.bucketize.Tensor": _bucketize,
    "aten.index.Tensor": _some_index,
    "aten.index_put.default": _index_put,
    "aten._unsafe_index_put.default": _index_put,
    "aten._pdist_forward.default": _pdist,
    "aten.im2col.default": _im2col,
    "aten._fft_c2r.default": _fft_c2r,
    "aten.bitwise_not.default": _bitwise_not,
    "aten.threshold_backward.default": _threshold_backward,
    "aten.logsumexp.default": _logsumexp,
    "aten.max.default": _nonempty_reduce,
    "aten.min.default": _nonempty_reduce,
    "aten.max.dim": _nonempty_dim,
    "aten.min.dim": _nonempty_dim,
    "aten.linalg_lu_factor_ex.default": _lu_factor,
    "aten.channel_shuffle.default": _channel_shuffle,
    "aten._int_mm.default": _int_mm,
    "aten.binary_cross_entropy.default": _bce,
    "aten.binary_cross_entropy_backward.default": _bce,
    "aten.binary_cross_entropy_with_logits.default": _bce_logits,
    "aten.soft_margin_loss.default": _soft_margin,
    "aten.soft_margin_loss_backward.default": _soft_margin,
    "aten.gelu_backward.default": _gelu_backward,
    "aten.glu_backward.default": _glu_backward,
    "aten.nll_loss_forward.default": _nll(False, False),
    "aten.nll_loss_backward.default": _nll(False, True),
    "aten.nll_loss2d_forward.default": _nll(True, False),
    "aten.nll_loss2d_backward.default": _nll(True, True),
    "aten.convolution.default": _convolution,
    "aten.convolution_backward.default": _convolution_backward,
    "aten.native_batch_norm.default": _batch_norm("training"),
    "aten._native_batch_norm_legit.default": _batch_norm("training"),
    "aten._native_batch_norm_legit.no_stats": _batch_norm("training"),
    "aten._native_batch_norm_legit_functional.default": _batch_norm("training"),
    "aten._native_batch_norm_legit_no_training.default": _batch_norm(None),
    "aten.native_batch_norm_backward.default": _batch_norm("train"),
    "aten.native_layer_norm.default": _layer_norm,
    "aten.native_layer_norm_backward.default": _layer_norm,
    "aten.native_group_norm.default": _group_norm,
    "aten.native_group_norm_backward.default": _group_norm,
    "aten._fused_rms_norm.default": _rms_norm,
    "aten._fused_rms_norm_backward.default": _rms_norm,
    # The fused cells' CUDA kernels read every tensor as the first's dtype.
    "aten._thnn_fused_lstm_cell.default": _same("input_gates", "hidden_gates", "cx", "input_bias", "hidden_bias"),
    "aten._thnn_fused_lstm_cell_backward_impl.default": _same("grad_hy", "grad_cy", "cx", "cy", "workspace"),
    "aten._thnn_fused_gru_cell.default": _same("input_gates", "hidden_gates", "hx", "input_bias", "hidden_bias"),
    "aten._thnn_fused_gru_cell_backward.default": _same("grad_hy", "workspace"),
    **_RANDOM,
}


def check(name: str, flags: int, a: dict, cuda: bool) -> None:
    """Reject what the kernel on the session's device would for op `name`
    (with opset `flags`) and arguments `a` (by schema name)."""
    pin_memory(a)
    float8(name, flags, a)
    c = CHECKS.get(name)
    if c is not None:
        c(a, cuda)
