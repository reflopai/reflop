# SPDX-License-Identifier: Apache-2.0
"""Seed programs for the meta/kernel parity test (parity.py, #21).

PROGRAMS run small PyTorch computations on CPU tensors; every aten call in
the opset they make becomes a seed. DIRECT builds seeds for ops no CPU
program reaches through the dispatcher (reflop natives, .vec overloads)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from parity import flat_tensors

A = torch.ops.aten


def _g(*shape, seed=0, dtype=torch.float32, grad=False):
    t = torch.randn(shape, generator=torch.Generator().manual_seed(seed + 7 * len(shape))).to(dtype)
    return t.requires_grad_() if grad else t


def _bk(f, *shapes):
    """f(*inputs) forward and backward on fresh leaf tensors."""
    def p():
        xs = [_g(*s, seed=i, grad=True) for i, s in enumerate(shapes)]
        y = f(*xs)
        ys = flat_tensors(y)
        sum(z.float().sum() for z in ys if z.requires_grad).backward()
    return p


def _fw(f, *shapes):
    return lambda: f(*[_g(*s, seed=i) for i, s in enumerate(shapes)])


IDX = torch.tensor([[0, 2], [1, 1], [3, 0]])


def _bag_dense_backward(mode):
    w, idx, off = _g(6, 4), torch.tensor([1, 2, 4, 5, 4, 3]), torch.tensor([0, 2, 4])
    out, o2b, bag_size, maxi = A._embedding_bag.default(w, idx, off, False, mode)
    A._embedding_bag_dense_backward.default(torch.ones_like(out), idx, o2b, bag_size, maxi, 6, False, mode, None, -1)

PROGRAMS = [
    # reductions over dims
    _bk(lambda x: (x.sum(1), x.mean((0, 2), keepdim=True), x.prod(1), x.amax(-1), x.amin(0),
                   x.var(1), x.std(0, correction=0), x.logsumexp(2), x.norm(dim=1),
                   torch.linalg.vector_norm(x, 3, dim=(0, 1)), x.nansum(1), x.cumsum(1), x.cumprod(2),
                   x.logcumsumexp(0), torch.var_mean(x, 1)[0], torch.std_mean(x, 2)[1]), (3, 4, 5)),
    _fw(lambda x: (x.argmax(1), x.argmin(), x.max(1), x.min(0), x.sort(1), x.topk(2, 1), x.kthvalue(2, 2),
                   x.median(1), x.nanmedian(0), x.mode(1), x.cummax(1), x.cummin(0), x.aminmax(dim=1),
                   (x > 0).any(1), (x > 0).all(0), (x > 0).any((0, 1)), (x > 0).all((1,)),
                   torch.nonzero_static(x > 0, size=5)), (3, 4, 5)),
    _bk(lambda x: (x.softmax(1), x.log_softmax(-1), F.glu(x, 1)), (3, 4, 5)),
    # conv, pool, padding, upsample
    _bk(lambda x, w: F.conv2d(x, w, padding=1, stride=2), (2, 3, 6, 6), (4, 3, 3, 3)),
    _bk(lambda x, w: F.conv1d(x, w, groups=2, dilation=2), (2, 4, 9), (6, 2, 3)),
    _bk(lambda x, w: F.conv_transpose2d(x, w, stride=2), (1, 2, 4, 4), (2, 3, 3, 3)),
    _bk(lambda x: (F.max_pool2d(x, 2), F.avg_pool2d(x, 3, 1, 1), F.adaptive_avg_pool2d(x, 2),
                   F.adaptive_max_pool2d(x, 3), F.avg_pool1d(x[0], 2), F.adaptive_avg_pool1d(x[0], 3)),
        (2, 3, 6, 6)),
    _bk(lambda x: (F.max_pool3d(x, 2), F.avg_pool3d(x, 2), F.adaptive_avg_pool3d(x, 2),
                   F.adaptive_max_pool3d(x, 2)), (1, 2, 4, 4, 4)),
    _fw(lambda x: F.max_unpool2d(*F.max_pool2d(x, 2, return_indices=True), 2), (1, 2, 4, 4)),
    _fw(lambda x: F.max_unpool3d(*F.max_pool3d(x, 2, return_indices=True), 2), (1, 1, 4, 4, 4)),
    _bk(lambda x: (F.pad(x, (1, 2, 0, 1)), F.pad(x, (1, 1, 1, 1), mode="reflect"),
                   F.pad(x, (2, 2, 0, 1), mode="replicate"), F.pad(x[0], (1, 2), mode="reflect")), (1, 2, 4, 4)),
    _bk(lambda x: (F.pad(x, (1, 1, 1, 1, 1, 1), mode="reflect"), F.pad(x, (1, 0, 1, 0, 0, 1), mode="replicate")),
        (1, 1, 3, 3, 3)),
    _bk(lambda x: (F.interpolate(x, scale_factor=2), F.interpolate(x, size=(5, 7), mode="bilinear"),
                   F.interpolate(x, size=(3, 5), mode="bicubic", align_corners=True),
                   F.interpolate(x, size=(5, 5), mode="nearest-exact"),
                   F.interpolate(x, size=(6, 3), mode="bilinear", antialias=True)), (1, 2, 3, 4)),
    _bk(lambda x: (F.interpolate(x, size=5), F.interpolate(x, size=7, mode="linear"),
                   F.interpolate(x, size=6, mode="nearest-exact")), (1, 2, 4)),
    _bk(lambda x: (F.interpolate(x, size=(3, 4, 5)), F.interpolate(x, size=(3, 3, 3), mode="trilinear"),
                   F.interpolate(x, size=(2, 3, 4), mode="nearest-exact")), (1, 1, 2, 2, 2)),
    _fw(lambda x: (A.upsample_nearest2d.default(x, [4, 6]), A.upsample_bilinear2d.default(x, [4, 6], False),
                   A.upsample_nearest1d.default(x[0], [5]), A.upsample_linear1d.default(x[0], [5], True),
                   A.upsample_bicubic2d.default(x, [5, 5], False),
                   A._upsample_nearest_exact2d.default(x, [5, 5]),
                   A._upsample_nearest_exact1d.default(x[0], [6]),
                   A._upsample_bilinear2d_aa.default(x, [2, 2], False)), (1, 2, 3, 4)),
    _fw(lambda x: (A.upsample_nearest3d.default(x, [3, 4, 5]), A.upsample_trilinear3d.default(x, [3, 3, 3], False),
                   A._upsample_nearest_exact3d.default(x, [3, 3, 3])), (1, 1, 2, 2, 2)),
    _bk(lambda x: F.affine_grid(x, [2, 3, 4, 5], align_corners=False), (2, 2, 3)),
    _bk(lambda x: F.fold(F.unfold(x, 2, stride=1, padding=1), (5, 5), 2, stride=1, padding=1), (1, 2, 5, 5)),
    _bk(lambda x, w, b: torch._weight_norm(w, b, 0) @ x, (3, 2), (4, 3), (4, 1)),
    # shape and views
    _bk(lambda x: (x.select(1, 2), x[:, 1:3], x[..., ::2], x.view(12, 5), x.expand(2, 3, 4, 5), x.permute(2, 0, 1),
                   x.unsqueeze(1), x[:, :1].squeeze(1), x[:1, :1].squeeze((0, 1)), x.diagonal(1, 0, 2),
                   x.split([1, 3], 1), x.split(2, 2), x.chunk(2, 0), x.flip(0, 2), x.repeat(1, 2, 1),
                   x.transpose(0, 2), x.unfold(2, 2, 1), x.narrow(1, 1, 2), x.unbind(1), x.tile(2, 1),
                   x.roll(1, 2), x.roll((1, -1), (0, 1)), x.rot90(1, (1, 2)), x.tril(1), x.triu(-1),
                   x.as_strided((2, 2), (5, 1), 3), torch.diag_embed(x, 1, 0, 2),
                   x.repeat_interleave(2, 1)), (3, 4, 5)),
    _bk(lambda a, b: (torch.cat([a, b], 1), torch.stack([a, b], 2)), (2, 3), (2, 3)),
    _fw(lambda x: (A.unsqueeze_copy.default(x, 1), A.view_copy.default(x, [4, 15]),
                   A.permute_copy.default(x, [1, 2, 0]), A.expand_copy.default(x[:1], [2, 4, 5]),
                   A.transpose_copy.int(x, 0, 1), A.squeeze_copy.dims(x[:1], [0]),
                   A.slice_copy.Tensor(x, 1, 1, 3, 1), A.select_copy.int(x, 2, 1),
                   A.split_with_sizes_copy.default(x, [2, 2], 1), A.unbind_copy.int(x, 0),
                   A.diagonal_copy.default(x, 0, 1, 2), A.unfold_copy.default(x, 1, 2, 2),
                   A._unsafe_view.default(x, [60])), (3, 4, 5)),
    _fw(lambda x: (A.full.default([2, 3], 1.5), A.zeros.default([3]), A.ones.default([2, 2]),
                   A.new_zeros.default(x, [2]), A.new_ones.default(x, [3, 1]), A.new_full.default(x, [2], 2.0),
                   A.arange.default(5), A.arange.start(1, 6), A.arange.start_step(0, 10, 3),
                   A.linspace.default(0, 1, 5), A.logspace.default(0, 2, 4, 10.0),
                   A.empty.memory_format([2, 3]), A.empty_strided.default([2, 3], [3, 1]),
                   A.new_empty.default(x, [4]), A.new_empty_strided.default(x, [2, 2], [2, 1])), (3, 4)),
    # indexing
    _bk(lambda x: (x.gather(1, IDX), x.index_select(1, IDX[0]), x[IDX[:, 0]], x[:, IDX[0]],
                   x.take(IDX), torch.zeros(3, 5).scatter_add(1, IDX, x[:3, :2]),
                   torch.zeros(3, 5).scatter(1, IDX, x[:3, :2]), x.scatter(1, IDX, 1.5),
                   x[:3].scatter_reduce(1, IDX, x[:3, :2], "amax"), x.index_add(1, IDX[0], x[:, :2]),
                   x.index_copy(0, IDX[:2, 0], x[:2]), x.index_fill(1, IDX[0], -1.0),
                   x.index_fill(1, IDX[0], torch.tensor(2.0)),
                   x.index_put((IDX[:, 0], IDX[:, 1]), torch.ones(3), accumulate=True)), (4, 5)),
    _bk(lambda w: F.embedding(IDX, w, padding_idx=0), (6, 4)),
    _bk(lambda w: F.embedding_bag(IDX.reshape(-1), w, torch.tensor([0, 2, 4]), mode="mean"), (6, 4)),
    _bk(lambda w: F.embedding_bag(IDX.reshape(-1), w, torch.tensor([0, 2, 4]), mode="sum"), (6, 4)),
    _fw(lambda x: (A._unsafe_index.Tensor(x, [IDX[:, 0]]),
                   A._unsafe_index_put.default(x, [IDX[:, 0]], torch.ones(3, 5), False)), (4, 5)),
    lambda: _bag_dense_backward(1),
    lambda: _bag_dense_backward(2),
    # misc
    _bk(lambda a, b, w: (torch.lerp(a, b, 0.3), torch.lerp(a, b, w.sigmoid())), (3, 4), (3, 4), (3, 4)),
]

T = torch.tensor
L = torch.long


def _pd(x):
    """A symmetric positive-definite batch from x."""
    return x @ x.transpose(-1, -2) + 3 * torch.eye(x.shape[-1])


def _linalg():
    a, b = _g(3, 3, grad=True), _g(3, 2, seed=1, grad=True)
    pd = _pd(_g(2, 3, 3, seed=2)).requires_grad_()
    outs = [torch.linalg.inv(a), torch.linalg.solve(a, b), torch.linalg.det(a), torch.linalg.slogdet(a)[1],
            torch.linalg.cholesky(pd), torch.linalg.eigh(pd)[0], torch.linalg.qr(a)[0], torch.linalg.svd(a)[1],
            torch.linalg.matrix_exp(a), torch.linalg.pinv(a), torch.linalg.solve_triangular(a.triu(), b, upper=True),
            torch.cholesky_solve(b, torch.linalg.cholesky(pd)[0]), torch.triangular_solve(b, a.triu())[0]]
    lu, piv = torch.linalg.lu_factor(a)
    outs.append(torch.linalg.lu_solve(lu, piv, b))
    sum(o.sum() for o in outs).backward()


def _ctc(tensor_lengths):
    def p():
        lp = _g(12, 3, 5, grad=True)
        tl, il = T([4, 3, 2], dtype=torch.int32), T([12, 10, 12], dtype=torch.int32)
        tg = torch.randint(1, 5, (3, 4), generator=torch.Generator().manual_seed(1))
        if not tensor_lengths:
            tl, il = tl.tolist(), il.tolist()
        F.ctc_loss(lp.log_softmax(2), tg, il, tl).backward()
    return p


def _norms():
    x = _g(4, 3, 5, grad=True)
    w, b = _g(3, seed=1, grad=True), _g(3, seed=2, grad=True)
    rm, rv = torch.zeros(3), torch.ones(3)
    ys = [F.batch_norm(x, rm, rv, w, b, training=True), F.batch_norm(x, rm, rv, w, b, training=False),
          F.batch_norm(x, None, None, w, b, training=True), F.group_norm(x, 3, w, b),
          F.layer_norm(x, (5,), _g(5, seed=3, grad=True), _g(5, seed=4, grad=True)),
          F.rms_norm(x, (5,), _g(5, seed=5, grad=True))]
    sum(y.sum() for y in ys).backward()
    A._native_batch_norm_legit_functional.default(x.detach(), w.detach(), b.detach(), rm, rv, True, 0.1, 1e-5)
    A._native_batch_norm_legit.no_stats(x.detach(), w.detach(), b.detach(), True, 0.1, 1e-5)
    A._native_batch_norm_legit_no_training.default(x.detach(), w.detach(), b.detach(), rm, rv, 0.1, 1e-5)


def _foreach():
    xs = [_g(3, 4), _g(5, seed=1)]
    ys = [_g(3, 4, seed=2).abs() + 0.5, _g(5, seed=3).abs() + 0.5]
    zs = [_g(3, 4, seed=4), _g(5, seed=5)]
    s = T(0.5)
    for f in (torch._foreach_abs, torch._foreach_exp, torch._foreach_neg, torch._foreach_reciprocal,
              torch._foreach_sign, torch._foreach_sqrt, torch._foreach_norm):
        f(ys)
    for name in ("add", "sub", "mul", "div", "maximum", "minimum", "clamp_max", "clamp_min", "pow"):
        f = getattr(torch, f"_foreach_{name}")
        f(ys, zs)
        f(ys, 1.5)
        f(ys, [1.5, 2.0])
    for name in ("add", "sub", "mul", "div"):
        getattr(torch, f"_foreach_{name}")(ys, s) if name in ("mul", "div") else None
    torch._foreach_add(ys, s, alpha=2)
    torch._foreach_pow(2.0, ys)
    for f in (torch._foreach_addcmul, torch._foreach_addcdiv):
        f(xs, ys, zs, 0.5)
        f(xs, ys, zs, [0.5, 1.0])
        f(xs, ys, zs, T([0.5, 1.0]))
    torch._foreach_lerp(xs, zs, ys)
    torch._foreach_lerp(xs, zs, 0.3)
    torch._foreach_lerp(xs, zs, [0.3, 0.6])
    torch._foreach_zero_([x.clone() for x in xs])


PROGRAMS += [
    # activations and their backward
    _bk(lambda x: (F.relu(x), x.clamp(-0.5, 0.5), x.clamp(min=T(-0.5), max=T(0.5)), F.elu(x), F.gelu(x),
                   F.gelu(x, approximate="tanh"), F.hardtanh(x), F.leaky_relu(x, 0.2), F.hardsigmoid(x),
                   F.hardswish(x), F.logsigmoid(x), F.mish(x), F.silu(x), F.softplus(x), x.tanh(),
                   F.threshold(x, 0.1, 2.0), F.selu(x), F.celu(x)), (3, 4)),
    _bk(lambda x, w: F.prelu(x, w), (2, 3, 4), (3,)),
    # losses and their backward
    lambda: _bk(lambda x, t: (F.binary_cross_entropy(x.sigmoid(), t.sigmoid()),
                              F.binary_cross_entropy(x.sigmoid(), t.sigmoid(), weight=torch.rand(4, 5),
                                                     reduction="none"),
                              F.binary_cross_entropy_with_logits(x, t.sigmoid(), weight=torch.rand(5),
                                                                 pos_weight=torch.rand(5)),
                              F.huber_loss(x, t, delta=0.5), F.mse_loss(x, t, reduction="sum"),
                              F.smooth_l1_loss(x, t, beta=0.5, reduction="none"), F.soft_margin_loss(x, t.detach().sign())),
                (4, 5), (4, 5))(),
    _bk(lambda x: (F.nll_loss(x.log_softmax(1), T([0, 2, 1, 4]), weight=torch.rand(5)),
                   F.nll_loss(x.log_softmax(1), T([0, -100, 1, 4]), reduction="none"),
                   F.cross_entropy(x, T([1, 1, 0, 3]), reduction="sum")), (4, 5)),
    _bk(lambda x: F.nll_loss(x.log_softmax(1), torch.randint(0, 3, (2, 4, 5), generator=torch.Generator()
                                                             .manual_seed(3)), weight=torch.rand(3)), (2, 3, 4, 5)),
    # norms
    _norms,
    # matmul family
    _bk(lambda m, a, b, v, bb, cc: (torch.addmm(m, a, b), torch.addbmm(m, bb, cc), torch.baddbmm(m, bb, cc),
                                    torch.addmv(v, a, b[:, 0]), torch.mv(a, b[:, 0]), torch.dot(v, v),
                                    torch.vdot(v, v), torch.outer(v, v)),
        (3, 3), (3, 4), (4, 3), (3,), (2, 3, 5), (2, 5, 3)),
    _fw(lambda m, a, b: A._addmm_activation.default(m, a, b, use_gelu=True), (3, 3), (3, 4), (4, 3)),
    lambda: A._int_mm.default(torch.randint(-5, 5, (32, 32), dtype=torch.int8),
                              torch.randint(-5, 5, (32, 32), dtype=torch.int8)),
    # linear algebra
    _linalg,
    # distances
    _bk(lambda a, b: (torch.cdist(a, b), torch.cdist(a, b, p=1), F.pdist(a[0]), F.pdist(a[0], p=3)),
        (2, 4, 3), (2, 5, 3)),
    # ctc
    _ctc(False), _ctc(True),
    # resampling and spatial
    _bk(lambda x, g: (F.grid_sample(x, g.tanh(), align_corners=False),
                      F.grid_sample(x, g.tanh(), mode="nearest", padding_mode="border", align_corners=True)),
        (1, 2, 4, 5), (1, 3, 3, 2)),
    _bk(lambda x, g: F.grid_sample(x, g.tanh(), align_corners=False), (1, 2, 3, 4, 5), (1, 2, 3, 3, 3)),
    _bk(lambda x: (F.pixel_shuffle(x, 2), F.pixel_unshuffle(x, 2), F.channel_shuffle(x, 2)), (1, 4, 4, 6)),
    # complex and fft
    _bk(lambda a, b: (torch.polar(a.abs(), b), torch.complex(a, b).angle(),
                      torch.view_as_real(torch.view_as_complex(torch.stack([a, b], -1))),
                      torch.complex(a, b).conj_physical().real, torch.complex(a, b).imag,
                      torch.fft.rfft(a, dim=1).abs(), torch.fft.irfft(torch.fft.rfft(a), n=6),
                      torch.fft.fft(torch.complex(a, b), dim=0).abs(), torch.fft.ifft2(torch.complex(a, b)).real),
        (3, 6), (3, 6)),
    # indexing
    lambda: (torch.bucketize(_g(3, 4), T([-1.0, 0.0, 1.0])), torch.bucketize(_g(5), T([-1.0, 1.0]), right=True),
             torch.searchsorted(T([[-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]), _g(2, 3), side="right"),
             torch.searchsorted(T([1.0, -1.0, 0.0]), _g(4), sorter=T([1, 2, 0])),
             _g(3, 4).masked_fill(_g(3, 4) > 0, T(2.0)), _g(3, 4).masked_scatter(_g(3, 4) > 0, _g(12, seed=4))),
    lambda: F.embedding(IDX, _g(6, 4), max_norm=1.0),
    # creation
    _fw(lambda x: (torch.eye(3), torch.eye(2, 4), x.fill_(2.5).clone(), torch.full_like(x, 3.0)), (3, 4)),
    # whole-tensor reductions
    _bk(lambda x: (x.max(), x.min(), x.median(), x.nanmedian(), x.mean(), x.prod(), x.var(1, unbiased=False),
                   x.norm(2, dim=1)), (3, 4)),
    _fw(lambda x: ((x > 0).all(), (x > 0).any()), (3, 4)),
    # foreach (optimizer building blocks)
    _foreach,
]


def DIRECT():
    """Seeds no CPU program reaches through the dispatcher."""
    from parity import Call

    x = _g(1, 2, 3, 4)
    p, gr = [_g(3, 4), _g(5, seed=1)], [_g(3, 4, seed=2), _g(5, seed=3)]
    m, v = [torch.zeros(3, 4), torch.zeros(5)], [torch.zeros(3, 4), torch.zeros(5)]
    mx, st = [torch.zeros(3, 4), torch.zeros(5)], [torch.ones(()), torch.ones(())]
    out = [Call("aten.upsample_nearest2d.vec", [x, [4, 6], None]),
           Call("aten.upsample_nearest2d.vec", [x, None, [2.0, 1.5]]),
           Call("aten.upsample_bilinear2d.vec", [x, [5, 7], False, None]),
           Call("aten.upsample_bicubic2d.vec", [x, None, True, [1.5, 2.0]]),
           Call("aten._upsample_bilinear2d_aa.vec", [x, [2, 2], False, None]),
           Call("aten._upsample_nearest_exact2d.vec", [x, [5, 5], None]),
           Call("aten.upsample_linear1d.vec", [x[0], [7], True, None]),
           Call("aten.upsample_nearest1d.vec", [x[0], None, [2.0]]),
           Call("aten._upsample_nearest_exact1d.vec", [x[0], [6], None]),
           Call("aten.upsample_nearest3d.vec", [x.unsqueeze(2), [2, 4, 6], None]),
           Call("aten._upsample_nearest_exact3d.vec", [x.unsqueeze(2), [2, 3, 3], None]),
           Call("aten.upsample_trilinear3d.vec", [x.unsqueeze(2), [2, 3, 3], False, None])]
    for amsgrad in (False, True):
        for name in ("reflop.fused_adam.default", "reflop.fused_adamw.default"):
            out.append(Call(name, [p, gr, m, v, mx if amsgrad else [], st, 1e-3, 0.9, 0.999, 0.01, 1e-8, amsgrad,
                                   False, None, None]))
    xb, w, b = _g(4, 3, 5), _g(3, seed=1), _g(3, seed=2)
    out += [Call("aten._native_batch_norm_legit.default", [xb, w, b, torch.zeros(3), torch.ones(3), True, 0.1, 1e-5]),
            Call("aten._native_batch_norm_legit.default", [xb, w, None, torch.zeros(3), torch.ones(3), False, 0.1,
                                                           1e-5])]
    out += [Call("reflop.fused_sgd.default", [p, gr, m, 0.0, 0.9, 0.1, 0.0, False, False, False, None, None]),
            Call("reflop.fused_sgd.default", [p, gr, [], 0.0, 0.0, 0.1, 0.0, False, False, True, None, None]),
            Call("reflop.fused_adagrad.default", [p, gr, m, st, 1e-2, 0.0, 0.0, 1e-10, False, None, None]),
            Call("reflop.amp_unscale_and_check.default", [gr, torch.zeros(1), T([0.5])]),
            Call("reflop.amp_update_scale.default", [T([1024.0]), T([0], dtype=torch.int32), T([0.0]), 2.0, 0.5,
                                                     2000])]
    return out
