"""a02 tensor operators: ~150 forward ops and autograd checks vs the local CPU, on the GPU and the server CPU,
under strict "remote" placement (no silent fallback); data-dependent shapes, RNG, dtypes, views/in-place."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

from live.harness import Expected, Skip, Suite, check, close_to, q

G = torch.Generator().manual_seed(1234)


def r(*shape, dtype=torch.float32, lo=None):
    t = torch.randn(*shape, generator=G)
    if lo is not None:
        t = t.abs() + lo
    return t.to(dtype)


def ri(*shape, hi=10, dtype=torch.int64):
    return torch.randint(0, hi, shape, generator=G, dtype=dtype)


A, B = r(16, 32), r(16, 32)
M1, M2 = r(24, 40), r(40, 12)
POS = r(16, 32, lo=0.1)
IMG = r(2, 3, 16, 16)
SEQ = r(2, 4, 10, 8)
I8 = ri(4, 6, hi=100)

# (name, inputs, fn, rtol, atol). fn receives tensors already on the target device.
OPS = [
    # elementwise
    ("add", (A, B), lambda a, b: a + b), ("sub", (A, B), lambda a, b: a - b), ("mul", (A, B), lambda a, b: a * b),
    ("div", (A, POS), lambda a, b: a / b), ("pow", (POS, A), lambda a, b: a ** b.clamp(-2, 2)),
    ("neg_abs", (A,), lambda a: (-a).abs()), ("exp", (A,), torch.exp), ("log", (POS,), torch.log),
    ("log1p", (POS,), torch.log1p), ("expm1", (A,), torch.expm1), ("sqrt", (POS,), torch.sqrt),
    ("rsqrt", (POS,), torch.rsqrt), ("sin_cos_tan", (A,), lambda a: a.sin() + a.cos() + a.clamp(-1, 1).tan()),
    ("asin_acos_atan", (A,), lambda a: a.tanh().asin() + a.tanh().acos() + a.atan()),
    ("sinh_cosh", (A,), lambda a: a.clamp(-3, 3).sinh() + a.clamp(-3, 3).cosh()),
    ("atan2", (A, B), torch.atan2), ("erf_erfc_erfinv", (A,), lambda a: a.erf() + a.erfc() + a.tanh().mul(0.9).erfinv()),
    ("lgamma_digamma", (POS,), lambda a: a.lgamma() + a.digamma()), ("sigmoid", (A,), torch.sigmoid),
    ("floor_ceil_round_trunc", (A,), lambda a: (a * 3).floor() + (a * 3).ceil() + (a * 3).round() + (a * 3).trunc()),
    ("frac_sign", (A,), lambda a: a.frac() + a.sign()), ("clamp", (A,), lambda a: a.clamp(-0.5, 0.7)),
    ("where", (A, B), lambda a, b: torch.where(a > b, a, b)), ("lerp", (A, B), lambda a, b: torch.lerp(a, b, 0.3)),
    ("addcmul_addcdiv", (A, B, POS), lambda a, b, c: torch.addcmul(a, b, c, value=0.5) + torch.addcdiv(a, b, c)),
    ("fmod_remainder", (A, POS), lambda a, b: torch.fmod(a, b) + torch.remainder(a, b)),
    ("maximum_minimum", (A, B), lambda a, b: torch.maximum(a, b) - torch.minimum(a, b)),
    ("hypot", (A, B), torch.hypot), ("xlogy", (POS, POS), torch.xlogy), ("nan_to_num", (A,),
                                                                         lambda a: torch.nan_to_num(a.log())),
    ("comparisons", (A, B), lambda a, b: (a > b).float() + (a <= b).float() * 2 + (a == a).float() + (a != b).float()),
    ("logical", (A, B), lambda a, b: torch.logical_xor(a > 0, b > 0) | (torch.logical_not(a > 0) & (b < 0))),
    ("isfinite_isnan", (A,), lambda a: torch.isfinite(a.log()).float() + torch.isnan(a.log()).float()),
    # activations
    ("relu_gelu_silu", (A,), lambda a: F.relu(a) + F.gelu(a) + F.silu(a) + F.gelu(a, approximate="tanh")),
    ("elu_selu_celu", (A,), lambda a: F.elu(a) + F.selu(a) + F.celu(a)),
    ("leaky_hardtanh", (A,), lambda a: F.leaky_relu(a, 0.1) + F.hardtanh(a) + F.hardswish(a) + F.hardsigmoid(a)),
    ("softplus_mish", (A,), lambda a: F.softplus(a) + F.mish(a) + F.softsign(a) + F.logsigmoid(a)),
    ("softmax", (A,), lambda a: F.softmax(a, -1) + F.log_softmax(a, 0)), ("glu", (A,), lambda a: F.glu(a, -1)),
    # reductions
    ("sum_mean", (A,), lambda a: torch.stack([a.sum(), a.mean(), a.sum(0).mean(), a.mean(1, keepdim=True).sum()])),
    ("prod", (POS,), lambda a: a[:4, :6].prod(1)), ("var_std", (A,), lambda a: torch.stack([a.var(), a.std(0).sum(),
                                                                                             a.var(1, unbiased=False).sum()])),
    ("min_max", (A,), lambda a: torch.stack([a.max(), a.min(), a.amax(0).sum(), a.amin(1).sum()])),
    ("argmax_argmin", (A,), lambda a: torch.stack([a.argmax(), a.argmin(1).sum()])),
    ("max_dim", (A,), lambda a: a.max(1).values + a.max(1).indices.float()),
    ("norms", (A,), lambda a: torch.stack([a.norm(), a.norm(p=1), torch.linalg.vector_norm(a, 3), a.norm(dim=1).sum()])),
    ("logsumexp", (A,), lambda a: torch.logsumexp(a, 1)), ("cumsum_cumprod", (A,), lambda a: a.cumsum(1) + (a * 0.1 + 1).cumprod(0)),
    ("cummax", (A,), lambda a: torch.cummax(a, 1).values), ("all_any", (A,), lambda a: torch.stack([(a > -10).all(), (a > 3).any()])),
    ("count_nonzero", (A,), lambda a: torch.count_nonzero(a > 0, 1)),
    ("sort", (A,), lambda a: torch.sort(a, 1).values), ("argsort", (A,), lambda a: torch.argsort(a, dim=0, stable=True)),
    ("topk", (A,), lambda a: torch.topk(a, 5, 1).values), ("kthvalue_median", (A,), lambda a: a.kthvalue(3, 1).values + a.median(1).values),
    ("quantile", (A,), lambda a: torch.quantile(a, torch.tensor([0.1, 0.5, 0.9], device=a.device), dim=1)),
    ("aminmax", (A,), lambda a: torch.stack(torch.aminmax(a))),
    # linear algebra
    ("mm", (M1, M2), torch.mm), ("matmul_batched", (r(3, 8, 5), r(3, 5, 7)), torch.matmul),
    ("bmm_baddbmm", (r(3, 8, 5), r(3, 5, 7), r(3, 8, 7)), lambda a, b, c: torch.baddbmm(c, a, b, beta=0.5)),
    ("addmm_addmv", (M1, M2, r(24, 12), r(40)), lambda a, b, c, v: torch.addmm(c, a, b) .sum() + torch.addmv(a[:, 0], a, v).sum()),
    ("einsum", (r(4, 5, 6), r(6, 7)), lambda a, b: torch.einsum("ijk,kl->ijl", a, b)),
    ("outer_dot", (r(8), r(8)), lambda a, b: torch.outer(a, b).sum() + torch.dot(a, b)),
    ("linalg_inv_det", (r(6, 6) + 6 * torch.eye(6),), lambda a: torch.linalg.inv(a).sum() + torch.linalg.det(a)),
    ("linalg_solve", (r(6, 6) + 6 * torch.eye(6), r(6, 3)), torch.linalg.solve),
    ("cholesky", (torch.eye(5) * 3 + 0.1,), torch.linalg.cholesky), ("qr", (r(8, 5),), lambda a: torch.linalg.qr(a).R.abs()),
    ("svdvals", (r(8, 5),), torch.linalg.svdvals), ("eigvalsh", (r(5, 5) @ r(5, 5).T,), torch.linalg.eigvalsh),
    ("cross_tril_triu", (r(4, 3), r(4, 3)), lambda a, b: torch.linalg.cross(a, b).sum() + torch.tril(a @ b.T).sum() + torch.triu(a @ b.T, 1).sum()),
    ("cdist", (r(10, 4), r(7, 4)), torch.cdist), ("trace_diag", (r(6, 6),), lambda a: a.trace() + torch.diag(a).sum() + torch.diag_embed(a[0]).sum()),
    # convolution / pooling / norms
    ("conv1d", (r(2, 3, 20), r(5, 3, 3)), lambda x, w: F.conv1d(x, w, padding=1)),
    ("conv2d", (IMG, r(4, 3, 3, 3)), lambda x, w: F.conv2d(x, w, stride=2, padding=1)),
    ("conv2d_dilated_groups", (r(2, 4, 12, 12), r(4, 2, 3, 3)), lambda x, w: F.conv2d(x, w, dilation=2, groups=2)),
    ("conv3d", (r(1, 2, 6, 6, 6), r(3, 2, 3, 3, 3)), F.conv3d),
    ("conv_transpose2d", (r(2, 4, 8, 8), r(4, 3, 3, 3)), lambda x, w: F.conv_transpose2d(x, w, stride=2)),
    ("max_avg_pool", (IMG,), lambda x: F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)),
    ("adaptive_pool", (IMG,), lambda x: F.adaptive_avg_pool2d(x, 3) + F.adaptive_max_pool2d(x, 3)),
    ("max_pool_indices", (IMG,), lambda x: F.max_pool2d(x, 2, return_indices=True)[1]),
    ("batch_norm_train", (IMG,), lambda x: F.batch_norm(x, None, None, training=True)),
    ("layer_group_instance_norm", (IMG,), lambda x: F.layer_norm(x, (16,)) + F.group_norm(x, 3) + F.instance_norm(x)),
    ("rms_norm", (A,), lambda a: F.rms_norm(a, (32,))), ("normalize", (A,), lambda a: F.normalize(a, dim=1)),
    ("interpolate", (IMG,), lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False).sum((2, 3))
     + F.interpolate(x, size=(5, 7), mode="nearest").sum((2, 3)) + F.interpolate(x, scale_factor=2, mode="bicubic").sum((2, 3))),
    ("pad", (IMG,), lambda x: F.pad(x, (1, 2, 3, 0)) .sum() + F.pad(x, (2, 2, 2, 2), mode="reflect").sum() + F.pad(x, (1, 1, 1, 1), mode="replicate").sum()),
    ("grid_sample", (IMG, r(2, 5, 5, 2).tanh()), lambda x, g: F.grid_sample(x, g, align_corners=False)),
    ("pixel_shuffle_unfold", (r(1, 8, 4, 4),), lambda x: F.pixel_shuffle(x, 2).sum() + F.unfold(x, 2).sum()),
    ("dropout_eval", (A,), lambda a: F.dropout(a, 0.5, training=False)),
    ("sdpa", (SEQ, SEQ * 0.5, SEQ + 1), lambda q_, k, v: F.scaled_dot_product_attention(q_, k, v)),
    ("sdpa_causal", (SEQ, SEQ * 0.5, SEQ + 1), lambda q_, k, v: F.scaled_dot_product_attention(q_, k, v, is_causal=True)),
    ("embedding", (ri(4, 7, hi=20), r(20, 6)), lambda i, w: F.embedding(i, w)),
    ("embedding_bag", (ri(12, hi=20), r(20, 6)), lambda i, w: F.embedding_bag(i, w, torch.tensor([0, 4, 9], device=i.device))),
    ("one_hot", (ri(9, hi=5),), lambda i: F.one_hot(i, 5)),
    # losses
    ("cross_entropy", (r(8, 5), ri(8, hi=5)), lambda x, y: F.cross_entropy(x, y) + F.cross_entropy(x, y, label_smoothing=0.1)),
    ("nll_mse_l1", (r(8, 5), ri(8, hi=5), r(8, 5)), lambda x, y, t: F.nll_loss(F.log_softmax(x, 1), y) + F.mse_loss(x, t) + F.l1_loss(x, t) + F.smooth_l1_loss(x, t)),
    ("bce", (r(8, 5), r(8, 5).sigmoid()), lambda x, t: F.binary_cross_entropy_with_logits(x, t) + F.binary_cross_entropy(x.sigmoid(), t)),
    ("kl_huber_cos", (r(8, 5), r(8, 5)), lambda a, b: F.kl_div(F.log_softmax(a, 1), F.softmax(b, 1), reduction="batchmean") + F.huber_loss(a, b) + F.cosine_similarity(a, b).sum()),
    ("ctc_loss", (r(12, 2, 6).log_softmax(2), ri(2, 4, hi=5) + 1), lambda lp, tg: F.ctc_loss(lp, tg, [12, 12], [4, 3])),
    ("ctc_loss_tensor_lengths", (r(12, 2, 6).log_softmax(2), ri(2, 4, hi=5) + 1), lambda lp, tg: F.ctc_loss(lp, tg, torch.tensor([12, 12]), torch.tensor([4, 3]))),
    # indexing / shape
    ("index_select_gather", (A, ri(5, hi=16), ri(16, 3, hi=32)), lambda a, i, j: a.index_select(0, i).sum() + a.gather(1, j).sum()),
    ("scatter_add", (torch.zeros(5, 6), ri(3, 6, hi=5), r(3, 6)), lambda z, i, s: z.scatter_add(0, i, s)),
    ("scatter_reduce", (torch.zeros(5, 6), ri(3, 6, hi=5), r(3, 6)), lambda z, i, s: z.scatter_reduce(0, i, s, "amax", include_self=False)),
    ("index_add_put", (torch.zeros(6, 4), ri(5, hi=6), r(5, 4)), lambda z, i, s: z.index_add(0, i, s).sum() + z.index_put((i,), s, accumulate=True).sum()),
    ("advanced_index", (A, ri(4, hi=16), ri(4, hi=32)), lambda a, i, j: a[i, j] + a[i][:, j].sum(1)),
    ("masked_fill_select", (A,), lambda a: a.masked_fill(a > 0.5, -1.0)),
    ("bool_mask_index", (A,), lambda a: a[a > 0.3]), ("nonzero", (A,), lambda a: torch.nonzero(a > 1.0)),
    ("masked_select", (A,), lambda a: torch.masked_select(a, a < -0.5)),
    ("cat_stack_split", (A, B), lambda a, b: torch.cat([a, b], 1).sum(0) + torch.stack([a, b]).sum((0, 1)).sum() + torch.split(a, 5, 1)[2].sum()),
    ("chunk_unbind", (A,), lambda a: torch.stack([c.sum() for c in a.chunk(3, 0)] + [u.sum() for u in a[:3].unbind(0)])),
    ("reshape_permute", (IMG,), lambda x: x.permute(0, 2, 3, 1).reshape(2, -1, 3).transpose(1, 2).contiguous()),
    ("expand_repeat_tile", (r(1, 4),), lambda a: a.expand(3, 4).sum() + a.repeat(2, 3).sum() + a.tile(3).sum() + a.repeat_interleave(2, 1).sum()),
    ("flip_roll_rot90", (A,), lambda a: a.flip(0) + a.roll(3, 1) + torch.rot90(a.reshape(32, 16), 1).reshape(16, 32)),
    ("narrow_select_unfold", (A,), lambda a: a.narrow(1, 3, 5).sum() + a.select(0, 2).sum() + a.unfold(1, 4, 2).sum()),
    ("squeeze_unsqueeze_flatten", (r(1, 4, 1, 5),), lambda a: a.squeeze().unsqueeze(0).flatten().unflatten(0, (2, 10))),
    ("diff_cumsum", (A,), lambda a: torch.diff(a, dim=1)), ("searchsorted_bucketize", (torch.linspace(-2, 2, 9), A),
                                                             lambda b, a: torch.searchsorted(b, a) + torch.bucketize(a, b)),
    ("tril_indices_meshgrid", (r(3),), lambda a: torch.stack(torch.meshgrid(a, a, indexing="ij")).sum() + a.new_ones(4, 4).tril().sum()),
    ("take_along_dim", (A, ri(16, 2, hi=32)), lambda a, i: torch.take_along_dim(a, i, 1)),
    ("histc_bincount", (A, ri(30, hi=7)), lambda a, i: torch.histc(a, 10, -2, 2).sum() + torch.bincount(i, minlength=7).sum()),
    # dtypes and casts
    ("int_arith", (I8, I8 + 1), lambda a, b: (a * b + a // b - a % b) & 0xFF),
    ("int_shift_bitwise", (I8,), lambda a: (a << 2) ^ (a >> 1) | a.bitwise_not()),
    ("int_div_true", (I8, I8 + 1), lambda a, b: torch.div(a, b) + torch.div(a, b, rounding_mode="floor")),
    ("bool_ops", (A,), lambda a: ((a > 0) & (a < 1)).sum() + (a > 0).float().mean()),
    ("casts", (A,), lambda a: a.half().float() + a.bfloat16().float() + a.double().float() + a.int().float() + a.to(torch.uint8).float()),
    ("fp16_math", (A.half(), B.half()), lambda a, b: (a @ b.T).relu()),
    ("bf16_math", (A.bfloat16(), B.bfloat16()), lambda a, b: (a @ b.T).relu()),
    ("fp64_math", (A.double(), B.double()), lambda a, b: (a @ b.T).exp().log1p()),
    ("complex", (A, B), lambda a, b: (torch.complex(a, b) * torch.complex(b, a)).abs()),
    ("fft", (r(4, 16),), lambda a: torch.fft.rfft(a).abs() + 0), ("fft_ifft", (r(4, 16),), lambda a: torch.fft.ifft(torch.fft.fft(a)).real),
    # factories (device-side)
    ("factories", (A,), lambda a: torch.zeros(3, 4, device=a.device).sum() + torch.ones(3, device=a.device).sum()
     + torch.full((2, 2), 7.0, device=a.device).sum() + torch.arange(10, device=a.device).sum() + torch.linspace(0, 1, 5, device=a.device).sum()
     + torch.eye(4, device=a.device).sum()),
    ("like_factories", (A,), lambda a: torch.zeros_like(a).sum() + torch.ones_like(a).sum() + torch.full_like(a, 2).sum() + a.new_zeros(3).sum() + torch.empty_like(a).zero_().sum()),
    # in-place and views
    ("inplace_chain", (A.clone(),), lambda a: a.mul_(2).add_(1).clamp_(-1, 3).sqrt_()),
    ("view_inplace_aliasing", (A.clone(),), lambda a: (a[:, :4].fill_(0.5), a.view(-1)[::7].mul_(3), a.T[2].zero_(), a)[-1]),
    ("out_arguments", (A, B), lambda a, b: torch.add(a, b, out=torch.empty_like(a))),
    ("copy_and_set", (A, B), lambda a, b: a.clone().copy_(b) + a.detach().clone().index_fill_(1, torch.tensor([1, 3], device=a.device), 9.0)),
    ("strided_noncontig", (r(8, 9, 10),), lambda a: a.transpose(0, 2)[::2, 1::3].contiguous() + 1),
    ("channels_last", (IMG, r(4, 3, 3, 3)), lambda x, w: F.conv2d(x.contiguous(memory_format=torch.channels_last), w)),
    ("zero_size", (r(0, 5), r(5, 3)), lambda a, b: a @ b), ("scalar_tensor", (torch.tensor(3.5),), lambda a: a * 2 + a.exp()),
    ("high_rank", (r(2, 1, 3, 1, 2, 2, 1),), lambda a: a.sum((1, 3, 6)) * a.squeeze().mean()),
    ("special_values", (torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-40, 1e38]),),
     lambda a: torch.stack([a * 2, a.exp(), torch.sign(a), 1 / a, a.isinf().float()])),
]

# RNG-dependent ops: compared for reproducibility (same seed -> same values) and distribution, not vs CPU.
RNG = [
    ("randn", lambda d: torch.randn(4000, device=d)),
    ("rand", lambda d: torch.rand(4000, device=d)),
    ("randint", lambda d: torch.randint(0, 10, (4000,), device=d).float()),
    ("randint_int32", lambda d: torch.randint(0, 10, (4000,), device=d, dtype=torch.int32).float()),
    ("dropout_train", lambda d: F.dropout(torch.ones(4000, device=d), 0.5, training=True)),
    ("bernoulli", lambda d: torch.bernoulli(torch.full((4000,), 0.3, device=d))),
    ("normal_", lambda d: torch.empty(4000, device=d).normal_(2.0, 3.0)),
    ("uniform_", lambda d: torch.empty(4000, device=d).uniform_(-1, 1)),
    ("multinomial", lambda d: torch.multinomial(torch.ones(50, device=d), 20).float()),
    ("randperm", lambda d: torch.randperm(1000, device=d).float()),
]

GRAD = [
    ("grad_matmul_chain", lambda a, b: (a @ b.T).tanh().sum(), (A, B)),
    ("grad_conv_bn", lambda x, w: F.batch_norm(F.conv2d(x, w, padding=1), None, None, training=True).relu().pow(2).mean(), (IMG, r(4, 3, 3, 3))),
    ("grad_attention", lambda q_, k, v: F.scaled_dot_product_attention(q_, k, v, is_causal=True).square().sum(), (SEQ, SEQ * 0.5, SEQ + 1)),
    ("grad_embedding_ce", lambda w, t: F.cross_entropy(F.embedding(torch.arange(6, device=w.device), w), t.argmax(1)), (r(6, 5), r(6, 5))),
    ("grad_views_index", lambda a: (a[::2, 1:7].exp() * a.T[:8, :6]).sum() + a[a > 0].sum(), (A,)),
    ("grad_layernorm_gelu", lambda a: F.gelu(F.layer_norm(a, (32,))).pow(3).mean(), (A,)),
    ("grad_lstm", None, None),
    ("grad_second_order", None, None),
]


def run_op(dev, fn, ins, rtol, atol, name):
    import reflop

    ref = fn(*[x.clone() for x in ins])
    try:
        with torch.no_grad():
            out = fn(*[x.to(dev) for x in ins])       # under placement("remote"): no silent client fallback
            if isinstance(out, tuple):
                out = out[0]
            out = out.cpu()
    except reflop.UnsupportedOperatorError as e:
        raise Expected(f"unsupported remotely: {str(e)[:200]}") from None
    if isinstance(ref, tuple):
        ref = ref[0]
    return f"max abs diff {close_to(out, ref, rtol, atol, name):.2e}"


def op_tol(name):
    if "fp16" in name or "bf16" in name:
        return 2e-2, 2e-2
    if "conv" in name or name == "channels_last":      # TF32 is on for cuDNN convolutions (documented)
        return 5e-3, 5e-3
    if name in ("linalg_inv_det", "linalg_solve", "qr", "svdvals", "eigvalsh", "cholesky", "pow", "lgamma_digamma",
                "special_values", "erf_erfc_erfinv", "cumsum_cumprod", "cdist", "quantile", "interpolate",
                "ctc_loss", "conv3d", "conv_transpose2d", "grid_sample"):
        return 1e-3, 1e-3
    return 2e-4, 2e-4


def grad_case(dev, name, fn, ins):
    import reflop

    if name == "grad_lstm":
        torch.manual_seed(0)
        m = torch.nn.LSTM(8, 16, 2, batch_first=True)
        x = r(3, 7, 8)
        ref_m = m
        rem_m = torch.nn.LSTM(8, 16, 2, batch_first=True)
        rem_m.load_state_dict(m.state_dict())
        rem_m = rem_m.to(dev)
        ref_m(x)[0].square().sum().backward()
        rem_m(x.to(dev))[0].square().sum().backward()
        worst = 0.0
        for (n, p), (_, q_) in zip(ref_m.named_parameters(), rem_m.named_parameters()):
            worst = max(worst, close_to(q_.grad, p.grad, 1e-3, 1e-4, n))
        return worst
    if name == "grad_second_order":
        x = r(10)
        xr = x.clone().requires_grad_(True)
        g1, = torch.autograd.grad((xr.sin() * xr ** 2).sum(), xr, create_graph=True)
        g2, = torch.autograd.grad(g1.sum(), xr)
        xd = x.to(dev).requires_grad_(True)
        h1, = torch.autograd.grad((xd.sin() * xd ** 2).sum(), xd, create_graph=True)
        h2, = torch.autograd.grad(h1.sum(), xd)
        return max(close_to(h1, g1, 1e-4, 1e-5, "d1"), close_to(h2, g2, 1e-4, 1e-5, "d2"))
    ref_in = [x.clone().requires_grad_(True) for x in ins]
    fn(*ref_in).backward()
    rem_in = [x.to(dev).detach().requires_grad_(True) for x in ins]
    loss = fn(*rem_in)
    loss.backward()
    worst = 0.0
    for i, (a, b) in enumerate(zip(rem_in, ref_in)):
        if a.grad is None or b.grad is None:
            check(a.grad is None and b.grad is None, f"grad[{i}] is None on one side only")
            continue
        worst = max(worst, close_to(a.grad, b.grad, 3e-3, 1e-3, f"grad[{i}]"))
    reflop.synchronize()
    return worst


def rng_case(dev, fn):
    import reflop

    reflop.seed(7)
    a = fn(dev).cpu()
    reflop.seed(7)
    b = fn(dev).cpu()
    c = fn(dev).cpu()
    check(torch.equal(a, b), "same seed gave different values")
    check(not torch.equal(b, c) or a.numel() < 100, "consecutive draws are identical")
    torch.manual_seed(7)
    d = fn(dev).cpu()
    check(torch.equal(a, d), "torch.manual_seed does not seed the server generator")
    return f"mean {a.float().mean():.3f} std {a.float().std():.3f}"


def unsupported_paths(dev):
    import reflop

    a = torch.tensor([3, 1, 3, 2, 1]).to(dev)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        u = torch.unique(a)                       # data-dependent: falls back to the client in "auto"
    check(torch.equal(u.cpu() if hasattr(u, "cpu") else u, torch.tensor([1, 2, 3])), f"unique gave {u}")
    msg = "; ".join(str(x.message)[:80] for x in w) or "no warning"
    with reflop.placement("remote"):
        try:
            torch.unique(a)
            strict = "unique ran remotely under 'remote'"
        except reflop.UnsupportedOperatorError as e:
            strict = f"UnsupportedOperatorError: {str(e)[:100]}"
    return f"auto fallback ok ({msg}); remote: {strict}"


def generator_object(dev):
    g = torch.Generator().manual_seed(0)
    try:
        torch.randn(10, generator=g).to(dev)                  # CPU generator on the client is fine
        torch.randn(10, device=dev, generator=g).sum().item()
    except Exception as e:  # noqa: BLE001
        raise Expected(f"explicit torch.Generator on a reflop tensor is documented as unsupported: {type(e).__name__}: {str(e)[:120]}")
    return "explicit generator accepted"


def main():
    import reflop

    s = Suite("a02_ops", __doc__.splitlines()[0])
    targets = [("gpu", lambda: reflop.device(0))]
    targets.append(("cpu", lambda: reflop.device(1, kind="cpu")))
    ops = OPS if not q(False, True) else OPS[::6]
    for tname, mk in targets:
        dev = mk()
        with reflop.placement("remote"):
            for name, ins, fn, *tol in [(o[0], o[1], o[2]) for o in ops]:
                rt, at = op_tol(name)
                s.case(f"{tname}/op/{name}", run_op, dev, fn, ins, rt, at, name)
            for name, fn, ins in GRAD:
                s.case(f"{tname}/autograd/{name}", grad_case, dev, name, fn, ins)
        for name, fn in RNG:
            s.case(f"{tname}/rng/{name}", rng_case, dev, fn)
        s.case(f"{tname}/unsupported_unique_fallback", unsupported_paths, dev)
        s.case(f"{tname}/explicit_generator", generator_object, dev)
        s.case(f"{tname}/session_healthy_after_battery", lambda d=dev: (torch.ones(3, device=d) * 2).sum().item() == 6.0 or check(False))
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
