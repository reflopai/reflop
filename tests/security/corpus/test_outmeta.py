# SPDX-License-Identifier: Apache-2.0
"""outmeta.py against the kernels (#25 E): for each op it fixes up, the meta
output after fixup() has the dtype and shape of every output that the CPU
and CUDA kernels return. E3 rejects an execute whose consumed outputs differ
from V4's metas, and E1 rejects a handle bound with metas other than the
worker's, so a wrong rule here breaks real programs."""

from __future__ import annotations

import pytest
import torch
from reflop_worker import outmeta

A = torch.ops.aten
f16, f32, f64, i32, i64 = torch.float16, torch.float32, torch.float64, torch.int32, torch.int64


def T(*shape, dt=f32):
    g = torch.Generator().manual_seed(len(shape) * 7 + 1)
    if dt.is_floating_point:
        return torch.randn(shape, generator=g).to(dt)
    return torch.randint(0, 3, shape, generator=g, dtype=dt)


def P(*shape, dt=f32):  # in (0, 1), for BCE
    return T(*shape, dt=dt).sigmoid()


def _ebag(mode, padding_idx=-1, psw=False, ilo=False, it=i64, ot=i64, wdt=f32):
    w = T(10, 5, dt=wdt)
    idx = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=it)
    off = torch.tensor([0, 2, 4, 6] if ilo else [0, 2, 4], dtype=ot)
    return (A._embedding_bag.default, [w, idx, off, False, mode, False, T(8) if psw else None, ilo, padding_idx])


def _bn(op, *rest, x=f32, p=None, rm=f32):
    w = None if p is None else T(3, dt=p)
    return (op, [T(2, 3, 4, dt=x), w, w, T(3, dt=rm), (T(3).abs() + 0.5).to(rm), *rest])


def _bn_bw(x, p, sv, mask, train=True, rm=None):
    rm = p if rm is None else rm
    w = None if p is None else T(3, dt=p)
    r = None if rm is None else T(3, dt=rm)
    return (A.native_batch_norm_backward.default,
            [T(2, 3, 4, dt=x), T(2, 3, 4, dt=x), w, r, None if r is None else r.abs() + 0.5, T(3, dt=sv),
             (T(3).abs() + 0.5).to(sv), train, 1e-5, mask])


def _gn_bw(x, st, p, mask=(True, True, True)):
    return (A.native_group_norm_backward.default,
            [T(2, 4, 3, dt=x), T(2, 4, 3, dt=x), T(2, 2, dt=st), (T(2, 2).abs() + 0.5).to(st),
             None if p is None else T(4, dt=p), 2, 4, 3, 2, list(mask)])


CPU, CUDA = ("cpu",), ("cuda",)


CASES = {
    "huber f32,f64": (A.huber_loss.default, [T(3, 4), T(3, 4, dt=f64), 1, 1.0]),
    "huber f32,f64 none": (A.huber_loss.default, [T(3, 4), T(3, 4, dt=f64), 0, 1.0]),
    "huber f64,f32": (A.huber_loss.default, [T(3, 4, dt=f64), T(3, 4), 1, 1.0]),
    "soft_margin f32,f64": (A.soft_margin_loss.default, [T(3, 4), T(3, 4, dt=f64), 1]),
    "soft_margin f32,f64 none": (A.soft_margin_loss.default, [T(3, 4), T(3, 4, dt=f64), 0]),
    "soft_margin_bw f32,f64": (A.soft_margin_loss_backward.default, [T(), T(3, 4), T(3, 4, dt=f64), 1]),
    "soft_margin_bw grad f64": (A.soft_margin_loss_backward.default, [T(dt=f64), T(3, 4), T(3, 4), 1]),
    "mse_bw f32,f64": (A.mse_loss_backward.default, [T(), T(3, 4), T(3, 4, dt=f64), 1]),
    "mse_bw grad f64": (A.mse_loss_backward.default, [T(dt=f64), T(3, 4), T(3, 4), 1]),
    "mse_bw grad f64 none": (A.mse_loss_backward.default, [T(3, 4, dt=f64), T(3, 4), T(3, 4), 0]),
    "smooth_l1_bw f32,f64": (A.smooth_l1_loss_backward.default, [T(), T(3, 4), T(3, 4, dt=f64), 1, 1.0]),
    "smooth_l1_bw grad f64": (A.smooth_l1_loss_backward.default, [T(dt=f64), T(3, 4), T(3, 4), 1, 1.0]),
    "bce weight f64": (A.binary_cross_entropy.default, [P(3, 4), P(3, 4), T(3, 4, dt=f64), 1]),
    "bce weight f64 none": (A.binary_cross_entropy.default, [P(3, 4), P(3, 4), T(3, 4, dt=f64), 0]),
    "bce_bw weight f64": (A.binary_cross_entropy_backward.default, [T(), P(3, 4), P(3, 4), T(3, 4, dt=f64), 1]),
    # these already match their meta kernels; kept so a PyTorch change shows
    "mse f32,f64": (A.mse_loss.default, [T(3, 4), T(3, 4, dt=f64), 1]),
    "smooth_l1 f32,f64": (A.smooth_l1_loss.default, [T(3, 4), T(3, 4, dt=f64), 1, 1.0]),
    "bce_logits target f64": (A.binary_cross_entropy_with_logits.default, [T(3, 4), T(3, 4, dt=f64), None, None, 1]),
    "bce_logits pos_weight f64": (A.binary_cross_entropy_with_logits.default,
                                  [T(3, 4), T(3, 4), None, T(4, dt=f64), 0]),
    "mish_bw f64,f32": (A.mish_backward.default, [T(3, 4, dt=f64), T(3, 4)]),
    "mish_bw f32,f64": (A.mish_backward.default, [T(3, 4), T(3, 4, dt=f64)]),
    "mish_bw 0-d f64,f32": (A.mish_backward.default, [T(dt=f64), T(3, 4)]),
    "threshold_bw i32,f32": (A.threshold_backward.default, [T(3, 4, dt=i32), T(3, 4), 0.0]),
    "threshold_bw f32,f64": (A.threshold_backward.default, [T(3, 4), T(3, 4, dt=f64), 0.0]),
    "threshold_bw f32,0-d f64": (A.threshold_backward.default, [T(3, 4), T(dt=f64), 0.0]),
    "threshold_bw 0-d f64,f32": (A.threshold_backward.default, [T(dt=f64), T(3, 4), 0.0]),
    "threshold_bw f16,f32": (A.threshold_backward.default, [T(3, 4, dt=f16), T(3, 4), 0.0]),
    "log_sigmoid_forward": (A.log_sigmoid_forward.default, [T(3, 4)]),
    "log_sigmoid_forward f16": (A.log_sigmoid_forward.default, [T(3, 4, dt=f16)]),
    "prelu_bw weight of higher rank": (A._prelu_kernel_backward.default, [T(3, 4), T(3, 4), T(1, 3, 4)]),
    "prelu_bw weight broadcast": (A._prelu_kernel_backward.default, [T(2, 3, 4), T(2, 3, 4), T(3, 1)]),
    "bn eval": _bn(A.native_batch_norm.default, False, 0.1, 1e-5),
    "bn train": _bn(A.native_batch_norm.default, True, 0.1, 1e-5),
    "bn legit eval": _bn(A._native_batch_norm_legit.default, False, 0.1, 1e-5),
    "bn legit_no_training": _bn(A._native_batch_norm_legit_no_training.default, 0.1, 1e-5),
    "bn legit_functional eval": _bn(A._native_batch_norm_legit_functional.default, False, 0.1, 1e-5),
    "bn legit_functional train": _bn(A._native_batch_norm_legit_functional.default, True, 0.1, 1e-5),
    "bn f16 train": _bn(A.native_batch_norm.default, True, 0.1, 1e-5, x=f16, rm=f16),
    "bn f16 eval": _bn(A.native_batch_norm.default, False, 0.1, 1e-5, x=f16, rm=f16),
    "bn f16 f32 params train": _bn(A.native_batch_norm.default, True, 0.1, 1e-5, x=f16, p=f32),
    "bn bf16 eval": _bn(A.native_batch_norm.default, False, 0.1, 1e-5, x=torch.bfloat16, p=torch.bfloat16,
                        rm=torch.bfloat16),
    "bn legit_functional f16 train": _bn(A._native_batch_norm_legit_functional.default, True, 0.1, 1e-5, x=f16,
                                         p=f16, rm=f16),
    "bn legit_functional f64 f32 stats": _bn(A._native_batch_norm_legit_functional.default, True, 0.1, 1e-5,
                                             x=f64, p=f64) + (CUDA,),
    "bn no_stats": (A._native_batch_norm_legit.no_stats, [T(2, 3, 4), None, None, True, 0.1, 1e-5]),
    "bn no_stats f16": (A._native_batch_norm_legit.no_stats, [T(2, 3, 4, dt=f16), None, None, True, 0.1, 1e-5]),
    "bn_bw": _bn_bw(f32, f32, f32, [True, True, True]),
    "bn_bw no grad_input": _bn_bw(f32, f32, f32, [False, True, True]),
    "bn_bw eval no grad_input": _bn_bw(f32, f32, f32, [False, True, True], train=False),
    "bn_bw f16 f32 params": _bn_bw(f16, f32, f32, [True, True, True]),
    "bn_bw f16 f32 params no grad_input": _bn_bw(f16, f32, f32, [False, True, False]),
    # the CPU forward saves float16 statistics, CUDA's float32
    "bn_bw f16": _bn_bw(f16, f16, f16, [True, True, True]) + (CPU,),
    "bn_bw f16 no weight": _bn_bw(f16, None, f32, [False, True, True]) + (CPU,),
    "bn_bw f16 no weight cuda": _bn_bw(f16, None, f32, [True, False, False]) + (CUDA,),
    "log_sigmoid_bw": (A.log_sigmoid_backward.default, [T(3, 4), T(3, 4), T(3, 4)]),
    "log_sigmoid_bw broadcast": (A.log_sigmoid_backward.default, [T(1, 4), T(3, 4), T(3, 4)]),
    # the buffer CUDA's forward returns
    "log_sigmoid_bw empty buffer": (A.log_sigmoid_backward.default, [T(3, 4), T(3, 4), T(0)], CUDA),
    "ln f16": (A.native_layer_norm.default, [T(2, 3, 4, dt=f16), [4], None, None, 1e-5]),
    "ln bf16": (A.native_layer_norm.default, [T(2, 3, 4, dt=torch.bfloat16), [4], T(4, dt=torch.bfloat16),
                                              T(4, dt=torch.bfloat16), 1e-5]),
    "ln f16 f32 params": (A.native_layer_norm.default, [T(2, 3, 4, dt=f16), [4], T(4), T(4), 1e-5], CPU),
    "gn f16": (A.native_group_norm.default, [T(2, 4, 3, dt=f16), None, None, 2, 4, 3, 2, 1e-5]),
    "gn f16 f32 params": (A.native_group_norm.default, [T(2, 4, 3, dt=f16), T(4), T(4), 2, 4, 3, 2, 1e-5], CPU),
    "gn_bw f16": _gn_bw(f16, f16, f16),
    "gn_bw f16 no weight": _gn_bw(f16, f16, None),
    "gn_bw f16 f32 params": _gn_bw(f16, f32, f32) + (CPU,),
    "gn_bw f16 f32 params no grad_input": _gn_bw(f16, f32, f32, (False, True, True)) + (CPU,),
    "pixel_unshuffle empty width": (A.pixel_unshuffle.default, [T(1, 2, 4, 0), 2]),
    "pixel_unshuffle empty batch": (A.pixel_unshuffle.default, [T(0, 2, 4, 4), 2]),
    "pixel_unshuffle": (A.pixel_unshuffle.default, [T(1, 2, 4, 4), 2]),
    "pixel_shuffle empty": (A.pixel_shuffle.default, [T(1, 4, 2, 0), 2]),
    "ebag f64 weight": _ebag(0, wdt=f64),
    "ebag f16 weight": _ebag(0, wdt=f16),
    "softmax_bw f32 grad of half": (A._softmax_backward_data.default, [T(3, 4), T(3, 4), 1, f16]) + (CUDA,),
    "softmax_bw f64 grad of f32": (A._softmax_backward_data.default, [T(3, 4, dt=f64), T(3, 4, dt=f64), 1, f32]) + (CPU,),
    "log_softmax_bw f32 grad of half": (A._log_softmax_backward_data.default, [T(3, 4), T(3, 4), 1, f16]) + (CUDA,),
    "dropout empty": (A.native_dropout.default, [T(0, 3), 0.5, True]),
    "dropout": (A.native_dropout.default, [T(3, 4), 0.5, True]),
    "dropout_bw int grad": (A.native_dropout_backward.default, [T(3, 4, dt=i64), T(3, 4) > 0, 2.0]) + (CPU,),
    "dropout_bw f32": (A.native_dropout_backward.default, [T(3, 4), T(3, 4) > 0, 2.0]),
    "randperm": (A.randperm.default, [5, {"device": "device"}]),
    "randperm int32": (A.randperm.default, [5, {"dtype": i32, "device": "device"}]),
}
for _mode in (0, 1, 2):
    for _pad in (-1, 2):
        for _psw in ((False, True) if _mode == 0 else (False,)):
            for _ilo in (False, True):
                for _it, _ot in ((i64, i64), (i32, i64), (i32, i32)):
                    CASES[f"ebag mode{_mode} pad{_pad} psw{int(_psw)} ilo{int(_ilo)} {_it},{_ot}"] = \
                        _ebag(_mode, _pad, _psw, _ilo, _it, _ot)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


def _flat(out):
    return [out] if isinstance(out, torch.Tensor) else [x for x in out if isinstance(x, torch.Tensor)]


def _on(v, dev):
    if isinstance(v, torch.Tensor):
        return v.to(dev)
    if isinstance(v, dict):  # keyword-only arguments; "device" is the device under test
        return {k: dev if x == "device" else x for k, x in v.items()}
    return v


def _split(args):
    return (args[:-1], args[-1]) if args and isinstance(args[-1], dict) else (args, {})


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("case", sorted(CASES))
def test_fixup_matches_kernel(case, dev):
    op, args, *devs = CASES[case]
    if devs and dev not in devs[0]:
        pytest.skip(f"{dev} kernels refuse these arguments")
    pos, kw = _split([_on(v, dev) for v in args])
    want = [(t.dtype, tuple(t.shape)) for t in _flat(op(*pos, **kw))]
    pos, kw = _split([_on(v, "meta") for v in args])
    a = {x.name: None for x in op._schema.arguments} | dict(zip((x.name for x in op._schema.arguments), pos)) | kw
    got = _flat(outmeta.fixup(f"aten.{op.__name__}", op(*pos, **kw), a, dev == "cuda"))
    assert all(t.device.type == "meta" for t in got)
    assert [(t.dtype, tuple(t.shape)) for t in got] == want


def test_every_fixup_has_a_case():
    names = {f"aten.{op.__name__}" for op, *_ in CASES.values()}
    assert set(outmeta.FIXUPS) <= names
