"""a05 mixed precision: bf16/fp16 autocast policy (matmul/conv low precision, reductions/losses fp32), GradScaler
(incl. inf-skip), fp16/bf16/fp64 storage training, AMP under capture, vs CPU references."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from live import workloads as wl
from live.harness import Expected, Suite, check, q

STEPS = q(20, 5)


def autocast_policy(dev, dtype):
    lin = nn.Linear(16, 8).to(dev)
    x = torch.randn(4, 16).to(dev)
    with torch.autocast("reflop", dtype=dtype):
        y = lin(x)
        s = y.float().softmax(-1)
        l = F.cross_entropy(y, torch.zeros(4, dtype=torch.long).to(dev))
        c = F.conv2d(torch.randn(1, 3, 8, 8).to(dev), torch.randn(4, 3, 3, 3).to(dev))
        n = F.layer_norm(y, (8,))
        r = y.sum()
    got = {"linear": y.dtype, "conv": c.dtype, "loss": l.dtype, "softmax": s.dtype, "layer_norm": n.dtype,
           "sum": r.dtype}
    check(y.dtype == dtype and c.dtype == dtype, f"matmul/conv not in {dtype}: {got}")
    check(l.dtype == torch.float32 and n.dtype == torch.float32, f"loss/norm not fp32: {got}")
    return {k: str(v) for k, v in got.items()}


def amp_train(dev, name, dtype, scaler, capture=False):
    import reflop

    w = wl.WORKLOADS[name]
    m = wl.build(name, 0).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9)
    sc = torch.amp.GradScaler("reflop") if scaler else None

    def step(x, y):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("reflop", dtype=dtype):
            loss = w.loss(m(x), y)
        if sc:
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
        else:
            loss.backward()
            opt.step()
        return loss

    f = reflop.capture(step) if capture else step
    x, y = w.batch(32, 0)
    xd, yd = x.to(dev), y.to(dev)
    losses = []
    try:
        for _ in range(STEPS):
            losses.append(float(f(xd, yd).item()))
    except reflop.CaptureError as e:
        raise Expected(f"AMP{'+GradScaler' if sc else ''} under capture refused: {str(e)[:160]}") from None
    # CPU fp32 reference on the same data: loss must fall comparably
    mr = wl.build(name, 0)
    o2 = torch.optim.SGD(mr.parameters(), lr=0.02, momentum=0.9)
    ref = []
    for _ in range(STEPS):
        o2.zero_grad()
        l = w.loss(mr(x), y)
        l.backward()
        o2.step()
        ref.append(l.item())
    check(all(v == v and abs(v) < 1e6 for v in losses), f"non-finite: {losses}")
    check(abs(losses[0] - ref[0]) / abs(ref[0]) < 3e-2, f"first loss {losses[0]:.4f} vs fp32 {ref[0]:.4f}")
    check(losses[-1] < losses[0], f"no progress: {losses[0]:.3f}->{losses[-1]:.3f}")
    rep = f", replays {f.replays}" if capture else ""
    scale = f", scale {sc.get_scale():.0f}" if sc else ""
    return f"{losses[0]:.3f}->{losses[-1]:.3f} (fp32 cpu {ref[0]:.3f}->{ref[-1]:.3f}){scale}{rep}"


def scaler_skips_inf(dev):
    m = nn.Linear(4, 1).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sc = torch.amp.GradScaler("reflop", init_scale=2.0 ** 16)
    w0 = m.weight.detach().clone().cpu()
    opt.zero_grad()
    with torch.autocast("reflop", dtype=torch.float16):
        loss = (m(torch.full((2, 4), 1e4).to(dev)) * 1e4).sum()     # overflows in fp16
    sc.scale(loss).backward()
    sc.step(opt)
    sc.update()
    check(torch.equal(m.weight.detach().cpu(), w0), "GradScaler applied a step with inf/nan grads")
    check(sc.get_scale() < 2.0 ** 16, f"scale not reduced: {sc.get_scale()}")
    return f"step skipped, scale {sc.get_scale():.0f}"


def storage_dtype(dev, dtype):
    torch.manual_seed(0)
    m = wl.MLP().to(dtype)
    mr = wl.MLP().to(dtype)
    mr.load_state_dict(m.state_dict())
    mr = mr.to(dev)
    x, y = torch.randn(32, 784).to(dtype), torch.randint(0, 10, (32,))
    o1, o2 = torch.optim.SGD(m.parameters(), lr=0.05), torch.optim.SGD(mr.parameters(), lr=0.05)
    worst = 0
    for _ in range(STEPS):
        for mm, oo, xx, yy in ((m, o1, x, y), (mr, o2, x.to(dev), y.to(dev))):
            oo.zero_grad()
            l = F.cross_entropy(mm(xx).float(), yy)
            l.backward()
            oo.step()
            if mm is m:
                a = l.item()
            else:
                worst = max(worst, abs(l.item() - a) / abs(a))
    tol = {torch.float64: 1e-6, torch.float16: 2e-2, torch.bfloat16: 5e-2}[dtype]
    check(worst < tol, f"{dtype}: rel loss diff {worst:.2e} > {tol}")
    check(next(mr.parameters()).dtype == dtype, "dtype changed")
    return f"max rel loss diff {worst:.1e}"


def main():
    import reflop

    s = Suite("a05_precision", __doc__.splitlines()[0])
    for tname, mk in (("gpu", lambda: reflop.device(0)), ("cpu", lambda: reflop.device(1, kind="cpu"))):
        dev = mk()
        for dt in (torch.bfloat16, torch.float16):
            s.case(f"{tname}/autocast_policy/{str(dt)[6:]}", autocast_policy, dev, dt)
        models = ("cnn", "gpt", "resnet18", "lstm") if tname == "gpu" else ("cnn", "gpt")
        for n in models:
            s.case(f"{tname}/amp/{n}/bf16", amp_train, dev, n, torch.bfloat16, False, timeout=600)
            s.case(f"{tname}/amp/{n}/fp16_gradscaler", amp_train, dev, n, torch.float16, True, timeout=600)
        if tname == "gpu":
            s.case("gpu/amp_capture/cnn/bf16", amp_train, dev, "cnn", torch.bfloat16, False, True)
            s.case("gpu/amp_capture/cnn/fp16_gradscaler", amp_train, dev, "cnn", torch.float16, True, True)
            s.case("gpu/amp_capture/gpt/bf16", amp_train, dev, "gpt", torch.bfloat16, False, True)
            s.case("gpu/gradscaler_skips_inf_step", scaler_skips_inf, dev)
        for dt in (torch.float64, torch.float16, torch.bfloat16):
            s.case(f"{tname}/storage/{str(dt)[6:]}", storage_dtype, dev, dt)
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
