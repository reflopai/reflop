# SPDX-License-Identifier: Apache-2.0
"""a04 optimizers and reflop.capture: 12 optimizers eager vs CPU; capture with every optimizer (replay + correct, or
an explicit CaptureError); capture rules: same output objects, shape/dtype re-capture, nested args, CPU args moved
inside, mutated args stay eager with a warning, Python scalars refused, tensor LR schedules, branches, RNG,
accumulation, server CPU, one request per replay."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from live import workloads as wl
from live.harness import Expected, Suite, check, q, stats, stats_delta

OPTS = ["sgd", "momentum", "nesterov", "adam", "adamw", "adam_cap", "adamw_cap", "adamw_fused", "rmsprop", "adagrad",
        "adadelta", "adamax"]
N = q(12, 5)


def data(n=N, B=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(B, 784, generator=g), torch.randint(0, 10, (B,), generator=g)) for _ in range(n)]


# capturable/fused variants cannot run on the local CPU; their reference is the plain optimizer
REF_KIND = {"adam_cap": "adam", "adamw_cap": "adamw", "adamw_fused": "adamw"}


def ref_losses(kind, batches, model=None):
    m = model or wl.build("mlp", 0)
    opt = wl.optimizer(m.parameters(), REF_KIND.get(kind, kind))
    step = wl.make_step(m, opt, F.cross_entropy)
    return [step(x, y).item() for x, y in batches], m


def optim_eager(dev, kind):
    import reflop

    b = data()
    ref, mref = ref_losses(kind, b)
    m = wl.build("mlp", 0).to(dev)
    opt = wl.optimizer(m.parameters(), kind)
    step = wl.make_step(m, opt, F.cross_entropy)
    rem = [step(x.to(dev), y.to(dev)).item() for x, y in b]
    worst = max(abs(a - c) / max(abs(c), 1e-9) for a, c in zip(rem, ref))
    st = reflop.fetch(m.state_dict())
    pd = max((st[k] - v).abs().max().item() for k, v in mref.state_dict().items())
    check(worst < 1e-4, f"max rel loss diff {worst:.2e}")
    ptol = 1e-3 if kind in ("rmsprop", "adagrad", "adadelta", "adamax") else 1e-4   # sqrt(v) amplifies rounding
    check(pd < ptol, f"param diff {pd:.2e}")
    return f"max rel loss diff {worst:.1e}, param diff {pd:.1e}"


def optim_capture(dev, kind):
    import reflop

    b = data()
    ref, _ = ref_losses(kind, b)
    m = wl.build("mlp", 0).to(dev)
    opt = wl.optimizer(m.parameters(), kind)
    step = reflop.capture(wl.make_step(m, opt, F.cross_entropy))
    rem = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for x, y in b:
                rem.append(step(x.to(dev), y.to(dev)).item())
    except reflop.CaptureError as e:
        if kind in ("sgd", "momentum", "nesterov", "adam_cap", "adamw_cap", "adamw_fused"):
            raise AssertionError(f"{kind} should be capturable: {e}") from None
        raise Expected(f"explicit CaptureError (no silent freeze): {str(e)[:160]}") from None
    worst = max(abs(a - c) / max(abs(c), 1e-9) for a, c in zip(rem, ref))
    check(worst < 1e-4, f"captured {kind}: max rel loss diff vs CPU {worst:.2e} (replays {step.replays})")
    warn = "; ".join(str(x.message)[:100] for x in w if "reflop" in str(x.message))
    if step.replays == 0:
        raise Expected(f"correct but never replayed ({warn or 'no warning'})")
    return f"replays {step.replays}/{len(b)}, max rel diff {worst:.1e}" + (f"; warn: {warn}" if warn else "")


# ----------------------------------------------------------------------------- capture rules

def _mlp_step(dev, opt_kind="momentum"):
    m = wl.build("mlp", 0).to(dev)
    opt = wl.optimizer(m.parameters(), opt_kind)
    return m, opt, wl.make_step(m, opt, F.cross_entropy)


def same_output_object(dev):
    import reflop

    _, _, f = _mlp_step(dev)
    step = reflop.capture(f)
    outs = [step(x.to(dev), y.to(dev)) for x, y in data(6)]
    vals = [o.item() for o in outs]
    check(step.replays >= 3, f"replays {step.replays}")
    check(outs[3] is outs[4] is outs[5], "replayed outputs are not the same object (CUDA-graph semantics promised)")
    return f"replays {step.replays}; last values {vals[-3:]}"


def shape_dtype_recapture(dev):
    import reflop

    m, opt, f = _mlp_step(dev)
    mref = wl.build("mlp", 0)
    fref = wl.make_step(mref, wl.optimizer(mref.parameters(), "momentum"), F.cross_entropy)
    step = reflop.capture(f)
    seq = data(4, 32) + data(4, 16, 1) + data(4, 32, 2) + [(x.double().float(), y) for x, y in data(2, 8, 3)]
    worst = 0
    for x, y in seq:
        a = step(x.to(dev), y.to(dev)).item()
        r = fref(x, y).item()
        worst = max(worst, abs(a - r) / abs(r))
    check(worst < 1e-4, f"rel diff {worst:.2e} across shape changes")
    return f"replays {step.replays}/{len(seq)}, worst rel {worst:.1e}"


def nested_args(dev):
    import reflop

    m, opt, _ = _mlp_step(dev)

    def f(batch):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(batch["x"][0]), batch["y"]["label"])
        loss.backward()
        opt.step()
        return {"loss": loss, "both": (loss * 2, loss * 3)}

    step = reflop.capture(f)
    for x, y in data(8):
        out = step({"x": [x.to(dev)], "y": {"label": y.to(dev)}})
    check(step.replays >= 4, f"nested args never replayed (replays {step.replays})")
    check(abs(out["both"][1].item() - 3 * out["loss"].item()) < 1e-4, "nested outputs inconsistent")
    return f"replays {step.replays}/8"


def cpu_args_moved_inside(dev):
    import reflop

    m, opt, _ = _mlp_step(dev)
    mref = wl.build("mlp", 0)
    fref = wl.make_step(mref, wl.optimizer(mref.parameters(), "momentum"), F.cross_entropy)

    def f(x, y):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        return loss

    step = reflop.capture(f)
    worst = 0
    for x, y in data(10):
        worst = max(worst, abs(step(x, y).item() - fref(x, y).item()))
    check(step.replays >= 6, f"replays {step.replays}")
    check(worst < 1e-4, f"diff {worst}")
    return f"replays {step.replays}/10, max abs diff {worst:.1e}"


def mutated_arg_stays_eager(dev):
    import reflop

    m, opt, _ = _mlp_step(dev)
    mref = wl.build("mlp", 0)
    fref = wl.make_step(mref, wl.optimizer(mref.parameters(), "momentum"), F.cross_entropy)

    def f(x, y):
        x.sub_(0.5)                          # CPU work on the argument before upload: not replayable
        x = x.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y.to(dev))
        loss.backward()
        opt.step()
        return loss

    step = reflop.capture(f)
    worst = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for x, y in data(8):
            a = step(x.clone(), y).item()
            r = fref(x.clone() - 0.5, y).item()
            worst = max(worst, abs(a - r))
    msgs = [str(x.message) for x in w if "reflop.capture" in str(x.message)]
    check(worst < 1e-4, f"values wrong: {worst}")
    check(step.replays == 0, f"replayed a step that mutates its CPU argument ({step.replays})")
    check(len(msgs) == 1, f"expected exactly one 'runs eagerly' warning, got {len(msgs)}: {msgs[:2]}")
    return f"0 replays, correct, warning: {msgs[0][:140]}"


def python_scalar_refused(dev):
    import reflop

    m, opt, _ = _mlp_step(dev, "sgd")
    lr = [0.1]

    def f(x, y):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(p.grad, alpha=-lr[0])
        lr[0] *= 0.9                          # a Python schedule
        return loss

    step = reflop.capture(f)
    try:
        for x, y in data(6):
            step(x.to(dev), y.to(dev)).item()
    except reflop.CaptureError as e:
        return f"CaptureError: {str(e)[:160]}"
    raise AssertionError(f"Python-scalar schedule did not raise CaptureError (replays {step.replays}): silent freeze?")


def tensor_lr_schedule(dev):
    import reflop

    m = wl.build("mlp", 0).to(dev)
    mref = wl.build("mlp", 0)
    lr = torch.tensor(0.1).to(dev)
    lr_ref = torch.tensor(0.1)

    def mk(model, lrt):
        def f(x, y):
            for p in model.parameters():
                p.grad = None
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p.sub_(lrt * p.grad)
            return loss
        return f

    step = reflop.capture(mk(m, lr))
    fref = mk(mref, lr_ref)
    worst = 0
    for x, y in data(10):
        a = step(x.to(dev), y.to(dev)).item()
        r = fref(x, y).item()
        worst = max(worst, abs(a - r))
        lr.mul_(0.8)
        lr_ref.mul_(0.8)
    check(worst < 1e-4, f"tensor LR schedule diverged: {worst}")
    check(step.replays >= 6, f"replays {step.replays}")
    return f"replays {step.replays}/10, max diff {worst:.1e}"


def scheduler_with_capturable(dev):
    import reflop

    m = wl.build("mlp", 0).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(1e-3).to(dev), capturable=True)
    sched = torch.optim.lr_scheduler.StepLR(opt, 2, 0.5)
    step = reflop.capture(wl.make_step(m, opt, F.cross_entropy))
    try:
        for x, y in data(8):
            step(x.to(dev), y.to(dev)).item()
            sched.step()
    except Exception as e:  # noqa: BLE001
        raise Expected(f"LR scheduler + tensor lr + capturable AdamW: {type(e).__name__}: {str(e)[:160]}") from None
    return f"replays {step.replays}/8, lr now {float(opt.param_groups[0]['lr']):.2e}"


def branch_on_python_state(dev):
    import reflop

    m = nn.Linear(8, 1).to(dev)
    calls = [0]

    def f(x):
        calls[0] += 1
        return (m(x) * (2.0 if calls[0] % 2 else 3.0)).sum()

    step = reflop.capture(f)
    xs = [torch.randn(4, 8) for _ in range(8)]
    try:
        vals = [step(x.to(dev)).item() for x in xs]
    except reflop.CaptureError as e:
        return f"refused: CaptureError {str(e)[:120]}"
    m_cpu = nn.Linear(8, 1)
    m_cpu.load_state_dict(reflop.fetch(m.state_dict()))
    ref = [(m_cpu(x) * (2.0 if (i + 1) % 2 else 3.0)).sum().item() for i, x in enumerate(xs)]
    worst = max(abs(a - b) for a, b in zip(vals, ref))
    check(worst < 1e-4, f"branch frozen by replay: diff {worst} (replays {step.replays})")
    return f"correct, replays {step.replays}"


def rng_in_capture(dev):
    import reflop

    def f(x):
        return F.dropout(x, 0.5, training=True).sum()

    step = reflop.capture(f)
    x = torch.ones(1000).to(dev)
    reflop.seed(3)
    a = [step(x).item() for _ in range(8)]
    reflop.seed(3)
    b = [step(x).item() for _ in range(8)]
    check(len(set(a[3:])) > 1, f"replayed dropout masks are frozen: {a}")
    check(a == b, f"same seed, different replay values: {a} vs {b}")
    return f"replays {step.replays}, values vary and repeat with the seed"


def accumulation_in_capture(dev):
    import reflop

    m, opt, _ = _mlp_step(dev)
    mref = wl.build("mlp", 0)
    oref = wl.optimizer(mref.parameters(), "momentum")

    def mk(model, o):
        def f(x1, y1, x2, y2):
            o.zero_grad(set_to_none=True)
            l1 = F.cross_entropy(model(x1), y1) / 2
            l1.backward()
            l2 = F.cross_entropy(model(x2), y2) / 2
            l2.backward()
            o.step()
            return l1 + l2
        return f

    step = reflop.capture(mk(m, opt))
    fr = mk(mref, oref)
    b = data(16)
    worst = 0
    for i in range(0, 16, 2):
        (x1, y1), (x2, y2) = b[i], b[i + 1]
        worst = max(worst, abs(step(x1.to(dev), y1.to(dev), x2.to(dev), y2.to(dev)).item() - fr(x1, y1, x2, y2).item()))
    check(worst < 1e-4 and step.replays >= 4, f"diff {worst}, replays {step.replays}")
    return f"replays {step.replays}/8, diff {worst:.1e}"


def one_request_per_replay(dev):
    import reflop

    m, opt, f = _mlp_step(dev)
    step = reflop.capture(f)
    x, y = data(1)[0]
    xd, yd = x.to(dev), y.to(dev)
    for _ in range(4):
        step(xd, yd)
    reflop.synchronize()
    idx = dev.index or 0
    s0 = stats(idx)
    for _ in range(20):
        step(xd, yd)
    reflop.synchronize()
    d = stats_delta(s0, stats(idx))
    check(d.get("executes", 0) <= 21, f"{d['executes']} executes for 20 replays")
    check(d.get("registers", 0) == 0, f"replays re-registered graphs: {d}")
    check(d.get("bytes_up", 0) < 4096, f"{d['bytes_up']} bytes up for 20 resident replays")
    return f"20 replays: {d}"


def adam_plain_capture_error(dev):
    import reflop

    _, _, f = _mlp_step(dev, "adam")
    step = reflop.capture(f)
    try:
        for x, y in data(5):
            step(x.to(dev), y.to(dev))
        reflop.synchronize()
    except reflop.CaptureError as e:
        check("capturable" in str(e), f"message does not suggest capturable=True: {e}")
        return str(e)[:200]
    raise AssertionError("plain Adam under capture did not raise CaptureError")


def main():
    import reflop

    s = Suite("a04_capture", __doc__.splitlines()[0])
    gpu = reflop.device(0)
    for k in OPTS:
        s.case(f"gpu/eager/{k}", optim_eager, gpu, k)
    for k in OPTS:
        s.case(f"gpu/capture/{k}", optim_capture, gpu, k)
    for fn in (same_output_object, shape_dtype_recapture, nested_args, cpu_args_moved_inside, mutated_arg_stays_eager,
               python_scalar_refused, tensor_lr_schedule, scheduler_with_capturable, branch_on_python_state,
               rng_in_capture, accumulation_in_capture, one_request_per_replay, adam_plain_capture_error):
        s.case(f"gpu/rule/{fn.__name__}", fn, gpu, timeout=300)
    cpu = reflop.device(1, kind="cpu")
    for k in ("momentum", "adamw_cap"):
        s.case(f"cpu/capture/{k}", optim_capture, cpu, k)
    for fn in (cpu_args_moved_inside, rng_in_capture, one_request_per_replay):
        s.case(f"cpu/rule/{fn.__name__}", fn, cpu, timeout=300)
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
