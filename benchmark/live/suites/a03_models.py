"""a03 model zoo correctness: 10 workloads trained from one init on the local CPU (reference), the server GPU and
the server CPU; drift judged against a 1e-6 weight-perturbation baseline. Plus training-feature compatibility:
activation checkpointing, clipping, hooks, accumulation, custom autograd, eval mode, DataLoader workers,
torch.compile / TorchScript / torch.save boundaries."""

from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from live import workloads as wl
from live.harness import Expected, Suite, check, close_to, q

STEPS = q(15, 4)
BATCH = 16


def train(model, opt, loss_fn, batches, dev=None):
    losses = []
    model.train()
    for x, y in batches:
        if dev is not None:
            x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def compare(name, dev, opt_kind="momentum"):
    import reflop

    w = wl.WORKLOADS[name]
    ref = wl.build(name, 0)
    init = copy.deepcopy(ref)
    rem = copy.deepcopy(ref).to(dev)
    batches = [w.batch(BATCH, 100 + i) for i in range(STEPS)]
    t0 = time.time()
    ref_l = train(ref, wl.optimizer(ref.parameters(), opt_kind), w.loss, batches)
    t_ref = time.time() - t0
    t0 = time.time()
    rem_l = train(rem, wl.optimizer(rem.parameters(), opt_kind), w.loss, batches, dev)
    t_rem = time.time() - t0
    torch.manual_seed(1)
    pert = copy.deepcopy(init)
    with torch.no_grad():
        for p in pert.parameters():
            p.add_(torch.randn_like(p) * 1e-6)
    pert_l = train(pert, wl.optimizer(pert.parameters(), opt_kind), w.loss, batches)
    chaos = max(abs(a - b) for a, b in zip(ref_l, pert_l))
    diffs = [abs(a - b) for a, b in zip(ref_l, rem_l)]
    first_rel = diffs[0] / max(abs(ref_l[0]), 1e-12)
    remote_state = reflop.fetch(rem.state_dict())
    pdiff = max(((v.double() - remote_state[k].double()).abs().max().item() for k, v in ref.state_dict().items()
                 if v.is_floating_point()), default=0.0)
    data = {"ref_losses": ref_l, "remote_losses": rem_l, "first_step_rel": first_rel, "max_abs": max(diffs),
            "chaos": chaos, "param_max_abs": pdiff, "ref_s": round(t_ref, 2), "remote_s": round(t_rem, 2)}
    data["summary"] = (f"first-step rel {first_rel:.1e}, max loss diff {max(diffs):.1e} (chaos {chaos:.1e}), "
                       f"loss {ref_l[0]:.3f}->{ref_l[-1]:.3f} vs {rem_l[-1]:.3f}, {t_rem:.1f}s remote")
    check(all(x == x and abs(x) != float("inf") for x in rem_l), "non-finite remote loss")
    tol = 5e-3 if name in ("cnn", "resnet18", "unet", "autoenc") else 1e-4   # TF32 convolutions on the GPU
    check(first_rel < tol, f"first-step loss differs by {first_rel:.2e} (> {tol}): " + data["summary"])
    check(max(diffs) <= max(20 * chaos, 1e-3 * abs(ref_l[0]), 1e-4),
          "trajectory drifts more than a 1e-6 perturbation does: " + data["summary"])
    return data


# ----------------------------------------------------------------------------- training features

def activation_checkpointing(dev):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
    x = torch.randn(8, 32)
    ref = checkpoint(m, x.requires_grad_(), use_reentrant=False).square().sum()
    ref.backward()
    g_ref = [p.grad.clone() for p in m.parameters()]
    mr = copy.deepcopy(m).to(dev)
    for p in mr.parameters():
        p.grad = None
    out = checkpoint(mr, x.detach().to(dev).requires_grad_(), use_reentrant=False).square().sum()
    out.backward()
    return max(close_to(p.grad, g, 1e-4, 1e-5) for p, g in zip(mr.parameters(), g_ref))


def clip_hooks_accumulation(dev):
    torch.manual_seed(0)
    m = nn.Linear(16, 4)
    mr = copy.deepcopy(m).to(dev)
    seen = []
    hr = mr.weight.register_hook(lambda g: seen.append(g.shape) or g * 0.5)
    hc = m.weight.register_hook(lambda g: g * 0.5)
    xs = [torch.randn(8, 16) for _ in range(3)]
    for x in xs:                                # gradient accumulation over 3 micro-batches
        m(x).pow(2).mean().backward()
        mr(x.to(dev)).pow(2).mean().backward()
    n1 = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1)
    n2 = torch.nn.utils.clip_grad_norm_(mr.parameters(), 0.1)
    torch.nn.utils.clip_grad_value_(m.parameters(), 0.01)
    torch.nn.utils.clip_grad_value_(mr.parameters(), 0.01)
    hr.remove(), hc.remove()
    check(len(seen) == 3, f"hook ran {len(seen)} times")
    return max(close_to(n2, n1, 1e-5, 1e-6, "norm"), close_to(mr.weight.grad, m.weight.grad, 1e-5, 1e-6, "grad"))


class Cube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 3

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return 3 * x ** 2 * g


def custom_autograd(dev):
    x = torch.randn(10)
    xr = x.to(dev).requires_grad_()
    Cube.apply(xr).sum().backward()
    return close_to(xr.grad, 3 * x ** 2, 1e-5, 1e-6)


def eval_mode_dropout_bn(dev):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Dropout(0.5), nn.Linear(16, 2))
    mr = copy.deepcopy(m).to(dev)
    x = torch.randn(32, 8)
    m.train(), mr.train()
    keep = (m(x), mr(x.to(dev)))                # updates running stats (outputs kept alive)
    m.eval(), mr.eval()
    with torch.no_grad():
        a, b = m(x), mr(x.to(dev))
    close_to(mr[1].running_mean, m[1].running_mean, 1e-5, 1e-6, "running_mean")
    check(int(mr[1].num_batches_tracked.item()) == 1, "num_batches_tracked")
    return close_to(b, a, 1e-4, 1e-5, "eval outputs")


def bn_stats_output_discarded(dev):
    """A forward whose output is dropped at once (BN recalibration, SWA bn_update) must still update running stats."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16))
    mr = copy.deepcopy(m).to(dev)
    with torch.no_grad():
        for i in range(3):
            x = torch.randn(32, 8, generator=torch.Generator().manual_seed(i))
            m(x)
            mr(x.to(dev))                           # output discarded immediately
    nbt = int(mr[1].num_batches_tracked.item())
    d_mean = close_to(mr[1].running_mean, m[1].running_mean, 1e-4, 1e-5, f"running_mean (nbt={nbt}, "
                      f"remote mean |{mr[1].running_mean.cpu().abs().max():.3g}|)")
    return max(d_mean, close_to(mr[1].running_var, m[1].running_var, 1e-4, 1e-5, "running_var"))


def module_roundtrip(dev):
    torch.manual_seed(0)
    m = wl.build("cnn", 0)
    sd = copy.deepcopy(m.state_dict())
    m.to(dev)
    m.to("cpu")
    for k, v in m.state_dict().items():
        close_to(v, sd[k], 0, 0, k)
    m2 = wl.build("cnn", 5).to(dev)
    m2.load_state_dict(sd)                       # CPU state dict into a remote model
    x = torch.randn(2, 3, 32, 32)
    m.eval(), m2.eval()
    with torch.no_grad():
        return close_to(m2(x.to(dev)), m(x), 5e-3, 5e-3, "outputs")


def dataloader_workers(dev):
    ds = torch.utils.data.TensorDataset(torch.randn(256, 784), torch.randint(0, 10, (256,)))
    out = {}
    for nw in (0, 2):
        torch.manual_seed(0)
        m = wl.MLP().to(dev)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=nw,
                                         generator=torch.Generator().manual_seed(0),
                                         multiprocessing_context="spawn" if nw else None, persistent_workers=False)
        for x, y in dl:
            opt.zero_grad()
            loss = F.cross_entropy(m(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
        out[nw] = loss.item()
    check(abs(out[0] - out[2]) < 1e-4, f"num_workers=0 vs 2 losses differ: {out}")
    return f"final loss {out}"


def torch_compile_boundary(dev):
    m = nn.Linear(4, 4).to(dev)
    try:
        cm = torch.compile(m)
        cm(torch.randn(2, 4).to(dev)).sum().item()
    except Exception as e:  # noqa: BLE001
        raise Expected(f"torch.compile is documented as not working: {type(e).__name__}: {str(e)[:150]}")
    return "torch.compile worked (documented as unsupported)"


def torch_save_refuses(dev):
    import io

    t = torch.randn(3).to(dev)
    try:
        torch.save(t, io.BytesIO())
    except Exception as e:  # noqa: BLE001
        return f"refused as documented: {type(e).__name__}: {str(e)[:120]}"
    raise AssertionError("torch.save of a reflop tensor did not raise (docs: refuses to be pickled)")


def torchscript_boundary(dev):
    m = nn.Linear(4, 4)
    try:
        sm = torch.jit.script(m).to(dev)
        sm(torch.randn(2, 4).to(dev)).sum().item()
    except Exception as e:  # noqa: BLE001
        raise Expected(f"TorchScript on reflop: {type(e).__name__}: {str(e)[:150]}")
    return "TorchScript module ran on reflop"


def main():
    import reflop

    s = Suite("a03_models", __doc__.splitlines()[0])
    for tname, mk in (("gpu", lambda: reflop.device(0)), ("cpu", lambda: reflop.device(1, kind="cpu"))):
        dev = mk()
        names = wl.ALL if tname == "gpu" else ["mlp", "cnn", "gpt", "lstm", "gru", "attn", "lora", "autoenc", "unet"]
        for n in names:
            s.case(f"{tname}/train/{n}", compare, n, dev, timeout=900)
        for fn in (activation_checkpointing, clip_hooks_accumulation, custom_autograd, eval_mode_dropout_bn,
                   bn_stats_output_discarded,
                   module_roundtrip, torch_compile_boundary, torch_save_refuses, torchscript_boundary):
            s.case(f"{tname}/feature/{fn.__name__}", fn, dev, timeout=600)
        if tname == "gpu":
            s.case("gpu/feature/dataloader_workers", dataloader_workers, dev, timeout=600)
    s.case("optim/adam_plain_eager", compare, "mlp", reflop.device(0), "adam")
    s.case("optim/adamw_eager", compare, "gpt", reflop.device(0), "adamw")
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
