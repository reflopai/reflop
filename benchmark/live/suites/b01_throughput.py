"""b01 training throughput matrix on the reflop GPU (and a server-CPU subset): 10 workloads x batch {8,32,128} x
eager/capture x resident/upload inputs; ms/step, samples/s, bytes up/down and graph registrations per step.
The native-CUDA counterpart (b02) runs the same workloads.py on the Owner."""

from __future__ import annotations

import time

import torch

from live import workloads as wl
from live.harness import Suite, q, stats, stats_delta

BATCHES = q((8, 32, 128), (32,))
WARMUP, STEPS, REPEATS = q(5, 3), q(20, 4), q(3, 1)


def bench(s, dev, idx, name, B, mode, policy, kind):
    import reflop

    w = wl.WORKLOADS[name]
    m = wl.build(name, 0).to(dev)
    opt = wl.optimizer(m.parameters(), "momentum")
    f = wl.make_step(m, opt, w.loss)
    if mode == "capture":
        f = reflop.capture(f)
    else:
        inner = f

        def f(x, y):
            loss = inner(x, y)
            reflop.step_boundary()
            return loss
    xc, yc = w.batch(B, 0)
    x, y = xc.to(dev), yc.to(dev)
    s0 = None

    def sync(loss):
        loss.item()

    # warm-up + timed windows
    r = wl.time_training(f, x, y, device=dev, upload=policy == "upload", warmup=WARMUP, steps=1, repeats=1, sync=sync,
                         x_cpu=xc, y_cpu=yc)
    s0 = stats(idx)
    r = wl.time_training(f, x, y, device=dev, upload=policy == "upload", warmup=0, steps=STEPS, repeats=REPEATS,
                         sync=sync, x_cpu=xc, y_cpu=yc)
    d = stats_delta(s0, stats(idx))
    n = STEPS * REPEATS
    row = dict(device=kind, workload=name, batch=B, mode=mode, policy=policy, ms_per_step=r["ms_per_step"],
               ms_reps=r["ms_reps"], samples_per_s=B * 1000 / r["ms_per_step"], bytes_up_per_step=d["bytes_up"] / n,
               bytes_down_per_step=d["bytes_down"] / n, registers_per_step=d["registers"] / n,
               executes_per_step=d["executes"] / n, replays=getattr(f, "replays", None), final_loss=r["final_loss"])
    s.metric("throughput", **row)
    del m, opt, f, x, y
    return f"{r['ms_per_step']:.2f} ms/step, {row['samples_per_s']:.0f} samples/s, {row['bytes_up_per_step'] / 1024:.1f} KiB up/step"


def main():
    import reflop

    s = Suite("b01_throughput", __doc__.splitlines()[0])
    dev = reflop.device(0)
    for name in wl.ALL:
        for B in BATCHES:
            for mode in ("eager", "capture"):
                for policy in ("resident", "upload"):
                    s.case(f"gpu/{name}/b{B}/{mode}/{policy}", bench, s, dev, 0, name, B, mode, policy, "cuda",
                           timeout=900)
            reflop.synchronize()
    cpu = reflop.device(1, kind="cpu")
    for name in q(("mlp", "cnn", "gpt", "lstm", "lora"), ("mlp",)):
        for mode in ("eager", "capture"):
            s.case(f"cpu/{name}/b32/{mode}/resident", bench, s, cpu, 1, name, 32, mode, "resident", "cpu", timeout=900)
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
