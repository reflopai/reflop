# SPDX-License-Identifier: Apache-2.0
"""Child training program for the Owner-control, checkpoint, quota and concurrency suites.

Prints JSON lines (flushed): {"sid"}, {"step","t","loss"}, {"event"}, {"lost"}, {"error"}, ...

  --mode train   deterministic MLP training with server-RNG dropout; .item() every step (for gap detection)
  --mode idle    open a session and wait
  --mode flood   warm up, wait for --go file, send --flood step_boundary() requests without waiting, then sync
  --mode ckpt    checkpoint=True training; after a checkpoint prints ready_to_kill; after SessionLostError calls
                 reflop.recover() and verifies values, RNG and late tensors
  --mode alloc   allocate --alloc-mib on the device and hold it for --seconds
  --mode bench   capture-trained CNN for --seconds, report steps/s (concurrency)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import reflop  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from reflop._pb import compute_pb2 as cpb  # noqa: E402


def out(**kw):
    kw.setdefault("t", time.time())
    print(json.dumps(kw, default=str), flush=True)


def hook_events(idx):
    from reflop import session as ss

    conn = ss._sessions[idx].conn
    orig = conn.on_event

    def rec(ev):
        out(event=cpb.Event.Kind.Name(ev.kind), text=ev.text[:300])
        orig(ev)

    conn.on_event = rec


def open_dev(a, checkpoint=False):
    idx = 1 if a.kind == "cpu" else 0
    if checkpoint:
        d = reflop.device(idx, kind=a.kind, checkpoint=True,
                          checkpoint_interval=a.interval if a.interval else None)
    else:
        d = reflop.device(idx, kind=a.kind)
    from reflop import session as ss

    hook_events(idx)
    out(sid=bytes(ss._sessions[idx].conn.session_id).hex(), idx=idx, pid=os.getpid())
    return d, idx


def small_model(d_in, h):
    return nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))


def batches(n, B, d_in, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(B, d_in, generator=g), torch.randint(0, 10, (B,), generator=g)) for _ in range(n)]


def mode_train(a):
    d, idx = open_dev(a)
    torch.manual_seed(a.seed)
    m = small_model(784, 512).to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9)
    data = [(x.to(d), y.to(d)) for x, y in batches(8, 64, 784)]
    ballast = torch.ones(a.state_mib * 2**18, device=d) if a.state_mib else None
    if ballast is not None:
        ballast.sum().item()
        out(ballast_mib=a.state_mib)
    reflop.seed(a.seed)
    for i in range(a.steps):
        x, y = data[i % len(data)]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        v = loss.item()
        out(step=i, loss=v)
        if a.sleep:
            time.sleep(a.sleep)
    if ballast is not None:
        out(ballast_ok=ballast[:: 2**16].sum().item() == len(range(0, ballast.numel(), 2**16)))
    out(done=True)


def mode_idle(a):
    open_dev(a)
    t = time.time()
    while time.time() - t < a.seconds:
        time.sleep(0.2)
    torch.ones(1).to(reflop.device(1 if a.kind == "cpu" else 0)).item()
    out(done=True)


def wait_file(p, timeout=600):
    t = time.time()
    while not os.path.exists(p):
        if time.time() - t > timeout:
            raise TimeoutError(f"{p} never appeared")
        time.sleep(0.1)


def mode_flood(a):
    d, idx = open_dev(a)
    acc = torch.zeros(1).to(d)
    for _ in range(3):
        acc.add_(1)
        reflop.step_boundary()
    reflop.synchronize()
    base = acc.item()
    out(ready=True, base=base)
    wait_file(a.go)
    t = time.time()
    sent, send_err = 0, None
    try:
        for _ in range(a.flood):
            acc.add_(1)
            reflop.step_boundary()
            sent += 1
    except reflop.ReflopError as e:          # an earlier refusal can surface while still sending
        send_err = f"{type(e).__name__}: {e}"
    out(sent=sent, send_error=send_err, send_s=time.time() - t)
    err = None
    try:
        reflop.synchronize()
    except reflop.ReflopError as e:
        err = f"{type(e).__name__}: {e}"
    out(sync_error=err, sync_t=time.time())
    errs = []
    v = None
    for _ in range(3):              # several async errors may be queued; each read raises the next one
        try:
            v = acc.item()
            break
        except reflop.ReflopError as e:
            errs.append(str(e)[:200])
    out(value=v, accepted=None if v is None else v - base, later_errors=errs)
    healthy = (torch.ones(4).to(d) * 2).sum().item() == 8.0
    out(healthy_after=healthy, done=True)


def params_vec(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).cpu()


def mode_ckpt(a):
    d, idx = open_dev(a, checkpoint=True)
    from reflop import session as ss

    sess = ss._sessions[idx]
    torch.manual_seed(a.seed)
    m = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10)).to(d)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, capturable=True)
    data = [(x.to(d), y.to(d)) for x, y in batches(8, 32, 32)]
    ballast = torch.full((a.state_mib * 2**18,), 1.0, device=d) if a.state_mib else None
    reflop.seed(a.seed)
    hist, losses, times = {}, {}, []
    step = 0
    saved_seen, saved_step, kill_announced, late = 0, None, False, None
    t_start = time.time()

    def do_step(i):
        x, y = data[i % len(data)]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        if ballast is not None:
            ballast.mul_(1.0)
        return loss.item()

    while True:
        try:
            t0 = time.time()
            losses[step] = do_step(step)
            step += 1
            reflop.checkpoint_tag({"step": step})
            reflop.step_boundary()
            hist[step] = params_vec(m)
            if len(hist) > 400:
                hist.pop(min(hist))
            times.append(time.time() - t0)
            ck = sess.checkpoints
            if ck.saved > saved_seen:
                saved_seen = ck.saved
                saved_step = step
                out(ckpt_saved=ck.saved, generation=ck.generation, at_step=step, after_s=time.time() - t_start)
            if saved_step is not None and step >= saved_step + 5 and not kill_announced:
                late = torch.full((3,), 7.0, device=d)       # created after the checkpoint
                late.sum().item()
                out(ready_to_kill=True, at_step=step, saved_step=saved_step)
                kill_announced = True
            if step % 50 == 0:
                out(step=step, loss=losses[step - 1], step_ms_median=sorted(times[-50:])[25] * 1000)
            if time.time() - t_start > a.seconds:
                out(error="timeout waiting for a checkpoint or the kill", steps=step)
                return 5
            time.sleep(a.sleep if not kill_announced else max(a.sleep, 0.3))
        except reflop.SessionLostError as e:
            out(lost=str(e)[:200], at_step=step)
            break
    big = sorted(times)
    out(step_time_ms={"median": big[len(big) // 2] * 1000, "p99": big[int(len(big) * 0.99)] * 1000,
                      "max": big[-1] * 1000, "n": len(big)})
    t0 = time.time()
    try:
        tag = reflop.recover(idx)
    except reflop.ReflopError as e:
        out(recover_refused=str(e)[:300], recover_s=time.time() - t0)
        return 0 if a.expect == "refused" else 6
    out(recovered=tag, recover_s=time.time() - t0)
    rs = tag["step"]
    match = rs in hist and torch.equal(params_vec(m), hist[rs])
    out(params_match=match, restored_step=rs, have_hist=rs in hist)
    try:
        late.sum().item()
        out(late_tensor="usable (should raise SessionLostError)")
    except reflop.SessionLostError as e:
        out(late_tensor="SessionLostError", msg=str(e)[:150])
    if ballast is not None:
        out(ballast_ok=ballast[:: 2**16].sum().item() == len(range(0, ballast.numel(), 2**16)))
    rerun = do_step(rs)
    out(rerun_loss=rerun, original_loss=losses.get(rs), rng_repeats=rerun == losses.get(rs))
    for i in range(rs + 1, rs + 6):
        do_step(i)
    reflop.step_boundary()
    out(continued=True, done=True)
    return 0 if a.expect == "recover" else 7


def mode_alloc(a):
    d, idx = open_dev(a)
    try:
        t = torch.empty(a.alloc_mib * 2**18, device=d)
        t.fill_(1.0)
        reflop.synchronize()
        out(allocated_mib=a.alloc_mib)
    except Exception as e:  # noqa: BLE001
        out(alloc_error=f"{type(e).__name__}: {str(e)[:300]}", is_oom=isinstance(e, torch.OutOfMemoryError))
        healthy = (torch.ones(4).to(d) * 2).sum().item() == 8.0
        out(healthy_after=healthy, done=True)
        return
    end = time.time() + a.seconds
    while time.time() < end:
        time.sleep(0.5)
    out(still_ok=t[:: 2**20].sum().item() == len(range(0, t.numel(), 2**20)), done=True)


def mode_bench(a):
    d, idx = open_dev(a)
    from live import workloads as wl

    m = wl.build("cnn", 0).to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    step = reflop.capture(wl.make_step(m, opt, F.cross_entropy))
    x, y = wl.WORKLOADS["cnn"].batch(a.batch, 0)
    x, y = x.to(d), y.to(d)
    for _ in range(5):
        step(x, y)
    step(x, y).item()
    out(ready=True)
    if a.go:
        wait_file(a.go)
    lat = []
    n = 0
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        t = time.time()
        for _ in range(5):
            l = step(x, y)
        l.item()
        lat.append((time.time() - t) / 5)
        n += 5
    dt = time.time() - t0
    lat.sort()
    out(steps=n, seconds=dt, steps_per_s=n / dt, samples_per_s=n * a.batch / dt, ms_median=lat[len(lat) // 2] * 1000,
        ms_p90=lat[int(len(lat) * 0.9)] * 1000, done=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train")
    p.add_argument("--kind", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-mib", type=int, default=0)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--flood", type=int, default=5000)
    p.add_argument("--go", default=None)
    p.add_argument("--interval", type=float, default=0)
    p.add_argument("--expect", default="recover")
    p.add_argument("--alloc-mib", type=int, default=1024)
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    try:
        rc = {"train": mode_train, "idle": mode_idle, "flood": mode_flood, "ckpt": mode_ckpt, "alloc": mode_alloc,
              "bench": mode_bench}[a.mode](a)
    except reflop.SessionLostError as e:
        out(lost=str(e)[:300])
        return 3
    except Exception as e:  # noqa: BLE001
        out(error=f"{type(e).__name__}: {str(e)[:400]}", tb=traceback.format_exc()[-1500:])
        return 4
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
