# SPDX-License-Identifier: Apache-2.0
"""a06 sessions, placement and errors: GPU + server-CPU sessions, t.device, moves (also differentiable), mixing
errors, module.to, close/reopen, threads, seeds; placement auto/remote/local/remote-cpu; when work is sent (flush at
20k ops, step_boundary, nonzero round trips), stats; async and exit-time errors, OOM (oversized and real fill) as
torch.OutOfMemoryError, bad indices, session health after every error, handle churn."""

from __future__ import annotations

import gc
import textwrap
import threading
import time

import torch
import torch.nn as nn

from live.harness import (
    SCRATCH,
    Suite,
    check,
    close_to,
    py,
    q,
    rss_mib,
    stats,
    stats_delta,
    tensor_session_index,
)


def gpu():
    import reflop

    return reflop.device(0)


def cpu():
    import reflop

    return reflop.device(1, kind="cpu")


def healthy(dev):
    v = (torch.arange(5, device=dev).float() * 2).sum().item()
    check(v == 20.0, f"session unhealthy: {v}")


# ----------------------------------------------------------------------------- sessions

def device_reports_session():
    g, c = gpu(), cpu()
    a, b = torch.ones(3).to(g), torch.ones(3).to(c)
    check(str(a.device) == "reflop:0" and str(b.device) == "reflop:1", f"{a.device} / {b.device}")
    z = torch.zeros(2, device=b.device)
    r = torch.arange(4, device=b.device)
    e = torch.empty_like(b)
    n = b.new_ones(2)
    for t, nm in ((z, "zeros"), (r, "arange"), (e, "empty_like"), (n, "new_ones")):
        check(tensor_session_index(t) == 1, f"{nm}(device=cpu-session tensor.device) went to session "
                                            f"{tensor_session_index(t)}")
    return f"{a.device} / {b.device}; factories follow the session"


def moves_between_sessions():
    g, c = gpu(), cpu()
    x = torch.randn(64, 64)
    a = x.to(c)
    b = (a @ a).to(g)
    d = b.to(a)                                   # to(other tensor)
    e = b.to(a.device)
    check(tensor_session_index(b) == 0 and tensor_session_index(d) == 1 and tensor_session_index(e) == 1, "moves")
    close_to(d, x @ x, 1e-4, 1e-4, "cpu->gpu->cpu")
    return "values preserved across cpu->gpu->cpu"


def differentiable_move():
    g, c = gpu(), cpu()
    w = torch.randn(8, 8)
    wc = w.clone().to(c).requires_grad_()
    out = (wc.to(g) * 3).sum()
    out.backward()
    close_to(wc.grad, torch.full((8, 8), 3.0), 0, 0, "grad through move")
    return "grad flows across sessions"


def mixing_raises():
    g, c = gpu(), cpu()
    try:
        (torch.ones(3).to(g) + torch.ones(3).to(c)).sum().item()
    except Exception as e:  # noqa: BLE001
        check("session" in str(e) and ".to(" in str(e), f"unhelpful message: {e}")
        return f"{type(e).__name__}: {str(e)[:150]}"
    raise AssertionError("mixing tensors from two sessions did not raise")


def module_to_and_data_assign():
    g, c = gpu(), cpu()
    m = nn.Linear(4, 4)
    ref = m(torch.ones(1, 4))
    m.to(c)
    check(all(tensor_session_index(p) == 1 for p in m.parameters()), "module.to(cpu session)")
    m.to(g)
    check(all(tensor_session_index(p) == 0 for p in m.parameters()), "module.to(gpu session)")
    close_to(m(torch.ones(1, 4).to(g)), ref, 1e-5, 1e-6)
    p = m.weight
    p.data = torch.zeros(4, 4).to(g)
    check(m.weight.abs().sum().item() == 0, "param.data = assignment ignored")
    return "module.to across sessions and param.data = work"


def cpu_session_training_with_gpu_open():
    g, c = gpu(), cpu()
    from live import workloads as wl

    for name in ("gpt", "lstm"):
        m = wl.build(name, 0).to(c)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x, y = wl.WORKLOADS[name].batch(8, 0)
        for _ in range(3):
            opt.zero_grad()
            l = wl.WORKLOADS[name].loss(m(x.to(c)), y.to(c))
            l.backward()
            opt.step()
        check(all(tensor_session_index(p) == 1 for p in m.parameters()), f"{name} params left the CPU session")
    _ = torch.ones(1).to(g).item()
    return "GPT (causal SDPA) and LSTM train on the server CPU while a GPU session is open"


def close_and_reopen():
    import reflop

    d = gpu()
    t = torch.ones(3).to(d)
    t0 = time.perf_counter()
    reflop.close()
    t_close = time.perf_counter() - t0
    try:
        t.sum().item()
        stale = "stale tensor still usable"
    except Exception as e:  # noqa: BLE001
        stale = f"{type(e).__name__}: {str(e)[:100]}"
    t0 = time.perf_counter()
    d2 = gpu()
    healthy(d2)
    t_open = time.perf_counter() - t0
    return {"summary": f"close {t_close * 1000:.0f} ms, reopen+first op {t_open * 1000:.0f} ms; stale tensor: {stale}",
            "close_s": t_close, "open_s": t_open}


def open_close_latency(s):
    import reflop

    for kind in ("cuda", "cpu"):
        ts_open, ts_close = [], []
        for _ in range(q(5, 2)):
            reflop.close()
            t0 = time.perf_counter()
            d = reflop.device(0, kind=kind)
            torch.ones(1).to(d).item()
            ts_open.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            reflop.close()
            ts_close.append(time.perf_counter() - t0)
        s.metric("session_latency", kind=kind, open_first_op_ms=1000 * sorted(ts_open)[len(ts_open) // 2],
                 close_ms=1000 * sorted(ts_close)[len(ts_close) // 2], samples=len(ts_open))
    return "see metrics"


def threads_share_session():
    d = gpu()
    errs, res = [], {}

    def work(i):
        try:
            x = torch.full((128, 128), float(i)).to(d)
            for _ in range(20):
                x = x * 1.0 + 0.0
            res[i] = (x @ torch.eye(128).to(d)).mean().item()
        except Exception as e:  # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}")

    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    check(not errs, f"thread errors: {errs[:2]}")
    check(all(abs(res[i] - i) < 1e-4 for i in range(8)), f"wrong results: {res}")
    return "8 threads, correct results"


def seed_across_sessions():
    import reflop

    reflop.close()
    d = gpu()
    reflop.seed(11)
    a = torch.randn(100, device=d).cpu()
    reflop.close()
    d = gpu()
    reflop.seed(11)
    b = torch.randn(100, device=d).cpu()
    check(torch.equal(a, b), "same seed in a new session gives different numbers")
    return "deterministic across sessions (one session open)"


def seed_with_two_sessions():
    """reflop.seed(n) / torch.manual_seed(n) with a GPU and a server-CPU session open: both should be reproducible."""
    import reflop

    reflop.close()
    g, c = gpu(), cpu()
    res = {}
    for name, seeder in (("torch.manual_seed", torch.manual_seed), ("reflop.seed", reflop.seed)):
        seeder(5)
        a, ac = torch.randn(50, device=g).cpu(), torch.randn(50, device=c).cpu()
        seeder(5)
        b, bc = torch.randn(50, device=g).cpu(), torch.randn(50, device=c).cpu()
        res[name] = {"gpu": torch.equal(a, b), "cpu": torch.equal(ac, bc)}
    check(all(v["gpu"] and v["cpu"] for v in res.values()), f"not reproducible with two sessions: {res}")
    return f"{res}"


# ----------------------------------------------------------------------------- placement

def placement_modes():
    import reflop

    d = gpu()
    x = torch.randn(32, 32).to(d)
    with reflop.placement("local"):
        y = x @ x
    check(not hasattr(y, "_rf_storage"), f"'local' result is still remote: {type(y)}")
    close_to(y, x.cpu() @ x.cpu(), 1e-5, 1e-5)
    with reflop.placement("remote-cpu"):
        z = torch.randn(4, 4).to(reflop.device())
        n = torch.zeros(3, device=reflop.device())
    check(tensor_session_index(z) == 1 and tensor_session_index(n) == 1, "'remote-cpu' did not use the server CPU")
    with reflop.placement("remote"):
        w = x @ x
    check(hasattr(w, "_rf_storage"), "'remote' result is local")
    with reflop.placement("auto"):
        small = torch.ones(2).to(d)
        big_local = torch.randn(2000, 2000)
        mixed = big_local @ big_local[:, :2] + small     # cost model may choose local
    return f"local -> CPU tensor, remote-cpu -> session 1, auto mixed op result type {type(mixed).__name__}"


def placement_invalid():
    import reflop

    try:
        with reflop.placement("bogus"):
            pass
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:100]}"
    raise AssertionError("placement('bogus') accepted")


# ----------------------------------------------------------------------------- when work is sent

def flush_at_20k_ops():
    d = gpu()
    idx = 0
    s0 = stats(idx)
    x = torch.zeros(1).to(d)
    for _ in range(25000):
        x = x + 1
    s1 = stats(idx)
    check(s1["flushes"] - s0["flushes"] >= 1, f"no automatic flush after 25k ops: {stats_delta(s0, s1)}")
    check(x.item() == 25000, f"value {x.item()}")
    return f"flushes during recording: {s1['flushes'] - s0['flushes']}"


def step_boundary_one_graph_per_step():
    import reflop

    d = gpu()
    m = nn.Linear(64, 64).to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x = torch.randn(32, 64).to(d)
    for _ in range(3):
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
        reflop.step_boundary()
    reflop.synchronize()
    s0 = stats(0)
    for _ in range(10):
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
        reflop.step_boundary()
    reflop.synchronize()
    dd = stats_delta(s0, stats(0))
    check(dd["registers"] == 0, f"identical steps registered new graphs: {dd}")
    check(9 <= dd["executes"] <= 11, f"expected ~10 executes: {dd}")
    check(dd["bytes_up"] < 2048, f"bytes up {dd['bytes_up']}")
    return f"10 steps: {dd}"


def nonzero_mask_roundtrips():
    d = gpu()
    x = torch.randn(1000)
    xd = x.to(d)
    s0 = stats(0)
    a = torch.nonzero(xd > 1).cpu()
    b = xd[xd < -1].cpu()
    dd = stats_delta(s0, stats(0))
    check(torch.equal(a, torch.nonzero(x > 1)) and torch.equal(b, x[x < -1]), "values")
    return f"stats delta {dd}"


def stats_bytes():
    d = gpu()
    s0 = stats(0)
    t = torch.randn(1024, 1024).to(d)        # 4 MiB
    t.sum().item()
    back = t.cpu()
    dd = stats_delta(s0, stats(0))
    check(dd["bytes_up"] >= 4 * 2**20, f"bytes_up {dd['bytes_up']}")
    check(dd["bytes_down"] >= 4 * 2**20, f"bytes_down {dd['bytes_down']}")
    check(back.shape == (1024, 1024), "shape")
    return f"{dd}"


def lazy_shape_errors_are_local():
    d = gpu()
    try:
        torch.randn(3, 4).to(d) @ torch.randn(5, 6).to(d)
    except RuntimeError as e:
        healthy(d)
        return f"raised at the call site: {str(e)[:100]}"
    raise AssertionError("shape mismatch not raised eagerly")


# ----------------------------------------------------------------------------- errors

def async_error_surfaces_later():
    import reflop

    d = gpu()
    t = torch.arange(10).to(d)
    idx = torch.tensor([3, 50]).to(d)
    try:
        bad = t[idx]                       # out of bounds: only the server knows
        reflop.step_boundary()
        _ = (torch.ones(2).to(d) * 2)      # unrelated work
        bad.sum().item()
    except reflop.ReflopRemoteError as e:
        msg = str(e)
        check("running a graph failed" in msg or "INDEX" in msg.upper(), f"message: {msg[:200]}")
        healthy(d)
        return f"{type(e).__name__}: {msg[:180]}"
    raise AssertionError("out-of-bounds index did not raise")


def error_at_exit_reported():
    script = SCRATCH / "exit_error.py"
    script.write_text(textwrap.dedent("""
        import torch, reflop
        d = reflop.device()
        t = torch.arange(10).to(d)
        bad = t[torch.tensor([99]).to(d)]
        reflop.step_boundary()          # sent, never waited on
    """))
    r = py([script], timeout=120)
    check("an error was not reported before the program ended" in r.stderr,
          f"rc={r.returncode}, stderr: {r.stderr[-600:]}")
    return f"rc={r.returncode}; " + r.stderr.strip().splitlines()[-1][:160]


def oom_oversized():
    import reflop

    d = gpu()
    try:
        x = torch.empty(2**38, dtype=torch.float32, device=d)    # 1 TiB
        x.fill_(1)
        x.sum().item()
    except reflop.OutOfMemoryError as e:
        check(isinstance(e, torch.OutOfMemoryError), "reflop.OutOfMemoryError is not a torch.OutOfMemoryError")
        healthy(d)
        return f"OutOfMemoryError (also torch.OutOfMemoryError): {str(e)[:150]}"
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"oversized allocation raised {type(e).__name__} instead of OutOfMemoryError: {e}") from e
    raise AssertionError("1 TiB allocation succeeded?")


def oom_real_fill_and_recover():
    import reflop

    d = gpu()
    held = []
    try:
        for i in range(40):
            t = torch.empty(256 * 2**20, dtype=torch.float32, device=d)  # 1 GiB each
            t.fill_(float(i))
            held.append(t)
            reflop.synchronize()
        raise AssertionError("allocated 40 GiB without OOM?")
    except torch.OutOfMemoryError as e:
        n = len(held)
        catch_type = type(e).__name__
    held.clear()
    gc.collect()
    reflop.synchronize()
    healthy(d)
    big = torch.empty(256 * 2**20, device=d).fill_(1)
    v = big[:10].sum().item()
    check(v == 10, "re-allocation after OOM failed")
    return f"held {n} GiB before {catch_type}; freed; 1 GiB allocates again"


def bad_index_scatter():
    import reflop

    d = gpu()
    z = torch.zeros(5).to(d)
    try:
        z.scatter_(0, torch.tensor([7]).to(d), torch.ones(1).to(d))
        z.sum().item()
    except reflop.ReflopRemoteError as e:
        healthy(d)
        return f"{e.__class__.__name__}: {str(e)[:150]}"
    raise AssertionError("scatter out of bounds not caught")


def handle_churn_rss():
    import reflop

    d = gpu()
    x = torch.ones(1024).to(d)
    for _ in range(200):
        x = (x * 1.0001).relu()
    x.sum().item()
    gc.collect()
    r0 = rss_mib()
    for i in range(q(3000, 300)):
        t = torch.randn(256).to(d)
        x = x + t.sum() * 0
        if i % 200 == 0:
            reflop.step_boundary()
    x.sum().item()
    gc.collect()
    r1 = rss_mib()
    check(r1 - r0 < 200, f"RSS grew {r1 - r0:.0f} MiB over 3000 uploads")
    return f"RSS {r0:.0f} -> {r1:.0f} MiB"


def long_graph():
    d = gpu()
    x = torch.ones(16).to(d)
    for i in range(q(15000, 2000)):
        x = x * 1.0 + 0.0 if i % 2 else x.sin().cos()
    v = x.sum().item()
    ref = torch.ones(16)
    for i in range(q(15000, 2000)):
        ref = ref * 1.0 + 0.0 if i % 2 else ref.sin().cos()
    check(abs(v - ref.sum().item()) < 1e-3, f"{v} vs {ref.sum().item()}")
    return f"one graph of {q(15000, 2000)} ops ok"


def main():
    import reflop

    s = Suite("a06_sessions", __doc__.splitlines()[0])
    for fn in (device_reports_session, moves_between_sessions, differentiable_move, mixing_raises,
               module_to_and_data_assign, cpu_session_training_with_gpu_open, threads_share_session,
               seed_across_sessions, seed_with_two_sessions, close_and_reopen):
        s.case(f"session/{fn.__name__}", fn, timeout=600)
    s.case("session/open_close_latency", open_close_latency, s, timeout=600)
    for fn in (placement_modes, placement_invalid):
        s.case(f"placement/{fn.__name__}", fn, timeout=300)
    for fn in (flush_at_20k_ops, step_boundary_one_graph_per_step, nonzero_mask_roundtrips, stats_bytes,
               lazy_shape_errors_are_local, long_graph, handle_churn_rss):
        s.case(f"dispatch/{fn.__name__}", fn, timeout=900)
    for fn in (async_error_surfaces_later, error_at_exit_reported, bad_index_scatter, oom_oversized,
               oom_real_fill_and_recover):
        s.case(f"errors/{fn.__name__}", fn, timeout=600)
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
