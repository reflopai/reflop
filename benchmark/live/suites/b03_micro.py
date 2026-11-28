# SPDX-License-Identifier: Apache-2.0
"""b03 micro-benchmarks: round-trip latency (.item() of a tiny op, GPU and server CPU), transfer sweep (upload /
fetch / round trip 4 B .. 64 MiB, content verified), matmul TFLOP/s sweep fp32/bf16/fp16 (wall and server device
time from reflop.usage()), client recording cost per op vs graph length, graph cache (same graph vs shape churn),
step_boundary vs merged steps, dataset batch gather latency, and a link baseline (the same bytes through plain ssh to
the Owner, next to reFLOP's own transfers)."""

from __future__ import annotations

import os
import subprocess
import time

import torch

from live.harness import Skip, Suite, check, q, server_name, stats, stats_delta, summarize_times


def rtt(s):
    import reflop

    for kind, idx in (("cuda", 0), ("cpu", 1)):
        d = reflop.device(idx, kind=kind)
        x = torch.ones(4).to(d)
        for _ in range(5):
            (x + 1).sum().item()
        ts = []
        for _ in range(q(200, 20)):
            t = time.perf_counter()
            (x + 1).sum().item()
            ts.append((time.perf_counter() - t) * 1000)
        st = summarize_times(ts)
        s.metric("rtt", device=kind, **{f"ms_{k}": v for k, v in st.items() if k != "n"}, n=st["n"])
    return "see metrics"


def transfer(s):
    import reflop

    d = reflop.device(0)
    for nbytes in q((4, 1024, 65536, 2**20, 16 * 2**20, 64 * 2**20), (4, 2**20)):
        n = max(1, nbytes // 4)
        src = torch.randn(n)
        reps = 3 if nbytes >= 16 * 2**20 else 10
        up, down, rt = [], [], []
        for _ in range(reps):
            t = time.perf_counter()
            r = src.to(d)
            reflop.synchronize()
            up.append(time.perf_counter() - t)
            r.sum()          # keep it alive
            t = time.perf_counter()
            back = r.cpu()
            down.append(time.perf_counter() - t)
            t = time.perf_counter()
            back2 = (src.to(d) * 1).cpu()
            rt.append(time.perf_counter() - t)
        check(torch.equal(back, src) and torch.equal(back2, src), f"{nbytes} B: content changed")
        mu, md, mr = (sorted(v)[len(v) // 2] for v in (up, down, rt))
        s.metric("transfer", bytes=nbytes, upload_ms=mu * 1000, fetch_ms=md * 1000, roundtrip_ms=mr * 1000,
                 upload_mib_s=nbytes / 2**20 / mu, fetch_mib_s=nbytes / 2**20 / md)
    return "see metrics"


def matmul_sweep(s):
    import reflop

    d = reflop.device(0)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for n in q((512, 1024, 2048, 4096, 8192), (1024,)):
            x = torch.randn(n, n).to(dtype).to(d)
            k = max(4, min(200, int(4e12 / (2 * n ** 3))))
            y = x
            for _ in range(3):
                y = x @ x
            y[0, 0].item()
            u0 = reflop.usage()
            t = time.perf_counter()
            acc = None
            for _ in range(k):
                y = x @ x
                acc = y[0, 0] if acc is None else acc + y[0, 0]
            acc.item()
            dt = time.perf_counter() - t
            u1 = reflop.usage()
            dev_s = u1["device_seconds"] - u0["device_seconds"]
            flops = 2 * n ** 3 * k
            s.metric("matmul", dtype=str(dtype)[6:], n=n, reps=k, wall_tflops=flops / dt / 1e12,
                     device_tflops=flops / dev_s / 1e12 if dev_s > 0 else None, wall_s=dt, device_s=dev_s,
                     usage_flops_ratio=(u1["flops"] - u0["flops"]) / flops)
            del x, y
    return "see metrics"


def recording_cost(s):
    import reflop

    d = reflop.device(0)
    x = torch.ones(64).to(d)
    for n_ops in q((10, 100, 1000, 5000), (10, 100)):
        for _rep in range(2):
            t = time.perf_counter()
            y = x
            for _ in range(n_ops):
                y = y * 1.0001
            t_rec = time.perf_counter() - t
            t = time.perf_counter()
            y.sum().item()
            t_exec = time.perf_counter() - t
        s.metric("recording", ops=n_ops, record_us_per_op=t_rec / n_ops * 1e6, flush_and_run_ms=t_exec * 1000)
    return "see metrics"


def graph_cache(s):
    import reflop

    d = reflop.device(0)
    m = torch.nn.Linear(256, 256).to(d)
    res = {}
    for label, shapes in (("same_shape", [64] * 20), ("shape_churn", list(range(40, 60)))):
        xs = [torch.randn(b, 256).to(d) for b in shapes]
        reflop.synchronize()
        s0 = stats(0)
        t = time.perf_counter()
        for x in xs:
            m(x).relu().sum().item()
        dt = (time.perf_counter() - t) / len(xs)
        dd = stats_delta(s0, stats(0))
        res[label] = dt
        s.metric("graph_cache", case=label, ms_per_call=dt * 1000, registers=dd["registers"], executes=dd["executes"])
    return f"same {res['same_shape'] * 1000:.1f} ms vs churn {res['shape_churn'] * 1000:.1f} ms per call"


def step_merging(s):
    import reflop

    d = reflop.device(0)
    m = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10)).to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x, y = torch.randn(64, 512).to(d), torch.randint(0, 10, (64,)).to(d)
    for boundary in (True, False):
        for _ in range(5):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m(x), y).backward()
            opt.step()
            if boundary:
                reflop.step_boundary()
        reflop.synchronize()
        s0 = stats(0)
        t = time.perf_counter()
        for _ in range(50):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m(x), y).backward()
            opt.step()
            if boundary:
                reflop.step_boundary()
        reflop.synchronize()
        dt = (time.perf_counter() - t) / 50
        dd = stats_delta(s0, stats(0))
        s.metric("step_boundary", step_boundary=boundary, ms_per_step=dt * 1000, flushes=dd["flushes"],
                 registers=dd["registers"], executes=dd["executes"])
    return "see metrics"


def _ssh(host: str, cmd: str, data: bytes | None = None) -> tuple[float, bytes]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "Compression=no", host, cmd]
    t = time.perf_counter()
    r = subprocess.run(argv, input=data if data is not None else b"", capture_output=True, timeout=600)
    dt = time.perf_counter() - t
    if r.returncode != 0:
        raise RuntimeError(f"ssh {cmd!r}: exit {r.returncode}: {r.stderr.decode(errors='replace')[-200:]}")
    return dt, r.stdout


def link_baseline(s):
    """Is a slow transfer the link or reFLOP? Send the same random bytes through plain `ssh <Owner> cat` and read them
    back from the Owner, next to reFLOP's upload and fetch of the same size, in the same minute. ssh's connection
    setup (the median of `ssh <Owner> true`) is subtracted. Needs E2E_OWNER_SSH; meaningless on one machine."""
    import reflop
    from reflop.config import ClientConfig

    from live import owner

    if owner.LOCAL or not owner.HOST:
        raise Skip("needs E2E_OWNER_SSH: the baseline runs plain ssh to the Owner")
    owner.need()
    host = owner.HOST
    entry = ClientConfig.load().servers.get(server_name())
    ssh_target = ""
    r = subprocess.run(["ssh", "-G", host], capture_output=True, text=True, timeout=30)
    for line in r.stdout.splitlines():
        if line.startswith("hostname "):
            ssh_target = line.split(None, 1)[1]
    reflop_target = entry.addresses[0].rsplit(":", 1)[0].strip("[]") if entry and entry.addresses else ""
    setup = sorted(_ssh(host, "true")[0] for _ in range(3))[1]
    d = reflop.device(0)
    for nbytes in q((16 * 2**20, 64 * 2**20), (2**20,)):
        data = os.urandom(nbytes)
        up, down, rup, rdown = [], [], [], []
        src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        for _ in range(q(3, 1)):
            up.append(max(_ssh(host, "cat > /dev/null", data)[0] - setup, 1e-6))
            dt, got = _ssh(host, f"head -c {nbytes} /dev/urandom")
            check(len(got) == nbytes, f"ssh fetch returned {len(got)} of {nbytes} bytes")
            down.append(max(dt - setup, 1e-6))
            t = time.perf_counter()
            x = src.to(d)
            reflop.synchronize()
            rup.append(time.perf_counter() - t)
            t = time.perf_counter()
            back = x.cpu()
            rdown.append(time.perf_counter() - t)
            check(torch.equal(back, src), f"{nbytes} B: content changed")
        mu, md, ru, rd = (sorted(v)[len(v) // 2] for v in (up, down, rup, rdown))
        mib = nbytes / 2**20
        s.metric("link", bytes=nbytes, ssh_upload_mib_s=mib / mu, ssh_fetch_mib_s=mib / md,
                 reflop_upload_mib_s=mib / ru, reflop_fetch_mib_s=mib / rd,
                 upload_ratio=(mib / ru) / (mib / mu), fetch_ratio=(mib / rd) / (mib / md),
                 ssh_setup_ms=setup * 1000, transport=entry.transport if entry else None,
                 same_host=bool(ssh_target) and ssh_target == reflop_target)
    return "see metrics (ratio = reFLOP MiB/s / ssh MiB/s; near 1 means the link is the limit)"


def main():
    import reflop

    s = Suite("b03_micro", __doc__.splitlines()[0])
    for fn in (rtt, transfer, link_baseline, matmul_sweep, recording_cost, graph_cache, step_merging):
        s.case(fn.__name__, fn, s, timeout=1800)
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
