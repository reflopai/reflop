"""b04 soak: captured char-GPT (dropout on) trained from a server dataset for 20 minutes; every 30 s: steps/s, step
time, loss, client RSS, thread count, replays, bytes per step. Pass if it never stalls > 30 s, loss falls, RSS and
step time stay flat (last third vs first third within 25% / 100 MiB)."""

from __future__ import annotations

import os
import shutil
import threading
import time

import torch
import torch.nn.functional as F

from live import workloads as wl
from live.harness import PREFIX, SCRATCH, Suite, check, cli, q, rss_mib, stats, tail

MINUTES = float(os.environ.get("E2E_SOAK_MINUTES", q(20, 1)))
NAME = f"{PREFIX}-soak-{os.environ.get('E2E_RUN_ID', 'adhoc').replace('_', '-').lower()[-12:]}"


def soak(s):
    import reflop

    p = SCRATCH / "packs_soak"
    shutil.rmtree(p, ignore_errors=True)
    r = cli("data", "build", "--from", "live.helpers.packs:text_200k", "--out", p, timeout=600)
    check(r.returncode == 0, tail(r.stderr))
    r = cli("data", "push", p, "--name", NAME, timeout=1800)
    check(r.returncode == 0, tail(r.stderr))
    d = reflop.device()
    ds = reflop.dataset(NAME)
    m = wl.GPT(dropout=0.1).to(d)
    opt = torch.optim.AdamW(m.parameters(), lr=torch.tensor(3e-4).to(d), capturable=True)

    def step(batch):
        opt.zero_grad(set_to_none=True)
        out = m(batch["x"])
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), batch["y"].reshape(-1))
        loss.backward()
        opt.step()
        return loss

    f = reflop.capture(step)
    samples, last = [], time.time()
    t0 = time.time()
    n = 0
    s0 = stats(0)
    win_t, win_n = time.time(), 0
    stall = 0.0
    it = ds.loader(64, shuffle=True, seed=0, epochs=10_000)
    first_loss = None
    while time.time() - t0 < MINUTES * 60:
        batch = next(it)
        loss = f(batch)
        n += 1
        win_n += 1
        if n % 50 == 0:
            tl = time.time()
            v = loss.item()
            stall = max(stall, time.time() - tl)
            first_loss = first_loss if first_loss is not None else v
        if time.time() - win_t >= 30:
            v = loss.item()
            st = stats(0)
            row = dict(minute=(time.time() - t0) / 60, steps_per_s=win_n / (time.time() - win_t),
                       ms_per_step=(time.time() - win_t) / win_n * 1000, loss=v, rss_mib=rss_mib(),
                       threads=threading.active_count(), replays=f.replays, total_steps=n,
                       bytes_up_per_step=(st["bytes_up"] - s0["bytes_up"]) / n)
            samples.append(row)
            s.metric("soak", **row)
            win_t, win_n = time.time(), 0
    reflop.close()
    cli("data", "rm", NAME)
    k = max(1, len(samples) // 3)
    a, b = samples[:k], samples[-k:]
    ms_a, ms_b = sum(x["ms_per_step"] for x in a) / k, sum(x["ms_per_step"] for x in b) / k
    rss_a, rss_b = sum(x["rss_mib"] for x in a) / k, sum(x["rss_mib"] for x in b) / k
    summary = (f"{n} steps in {MINUTES:g} min; step {ms_a:.2f}->{ms_b:.2f} ms; RSS {rss_a:.0f}->{rss_b:.0f} MiB; "
               f"loss {first_loss:.3f}->{samples[-1]['loss']:.3f}; replays {f.replays}")
    check(stall < 30, f"stalled {stall:.1f}s: {summary}")
    check(samples[-1]["loss"] < first_loss, f"loss did not fall: {summary}")
    check(ms_b < ms_a * 1.25, f"step time grew: {summary}")
    check(rss_b - rss_a < 100, f"client RSS grew: {summary}")
    return summary


def main():
    s = Suite("b04_soak", __doc__.splitlines()[0])
    s.case("gpt_capture_dataset_soak", soak, s, timeout=MINUTES * 60 + 1800)
    s.finish()


if __name__ == "__main__":
    main()
