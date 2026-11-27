"""a14 concurrency: 1/2/4/8 Guest processes training a captured CNN at once (aggregate throughput, per-process
latency, Jain fairness), and one process filling the GPU while another hits OutOfMemoryError (both sessions survive)."""

from __future__ import annotations

import time

from live.harness import SCRATCH, LineReader, Suite, check, popen_helper, q

SECONDS = q(25, 8)


def run_n(n, batch=128):
    go = SCRATCH / f"conc_{n}.go"
    go.unlink(missing_ok=True)
    kids = [LineReader(popen_helper("train_loop", "--mode", "bench", "--seconds", SECONDS, "--batch", batch,
                                    "--go", go)) for _ in range(n)]
    for k in kids:
        check(k.wait_json(lambda j: j.get("ready") or "error" in j, 180), f"child not ready: {k.stderr_text()[-400:]}")
    go.touch()
    res = []
    for k in kids:
        rc = k.finish(SECONDS + 180)
        j = k.wait_json(lambda j: j.get("done"), 5)
        check(rc == 0 and j, f"child rc {rc}: {k.stdout_text()[-300:]} {k.stderr_text()[-300:]}")
        res.append(j)
    return res


def jain(xs):
    return sum(xs) ** 2 / (len(xs) * sum(x * x for x in xs)) if xs else float("nan")


def scaling(s):
    base = None
    lines = []
    for n in q((1, 2, 4, 8), (1, 2)):
        res = run_n(n)
        sps = [r["samples_per_s"] for r in res]
        agg = sum(sps)
        base = base or agg
        f = jain(sps)
        s.metric("concurrency", processes=n, aggregate_samples_per_s=agg, per_process_samples_per_s=sps,
                 scaling_vs_1=agg / base, jain_fairness=f, ms_median=[r["ms_median"] for r in res],
                 ms_p90=[r["ms_p90"] for r in res])
        check(f > 0.8, f"{n} processes: unfair split {sps}")
        lines.append(f"{n}: {agg:.0f} samples/s (x{agg / base:.2f}, Jain {f:.2f})")
    return "; ".join(lines)


def oom_isolation():
    hog = LineReader(popen_helper("train_loop", "--mode", "alloc", "--alloc-mib", 7 * 1024, "--seconds", 25))
    j = hog.wait_json(lambda j: "allocated_mib" in j or "alloc_error" in j, 180)
    check(j and "allocated_mib" in j, f"hog could not allocate 7 GiB (GPU busy?): {j}")
    victim = LineReader(popen_helper("train_loop", "--mode", "alloc", "--alloc-mib", 6 * 1024, "--seconds", 1))
    victim.finish(180)
    hog.finish(180)
    v = {k: x for d in victim.json() for k, x in d.items()}
    h = {k: x for d in hog.json() for k, x in d.items()}
    check(v.get("is_oom") is True, f"second process: {v}")
    check(v.get("healthy_after") is True, "second process's session broken after OOM")
    check(h.get("still_ok") is True, f"first process's memory damaged: {h}")
    return f"7 GiB holder intact; 6 GiB request -> {v.get('alloc_error', '')[:140]}"


def main():
    s = Suite("a14_concurrency", __doc__.splitlines()[0])
    s.case("scaling_1_2_4_8", scaling, s, timeout=2400)
    s.case("oom_isolation", oom_isolation, timeout=600)
    s.finish()


if __name__ == "__main__":
    main()
