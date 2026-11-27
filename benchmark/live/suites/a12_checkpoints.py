"""a12 automatic checkpoints and recovery (0.2.0, Owner-guarded): refused while off; interval rules; tag rules; with
`reflopd checkpoints on --interval 5m`, three Guests checkpoint in parallel, then lose their worker three ways:
GPU worker SIGKILL (crash) -> recover(); server-CPU worker SIGKILL -> recover() (pause --now freezes CPU sessions since 0.4.0, see a11); `reflopd kill` (on
purpose) -> recover refused. After recover: params bit-equal to the checkpoint, dropout RNG repeats, late tensors
raise SessionLostError, 1 GiB ballast intact, training continues. Checkpoint overhead on step time."""

from __future__ import annotations

import time

from live import owner
from live.harness import LineReader, Skip, Suite, check, popen_helper, q

INTERVAL_MIN = 5


def child(*args):
    return LineReader(popen_helper("train_loop", *map(str, args)))


def refused_when_off():
    owner.need()
    st = owner.checkpoints_state()
    if st["on"]:
        raise Skip("checkpoints already on at the Owner")
    import reflop

    reflop.close()
    try:
        reflop.device(checkpoint=True)
    except reflop.ReflopError as e:
        check("reflopd checkpoints on" in str(e), f"message: {e}")
        reflop.close()
        return str(e)[:200]
    reflop.close()
    raise AssertionError("checkpoint=True accepted while the Owner has checkpoints off")


def tag_without_checkpoints():
    import reflop

    reflop.close()
    reflop.device()
    try:
        reflop.checkpoint_tag({"step": 1})
    except reflop.ReflopError as e:
        reflop.close()
        return str(e)[:150]
    raise AssertionError("checkpoint_tag accepted without checkpoints")


def interval_rules():
    import reflop

    reflop.close()
    res = {}
    try:   # below the Owner's minimum (since 0.4.0 the 1-min RAM interval; before, the disk interval)
        reflop.device(checkpoint=True, checkpoint_interval=20)
        res["20s"] = "accepted"
    except reflop.ReflopError as e:
        res["20s"] = str(e)[:160]
    reflop.close()
    check(res["20s"] != "accepted" and "at most" in res["20s"], f"interval below the Owner's minimum: {res['20s']}")
    from reflop import session as ss

    reflop.device(checkpoint=True, checkpoint_interval=60)
    caps = next(iter(ss._sessions.values())).caps
    res["owner_min_s"] = int(caps.checkpoint_interval_s)
    res["owner_disk_s"] = int(getattr(caps, "checkpoint_disk_interval_s", 0))
    res["60s"] = "accepted"
    reflop.close()
    check(res["owner_min_s"] <= 60, f"Owner minimum {res['owner_min_s']} s: 60 s should be refused, but was accepted")
    reflop.device(checkpoint=True, checkpoint_interval=1800)
    res["1800s"] = "accepted"
    try:
        reflop.checkpoint_tag({"blob": "x" * 5000})
        res["tag_5000B"] = "accepted"
    except ValueError as e:
        res["tag_5000B"] = str(e)[:100]
    check(res["tag_5000B"] != "accepted", "oversized tag accepted")
    reflop.checkpoint_tag({"step": 3, "lr": 0.1})
    reflop.close()
    return res


def three_way_recovery(s):
    owner.wait_quiet()
    wait = INTERVAL_MIN * 60 + 120
    a = child("--mode", "ckpt", "--kind", "cuda", "--seconds", wait + 300, "--sleep", 0.1, "--state-mib", 1024)
    b = child("--mode", "ckpt", "--kind", "cpu", "--seconds", wait + 300, "--sleep", 0.1)
    c = child("--mode", "ckpt", "--kind", "cuda", "--seconds", wait + 300, "--sleep", 0.1, "--expect", "refused")
    sids = {}
    for k, lr in (("gpu_crash", a), ("cpu_crash", b), ("owner_kill", c)):
        j = lr.wait_json(lambda j: "sid" in j, 120)
        check(j, f"{k}: no session: {lr.stdout_text()[-300:]} {lr.stderr_text()[-500:]}")
        sids[k] = j["sid"]
    warn = a.stderr_text()
    check("doubles" in warn, f"no GPU-memory warning printed when turning checkpoints on: {warn[-300:]}")
    ready = {}
    for k, lr in (("gpu_crash", a), ("cpu_crash", b), ("owner_kill", c)):
        j = lr.wait_json(lambda j: j.get("ready_to_kill") or "error" in j, wait)
        check(j and j.get("ready_to_kill"), f"{k}: no checkpoint within {wait}s: {lr.stdout_text()[-500:]}")
        ready[k] = j
    saved = {k: next(j for j in lr.json() if "ckpt_saved" in j) for k, lr in
             (("gpu_crash", a), ("cpu_crash", b), ("owner_kill", c))}
    for k, v in saved.items():
        s.metric("checkpoint_first", case=k, after_s=v["after_s"], at_step=v["at_step"])
    ko = owner.kill_worker(sids["gpu_crash"])
    kp = owner.kill_worker(sids["cpu_crash"])
    kc = owner.kill(sids["owner_kill"][:8])
    res = {}
    for k, lr in (("gpu_crash", a), ("cpu_crash", b), ("owner_kill", c)):
        rc = lr.finish(600)
        js = {}
        for j in lr.json():
            for key in j:
                js.setdefault(key, j)
        res[k] = {"rc": rc, **{x: js[x] for x in ("lost", "recovered", "recover_refused", "params_match",
                                                   "late_tensor", "rng_repeats", "continued", "ballast_ok",
                                                   "step_time_ms", "error") if x in js}}
        if "step_time_ms" in js:
            s.metric("checkpoint_step_time", case=k, **js["step_time_ms"]["step_time_ms"])
        if "recovered" in js:
            s.metric("recover_latency", case=k, recover_s=js["recovered"]["recover_s"])
    return {"summary": "", "owner": {"kill_worker_gpu": ko[-200:], "kill_worker_cpu": kp[-200:], "kill": kc[-200:]},
            **res}


def check_recovered(r, k):
    v = r[k]
    check("lost" in v, f"{k}: worker loss not seen as SessionLostError: {v}")
    check("recovered" in v, f"{k}: recover() failed: {v}")
    check(v["params_match"]["params_match"], f"{k}: parameters differ from the checkpoint: {v['params_match']}")
    check(v["rng_repeats"]["rng_repeats"], f"{k}: dropout did not repeat after recover: {v['rng_repeats']}")
    check(v["late_tensor"]["late_tensor"] == "SessionLostError", f"{k}: late tensor: {v['late_tensor']}")
    check(v.get("continued"), f"{k}: training did not continue")
    if "ballast_ok" in v:
        check(v["ballast_ok"]["ballast_ok"], f"{k}: 1 GiB ballast changed")
    return (f"lost -> recovered step {v['recovered']['recovered']} in {v['recovered']['recover_s']:.1f}s, "
            f"params bit-equal, RNG repeats, late tensor raises, continued (rc {v['rc']})")


def main():
    s = Suite("a12_checkpoints", __doc__.splitlines()[0])
    if not owner.available():
        s.case("owner_access", owner.need)
        s.finish()
    owner.save_initial_state()
    initial = owner.checkpoints_state()
    s.case("refused_when_owner_off", refused_when_off)
    s.case("tag_without_checkpoints", tag_without_checkpoints)
    try:
        o = owner.checkpoints(True, f"{INTERVAL_MIN}m")
        s.meta["checkpoints_on_output"] = o
        too_short = owner.out("checkpoints", "on", "--interval", "1m")
        s.case("owner_interval_floor_5m", lambda: (check("5m" in too_short or "5 min" in too_short
                                                         or "at least" in too_short, too_short),
                                                   too_short.strip()[:200])[1])
        owner.checkpoints(True, f"{INTERVAL_MIN}m")
        s.case("interval_and_tag_rules", interval_rules)
        rec = s.case("three_way_recovery", three_way_recovery, s, timeout=q(1800, 1800))
        r = rec.get("data") or {}
        if r:
            s.case("gpu_worker_crash_recovers", check_recovered, r, "gpu_crash")
            s.case("cpu_worker_crash_recovers", check_recovered, r, "cpu_crash")
            s.case("owner_kill_not_recoverable", lambda: (check("recover_refused" in r["owner_kill"],
                                                                f"recover after reflopd kill: {r['owner_kill']}"),
                                                          r["owner_kill"]["recover_refused"]["recover_refused"][:200])[1])
    finally:
        owner.checkpoints(initial["on"], initial.get("interval"))
        owner.resume()
    s.finish()


if __name__ == "__main__":
    main()
