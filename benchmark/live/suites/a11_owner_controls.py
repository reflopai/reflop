# SPDX-License-Identifier: Apache-2.0
"""a11 Owner controls (over SSH, guarded): say (+ control-sequence stripping), status/top/usage/audit agree with the
Guest, pause/resume (work waits, nothing lost), pause --now suspends a GPU worker (2 GiB state survives, bit-identical
losses, latency), pause --now freezes a server-CPU session and resume thaws it (0.4.0; identical losses), bounded pause queue (0.2.2:
>4096 held requests -> ERROR_PAUSED, session fine after), a session killed while paused gets SESSION_LOST (0.2.2),
kill, auto-yield (+ --suspend) when the Owner uses the GPU."""

from __future__ import annotations

import re
import time

from live import owner
from live.harness import SCRATCH, LineReader, Suite, check, popen_helper

REF: dict = {}


def child(*args, **kw):
    return LineReader(popen_helper("train_loop", *map(str, args), **kw))


def sid_of(lr: LineReader, timeout=60) -> str:
    j = lr.wait_json(lambda j: "sid" in j, timeout)
    check(j, f"child never opened a session: {lr.stderr_text()[-600:]}")
    return j["sid"]


def steps(lr):
    return [j for j in lr.json() if "step" in j and "loss" in j]


def max_gap(lr):
    st = steps(lr)
    gaps = [(b["t"] - a["t"], a["step"]) for a, b in zip(st, st[1:])]
    return max(gaps) if gaps else (0, None)


def reference(kind="cuda", n=60, state=0):
    key = (kind, n, state)
    if key not in REF:
        lr = child("--mode", "train", "--kind", kind, "--steps", n, "--state-mib", state)
        rc = lr.finish(600)
        check(rc == 0, f"reference run failed: {lr.stdout_text()[-500:]} {lr.stderr_text()[-500:]}")
        REF[key] = [j["loss"] for j in steps(lr)]
    return REF[key]


# ----------------------------------------------------------------------------- read-only

def say_sanitized():
    owner.need()
    lr = child("--mode", "idle", "--seconds", 15)
    sid_of(lr)
    time.sleep(2)
    msg = "e2e hello \x1b[31mRED\x1b[0m \x07bell done"
    o = owner.say(msg)
    lr.finish(60)
    err = lr.stderr_text()
    ev = [j for j in lr.json() if j.get("event") == "KIND_OWNER_MESSAGE"]
    check(ev, f"no owner message event; owner said: {o!r}")
    line = next((l for l in err.splitlines() if "e2e hello" in l), "")
    check(line, f"message not printed:\n{err[-500:]}")
    check("\x1b" not in line and "\x07" not in line, f"control sequences reached the terminal: {line!r}")
    return f"printed: {line!r}"


def status_top_usage_audit():
    owner.need()
    lr = child("--mode", "train", "--steps", 40, "--sleep", 0.1)
    sid = sid_of(lr)
    time.sleep(3)
    st = owner.sessions()
    top = owner.out("top")
    mine = [x for x in st if sid.startswith(x.get("session", "-"))]
    check(mine and owner.guest_of(mine[0]["identity"]) == owner.OURS[0], f"session {sid[:8]} not in reflopd status: {st}")
    ident = mine[0]["identity"]
    check(sid[:8] in top, f"session not in reflopd top:\n{top[-600:]}")
    lr.finish(300)
    import reflop

    time.sleep(2)
    g = reflop.usage()
    o = owner.guest_usage(ident)
    check(abs(o["executes"] - g["executes"]) <= 5, f"executes: owner {o['executes']} guest {g['executes']}")
    check(abs(o["flops"] - g["flops"]) <= 0.001 * max(g["flops"], 1), "flops differ")
    aud = owner.audit(ident, since="10m")
    check(sid[:8] in aud and "session_end" in aud, f"audit lacks the session:\n{aud[-800:]}")
    (SCRATCH / "audit_sample.txt").write_text(aud)
    return {"summary": f"session {sid[:8]} in status/top/audit; usage agrees ({g['executes']} executes)",
            "status_row": mine[0], "top_excerpt": top[-600:]}


# ----------------------------------------------------------------------------- pause family

def pause_resume():
    owner.wait_quiet()
    ref = reference()
    lr = child("--mode", "train", "--steps", 60, "--sleep", 0.05)
    sid_of(lr)
    lr.wait_json(lambda j: j.get("step") == 15, 120)
    try:
        _t_p = time.time()
        po = owner.pause()
        time.sleep(10)
    finally:
        _ro = owner.resume()
    rc = lr.finish(300)
    check(rc == 0, f"child rc {rc}: {lr.stdout_text()[-400:]}")
    gap, at = max_gap(lr)
    got = [j["loss"] for j in steps(lr)]
    ev = sorted({j["event"] for j in lr.json() if "event" in j})
    check(gap >= 7, f"no stall during the pause (max gap {gap:.1f}s)")
    check(got == ref, "losses differ from the undisturbed run")
    check("KIND_PAUSED" in ev and "KIND_RESUMED" in ev, f"events {ev}")
    return f"stalled {gap:.1f}s at step {at}; losses identical; events {ev}; owner: {po.strip()[:80]!r}"


def pause_now_suspends_gpu(s):
    owner.wait_quiet()
    ref = reference(state=2048)
    lr = child("--mode", "train", "--steps", 60, "--sleep", 0.05, "--state-mib", 2048)
    _sid = sid_of(lr)
    lr.wait_json(lambda j: j.get("step") == 15, 300)
    try:
        t0 = time.time()
        po = owner.pause(now=True)
        t_pause = time.time() - t0
        st = owner.status_text()
        time.sleep(8)
    finally:
        t0 = time.time()
        _ro = owner.resume()
        t_resume = time.time() - t0
    rc = lr.finish(600)
    gap, at = max_gap(lr)
    got = [j["loss"] for j in steps(lr)]
    ev = sorted({j["event"] for j in lr.json() if "event" in j})
    ok_ballast = any(j.get("ballast_ok") for j in lr.json())
    s.metric("pause_now", state_mib=2048, owner_pause_cmd_s=t_pause, owner_resume_cmd_s=t_resume, guest_gap_s=gap,
             owner_output=po.strip()[:200])
    check(rc == 0, f"child rc {rc}: {lr.stdout_text()[-500:]}")
    check("suspend" in po.lower(), f"pause --now output does not report a suspend: {po!r}")
    check("suspend" in st.lower(), f"reflopd status does not mark the session suspended:\n{st[-500:]}")
    check(ok_ballast, "2 GiB ballast changed across suspend/resume")
    check(got == ref, "losses differ from the undisturbed run")
    return f"owner: {po.strip()[:120]!r}; guest stalled {gap:.1f}s; state intact; losses identical; events {ev}"


def pause_now_freezes_cpu_session(s):
    """reflopd >= 0.4.0: `pause --now` freezes server-CPU sessions (it used to kill them) and `resume` thaws them."""
    owner.wait_quiet()
    ref = reference(kind="cpu", n=80)
    lr = child("--mode", "train", "--kind", "cpu", "--steps", 80, "--sleep", 0.05)
    sid_of(lr)
    lr.wait_json(lambda j: j.get("step") == 10, 300)
    try:
        t0 = time.time()
        po = owner.pause(now=True)
        t_pause = time.time() - t0
        _st = owner.status_text()
        time.sleep(6)
    finally:
        t0 = time.time()
        owner.resume()
        t_resume = time.time() - t0
    rc = lr.finish(300)
    gap, at = max_gap(lr)
    got = [j["loss"] for j in steps(lr)]
    lost = [j for j in lr.json() if "lost" in j]
    s.metric("pause_now_cpu", owner_pause_cmd_s=t_pause, owner_resume_cmd_s=t_resume, guest_gap_s=gap,
             owner_output=po.strip()[:200])
    check(not lost, f"CPU session lost instead of frozen: {lost[0]['lost'][:200] if lost else ''}")
    check(rc == 0, f"child rc {rc}: {lr.stdout_text()[-500:]} {lr.stderr_text()[-300:]}")
    m = re.search(r"(\d+) CPU session\(s\) frozen, (\d+) killed", po)
    check(m and int(m.group(1)) >= 1 and m.group(2) == "0", f"pause --now did not report a frozen CPU session: {po!r}")
    check(gap >= 4, f"guest did not stall while frozen (max gap {gap:.1f}s at step {at})")
    check(got == ref, "losses differ from the undisturbed run")
    return f"owner: {po.strip()[:120]!r}; guest stalled {gap:.1f}s; resumed; {len(got)} steps, losses identical"


def bounded_pause_queue():
    owner.wait_quiet()
    go = SCRATCH / "flood.go"
    go.unlink(missing_ok=True)
    n = 6000
    lr = child("--mode", "flood", "--flood", n, "--go", go)
    sid_of(lr)
    check(lr.wait_json(lambda j: j.get("ready"), 120), f"flood child not ready: {lr.stderr_text()[-500:]}")
    try:
        owner.pause()
        time.sleep(1)
        go.touch()
        sent = lr.wait_json(lambda j: "sent" in j, 300)
        time.sleep(3)
        _st = owner.status_text()
    finally:
        owner.resume()
    rc = lr.finish(300)
    js = lr.json()
    sync = next((j for j in js if "sync_error" in j), {})
    val = next((j for j in js if "value" in j), {})
    healthy = next((j for j in js if "healthy_after" in j), {}).get("healthy_after")
    errtext = (sent or {}).get("send_error") or ""
    errtext += (sync.get("sync_error") or "") + " ".join(val.get("later_errors") or [])
    check(sent, f"child never finished sending (rc {rc}): {lr.stdout_text()[-600:]}")
    check("pause" in errtext.lower(), f"no ERROR_PAUSED for requests past the bound: sync={sync} value={val}")
    acc = val.get("accepted")
    check(acc is not None and 4000 <= acc < n, f"accepted {acc} of {n} sent={sent} (bound 4096 per session)")
    check(healthy, "session unusable after the flood")
    return f"{sent['sent']} sent while paused: {acc:.0f} accepted, rest refused: {errtext[:160]!r}; healthy after resume"


def killed_while_paused():
    owner.wait_quiet()
    lr = child("--mode", "train", "--steps", 400, "--sleep", 0.05)
    sid = sid_of(lr)
    lr.wait_json(lambda j: j.get("step") == 10, 120)
    try:
        owner.pause()
        time.sleep(3)
        t0 = time.time()
        ko = owner.kill(sid[:8])
        rc = lr.finish(60)
        dt = time.time() - t0
    finally:
        owner.resume()
    lost = [j for j in lr.json() if "lost" in j]
    check(lost, f"held request never answered (rc {rc}): {lr.stdout_text()[-300:]}")
    return f"SessionLostError {dt:.1f}s after kill: {lost[0]['lost'][:120]!r}; owner {ko.strip()[:80]!r}"


def kill_session():
    owner.need()
    lr = child("--mode", "train", "--steps", 1000, "--sleep", 0.05)
    sid = sid_of(lr)
    lr.wait_json(lambda j: j.get("step") == 10, 120)
    t0 = time.time()
    _ko = owner.kill(sid[:8])
    rc = lr.finish(60)
    dt = time.time() - t0
    lost = [j for j in lr.json() if "lost" in j]
    ev = sorted({j["event"] for j in lr.json() if "event" in j})
    check(lost, f"no SessionLostError after reflopd kill (rc {rc})")
    return f"SessionLostError after {dt:.1f}s; events {ev}; {lost[0]['lost'][:100]!r}"


GPU_HOG = """
import time, torch
x = torch.randn(4096, 4096, device="cuda")
t = time.time()
while time.time() - t < {secs}:
    y = x @ x
torch.cuda.synchronize()
print("hog done", flush=True)
"""


def auto_yield(suspend: bool):
    owner.wait_quiet()
    hog = SCRATCH / "gpu_hog.py"
    hog.write_text(GPU_HOG.format(secs=15))
    remote = owner.push(str(hog))
    pyr = owner.runtime_python()
    lr = child("--mode", "train", "--steps", 150, "--sleep", 0.05, "--state-mib", 512)
    sid_of(lr)
    lr.wait_json(lambda j: j.get("step") == 10, 120)
    try:
        yo = owner.yield_mode("auto-suspend" if suspend else "auto")
        hp = owner.popen(f"{pyr} -I {remote}")
        time.sleep(6)
        st = owner.status_text()
        hp.wait(120)
        time.sleep(6)
    finally:
        owner.yield_mode("off")
    rc = lr.finish(600)
    gap, at = max_gap(lr)
    ev = sorted({j["event"] for j in lr.json() if "event" in j})
    check(rc == 0, f"child rc {rc}: {lr.stdout_text()[-300:]}")
    check(gap >= 5, f"Guest was not paused while the Owner used the GPU (max gap {gap:.1f}s); events {ev}")
    check(any(j.get("ballast_ok") for j in lr.json()), "state changed")
    if suspend:
        check("suspend" in st.lower(), f"--suspend: session not marked suspended:\n{st[-400:]}")
    return f"yield {'--suspend ' if suspend else ''}paused the Guest {gap:.1f}s; events {ev}; {yo.strip()[:80]!r}"


def main():
    s = Suite("a11_owner_controls", __doc__.splitlines()[0])
    if not owner.available():
        s.case("owner_access", owner.need)
        s.finish()
    owner.save_initial_state()
    try:
        s.case("say_sanitized", say_sanitized, timeout=180)
        s.case("status_top_usage_audit", status_top_usage_audit, timeout=600)
        s.case("pause_resume", pause_resume, timeout=1200)
        s.case("pause_now_suspends_gpu", pause_now_suspends_gpu, s, timeout=1500)
        s.case("pause_now_freezes_cpu_session", pause_now_freezes_cpu_session, s, timeout=1200)
        s.case("bounded_pause_queue", bounded_pause_queue, timeout=1200)
        s.case("killed_while_paused", killed_while_paused, timeout=1200)
        s.case("kill_session", kill_session, timeout=600)
        s.case("auto_yield", auto_yield, False, timeout=1200)
        s.case("auto_yield_suspend", auto_yield, True, timeout=1200)
    finally:
        owner.resume()
        owner.yield_mode("off")
    s.finish()


if __name__ == "__main__":
    main()
