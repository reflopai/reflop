# SPDX-License-Identifier: Apache-2.0
"""a13 identities and quotas (Owner-guarded; only throwaway e2e-* identities): pair e2e-alice/bob/carol in separate
client homes (distinct keys, Keychain items); invite single-use, expired invite message, pairing while paired says
unpair first without using the invite; isolation of datasets and snapshots across identities; concurrent training;
quotas on e2e-alice (device memory, max sessions, execute wall time, execute FLOPs, storage, upload bandwidth); unpair
Guest-first and Owner-first, then pair again."""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import shutil
import subprocess
import textwrap
import time

from live import owner
from live.harness import (
    GUESTS,
    PREFIX,
    SCRATCH,
    LineReader,
    Skip,
    Suite,
    check,
    cli,
    guest_home,
    popen_helper,
    py,
    tail,
)

A, B, C = f"{PREFIX}-alice", f"{PREFIX}-bob", f"{PREFIX}-carol"


def keychain_delete(name):
    if os.uname().sysname != "Darwin":
        return
    acct = f"{getpass.getuser()}:{guest_home(name)}"
    subprocess.run(["security", "delete-generic-password", "-s", "reflop-client-key", "-a", acct],
                   capture_output=True)


def wipe_local(name):
    h = guest_home(name)
    with contextlib.suppress(Exception):
        cli("unpair", "--local", "--yes", guest=name, timeout=60)
    keychain_delete(name)
    shutil.rmtree(h, ignore_errors=True)


def pair(name, code=None):
    code = code or owner.invite(name, "30m")
    guest_home(name).mkdir(parents=True, exist_ok=True)
    r = cli("pair", code, "--name", "e2e", guest=name, timeout=120)
    return r, code


def gpy(name, code, timeout=600):
    f = SCRATCH / f"snippet_{name}_{abs(hash(code)) % 10**8}.py"
    f.write_text(textwrap.dedent(code))
    return py([f], guest=name, timeout=timeout)


# ----------------------------------------------------------------------------- pairing

def setup_pair_all():
    owner.need()
    for n in (A, B, C):
        wipe_local(n)
        with contextlib.suppress(Exception):
            owner.unpair(n)
    res = {}
    for n in (A, B):
        r, _ = pair(n)
        check(r.returncode == 0, f"pair {n}: {tail(r.stdout + r.stderr)}")
        st = cli("--server", "e2e", "status", guest=n)   # 0.4.x: plain `status` prints the pairings table
        check(re.search(rf"you are '{re.escape(n)}(@gpu-[0-9a-f-]+)?'", st.stdout), f"{n} status: {st.stdout[:300]}")
        res[n] = st.stdout.splitlines()[0]
    ka, kb = cli("key", guest=A).stdout.strip(), cli("key", guest=B).stdout.strip()
    kd = cli("key").stdout.strip()
    check(len({ka, kb, kd}) == 3, "Guest homes share a key")
    u = owner.users()
    check(A in u and B in u, f"owner users: {u}")
    return {"summary": f"{A}, {B} paired with distinct keys", **res}


def invite_single_use():
    code = owner.invite(C, "30m")
    r, _ = pair(C, code)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    wipe_local(C)
    owner.unpair(C)
    h = guest_home(C)
    h.mkdir(parents=True, exist_ok=True)
    r2 = cli("pair", code, "--name", "e2e", guest=C)
    o = r2.stdout + r2.stderr
    check(r2.returncode != 0 and "Traceback" not in o, f"reused invite accepted: {o[-300:]}")
    wipe_local(C)
    return o.strip().splitlines()[-1][:200]


def pair_while_paired_keeps_invite():
    code = owner.invite(C, "30m")
    r = cli("pair", code, "--name", "e2e-again", guest=A)
    o = r.stdout + r.stderr
    # refused without using the invite; 0.2.x said "run `reflop unpair e2e` first", 0.4.x "already paired as 'e2e'"
    check(r.returncode != 0 and ("unpair" in o.lower() or "already paired" in o.lower()),
          f"pairing a paired home: {o[-300:]}")
    r2, _ = pair(C, code)
    check(r2.returncode == 0, f"the invite was used up by the refused attempt: {tail(r2.stdout + r2.stderr)}")
    wipe_local(C)
    owner.unpair(C)
    return f"refused with {o.strip().splitlines()[-1][:140]!r}; invite still worked afterwards"


def expired_invite():
    t = owner.out("invite", "--name", C, "--expires", "1s")
    m = re.search(r"reflop1-[A-Za-z0-9_\-.~+=/]+", t)
    if not m:
        raise Skip(f"Owner refused a 1s invite: {t.strip()[:150]}")
    time.sleep(4)
    guest_home(C).mkdir(parents=True, exist_ok=True)
    r = cli("pair", m.group(0), guest=C)
    o = r.stdout + r.stderr
    wipe_local(C)
    check(r.returncode != 0, "expired invite accepted")
    check("UTC" in o and "expired" in o.lower(), f"message lacks expiry/clock: {o[-300:]}")
    return o.strip()[-220:]


# ----------------------------------------------------------------------------- isolation

def isolation():
    ds = f"{PREFIX}-alice-ds"
    snap = f"{PREFIX}-alice-snap"
    r = gpy(A, f"""
        import torch, reflop
        d = reflop.device()
        reflop.save({snap!r}, {{"w": torch.arange(10.0).to(d)}})
        print("SAVED", [s["name"] for s in reflop.snapshots()])
    """)
    check(r.returncode == 0 and "SAVED" in r.stdout, tail(r.stdout + r.stderr))
    p = SCRATCH / "packs_alice"
    shutil.rmtree(p, ignore_errors=True)
    b = cli("data", "build", "--from", "live.helpers.packs:images_4k", "--out", p, guest=A)
    check(b.returncode == 0, tail(b.stderr))
    pu = cli("data", "push", p, "--name", ds, guest=A, timeout=600)
    check(pu.returncode == 0, tail(pu.stderr))
    check(ds in cli("data", "ls", guest=A).stdout, "alice cannot see her dataset")
    bl = cli("data", "ls", guest=B).stdout
    check(ds not in bl, f"bob sees alice's dataset:\n{bl}")
    rb = cli("data", "rm", ds, guest=B)
    check(rb.returncode != 0, "bob deleted alice's dataset")
    r2 = gpy(B, f"""
        import torch, reflop
        d = reflop.device()
        names = [s["name"] for s in reflop.snapshots()]
        print("BOB_SNAPS", names)
        try:
            reflop.load({snap!r}); print("BOB_LOADED")
        except Exception as e:
            print("BOB_LOAD_ERR", type(e).__name__)
        try:
            reflop.dataset({ds!r}).batch(torch.tensor([0]))["labels"].cpu(); print("BOB_DS")
        except Exception as e:
            print("BOB_DS_ERR", type(e).__name__)
        try:
            reflop.delete_snapshot({snap!r}); print("BOB_DELETED")
        except Exception as e:
            print("BOB_DEL_ERR", type(e).__name__)
    """)
    o = r2.stdout
    check(snap not in o.split("BOB_SNAPS", 1)[1].splitlines()[0], f"bob lists alice's snapshot: {o}")
    check("BOB_LOADED" not in o and "BOB_DS\n" not in o and "BOB_DELETED" not in o, f"cross-identity access: {o}")
    r3 = gpy(A, f"""
        import reflop
        reflop.device()
        print("STILL", {snap!r} in [s["name"] for s in reflop.snapshots()])
        reflop.delete_snapshot({snap!r})
    """)
    check("STILL True" in r3.stdout, f"alice's snapshot vanished: {r3.stdout}")
    cli("data", "rm", ds, guest=A)
    return "bob cannot list, load, use or delete alice's datasets and snapshots"


def concurrent_identities():
    la = LineReader(popen_helper("train_loop", "--mode", "train", "--steps", 80, guest=A))
    lb = LineReader(popen_helper("train_loop", "--mode", "train", "--steps", 80, guest=B))
    ra, rb = la.finish(600), lb.finish(600)
    check(ra == 0 and rb == 0, f"rc {ra}/{rb}: {la.stdout_text()[-200:]} {lb.stdout_text()[-200:]}")
    fa = [j["loss"] for j in la.json() if "loss" in j]
    fb = [j["loss"] for j in lb.json() if "loss" in j]
    check(fa == fb, "same seed, different losses for two identities")
    return f"both finished 80 steps with identical losses ({fa[0]:.3f}->{fa[-1]:.3f})"


# ----------------------------------------------------------------------------- quotas

def q_device_memory():
    _o = owner.quota_set(A, device_memory="1GiB")
    r = gpy(A, """
        import torch, reflop
        d = reflop.device()
        t = torch.empty(128 * 2**18, device=d).fill_(1); print("SMALL_OK", t[0].item())
        try:
            u = torch.empty(2048 * 2**18, device=d).fill_(1); u[0].item(); print("BIG_OK")
        except Exception as e:
            print("BIG_ERR", type(e).__name__, isinstance(e, torch.OutOfMemoryError), str(e)[:200])
        print("HEALTHY", (torch.ones(3).to(d) * 2).sum().item())
    """)
    owner.quota_clear(A)
    o2 = r.stdout
    check("SMALL_OK" in o2 and "BIG_ERR" in o2 and "HEALTHY 6.0" in o2, f"{o2}\n{tail(r.stderr)}")
    return next(l for l in o2.splitlines() if l.startswith("BIG_ERR"))[:220]


def q_max_sessions():
    owner.quota_set(A, max_sessions=1)
    hold = LineReader(popen_helper("train_loop", "--mode", "idle", "--seconds", 25, guest=A))
    hold.wait_json(lambda j: "sid" in j, 60)
    r = gpy(A, """
        import torch, reflop
        try:
            d = reflop.device(); print("SECOND_OK", torch.ones(1).to(d).item())
        except Exception as e:
            print("SECOND_ERR", type(e).__name__, str(e)[:200])
    """, timeout=120)
    hold.finish(120)
    time.sleep(2)
    r2 = gpy(A, "import torch, reflop\nd = reflop.device(); print('AFTER_OK', torch.ones(1).to(d).item())\n")
    owner.quota_clear(A)
    check("SECOND_ERR" in r.stdout, f"second session allowed with max-sessions 1: {r.stdout} {tail(r.stderr)}")
    check("AFTER_OK" in r2.stdout, f"session refused after the first closed: {r2.stdout} {tail(r2.stderr)}")
    return next(l for l in r.stdout.splitlines() if "SECOND_ERR" in l)[:220]


def q_execute_wall():
    o = owner.quota_set(A, execute_wall_ms=1000)
    r = gpy(A, """
        import time, torch, reflop
        d = reflop.device()
        x = torch.randn(8192, 8192, device=d)
        t = time.time()
        try:
            for _ in range(150):
                x = (x @ x).tanh()
            x[0, 0].item(); print("LONG_OK", time.time() - t)
        except Exception as e:
            print("LONG_ERR", type(e).__name__, round(time.time() - t, 1), str(e)[:200])
    """, timeout=300)
    r2 = gpy(A, "import torch, reflop\nd = reflop.device(); print('NEW_OK', torch.ones(1).to(d).item())\n")
    owner.quota_clear(A)
    check("LONG_ERR" in r.stdout, f"a >1s execute ran under execute-wall-ms 1000 ({o.strip()}): {r.stdout} {tail(r.stderr)}")
    check("NEW_OK" in r2.stdout, "new session refused after the kill")
    return next(l for l in r.stdout.splitlines() if "LONG_ERR" in l)[:220]


def q_execute_flops():
    o = owner.quota_set(A, execute_flops="1000000000")
    r = gpy(A, """
        import torch, reflop
        d = reflop.device()
        print("SMALL", (torch.randn(64, 64, device=d) @ torch.randn(64, 64, device=d)).sum().item() is not None)
        try:
            x = torch.randn(4096, 4096, device=d); (x @ x).sum().item(); print("BIG_OK")
        except Exception as e:
            print("BIG_ERR", type(e).__name__, str(e)[:200])
    """)
    owner.quota_clear(A)
    check("BIG_ERR" in r.stdout and "SMALL True" in r.stdout, f"{o.strip()}: {r.stdout} {tail(r.stderr)}")
    return next(l for l in r.stdout.splitlines() if "BIG_ERR" in l)[:220]


def q_storage():
    o = owner.quota_set(A, storage="20MiB")
    p = SCRATCH / "packs_alice_q"
    shutil.rmtree(p, ignore_errors=True)
    cli("data", "build", "--from", "live.helpers.packs:images_4k", "--out", p, guest=A)   # ~12 MiB
    p2 = SCRATCH / "packs_alice_q2"
    shutil.rmtree(p2, ignore_errors=True)
    cli("data", "build", "--from", "live.helpers.packs:images_50k", "--out", p2, guest=A)  # ~147 MiB
    small = cli("data", "push", p, "--name", f"{PREFIX}-q-small", guest=A, timeout=600)
    big = cli("data", "push", p2, "--name", f"{PREFIX}-q-big", guest=A, timeout=600)
    r = gpy(A, f"""
        import torch, reflop
        d = reflop.device()
        try:
            reflop.save("{PREFIX}-q-snap", {{"t": torch.zeros(16 * 2**18, device=d)}}); print("SNAP_OK")
        except Exception as e:
            print("SNAP_ERR", type(e).__name__, str(e)[:200])
    """)
    owner.quota_clear(A)
    cli("data", "rm", f"{PREFIX}-q-small", guest=A)
    cli("data", "rm", f"{PREFIX}-q-big", guest=A)
    gpy(A, f"import reflop\nreflop.device()\ntry:\n    reflop.delete_snapshot('{PREFIX}-q-snap')\nexcept Exception: pass\n")
    check(small.returncode == 0, f"12 MiB push refused under 20 MiB: {tail(small.stderr)}")
    check(big.returncode != 0, f"147 MiB push accepted under a 20 MiB quota ({o.strip()})")
    check("SNAP_ERR" in r.stdout, f"16 MiB snapshot accepted on top of 12 MiB under 20 MiB: {r.stdout}")
    return f"big push: {tail(big.stderr, 150).strip()!r}; snapshot: {r.stdout.strip()[-150:]!r}"


def q_upload_bandwidth(s):
    p = SCRATCH / "packs_alice_bw"
    shutil.rmtree(p, ignore_errors=True)
    cli("data", "build", "--from", "live.helpers.packs:images_4k", "--out", p, guest=A)
    size = sum(f.stat().st_size for f in p.iterdir()) / 2**20
    t = time.time()
    r0 = cli("data", "push", p, "--name", f"{PREFIX}-bw0", guest=A, timeout=600)
    t0 = time.time() - t
    cli("data", "rm", f"{PREFIX}-bw0", guest=A)
    owner.quota_set(A, upload_bandwidth="1MiB")
    t = time.time()
    r1 = cli("data", "push", p, "--name", f"{PREFIX}-bw1", guest=A, timeout=900)
    t1 = time.time() - t
    owner.quota_clear(A)
    cli("data", "rm", f"{PREFIX}-bw1", guest=A)
    s.metric("upload_bandwidth_quota", mib=size, unlimited_s=t0, limited_1mib_s=t1, limited_rate=size / t1)
    check(r0.returncode == 0 and r1.returncode == 0, f"{tail(r0.stderr)} {tail(r1.stderr)}")
    check(t1 >= size * 0.8, f"{size:.1f} MiB at 1 MiB/s took {t1:.1f}s")
    return f"{size:.1f} MiB: {t0:.1f}s unlimited, {t1:.1f}s at 1 MiB/s"


def quota_show():
    owner.quota_set(A, device_memory="6GiB", max_sessions=2)
    t = owner.out("quota", "show")
    owner.quota_clear(A)
    t2 = owner.out("quota", "show")
    check(A in t and A not in t2, f"show:\n{t}\nafter clear:\n{t2}")
    r = owner.out("quota", "set", A, "--device-memory", "0")
    check("unlimited" in r.lower() or "error" in r.lower() or "invalid" in r.lower(), f"quota 0 accepted? {r}")
    owner.quota_clear(A)
    return f"listed then cleared; 0 rejected: {r.strip()[:120]!r}"


# ----------------------------------------------------------------------------- unpairing

def unpair_guest_first():
    r = cli("unpair", "--yes", guest=A, timeout=120)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    u = owner.users()
    line = next((l for l in u.splitlines() if l.startswith(A)), "")
    check("unpaired" in line.lower(), f"owner does not show {A} as unpaired by guest: {line!r}")
    _st = owner.status_text()
    st2 = cli("--server", "e2e", "status", guest=A)   # plain `status` lists (zero) pairings and exits 0 in 0.4.x
    check(st2.returncode != 0, f"status still works after unpair: {tail(st2.stdout + st2.stderr, 300)}")
    owner.unpair(A)
    keychain_delete(A)
    r2, _ = pair(A)
    check(r2.returncode == 0, f"pair again after both sides unpaired: {tail(r2.stdout + r2.stderr)}")
    return f"users: {line.strip()!r}; paired again"


def unpair_owner_first():
    owner.unpair(B)
    r = cli("--server", "e2e", "status", guest=B)
    o = r.stdout + r.stderr
    check(r.returncode != 0 and "reflop unpair" in o, f"message after the Owner unpaired: {o[-300:]}")
    r2 = cli("unpair", "--yes", guest=B, timeout=120)
    check(r2.returncode == 0, tail(r2.stdout + r2.stderr))
    keychain_delete(B)
    r3, _ = pair(B)
    check(r3.returncode == 0, f"pair again: {tail(r3.stdout + r3.stderr)}")
    return f"Guest told: {o.strip().splitlines()[-1][:160]!r}; paired again"


def cleanup_all():
    for n in (A, B, C):
        with contextlib.suppress(Exception):
            owner.quota_clear(n)
        with contextlib.suppress(Exception):
            owner.unpair(n)
        wipe_local(n)
    left = [n for n in (A, B, C) if re.search(rf"^{n}\s", owner.users(), re.M)]
    check(not left, f"identities left on the Owner: {left}")
    return "e2e identities unpaired on both sides, homes and Keychain items deleted"


def main():
    s = Suite("a13_identities_quotas", __doc__.splitlines()[0])
    if not owner.available():
        s.case("owner_access", owner.need)
        s.finish()
    owner.save_initial_state()
    GUESTS.mkdir(exist_ok=True)
    try:
        r = s.case("pair_alice_bob", setup_pair_all, timeout=600)
        if r["status"] == "pass":
            s.case("invite_single_use", invite_single_use, timeout=300)
            s.case("pair_while_paired_keeps_invite", pair_while_paired_keeps_invite, timeout=300)
            s.case("expired_invite_message", expired_invite, timeout=120)
            s.case("isolation_datasets_snapshots", isolation, timeout=900)
            s.case("concurrent_identities", concurrent_identities, timeout=900)
            s.case("quota_device_memory", q_device_memory, timeout=600)
            s.case("quota_max_sessions", q_max_sessions, timeout=600)
            s.case("quota_execute_wall_ms", q_execute_wall, timeout=600)
            s.case("quota_execute_flops", q_execute_flops, timeout=600)
            s.case("quota_storage", q_storage, timeout=1200)
            s.case("quota_upload_bandwidth", q_upload_bandwidth, s, timeout=1200)
            s.case("quota_show_clear_zero", quota_show, timeout=300)
            s.case("unpair_guest_first", unpair_guest_first, timeout=600)
            s.case("unpair_owner_first", unpair_owner_first, timeout=600)
    finally:
        s.case("cleanup", cleanup_all, timeout=600)
    s.finish()


if __name__ == "__main__":
    main()
