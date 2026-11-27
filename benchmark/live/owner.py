# SPDX-License-Identifier: Apache-2.0
"""The Owner machine: reflopd commands, worker kills and native CUDA runs.

Guards (agreed with the human):
  * anything that pauses or restarts Guest work first waits until no *other* identity (not `guest`,
    not `e2e-*`) has a live session, and is skipped if that does not happen in time;
  * kills only target our own sessions or e2e-* identities;
  * every change is undone in a `finally` (resume, yield --off, checkpoints back to the initial
    setting, quota clear, unpair e2e-*); `python -m live.owner cleanup` does all of it idempotently.

Configuration (nothing touches an Owner unless one of these is set):
  E2E_OWNER_SSH=<host>   drive the Owner over `ssh <host>` (an ~/.ssh/config alias works)
  E2E_LOCAL=1            the Owner is a throwaway reflopd on this machine, whose home is E2E_OWNER_HOME;
                         refused for the default home (~/.local/share/reflop), so a live daemon is never touched
  E2E_GUEST_IDENTITY     the default Guest's identity name on the Owner (default "guest")
  E2E_NO_OWNER=1         never touch the Owner (Owner suites skip)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

from live.harness import LOGS, PREFIX, Skip

HOST = os.environ.get("E2E_OWNER_SSH", "")
LOCAL = os.environ.get("E2E_LOCAL") == "1"
OURS = (os.environ.get("E2E_GUEST_IDENTITY", "guest"),)   # the default Guest's identity on the Owner
PRE = ('export PATH="$HOME/.local/bin:$PATH" XDG_RUNTIME_DIR="/run/user/$(id -u)"; '
       'CG="${E2E_OWNER_CGROUP:-/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/'
       'reflopd.service}"; RH="${REFLOP_HOME:-$HOME/.local/share/reflop}"; ')
STATE_FILE = LOGS / "owner_initial_state.json"
LOG = LOGS / "owner_commands.log"

_avail: bool | None = None


def guest_of(identity: str) -> str:
    """The Guest an identity belongs to: alice for alice@gpu-<uuid> (one identity per GPU since reflopd 0.2.10)."""
    return identity.split("@", 1)[0]


def is_ours(identity: str) -> bool:
    """The default Guest's or a throwaway e2e-* identity, on any GPU."""
    g = guest_of(identity)
    return g in OURS or g.startswith(PREFIX + "-")


def local_home() -> str:
    """E2E_LOCAL's Owner home. Refuses the default reflopd home: local mode is for a throwaway daemon only."""
    home = os.environ.get("E2E_OWNER_HOME", "")
    default = os.path.realpath(os.path.expanduser("~/.local/share/reflop"))
    if not home or os.path.realpath(os.path.expanduser(home)) == default:
        raise RuntimeError("E2E_LOCAL=1 needs E2E_OWNER_HOME set to a throwaway reflopd home (not "
                           "~/.local/share/reflop); see benchmark/README.md")
    return os.path.realpath(os.path.expanduser(home))


def _argv(cmd: str) -> list[str]:
    if LOCAL:
        return ["bash", "-c", f"export REFLOP_HOME={shlex.quote(local_home())}; " + PRE + cmd]
    if not HOST:
        raise Skip("no Owner configured (set E2E_OWNER_SSH, or E2E_LOCAL=1 with E2E_OWNER_HOME)")
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", HOST, PRE + cmd]


def where() -> str:
    return f"local reflopd ({os.environ.get('E2E_OWNER_HOME')})" if LOCAL else f"ssh {HOST or '(none)'}"


def _log(cmd: str, r: subprocess.CompletedProcess | None, dt: float):
    with open(LOG, "a") as f:
        f.write(f"--- {time.strftime('%H:%M:%S')} ({dt:.2f}s) $ {cmd}\n")
        if r is not None:
            f.write(f"rc={r.returncode}\n{r.stdout}{r.stderr}\n")


def run(cmd: str, timeout: float = 120, check: bool = False) -> subprocess.CompletedProcess:
    if os.environ.get("E2E_NO_OWNER") == "1":
        raise Skip("E2E_NO_OWNER=1")
    t = time.time()
    try:
        r = subprocess.run(_argv(cmd), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        _log(cmd, None, time.time() - t)
        raise
    _log(cmd, r, time.time() - t)
    if check and r.returncode != 0:
        raise RuntimeError(f"owner: `{cmd}` exited {r.returncode}: {(r.stdout + r.stderr)[-800:]}")
    return r


def available() -> bool:
    global _avail
    if os.environ.get("E2E_NO_OWNER") == "1" or not (LOCAL or HOST):
        return False
    if _avail is None:
        try:
            _avail = run("reflopd version", timeout=30).returncode == 0
        except Exception:  # noqa: BLE001
            _avail = False
    return _avail


def need():
    if not available():
        raise Skip(f"no Owner access ({where()})")


def reflopd(*args, timeout=120, check=False) -> subprocess.CompletedProcess:
    return run("reflopd " + " ".join(shlex.quote(str(a)) for a in args), timeout=timeout, check=check)


def out(*args, **kw) -> str:
    r = reflopd(*args, **kw)
    return r.stdout + r.stderr


# ----------------------------------------------------------------------------- status parsing

def status_text() -> str:
    return out("status")


def sessions() -> list[dict]:
    """Parse the session table of `reflopd status`."""
    txt = status_text()
    lines = txt.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("SESSION")), None)
    if hdr is None:
        return []
    h = lines[hdr]
    cols = [(m.group(0), m.start()) for m in re.finditer(r"\S+", h)]
    rows = []
    for l in lines[hdr + 1:]:
        if not l.strip():
            break
        row = {}
        for j, (name, start) in enumerate(cols):
            end = cols[j + 1][1] if j + 1 < len(cols) else None
            row[name.lower()] = l[start:end].strip() if end else l[start:].strip()
        rows.append(row)
    return rows


def foreign_sessions() -> list[dict]:
    return [s for s in sessions() if not is_ours(s.get("identity", ""))]


def wait_quiet(max_wait: float = 600, poll: float = 15) -> None:
    """Wait until no other identity has a session; Skip if that does not happen in max_wait."""
    need()
    end = time.time() + max_wait
    while True:
        fs = foreign_sessions()
        if not fs:
            return
        if time.time() > end:
            raise Skip("other Guests are using the server: " + ", ".join(sorted({s['identity'] for s in fs})))
        print(f"owner: waiting for other Guests' sessions to end: {[s['identity'] for s in fs]}", file=sys.stderr,
              flush=True)
        time.sleep(poll)


def our_sessions(identity: str | None = None) -> list[dict]:
    return [s for s in sessions() if guest_of(s.get("identity", "")) == (identity or OURS[0])]


# ----------------------------------------------------------------------------- controls

def pause(now: bool = False) -> str:
    return out("pause", *(["--now"] if now else []), timeout=180)


def resume() -> str:
    return out("resume", timeout=180)


@contextlib.contextmanager
def paused(now: bool = False, quiet_wait: float = 600):
    wait_quiet(quiet_wait)
    try:
        yield pause(now)
    finally:
        resume()


def say(msg: str) -> str:
    return out("say", msg)


def kill(target: str) -> str:
    check_target(target)
    return out("kill", target, timeout=60)


def check_target(target: str):
    """Only our own identities, or a session id prefix that belongs to one of them."""
    if is_ours(target):
        return
    ss = [s for s in sessions() if s.get("session", "").startswith(target[:8])]
    if not ss or any(not is_ours(s.get("identity", "")) for s in ss):
        raise RuntimeError(f"refusing to act on {target!r}: not one of our sessions")


def kill_worker(session_hex: str, sig: str = "KILL") -> str:
    """Kill the worker process of one of our sessions (a crash, as far as reflopd can tell)."""
    check_target(session_hex[:8])
    sid = re.sub(r"[^0-9a-f]", "", session_hex.lower())
    r = run(f'P=$(cat "$CG"/workers/{sid}*/cgroup.procs 2>/dev/null); echo "pids: $P"; '
            f'[ -n "$P" ] && kill -{sig} $P; echo rc=$?')
    return r.stdout + r.stderr


def worker_pids(session_hex: str) -> list[int]:
    sid = re.sub(r"[^0-9a-f]", "", session_hex.lower())
    r = run(f'cat "$CG"/workers/{sid}*/cgroup.procs 2>/dev/null')
    return [int(x) for x in r.stdout.split()]


def checkpoints_state() -> dict:
    t = out("checkpoints")
    m = re.search(r"interval[^0-9]*([0-9hms.]+)", t)
    return {"on": t.strip().startswith("checkpoints on"), "interval": m.group(1) if m else None, "text": t.strip()}


def checkpoints(on: bool, interval: str | None = None) -> str:
    args = ["checkpoints", "on" if on else "off"]
    if on and interval:
        args += ["--interval", interval]
    return out(*args)


def yield_mode(mode: str) -> str:
    """mode: 'auto', 'auto-suspend' or 'off'."""
    args = {"auto": ["--auto"], "auto-suspend": ["--auto", "--suspend"], "off": ["--off"]}[mode]
    return out("yield", *args)


def quota_set(name: str, **kw) -> str:
    check_target(name)
    args = ["quota", "set", name]
    for k, v in kw.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    return out(*args)


def quota_clear(name: str) -> str:
    check_target(name)
    return out("quota", "clear", name)


def invite(name: str, expires: str = "2h") -> str:
    if not name.startswith(PREFIX + "-"):
        raise RuntimeError("test identities must start with e2e-")
    t = out("invite", "--name", name, "--expires", expires)
    m = re.search(r"reflop1-[A-Za-z0-9_\-.~+=/]+", t)
    if not m:
        raise RuntimeError(f"no invite code in: {t[-600:]}")
    return m.group(0)


def users() -> str:
    return out("users")


def _identity_gpus(name: str) -> list[str]:
    """GPU UUIDs `name` is paired with (reflopd >= 0.2.10 scopes invites to a GPU); [""] for a host-wide pairing."""
    gpus = [m.group(1) or "" for m in re.finditer(rf"^{re.escape(name)}\s+(gpu-[0-9a-f-]+)?", users(), re.M)]
    return gpus or [""]


def unpair(name: str) -> str:
    if not name.startswith(PREFIX + "-"):
        raise RuntimeError("refusing to unpair a non-test identity")
    res = []
    for gpu in _identity_gpus(name):
        res.append(out("unpair", name, *(["--gpu", gpu] if gpu else []), "--yes", timeout=120))
    return "\n".join(res)


def usage_json(name: str | None = None) -> dict:
    t = reflopd("usage", *([name] if name else []), "--json").stdout
    return json.loads(t)


def guest_usage(identity: str | None = None) -> dict:
    """Usage of one identity: `identity` exactly, else the default Guest's (whichever GPU it is paired with)."""
    ids = usage_json()["identities"]
    hits = [v for k, v in ids.items() if (k == identity if identity else guest_of(k) == OURS[0])]
    if len(hits) != 1:
        raise RuntimeError(f"{len(hits)} usage entries for {identity or OURS[0]}: {sorted(ids)}")
    return hits[0]


def audit(name: str | None = None, since: str = "1h", limit: int = 500) -> str:
    args = ["audit", "--since", since, "--limit", str(limit)]
    if name:
        args += ["--name", name]
    return out(*args, timeout=60)


# ----------------------------------------------------------------------------- native runs on the Owner

def runtime_python() -> str:
    r = run('ls -d "$RH"/runtime/cu*/python/bin/python3 2>/dev/null | head -1')
    p = r.stdout.strip()
    if not p:
        raise Skip("no CUDA runtime python on the Owner")
    return p


def push(local: str, remote_dir: str = "/tmp/reflop-e2e") -> str:
    run(f"mkdir -p {remote_dir}")
    if LOCAL:
        r = subprocess.run(["cp", local, remote_dir + "/"], capture_output=True, text=True, timeout=120)
    else:
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", local, f"{HOST}:{remote_dir}/"], capture_output=True,
                           text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(f"scp failed: {r.stderr}")
    return f"{remote_dir}/{os.path.basename(local)}"


def popen(cmd: str) -> subprocess.Popen:
    """Start a long Owner command; its stdout/stderr come back through ssh."""
    return subprocess.Popen(_argv(cmd), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL)


# ----------------------------------------------------------------------------- state save / restore

def save_initial_state():
    if not available() or STATE_FILE.exists():
        return
    st = {"checkpoints": checkpoints_state(), "users": users(), "time": time.time()}
    STATE_FILE.write_text(json.dumps(st, indent=1))


def cleanup(verbose=True):
    """Undo everything this suite may have changed on the Owner. Idempotent."""
    if not available():
        return
    log = []
    log.append(resume())
    log.append(yield_mode("off"))
    want_on, interval = False, None
    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text())["checkpoints"]
        want_on, interval = st["on"], st.get("interval")
    cur = checkpoints_state()
    if cur["on"] != want_on:
        log.append(checkpoints(want_on, interval))
    u = users()
    for name in sorted(set(re.findall(rf"^({PREFIX}-[a-z0-9_-]+)\s", u, re.M))):
        with contextlib.suppress(Exception):
            log.append(quota_clear(name))
        with contextlib.suppress(Exception):
            log.append(unpair(name))
    if verbose:
        print("owner cleanup:\n" + "\n".join(x.strip() for x in log if x and x.strip()), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "cleanup":
        cleanup()
    elif cmd == "save-state":
        save_initial_state()
    elif cmd == "status":
        print(status_text())
        print(json.dumps(sessions(), indent=1))
    else:
        print(out(*sys.argv[1:]))
