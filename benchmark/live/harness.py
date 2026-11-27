# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the reFLOP end-to-end suite and benchmarks.

Every suite is a module `live.suites.<name>` with a `main()` that builds a `Suite`, runs cases and
benchmarks, and calls `finish()`. Results go to benchmark/results/<RUN_ID>/<suite>.json; logs to
benchmark/results/<RUN_ID>/logs/. RUN_ID comes from $E2E_RUN_ID (set by run_full.sh) or defaults to "adhoc".

Environment knobs:
  E2E_RUN_ID     results directory name
  E2E_RESULTS    results root (default benchmark/results)
  E2E_SRC        reFLOP source tree for the C API suite (default: this checkout; run_full.sh sets it to the
                 release's source when the installed client is a release)
  E2E_SCRATCH    directory for temporary files (run_full.sh uses a temp dir it deletes afterwards)
  E2E_QUICK=1    tiny matrices and short waits (smoke-testing the harness itself)
  E2E_NO_OWNER=1 never touch the Owner (owner-dependent cases are skipped)
  E2E_COLOR=1/0  force colored [PASS ]/[XFAIL]/[FAIL ] tags on or off (default: on for a terminal)
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # benchmark/
REPO = ROOT.parent                                      # the reFLOP checkout
E2E = ROOT / "live"
RUN_ID = os.environ.get("E2E_RUN_ID", "adhoc")
RESULTS = Path(os.environ.get("E2E_RESULTS") or ROOT / "results") / RUN_ID
LOGS = RESULTS / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(os.environ["E2E_SCRATCH"]) if os.environ.get("E2E_SCRATCH") else RESULTS / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
GUESTS = ROOT / "guests"
SRC = Path(os.environ.get("E2E_SRC") or REPO)
PY = sys.executable
REFLOP_EXE = str(Path(PY).parent / "reflop")
QUICK = os.environ.get("E2E_QUICK") == "1"
IS_MAC = platform.system() == "Darwin"
PREFIX = "e2e"          # names of snapshots, datasets, identities created by this suite
# Colored status tags: on when stdout is a terminal, or E2E_COLOR=1 (run_full.sh: its terminal); NO_COLOR turns off.
COLOR = not os.environ.get("NO_COLOR") and (os.environ.get("E2E_COLOR") == "1" or
                                            (os.environ.get("E2E_COLOR") is None and sys.stdout.isatty()))
_COLORS = {"pass": "32", "xfail": "33", "fail": "31"}   # green, yellow, red


class Skip(Exception):
    pass


class Expected(Exception):
    """A documented limitation was hit (recorded as 'xfail', not a failure)."""


def q(full, quick):
    """Pick the full or the quick value."""
    return quick if QUICK else full


# ----------------------------------------------------------------------------- suite runner

class Suite:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.cases: list[dict] = []
        self.metrics: list[dict] = []
        self.t0 = time.time()
        self.meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "quick": QUICK}
        print(f"== {name}: {description}", flush=True)

    def case(self, cid: str, fn, *a, timeout: float | None = None, **kw):
        """Run one case. fn returns a detail string/dict (pass), raises Skip/Expected/other (fail)."""
        t = time.time()
        extra = {}
        try:
            if timeout:
                detail = _with_timeout(fn, timeout, *a, **kw)
            else:
                detail = fn(*a, **kw)
            status = "pass"
        except Skip as e:
            status, detail = "skip", str(e)
        except Expected as e:
            status, detail = "xfail", str(e)
        except Exception as e:  # noqa: BLE001 - every failure is recorded with its traceback
            status, detail = "fail", f"{type(e).__name__}: {e}"
            extra["traceback"] = traceback.format_exc()[-6000:]
        if isinstance(detail, dict):
            extra["data"] = detail
            detail = detail.get("summary") or json.dumps({k: v for k, v in detail.items()}, default=str)[:400]
        rec = {"id": cid, "status": status, "detail": "" if detail is None else str(detail),
               "seconds": round(time.time() - t, 2), **extra}
        self.cases.append(rec)
        first = rec["detail"].splitlines()[0][:300] if rec["detail"] else ""
        tag = f"[{status.upper():5}]"
        if COLOR and status in _COLORS:
            tag = f"\033[{_COLORS[status]}m{tag}\033[0m"
        print(f"{tag} {self.name}/{cid} ({rec['seconds']} s): {first}", flush=True)
        if status == "fail" and "traceback" in rec:
            print(rec["traceback"], flush=True)
        self._save()
        return rec

    def metric(self, group: str, **values):
        """Record a benchmark measurement (one row of a table in the summary)."""
        row = {"group": group, **values}
        self.metrics.append(row)
        print(f"[METRIC] {self.name}/{group}: " + json.dumps(values, default=_jd)[:400], flush=True)
        self._save()
        return row

    def _save(self):
        out = {"suite": self.name, "description": self.description, "meta": self.meta, "cases": self.cases,
               "metrics": self.metrics, "seconds": round(time.time() - self.t0, 1)}
        tmp = RESULTS / f".{self.name}.json.tmp"
        tmp.write_text(json.dumps(out, indent=1, default=_jd))
        tmp.replace(RESULTS / f"{self.name}.json")

    def finish(self):
        self.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save()
        bad = [c for c in self.cases if c["status"] == "fail"]
        n = {s: sum(c["status"] == s for c in self.cases) for s in ("pass", "fail", "skip", "xfail")}
        print(f"== {self.name}: {n} in {round(time.time() - self.t0)} s", flush=True)
        sys.exit(1 if bad else 0)


def _with_timeout(fn, timeout, *a, **kw):
    box = {}

    def run():
        try:
            box["v"] = fn(*a, **kw)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"case did not finish within {timeout} s (hang?)")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _jd(o):
    try:
        import torch

        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, bytes):
        return o.hex()
    return repr(o)


# ----------------------------------------------------------------------------- assertions / numbers

def check(cond, msg="check failed"):
    if not cond:
        raise AssertionError(msg)


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def max_abs(a, b) -> float:
    import torch

    a = a.detach().cpu().double() if isinstance(a, torch.Tensor) else torch.tensor(a).double()
    b = b.detach().cpu().double() if isinstance(b, torch.Tensor) else torch.tensor(b).double()
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return (a - b).abs().max().item()


def close_to(remote, ref, rtol=1e-4, atol=1e-5, what=""):
    """Assert remote ≈ ref (torch.allclose semantics), return the max abs diff."""
    import torch

    r = remote.detach().cpu() if isinstance(remote, torch.Tensor) else torch.as_tensor(remote)
    e = ref.detach().cpu() if isinstance(ref, torch.Tensor) else torch.as_tensor(ref)
    check(tuple(r.shape) == tuple(e.shape), f"{what}: shape {tuple(r.shape)} != {tuple(e.shape)}")
    check(r.dtype == e.dtype, f"{what}: dtype {r.dtype} != {e.dtype}")
    if r.dtype in (torch.bool,) or not (r.is_floating_point() or r.is_complex()):
        check(torch.equal(r, e), f"{what}: values differ")
        return 0.0
    rr, ee = r.to(torch.complex128 if r.is_complex() else torch.float64), e.to(
        torch.complex128 if e.is_complex() else torch.float64)
    nan_ok = torch.equal(torch.isnan(rr), torch.isnan(ee))
    check(nan_ok, f"{what}: NaN pattern differs")
    fin = torch.isfinite(ee)
    check(torch.equal(rr[~fin & ~torch.isnan(ee)], ee[~fin & ~torch.isnan(ee)]), f"{what}: inf values differ")
    d = (rr - ee).abs()
    d = torch.where(fin, d, torch.zeros_like(d)) if d.numel() else d
    tol = atol + rtol * torch.where(fin, ee.abs(), torch.zeros_like(ee.abs()))
    worst = d.max().item() if d.numel() else 0.0
    check(bool((d <= tol).all()), f"{what}: max abs diff {worst:.3e} > atol {atol} + rtol {rtol}·|ref|")
    return worst


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize_times(ts: list[float]) -> dict:
    if not ts:
        return {}
    return {"n": len(ts), "mean": statistics.fmean(ts), "median": statistics.median(ts), "p10": pct(ts, 10),
            "p90": pct(ts, 90), "p99": pct(ts, 99), "min": min(ts), "max": max(ts),
            "stdev": statistics.pstdev(ts) if len(ts) > 1 else 0.0}


def rss_mib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# ----------------------------------------------------------------------------- reflop helpers (in-process)

def sessions():
    from reflop import session as s

    return s._sessions


def session_of(index=0):
    return sessions()[index]


def session_id_hex(index=0) -> str:
    return bytes(session_of(index).conn.session_id).hex()


def tensor_session_index(t) -> int:
    return t._rf_storage.session.device_index


def capture_events(index=0) -> list:
    """Replace the session's event printer with a recorder (also still prints). Returns the list."""
    from reflop._pb import compute_pb2 as cpb

    events: list = []
    conn = session_of(index).conn
    orig = conn.on_event

    def rec(ev):
        kind = cpb.Event.Kind.Name(ev.kind)
        events.append({"t": time.time(), "kind": kind, "text": ev.text})
        orig(ev)

    conn.on_event = rec
    return events


def stats(index=0) -> dict:
    import reflop

    return dict(reflop.stats().get(index, {}))


def stats_delta(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def reset_reflop():
    """Close every session (best effort) so the next reflop.device() opens fresh ones."""
    import reflop

    try:
        reflop.close()
    except Exception:  # noqa: BLE001
        for s in list(sessions().values()):
            with contextlib.suppress(Exception):
                s.conn.close()
        sessions().clear()
        from reflop import session as ss

        ss._default.clear()


def server_name() -> str:
    """The server under test: $REFLOP_SERVER, else the client's default pairing. Pass it as `--server` where the
    CLI would otherwise show the table of every pairing (`reflop status` with more than one paired)."""
    from reflop.config import ClientConfig

    return os.environ.get("REFLOP_SERVER") or ClientConfig.load().default


# ----------------------------------------------------------------------------- subprocesses

def guest_home(name: str | None) -> Path | None:
    return None if not name or name == "default" else GUESTS / name


def env_for(guest: str | None = None, **extra) -> dict:
    """Environment for a Guest subprocess. The default Guest keeps this process's REFLOP_CLIENT_HOME (a
    throwaway client home with E2E_LOCAL=1); a named Guest gets its own home under guests/."""
    env = dict(os.environ)
    h = guest_home(guest)
    if h:
        env["REFLOP_CLIENT_HOME"] = str(h)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    for k, v in extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def cli(*args, guest=None, env=None, timeout=600, input=None, cwd=None) -> subprocess.CompletedProcess:
    """Run the `reflop` CLI from this environment."""
    e = env or env_for(guest)
    return subprocess.run([REFLOP_EXE, *map(str, args)], capture_output=True, text=True, env=e, timeout=timeout,
                          input=input, cwd=cwd or ROOT, stdin=None if input is not None else subprocess.DEVNULL)


def py(args, guest=None, env=None, timeout=900, cwd=None) -> subprocess.CompletedProcess:
    e = env or env_for(guest)
    return subprocess.run([PY, "-u", *map(str, args)], capture_output=True, text=True, env=e, timeout=timeout,
                          cwd=cwd or ROOT, stdin=subprocess.DEVNULL)


def helper(name: str, *args, guest=None, env=None, timeout=900) -> subprocess.CompletedProcess:
    """Run e2e/helpers/<name>.py."""
    return py([E2E / "helpers" / f"{name}.py", *args], guest=guest, env=env, timeout=timeout)


def popen_helper(name: str, *args, guest=None, env=None, **kw) -> subprocess.Popen:
    e = env or env_for(guest)
    return subprocess.Popen([PY, "-u", str(E2E / "helpers" / f"{name}.py"), *map(str, args)], env=e, cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL,
                            **kw)


def json_lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
    return out


def last_json(text: str) -> dict | None:
    js = json_lines(text)
    return js[-1] if js else None


def tail(s: str | None, n=1500) -> str:
    return (s or "")[-n:]


class LineReader:
    """Collects a Popen's stdout/stderr lines in background threads, with timestamps."""

    def __init__(self, p: subprocess.Popen):
        self.p = p
        self.out: list[tuple[float, str]] = []
        self.err: list[tuple[float, str]] = []
        self._ts = [threading.Thread(target=self._pump, args=(p.stdout, self.out), daemon=True),
                    threading.Thread(target=self._pump, args=(p.stderr, self.err), daemon=True)]
        for t in self._ts:
            t.start()

    @staticmethod
    def _pump(f, dst):
        for line in iter(f.readline, ""):
            dst.append((time.time(), line.rstrip("\n")))

    def json(self) -> list[dict]:
        return json_lines("\n".join(l for _, l in self.out))

    def wait_json(self, pred, timeout: float) -> dict | None:
        end = time.time() + timeout
        while time.time() < end:
            for j in self.json():
                if pred(j):
                    return j
            if self.p.poll() is not None:
                time.sleep(0.3)
                for j in self.json():
                    if pred(j):
                        return j
                return None
            time.sleep(0.2)
        return None

    def finish(self, timeout: float = 120) -> int:
        try:
            rc = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = self.p.wait(10)
        for t in self._ts:
            t.join(5)
        return rc

    def stderr_text(self) -> str:
        return "\n".join(l for _, l in self.err)

    def stdout_text(self) -> str:
        return "\n".join(l for _, l in self.out)


def kill_tree(p: subprocess.Popen):
    with contextlib.suppress(Exception):
        p.send_signal(signal.SIGKILL)


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


def env_info() -> dict:
    import reflop
    import torch

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "reflop": reflop.__version__}
    with contextlib.suppress(Exception):
        info["src"] = subprocess.run(["git", "-C", str(SRC), "describe", "--tags", "--always"], capture_output=True,
                                     text=True).stdout.strip()
    return info
