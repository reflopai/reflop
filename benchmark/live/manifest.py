# SPDX-License-Identifier: Apache-2.0
"""results/<run>/run.json: what was tested (versions, server, settings), which suites ran, exit codes, timings and
case counts. Machine-readable only; run_full.sh calls it.

    python -m live.manifest init [suite ...]
    python -m live.manifest suite <name> <exit-code> <seconds>
    python -m live.manifest finish <completed|interrupted|failed>
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

from live.harness import QUICK, RESULTS, RUN_ID, SRC, cli, server_name

FILE = RESULTS / "run.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load() -> dict:
    return json.loads(FILE.read_text())


def _save(d: dict):
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(FILE)


def owner_label() -> str | None:
    """How the Owner is reached (an ssh host or a local throwaway home); the report's sanitizer drops it."""
    from live import owner

    return owner.where() if (owner.LOCAL or owner.HOST) else None


def _owner_status() -> str | None:
    if os.environ.get("E2E_NO_OWNER") == "1":
        return None
    from live import owner

    with contextlib.suppress(Exception):
        if owner.available():
            return owner.status_text()
    return None


def _sh(cmd: list[str], timeout: float = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout + r.stderr
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def owner_gpu() -> dict | None:
    """Read-only snapshot of the Owner's GPU: memory in use and every compute process (reflop workers run the reflop
    runtime Python; anything else, e.g. the Owner's own training, competes with the benchmark)."""
    if os.environ.get("E2E_NO_OWNER") == "1":
        return None
    from live import owner

    try:
        if not owner.available():
            return None
        r = owner.run("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,"
                      "clocks.sm,power.draw --format=csv,noheader,nounits; echo ---; "
                      "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do "
                      "m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | "
                      "awk -F', ' -v p=$p '$1==p{print $2}'); "
                      "echo \"$p|$m|$(ps -o user= -p $p)|$(ps -o args= -p $p | cut -c1-160)\"; done", timeout=60)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    head, _, procs = r.stdout.partition("---")
    out: dict = {"time": _now(), "gpus": [], "processes": []}
    for line in head.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 7:
            num = lambda x: float(x) if re.fullmatch(r"[\d.]+", x) else None  # noqa: E731
            out["gpus"].append({"name": f[0], "memory_used_mib": num(f[1]), "memory_total_mib": num(f[2]),
                                "utilization_pct": num(f[3]), "temperature_c": num(f[4]), "sm_clock_mhz": num(f[5]),
                                "power_w": num(f[6])})
    for line in procs.strip().splitlines():
        pid, mem, user, args = (line.split("|", 3) + ["", "", ""])[:4]
        if pid.strip().isdigit():
            out["processes"].append({"pid": int(pid), "used_memory_mib": float(mem) if mem.strip() else None,
                                     "user": user.strip(), "cmd": args.strip(),
                                     "reflop_worker": bool(re.search(r"/runtime/cu[^/]*/python/", args))})
    out["foreign_gpu_memory_mib"] = sum(p["used_memory_mib"] or 0 for p in out["processes"] if not p["reflop_worker"])
    return out


def network(server: str) -> dict:
    """Path to the server (latency changes between runs dominate small-step timings): ICMP ping, Tailscale path."""
    from reflop.config import ClientConfig

    out: dict = {"time": _now()}
    try:
        addr = ClientConfig.load().servers[server].addresses[0]
    except Exception as e:  # noqa: BLE001
        return {**out, "error": f"no address for {server!r}: {e}"}
    host = addr.rsplit(":", 1)[0].strip("[]")
    out["target"] = addr
    t = _sh(["ping", "-c", "20", "-q", host], timeout=60)
    m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", t)
    loss = re.search(r"([\d.]+)% packet loss", t)
    out["ping_ms"] = ({"n": 20, "min": float(m[1]), "avg": float(m[2]), "max": float(m[3]), "stddev": float(m[4]),
                       "loss_pct": float(loss[1]) if loss else None} if m else {"raw": t.strip()[-300:]})
    r = _sh(["route", "-n", "get", host]) if platform.system() == "Darwin" else _sh(["ip", "route", "get", host])
    m = re.search(r"interface: (\S+)", r) or re.search(r" dev (\S+)", r)
    out["interface"] = m[1] if m else None
    ts = shutil.which("tailscale")
    if ts and host.startswith("100."):
        pong = _sh([ts, "ping", "-c", "1", "--timeout", "5s", host]).strip()
        out["tailscale_ping"] = pong.splitlines()[-1] if pong else None
        with contextlib.suppress(Exception):
            st = json.loads(_sh([ts, "status", "--json"]))
            peer = next(v for v in st.get("Peer", {}).values() if host in v.get("TailscaleIPs", []))
            cur = peer.get("CurAddr") or ""
            out["tailscale"] = {"peer": peer.get("HostName"), "path": "direct" if cur else "relay",
                                "cur_addr": cur or None, "relay": peer.get("Relay"), "online": peer.get("Online")}
            if cur:   # the physical link the tunnel runs over (Wi-Fi vs Ethernet)
                ip = cur.rsplit(":", 1)[0].strip("[]")
                r = _sh(["route", "-n", "get", ip]) if platform.system() == "Darwin" else _sh(["ip", "route", "get", ip])
                m = re.search(r"interface: (\S+)", r) or re.search(r" dev (\S+)", r)
                out["tailscale"]["underlay_interface"] = m[1] if m else None
    return out


def init(selected: list[str]):
    import reflop
    import torch

    server = server_name()
    st = cli("--server", server, "status", timeout=60)
    status = st.stdout + st.stderr
    owner_status = _owner_status()
    m = re.search(r"reflopd v?(\d[0-9A-Za-z.+-]*)", owner_status or "") or re.search(r"reflopd v?(\d[0-9A-Za-z.+-]*)", status)
    src = subprocess.run(["git", "-C", str(SRC), "describe", "--tags", "--always"], capture_output=True,
                         text=True).stdout.strip()
    _save({
        "run_id": RUN_ID,
        "state": "running",
        "started": _now(),
        "finished": None,
        "seconds": None,
        "versions": {"reflop_client": reflop.__version__, "reflopd": m.group(1) if m else None,
                     "reflop_src": src or None, "torch": torch.__version__, "python": sys.version.split()[0],
                     "platform": platform.platform()},
        "settings": {"server": server, "quick": QUICK, "no_owner": os.environ.get("E2E_NO_OWNER") == "1",
                     "owner": owner_label(),
                     "soak_minutes": os.environ.get("E2E_SOAK_MINUTES"), "suites_selected": selected},
        "network_start": network(server),
        "network_end": None,
        "owner_gpu_start": owner_gpu(),
        "owner_gpu_end": None,
        "reflop_status": status,
        "reflopd_status_start": owner_status,
        "reflopd_status_end": None,
        "suites": [],
        "totals": {},
    })


def suite(name: str, rc: int, seconds: float):
    d = _load()
    row = {"suite": name, "exit_code": rc, "seconds": seconds, "result_file": None, "counts": None,
           "n_metrics": None}
    f = RESULTS / f"{name}.json"
    if f.exists():
        with contextlib.suppress(ValueError):
            j = json.loads(f.read_text())
            row["result_file"] = f.name
            row["counts"] = {s: sum(c["status"] == s for c in j["cases"]) for s in ("pass", "fail", "xfail", "skip")}
            row["n_metrics"] = len(j["metrics"])
    d["suites"].append(row)
    tot: dict = {"pass": 0, "fail": 0, "xfail": 0, "skip": 0}
    for r in d["suites"]:
        for k, v in (r["counts"] or {}).items():
            tot[k] += v
    tot["suites_run"] = len(d["suites"])
    tot["suites_nonzero_exit"] = sum(r["exit_code"] != 0 for r in d["suites"])
    d["totals"] = tot
    _save(d)


def finish(state: str):
    d = _load()
    d["state"] = state
    d["finished"] = _now()
    with contextlib.suppress(Exception):
        t0 = time.mktime(time.strptime(d["started"], "%Y-%m-%dT%H:%M:%S%z"))
        d["seconds"] = round(time.time() - t0)
    d["reflopd_status_end"] = _owner_status()
    with contextlib.suppress(Exception):
        d["network_end"] = network(d["settings"]["server"])
    d["owner_gpu_end"] = owner_gpu()
    _save(d)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "init":
        init(args)
    elif cmd == "suite":
        suite(args[0], int(args[1]), float(args[2]))
    elif cmd == "finish":
        finish(args[0])
    else:
        sys.exit(f"unknown command {cmd}")
