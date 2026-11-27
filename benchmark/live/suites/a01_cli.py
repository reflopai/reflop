# SPDX-License-Identifier: Apache-2.0
"""a01 CLI and install: version, status, usage, verify, doctor (OpenMP), key, servers, init, update --check,
REFLOP_DEBUG connection log, error messages for a wrong server name."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from live import owner
from live.harness import SCRATCH, Suite, check, cli, env_for, env_info, server_name, strip_ansi, tail


def version():
    r = cli("version")
    check(r.returncode == 0, r.stderr)
    import reflop

    check(f"reflop {reflop.__version__}" in r.stdout, f"CLI and module disagree: {r.stdout} vs {reflop.__version__}")
    return r.stdout.strip()


def status():
    t = time.time()
    r = cli("--server", server_name(), "status")
    dt = time.time() - t
    check(r.returncode == 0, tail(r.stderr))
    o = r.stdout
    for needle in ("server:", "device:", "opset:", "runtime:", "usage:"):
        check(needle in o, f"`reflop status` lacks {needle!r}:\n{o}")
    caps = re.search(r"capabilities: (.*)", o).group(1)
    for c in ("usage", "snapshots", "checkpoints"):
        check(c in caps, f"capability {c} missing (needs reflopd 0.2.x): {caps}")
    m = re.search(r"reflopd (\S+)", o)
    check(m, f"`reflop status` does not name the reflopd version:\n{o}")
    (SCRATCH / "reflop_status.txt").write_text(o)
    n_ops = int(re.search(r"(\d+) operators enabled", o).group(1))
    return {"summary": f"reflopd {m.group(1)}, {n_ops} ops, caps: {caps}, {dt:.2f} s", "reflopd": m.group(1),
            "ops": n_ops, "seconds": dt, "status": o}


def usage_cli_and_api():
    r = cli("usage", "--json")
    check(r.returncode == 0, r.stderr)
    j = json.loads(r.stdout)
    for k in ("identity", "device_seconds", "executes", "flops", "sessions", "bytes_up", "bytes_down", "by_device"):
        check(k in j, f"usage --json lacks {k}")
    import reflop

    u = reflop.usage()
    check(u["identity"] == j["identity"], f"API identity {u['identity']} != CLI {j['identity']}")
    check(u["executes"] >= j["executes"], "API executes went down")
    r2 = cli("usage")
    check(r2.returncode == 0 and "device time" in r2.stdout, r2.stdout)
    return {"summary": f"identity {j['identity']}, {j['executes']} executes, {j['device_seconds']:.0f} s device time",
            "json": j}


def usage_grows_with_work():
    import reflop
    import torch

    before = reflop.usage()
    dev = reflop.device()
    x = torch.randn(2048, 2048).to(dev)
    for _ in range(10):
        x = (x @ x).tanh()
    x.sum().item()
    reflop.close()
    time.sleep(1)
    after = reflop.usage()
    d_ex = after["executes"] - before["executes"]
    d_fl = after["flops"] - before["flops"]
    check(d_ex >= 1, f"executes did not grow ({d_ex})")
    check(d_fl >= 10 * 2 * 2048 ** 3 * 0.9, f"FLOPs grew by {d_fl:.3g}, expected >= {10 * 2 * 2048 ** 3:.3g}")
    if owner.available():
        o = owner.guest_usage()
        check(abs(o["executes"] - after["executes"]) <= 50, f"Owner sees {o['executes']} executes, Guest {after['executes']}")
    return f"+{d_ex} executes, +{d_fl / 1e12:.3f} TFLOP, device +{after['device_seconds'] - before['device_seconds']:.3f} s"


def verify():
    r = cli("verify", timeout=600)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    return tail(r.stdout.strip(), 300)


def doctor():
    r = cli("doctor", timeout=300)
    o = strip_ansi(r.stdout + r.stderr)
    (SCRATCH / "reflop_doctor.txt").write_text(o)
    check(r.returncode == 0, f"doctor exit {r.returncode}:\n{tail(o)}")
    check(re.search(r"openmp", o, re.I), "doctor does not mention OpenMP runtimes (0.1.14)")
    return {"summary": o.strip().splitlines()[-1][:200], "output": o}


def key_cmds():
    r = cli("key")
    check(r.returncode == 0 and r.stdout.strip(), r.stderr)
    r2 = cli("key", "--ssh")
    check(r2.returncode == 0 and r2.stdout.startswith("ssh-ed25519 "), r2.stdout)
    return r2.stdout.strip()[:60] + "…"


def servers_list():
    r = cli("servers")
    check(r.returncode == 0 and r.stdout.strip(), r.stderr)
    r2 = cli("servers", "list")
    check(r2.returncode == 0, r2.stderr)
    return r.stdout.strip().splitlines()[0]


def init_writes_toml():
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    r = cli("init", cwd=d, timeout=600)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    f = d / ".reflop.toml"
    check(f.exists(), f"no .reflop.toml in {d}: {r.stdout}")
    txt = f.read_text()
    shutil.copy(f, SCRATCH / "reflop_init.toml")
    return {"summary": " ".join(txt.split())[:250], "toml": txt}


def update_check():
    r = cli("update", "--check", timeout=120)
    o = r.stdout + r.stderr
    check("Traceback" not in o, o)
    check(r.returncode in (0, 1), f"rc {r.returncode}: {o}")
    check(re.search(r"0\.2\.2|up to date|newest|latest", o, re.I), f"update --check output: {o}")
    return o.strip()[:200]


def debug_log():
    r = cli("--server", server_name(), "status", env=env_for(REFLOP_DEBUG="1"))
    check(r.returncode == 0, r.stderr)
    e = r.stderr
    for step in ("tls", "handshake", "auth"):
        check(step in e.lower(), f"REFLOP_DEBUG=1 log lacks {step!r}:\n{tail(e)}")
    check("id_ed25519" not in e and "BEGIN" not in e, "debug log leaks key material?")
    r0 = cli("--server", server_name(), "status", env=env_for(REFLOP_DEBUG="0"))
    check(r0.stderr.strip() == "" or "handshake" not in r0.stderr.lower(), "REFLOP_DEBUG=0 still logs")
    return f"{len(e.splitlines())} debug lines; e.g. {e.splitlines()[0][:120] if e else ''}"


def wrong_server_name():
    r = cli("--server", "no-such-server-e2e", "status")
    o = r.stdout + r.stderr
    check(r.returncode != 0, "status with an unknown server succeeded")
    check("Traceback" not in o, "traceback instead of a message")
    check(len(o.strip()) > 0, "no message")
    return o.strip()[:200]


def pair_bad_code():
    r = cli("pair", "reflop1-not-a-real-code", "--name", "e2e-bogus")
    o = r.stdout + r.stderr
    check(r.returncode != 0, "pairing with garbage succeeded")
    check("Traceback" not in o, o)
    return o.strip()[:200]


def unknown_dataset_rm():
    r = cli("data", "rm", "e2e-does-not-exist")
    o = r.stdout + r.stderr
    check(r.returncode != 0, "rm of a missing dataset succeeded")
    check("Traceback" not in o, o)
    return o.strip()[:200]


def owner_status():
    owner.need()
    t = owner.status_text()
    (SCRATCH / "reflopd_status.txt").write_text(t)
    check(re.match(r"reflopd \S+", t), t[:200])
    return t.splitlines()[0]


def main():
    s = Suite("a01_cli", __doc__.splitlines()[0])
    s.meta["env"] = env_info()
    s.case("version", version)
    s.case("status", status)
    s.case("usage_cli_and_api", usage_cli_and_api)
    s.case("usage_grows_with_work", usage_grows_with_work)
    s.case("verify", verify)
    s.case("doctor_openmp", doctor)
    s.case("key", key_cmds)
    s.case("servers", servers_list)
    s.case("init_writes_reflop_toml", init_writes_toml)
    s.case("update_check", update_check)
    s.case("debug_connection_log", debug_log)
    s.case("wrong_server_name_message", wrong_server_name)
    s.case("pair_bad_code_message", pair_bad_code)
    s.case("data_rm_unknown_message", unknown_dataset_rm)
    s.case("owner_status", owner_status)
    s.finish()


if __name__ == "__main__":
    main()
