# SPDX-License-Identifier: Apache-2.0
"""Make a live-suite run safe to publish.

    python -m live.sanitize run <results/RUN> <out_dir>     copy run.json and <suite>.json, scrubbed (no logs/)
    python -m live.sanitize text <in.md> <out.md> --run <results/RUN>
                                                          scrub a document with the names that run recorded
    python -m live.sanitize check <path>...               exit 1 if anything private is left

A run records the Owner's and Guest's host names, addresses, user paths, GPU UUIDs, `reflop status` and
`reflopd status` output and the command lines of processes on the Owner's GPU. `run` drops the status texts,
process command lines and network peer details, and in every remaining string replaces IP addresses, home
directories, GPU UUIDs and the run's own server and host names with placeholders. `check` looks for what
`run` removes; it cannot know names a run did not record, so read a document before publishing it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6 = re.compile(r"(?<![\w:])(?=[0-9a-fA-F:]*[a-fA-F])[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){3,7}(?![\w:])")
HOMES = re.compile(r"(/Users/|/home/)(?!<user>)[^/\s'\"`:;,)\]]+")
GPU_UUID = re.compile(r"(?i)\bgpu-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
PATTERNS = {"IP address": IPV4, "IPv6 address": IPV6, "home directory": HOMES, "GPU UUID": GPU_UUID}

# Names that are not private and appear everywhere (the suite's own pairing and identity names).
GENERIC = {"e2e", "local", "guest", "default", "localhost"}
DROP_RUN_KEYS = ("reflop_status", "reflopd_status_start", "reflopd_status_end")


def names_of(run: dict) -> list[str]:
    """Host and server names a run recorded (pairing name, ssh host, Tailscale peers)."""
    s = run.get("settings") or {}
    out = [s.get("server"), s.get("owner_ssh")]
    owner = s.get("owner") or ""
    if owner.startswith("ssh "):
        out.append(owner[4:])
    for k in ("network_start", "network_end"):
        ts = (run.get(k) or {}).get("tailscale") or {}
        out.append(ts.get("peer"))
    found = set()
    for n in out:
        if isinstance(n, str) and len(n) >= 3 and n not in ("(none)", "none"):
            found.add(n)
            found.add(n.split("@")[-1].split(".")[0])     # user@host.domain -> host
    return sorted((n for n in found if len(n) >= 3 and n.lower() not in GENERIC), key=len, reverse=True)


def scrub(text: str, names: list[str]) -> str:
    for n in names:
        text = re.sub(rf"(?<![\w]){re.escape(n)}(?![\w])", "<server>", text, flags=re.I)
    text = GPU_UUID.sub("gpu-<uuid>", text)
    text = IPV4.sub("<ip>", text)
    text = IPV6.sub("<ip>", text)
    return HOMES.sub(lambda m: m.group(1) + "<user>", text)


def walk(obj, names):
    if isinstance(obj, str):
        return scrub(obj, names)
    if isinstance(obj, list):
        return [walk(x, names) for x in obj]
    if isinstance(obj, dict):
        return {scrub(k, names): walk(v, names) for k, v in obj.items()}
    return obj


def sanitize_run_json(run: dict) -> dict:
    run = {k: v for k, v in run.items() if k not in DROP_RUN_KEYS}
    s = dict(run.get("settings") or {})
    for k in ("owner_ssh", "owner"):
        if s.get(k):
            s[k] = "local" if str(s[k]).startswith("local") else "<owner>"
    run["settings"] = s
    for k in ("owner_gpu_start", "owner_gpu_end"):
        g = run.get(k)
        if isinstance(g, dict):
            run[k] = {kk: v for kk, v in g.items() if kk != "processes"}
            run[k]["processes"] = [{"used_memory_mib": p.get("used_memory_mib"), "reflop_worker": p.get("reflop_worker")}
                                   for p in g.get("processes", [])]
    for k in ("network_start", "network_end"):
        n = run.get(k)
        if isinstance(n, dict):
            ts = n.get("tailscale") or {}
            run[k] = {kk: v for kk, v in n.items() if kk.startswith("ping") and isinstance(v, (int, float))}
            if ts.get("path"):
                run[k]["tailscale_path"] = ts["path"]
    return run


def sanitize_run(src: Path, dst: Path) -> list[str]:
    run = json.loads((src / "run.json").read_text())
    names = names_of(run)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "run.json").write_text(json.dumps(walk(sanitize_run_json(run), names), separators=(",", ":")) + "\n")
    for f in sorted(src.glob("*.json")):
        if f.name != "run.json":
            d = json.loads(f.read_text())
            (dst / f.name).write_text(json.dumps(walk(d, names), separators=(",", ":")) + "\n")
    return names


def leaks(text: str) -> list[str]:
    return [f"{what}: {m.group(0)!r}" for what, rx in PATTERNS.items() for m in rx.finditer(text)]


def check(paths: list[Path]) -> int:
    bad = 0
    for p in paths:
        for f in sorted(p.rglob("*") if p.is_dir() else [p]):
            if f.is_file() and f.suffix in (".json", ".md", ".txt", ".csv"):
                for found in leaks(f.read_text(errors="replace"))[:5]:
                    print(f"{f}: {found}")
                    bad += 1
    return 1 if bad else 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["run"] and len(a) == 3:
        names = sanitize_run(Path(a[1]), Path(a[2]))
        print(f"wrote {a[2]} ({len(names)} names replaced)")
        return 0
    if a[:1] == ["text"] and len(a) == 5 and a[3] == "--run":
        names = names_of(json.loads((Path(a[4]) / "run.json").read_text()))
        Path(a[2]).write_text(scrub(Path(a[1]).read_text(), names))
        return 0
    if a[:1] == ["check"] and len(a) >= 2:
        return check([Path(x) for x in a[1:]])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
