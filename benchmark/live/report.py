# SPDX-License-Identifier: Apache-2.0
"""Write a Markdown report of a live-suite run, compared with an earlier run.

    python -m live.report results/<NEW> [results/<OLD>] [-o report.md] [--findings findings.md]

Every table is computed from the runs' JSON (run.json and <suite>.json). Section 2 holds the analysis, which no
script can write: --findings inserts it from a file, otherwise the section says it is missing. Numbers are
rounded to 4 significant digits. Run the output through `python -m live.sanitize` before publishing it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

STATUSES = ("pass", "fail", "xfail", "skip")


# ----------------------------------------------------------------------------- loading

class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.run = json.loads((self.path / "run.json").read_text())
        self.suites: dict[str, dict] = {}
        for f in sorted(self.path.glob("*.json")):
            if f.name != "run.json":
                d = json.loads(f.read_text())
                self.suites[d.get("suite", f.stem)] = d
        self.cases = {f"{s}/{c['id']}": c for s, d in self.suites.items() for c in d.get("cases", [])}

    @property
    def id(self) -> str:
        return self.run.get("run_id") or self.path.name

    def metrics(self, suite: str, group: str) -> list[dict]:
        return [m for m in self.suites.get(suite, {}).get("metrics", []) if m.get("group") == group]

    def version(self) -> str:
        return (self.run.get("versions") or {}).get("reflop_client") or "?"

    def rtt_ms(self) -> float | None:
        for m in self.metrics("b03_micro", "rtt"):
            if m.get("device") == "cuda":
                return m.get("ms_median")
        return None


# ----------------------------------------------------------------------------- formatting

def num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e6 or a < 1e-3:
            return f"{v:.3e}"
        return f"{v:.4g}" if a < 1e4 else f"{v:.0f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(num(x) for x in v) + "]"
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    return str(v)


def cell(v) -> str:
    return num(v).replace("|", "\\|").replace("\n", " ⏎ ")


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def delta(new, old) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new / old - 1) * 100


def pct(v) -> str:
    return "" if v is None else f"{v:+.1f}%"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def first_line(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return (s.splitlines()[0] if s else "")[:n]


# ----------------------------------------------------------------------------- sections

def identity(new: Run, old: Run | None) -> str:
    def flat(r: Run) -> dict:
        d = {}
        for k in ("versions", "settings"):
            for kk, v in (r.run.get(k) or {}).items():
                if kk != "suites_selected":
                    d[f"{k}.{kk}"] = v
        for k in ("started", "finished", "seconds", "state"):
            d[k] = r.run.get(k)
        env = (r.suites.get("b02_native", {}).get("meta") or {}).get("native_env") or {}
        for kk in ("gpu", "torch", "cuda", "tf32_matmul", "tf32_cudnn"):
            if kk in env:
                d[f"native_env.{kk}"] = env[kk]
        net = r.run.get("network_start") or {}
        if net.get("ping_ms") is not None:
            d["network.ping_ms"] = net.get("ping_ms")
        if (net.get("tailscale") or {}).get("path"):
            d["network.tailscale_path"] = net["tailscale"]["path"]
        return d
    n, o = flat(new), flat(old) if old else {}
    keys = list(dict.fromkeys([*n, *o]))
    hdr = ["field", "NEW"] + (["OLD"] if old else [])
    return table(hdr, [[k, n.get(k)] + ([o.get(k)] if old else []) for k in keys])


def totals(new: Run, old: Run | None) -> str:
    def per(r: Run):
        rows = {}
        for s in r.run.get("suites", []):
            rows[s["suite"]] = s
        return rows
    n, o = per(new), per(old) if old else {}
    hdr = ["suite"] + [f"NEW {x}" for x in (*STATUSES, "exit", "s")]
    if old:
        hdr += [f"OLD {x}" for x in (*STATUSES, "exit", "s")]
    rows, tn, to = [], dict.fromkeys(STATUSES, 0), dict.fromkeys(STATUSES, 0)

    def cols(s, tot):
        if not s:
            return [""] * 6
        c = s.get("counts") or {}
        for k in STATUSES:
            tot[k] += c.get(k, 0)
        return [c.get(k, 0) for k in STATUSES] + [s.get("exit_code"), s.get("seconds")]
    for name in sorted(set(n) | set(o)):
        rows.append([name] + cols(n.get(name), tn) + (cols(o.get(name), to) if old else []))
    bad = lambda d: f"{sum(1 for s in d.values() if s.get('exit_code'))} nonzero"  # noqa: E731
    rows.append(["TOTAL"] + [tn[k] for k in STATUSES] + [bad(n), new.run.get("seconds")]
                + ([to[k] for k in STATUSES] + [bad(o), old.run.get("seconds")] if old else []))
    out = table(hdr, rows)
    desc = [f"- `{name}`: {first_line(d.get('description', ''), 160)}" for name, d in sorted(new.suites.items())]
    return out + "\n\nSuite descriptions (from each suite's docstring):\n\n" + "\n".join(desc)


def failures(new: Run, old: Run | None) -> str:
    bad = [(k, c) for k, c in new.cases.items() if c["status"] == "fail"]
    if not bad:
        return "No case failed."
    rows = [[f"F{i}", f"`{k}`", (old.cases.get(k, {}).get("status", "absent") if old else ""),
             first_line(c.get("detail"), 240)] for i, (k, c) in enumerate(bad, 1)]
    out = [table(["#", "case", "OLD status", "detail"], rows), ""]
    for i, (k, c) in enumerate(bad, 1):
        out.append(f"### F{i}: `{k}`\n")
        o = old.cases.get(k) if old else None
        out.append(f"- NEW: fail in {c.get('seconds')} s. OLD: "
                   + (f"{o['status']}: {first_line(o.get('detail'), 240)!r}" if o else "absent"))
        out.append("- Detail:\n\n```text\n" + (c.get("detail") or "")[:2000] + "\n```")
        if c.get("traceback"):
            tb = c["traceback"].strip().splitlines()[-14:]
            out.append("- Traceback (last frames):\n\n```text\n" + "\n".join(tb) + "\n```")
        out.append("")
    return "\n".join(out)


def status_changes(new: Run, old: Run) -> str:
    keys = sorted(set(new.cases) | set(old.cases))
    pairs: dict[tuple, list] = {}
    for k in keys:
        a = old.cases.get(k, {}).get("status", "absent")
        b = new.cases.get(k, {}).get("status", "absent")
        pairs.setdefault((a, b), []).append(k)
    out = [table(["OLD status", "NEW status", "cases"],
                 [[a, b, len(v)] for (a, b), v in sorted(pairs.items(), key=lambda x: -len(x[1]))]), ""]
    rows = []
    for (a, b), ks in sorted(pairs.items()):
        if a == b:
            continue
        for k in ks:
            rows.append([k, a, b, first_line(new.cases.get(k, {}).get("detail"), 160),
                         first_line(old.cases.get(k, {}).get("detail"), 160)])
    out.append(table(["case", "OLD", "NEW", "NEW detail", "OLD detail"], rows) if rows else "No case changed status.")
    return "\n".join(out)


def xfails(new: Run, old: Run | None) -> str:
    rows = [[k, c.get("seconds"), first_line(c.get("detail"), 220), old.cases.get(k, {}).get("status", "absent")
             if old else ""] for k, c in new.cases.items() if c["status"] == "xfail"]
    return table(["case", "seconds", "detail", "OLD status"], rows) if rows else "No expected failures."


# ----------------------------------------------------------------------------- benchmarks

def _key(m):
    return (m.get("workload"), m.get("batch"), m.get("policy"))


def throughput_rows(r: Run, device: str) -> dict:
    return {(_key(m), m.get("mode")): m for m in r.metrics("b01_throughput", "throughput") if m.get("device") == device}


def native_rows(r: Run) -> dict:
    return {_key(m): m for m in r.metrics("b02_native", "native") if m.get("status", "ok") == "ok"}


def gpu_join(new: Run, old: Run | None) -> list[dict]:
    """b01 GPU rows with b02 native rows of the same run, and the OLD run's counterpart."""
    nt, nn = throughput_rows(new, "cuda"), native_rows(new)
    ot, on = (throughput_rows(old, "cuda"), native_rows(old)) if old else ({}, {})
    rtt_n, rtt_o = new.rtt_ms(), old.rtt_ms() if old else None
    out = []
    for (k, mode), m in sorted(nt.items(), key=lambda x: (str(x[0][0][0]), x[0][0][1] or 0, x[0][1], x[0][0][2])):
        o, nat, nat_o = ot.get((k, mode)), nn.get(k), on.get(k)
        r = {"workload": k[0], "batch": k[1], "mode": mode, "policy": k[2], "new": m, "old": o,
             "ms": m.get("ms_per_step"), "ms_old": o.get("ms_per_step") if o else None,
             "native": nat.get("ms_per_step") if nat else None, "native_old": nat_o.get("ms_per_step") if nat_o else None}
        r["overhead"] = r["ms"] - r["native"] if r["native"] is not None else None
        r["overhead_old"] = r["ms_old"] - r["native_old"] if None not in (r["ms_old"], r["native_old"]) else None
        r["corr"] = r["overhead"] - rtt_n / 20 if None not in (r["overhead"], rtt_n) else None
        r["corr_old"] = r["overhead_old"] - rtt_o / 20 if None not in (r["overhead_old"], rtt_o) else None
        r["retained"] = r["native"] / r["ms"] * 100 if r["native"] and r["ms"] else None
        r["retained_old"] = r["native_old"] / r["ms_old"] * 100 if r["native_old"] and r["ms_old"] else None
        out.append(r)
    return out


def gpu_summary(rows: list[dict], new: Run, old: Run | None) -> str:
    def group(rs):
        return [len(rs), pct(med([delta(r["ms"], r["ms_old"]) for r in rs])),
                pct(min((d for r in rs if (d := delta(r["ms"], r["ms_old"])) is not None), default=None)),
                pct(max((d for r in rs if (d := delta(r["ms"], r["ms_old"])) is not None), default=None)),
                med([r["retained"] for r in rs]), med([r["retained_old"] for r in rs]),
                med([r["overhead"] for r in rs]), med([r["overhead_old"] for r in rs]),
                pct(med([delta(r["overhead"], r["overhead_old"]) for r in rs])),
                pct(med([delta(r["corr"], r["corr_old"]) for r in rs]))]
    hdr = ["mode", "policy", "n", "median Δ% ms/step", "min Δ%", "max Δ%", "NEW median retained %",
           "OLD median retained %", "NEW median overhead ms", "OLD median overhead ms", "median Δ% overhead",
           "median Δ% overhead − RTT/20"]
    body = []
    for mode in ("eager", "capture"):
        for pol in ("resident", "upload"):
            rs = [r for r in rows if r["mode"] == mode and r["policy"] == pol]
            if rs:
                body.append([mode, pol, *group(rs)])
    out = [table(hdr, body), ""]
    rn, ro = new.rtt_ms(), old.rtt_ms() if old else None
    out.append(f"`overhead` = reflop ms − native ms in the same run. `overhead − RTT/20` also takes out one sync "
               f"round trip per 20 timed steps, using the b03 GPU RTT median (NEW {num(rn)} ms"
               + (f", OLD {num(ro)} ms" if old else "") + "). Bandwidth differences (upload policy) are not removed.")
    out.append("\nBy batch size:\n")
    body = []
    for b in sorted({r["batch"] for r in rows}):
        for mode in ("eager", "capture"):
            for pol in ("resident", "upload"):
                rs = [r for r in rows if r["batch"] == b and r["mode"] == mode and r["policy"] == pol]
                if rs:
                    body.append([b, mode, pol, len(rs), pct(med([delta(r["ms"], r["ms_old"]) for r in rs])),
                                 med([r["retained"] for r in rs]), med([r["retained_old"] for r in rs])])
    out.append(table(["batch", "mode", "policy", "n", "median Δ% ms/step", "NEW median retained %",
                      "OLD median retained %"], body))
    return "\n".join(out)


def gpu_configs(rows: list[dict]) -> str:
    hdr = ["workload", "batch", "mode", "policy", "reflop ms NEW", "reflop ms OLD", "Δ%", "native ms NEW",
           "native ms OLD", "overhead ms NEW", "overhead ms OLD", "overhead Δ%", "overhead−RTT/20 NEW",
           "overhead−RTT/20 OLD", "retained % NEW", "retained % OLD", "samples/s NEW", "B up/step", "B down/step",
           "registers/step", "executes/step", "replays", "final loss NEW", "final loss OLD", "ms_reps NEW"]
    body = []
    for r in rows:
        m, o = r["new"], r["old"] or {}
        body.append([r["workload"], r["batch"], r["mode"], r["policy"], r["ms"], r["ms_old"],
                     pct(delta(r["ms"], r["ms_old"])), r["native"], r["native_old"], r["overhead"], r["overhead_old"],
                     pct(delta(r["overhead"], r["overhead_old"])), r["corr"], r["corr_old"], r["retained"],
                     r["retained_old"], m.get("samples_per_s"), m.get("bytes_up_per_step"),
                     m.get("bytes_down_per_step"), m.get("registers_per_step"), m.get("executes_per_step"),
                     m.get("replays"), m.get("final_loss"), o.get("final_loss"), m.get("ms_reps")])
    return table(hdr, body)


def cpu_configs(new: Run, old: Run | None) -> str:
    nt, ot = throughput_rows(new, "cpu"), throughput_rows(old, "cpu") if old else {}
    body = []
    for (k, mode), m in sorted(nt.items(), key=lambda x: (str(x[0][0][0]), x[0][0][1] or 0, x[0][1])):
        o = ot.get((k, mode)) or {}
        body.append([k[0], k[1], mode, k[2], m.get("ms_per_step"), o.get("ms_per_step"),
                     pct(delta(m.get("ms_per_step"), o.get("ms_per_step"))), m.get("samples_per_s"),
                     o.get("samples_per_s"), m.get("final_loss"), o.get("final_loss"), m.get("ms_reps")])
    if not body:
        return "No server-CPU rows."
    return table(["workload", "batch", "mode", "policy", "ms NEW", "ms OLD", "Δ%", "samples/s NEW", "samples/s OLD",
                  "final loss NEW", "final loss OLD", "ms_reps NEW"], body)


def native_drift(new: Run, old: Run) -> str:
    nn, on = native_rows(new), native_rows(old)
    body = [[*k, m.get("ms_per_step"), on[k].get("ms_per_step"), pct(delta(m.get("ms_per_step"),
                                                                          on[k].get("ms_per_step")))]
            for k, m in sorted(nn.items(), key=lambda x: (str(x[0][0]), x[0][1] or 0, str(x[0][2]))) if k in on]
    ds = [delta(r[3], r[4]) for r in body]
    head = f"Median Δ% {pct(med(ds))} over {len(body)} configurations (native code is the same in both runs).\n\n"
    return head + table(["workload", "batch", "policy", "native ms NEW", "native ms OLD", "Δ%"], body)


def matmul(new: Run, old: Run | None) -> str:
    def idx(r, suite, group):
        return {(m.get("dtype"), m.get("n")): m for m in r.metrics(suite, group)} if r else {}
    rn, ro = idx(new, "b03_micro", "matmul"), idx(old, "b03_micro", "matmul")
    nn, no = idx(new, "b02_native", "native_matmul"), idx(old, "b02_native", "native_matmul")
    body = []
    for k in sorted(rn, key=lambda x: (str(x[0]), x[1] or 0)):
        m, o, a, b = rn[k], ro.get(k, {}), nn.get(k, {}), no.get(k, {})
        ret = m["wall_tflops"] / a["tflops"] * 100 if a.get("tflops") and m.get("wall_tflops") else None
        body.append([k[0], k[1], a.get("tflops"), b.get("tflops"), m.get("wall_tflops"), o.get("wall_tflops"),
                     m.get("device_tflops"), o.get("device_tflops"), ret, m.get("usage_flops_ratio")])
    if not body:
        return "No matmul rows."
    return table(["dtype", "n", "native TFLOP/s NEW", "native OLD", "reflop wall TFLOP/s NEW", "reflop wall OLD",
                  "reflop device TFLOP/s NEW", "reflop device OLD", "wall retained % NEW", "usage_flops_ratio NEW"],
                 body)


def generic(new: Run, old: Run | None, suite: str, group: str) -> str:
    rows = [("NEW", m) for m in new.metrics(suite, group)] + ([("OLD", m) for m in old.metrics(suite, group)]
                                                              if old else [])
    keys = list(dict.fromkeys(k for _, m in rows for k in m if k != "group"))
    return table(["run", *keys], [[r, *(m.get(k) for k in keys)] for r, m in rows])


def all_cases(new: Run, old: Run | None) -> str:
    out = []
    for name, d in sorted(new.suites.items()):
        out.append(f"### A.{name}\n")
        out.append(table(["case", "status", "seconds", "detail", "OLD"],
                         [[c["id"], c["status"], c.get("seconds"), first_line(c.get("detail"), 200),
                           old.cases.get(f"{name}/{c['id']}", {}).get("status", "absent") if old else ""]
                          for c in d.get("cases", [])]))
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- document

SPECIAL = {("b01_throughput", "throughput"), ("b02_native", "native"), ("b02_native", "native_matmul"),
           ("b03_micro", "matmul")}


def report(new: Run, old: Run | None, findings: str | None) -> str:
    vs = f" vs {old.version()} ({old.id})" if old else ""
    out = [f"# reFLOP live-suite report {new.id}: reflop {new.version()}{vs}", "",
           "Generated by `python -m live.report` from the runs' JSON. Status: `pass`, `fail`, `xfail` (a documented "
           "limitation or an explicit refusal), `skip`, `absent` (did not run). `Δ%` = NEW / OLD − 1 (negative: NEW "
           "takes less time). `retained %` = native ms / reflop ms × 100. `overhead` = reflop ms − native ms, the "
           "cost reFLOP adds; compare releases by overhead, because native speed varies between runs (§7.4).", ""]
    out += ["## 1. Run identity", "", identity(new, old), ""]
    out += ["## 2. Findings", "", findings.strip() if findings else
            "_No analysis was supplied (`--findings`). The tables below are complete; read §7.4 and the RTT in §7.6 "
            "before comparing speed between runs._", ""]
    out += ["## 3. Suite totals", "", totals(new, old), ""]
    out += ["## 4. Failures in NEW", "", failures(new, old), ""]
    if old:
        out += ["## 5. Status changes OLD → NEW", "", status_changes(new, old), ""]
    out += ["## 6. Expected failures (xfail) in NEW", "", xfails(new, old), ""]
    rows = gpu_join(new, old)
    out += ["## 7. Benchmarks", ""]
    if rows:
        out += ["### 7.1 GPU throughput summary (b01 vs b02 native)", "", gpu_summary(rows, new, old), ""]
        out += ["### 7.2 GPU throughput per configuration", "", gpu_configs(rows), ""]
    out += ["### 7.3 Server-CPU throughput (b01, device cpu)", "", cpu_configs(new, old), ""]
    if old and native_rows(new) and native_rows(old):
        out += ["### 7.4 Native CUDA on the Owner, NEW vs OLD (hardware and run drift)", "", native_drift(new, old), ""]
    out += ["### 7.5 Matmul TFLOP/s: native (b02) vs reflop (b03)", "", matmul(new, old), ""]
    seen = set()
    groups = [(s, m["group"]) for s, d in sorted(new.suites.items()) for m in d.get("metrics", [])]
    rest = [g for g in groups if g not in SPECIAL and not (g in seen or seen.add(g))]
    if rest:
        out += ["### 7.6 Other measurements", ""]
        for s, g in rest:
            out += [f"#### `{s}` / `{g}`", "", generic(new, old, s, g), ""]
    out += ["## Appendix A. Every NEW case", "", all_cases(new, old)]
    return "\n".join(out).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m live.report", description=__doc__.splitlines()[0])
    ap.add_argument("new", type=Path)
    ap.add_argument("old", type=Path, nargs="?")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--findings", type=Path, help="Markdown for section 2 (the analysis)")
    a = ap.parse_args(argv)
    text = report(Run(a.new), Run(a.old) if a.old else None, a.findings.read_text() if a.findings else None)
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
