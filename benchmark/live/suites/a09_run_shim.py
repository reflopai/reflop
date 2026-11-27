"""a09 `reflop run`: unmodified CUDA script (fp32 and AMP) trains on reflop; torch.cuda API compatibility map; exit
status passthrough, signals named and 128+n, args not echoed, PYTHONFAULTHANDLER and OMP_NUM_THREADS defaults (user
values kept), SIGTERM forwarded, Ctrl-C reaches the program first, REFLOP_DEBUG pid + SIGUSR1 stack dump, bad
commands."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time

from live.harness import E2E, IS_MAC, PY, REFLOP_EXE, ROOT, Skip, Suite, check, env_for, last_json, tail

PROBE = str(E2E / "helpers" / "run_probe.py")


def run(*args, env=None, timeout=300):
    return subprocess.run([REFLOP_EXE, "run", *map(str, args)], capture_output=True, text=True,
                          env=env or env_for(), timeout=timeout, cwd=ROOT, stdin=subprocess.DEVNULL)


def popen(*args, env=None, new_session=False):
    return subprocess.Popen([REFLOP_EXE, "run", *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env or env_for(), cwd=ROOT, stdin=subprocess.DEVNULL,
                            start_new_session=new_session)


def wait_line(p, pat, timeout=60):
    """Read p.stdout line by line (in a thread) until pat matches."""
    import threading

    lines, hit = [], threading.Event()

    def pump():
        for line in iter(p.stdout.readline, ""):
            lines.append(line)
            if re.search(pat, line):
                hit.set()
        hit.set()

    threading.Thread(target=pump, daemon=True).start()
    hit.wait(timeout)
    buf = "".join(lines)
    m = re.search(pat, buf)
    if not m:
        raise AssertionError(f"never printed {pat!r}: {buf[-300:]}")
    return m, buf


def cuda_script(amp):
    r = run(PY, E2E / "helpers" / "cuda_train.py", *(["--amp"] if amp else []), timeout=600)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    j = last_json(r.stdout)
    check(j and j["device"].startswith("reflop"), f"weights on {j and j['device']}")
    check(j["losses"][-1] < j["losses"][0], f"no learning: {j['losses']}")
    return f"device {j['device']} ({j['type']}), epoch losses {[round(v, 3) for v in j['losses']]}"


def cuda_api_map():
    r = run(PY, PROBE, "cuda_api", timeout=300)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    j = last_json(r.stdout)
    core = ["is_available", "tensor_cuda", "to_cuda", "device_kw", "device_obj", "module_cuda", "manual_seed"]
    bad = [k for k in core if not j[k]["ok"]]
    check(not bad, f"documented shim mappings failed: { {k: j[k] for k in bad} }")
    unsupported = sorted(k for k, v in j.items() if not v["ok"])
    return {"summary": f"{sum(v['ok'] for v in j.values())}/{len(j)} work; not supported: {unsupported}", "map": j}


def shim_autocast_effective():
    """A CUDA script's torch.autocast("cuda") under reflop run: low precision, or an error -- never a silent fp32."""
    r = run(PY, PROBE, "cuda_api", timeout=300)
    j = last_json(r.stdout)
    fp16, bf16 = j["amp_autocast_cuda_fp16"], j["amp_autocast_cuda_bf16"]
    detail = f"fp16: {fp16}; bf16: {bf16}"
    check(not (fp16["ok"] and "float32" in fp16["value"]), "autocast('cuda', float16) is a silent no-op (fp32 "
                                                           "output) under reflop run; " + detail)
    check(fp16["ok"] and bf16["ok"], detail)
    return detail


def exit_status():
    out = {}
    for code in (0, 3, 42):
        r = run(PY, PROBE, "exit", code)
        out[code] = r.returncode
    check(out == {0: 0, 3: 3, 42: 42}, f"exit codes {out}")
    r = run("sh", "-c", "exit 5")
    check(r.returncode == 5, f"sh exit 5 -> {r.returncode}")
    return f"{out}, sh -> 5"


def signals():
    res = {}
    for name, n in (("SIGSEGV", 11), ("SIGABRT", 6), ("SIGKILL", 9), ("SIGBUS", 10 if IS_MAC else 7)):
        r = run(PY, PROBE, "signal", name, "--token=SECRET-ARG-123")
        res[name] = r.returncode
        check(r.returncode == 128 + n, f"{name}: rc {r.returncode}, want {128 + n}\n{tail(r.stderr, 400)}")
        check(re.search(rf"killed by {name}", r.stderr), f"{name} not named:\n{tail(r.stderr, 400)}")
        reflop_lines = [l for l in r.stderr.splitlines() if l.startswith("reflop:")]
        check(not any("SECRET-ARG-123" in l for l in reflop_lines), f"reflop echoed the program's arguments: {reflop_lines}")
    r = run(PY, PROBE, "segv")
    check(r.returncode == 139 and "Fatal Python error" in r.stderr, f"segv: rc {r.returncode}\n{tail(r.stderr, 500)}")
    return f"{res}; real segfault -> 139 with faulthandler stack"


def env_defaults():
    r = run(PY, PROBE, "env")
    j = last_json(r.stdout)
    check(j["PYTHONFAULTHANDLER"] == "1", f"PYTHONFAULTHANDLER={j['PYTHONFAULTHANDLER']}")
    if IS_MAC:
        check(j["OMP_NUM_THREADS"] == "1", f"OMP_NUM_THREADS={j['OMP_NUM_THREADS']} on macOS")
        check("OMP_NUM_THREADS" in r.stderr, f"start-up note missing: {r.stderr[:300]}")
    r2 = run(PY, PROBE, "env", env=env_for(OMP_NUM_THREADS="8", PYTHONFAULTHANDLER="0"))
    j2 = last_json(r2.stdout)
    check(j2["OMP_NUM_THREADS"] == "8" and j2["PYTHONFAULTHANDLER"] == "0", f"user values not kept: {j2}")
    r3 = run(PY, PROBE, "env", env=env_for(OMP_NUM_THREADS=""))
    j3 = last_json(r3.stdout)
    check(j3["OMP_NUM_THREADS"] == "", f"empty OMP_NUM_THREADS not kept: {j3}")
    note = next((l for l in r.stderr.splitlines() if "OMP" in l), "")
    return f"defaults {j}; user values kept; note: {note[:120]}"


def sigterm_forwarded():
    p = popen(PY, PROBE, "sigterm")
    wait_line(p, r"ready")
    p.send_signal(signal.SIGTERM)
    rc = p.wait(30)
    out = ""
    check(rc == 7, f"rc {rc} (child handler exits 7); stdout {out[-200:]}")
    return "child received SIGTERM and its exit status came through"


def ctrl_c_reaches_program():
    p = popen(PY, PROBE, "sigint", new_session=True)
    wait_line(p, r"ready")
    os.killpg(p.pid, signal.SIGINT)        # what the terminal does on Ctrl-C
    rc = p.wait(30)
    err = p.stderr.read()
    check("KeyboardInterrupt" in err, f"program traceback lost:\n{tail(err, 500)}")
    check(rc == 130, f"rc {rc}")
    return f"rc {rc}; program printed its traceback"


def debug_pid_and_usr1():
    p = popen(PY, PROBE, "sleep", "6", env=env_for(REFLOP_DEBUG="1"))
    time.sleep(2.5)
    os.set_blocking(p.stderr.fileno(), False)
    err = p.stderr.read() or ""
    m = re.search(r"pid[ =:]+(\d+)", err)
    check(m, f"REFLOP_DEBUG=1 did not print the child's pid:\n{err[-500:]}")
    pid = int(m.group(1))
    os.kill(pid, signal.SIGUSR1)
    rc = p.wait(30)
    os.set_blocking(p.stderr.fileno(), True)
    err += p.stderr.read()
    out = p.stdout.read() if p.stdout else ""
    check(rc == 0 and "done" in out, f"program did not survive SIGUSR1: rc {rc}")
    check("Thread" in err or "File " in err, f"no stack dump after SIGUSR1:\n{err[-600:]}")
    return f"pid {pid} printed; SIGUSR1 dumped the stack and the program finished"


def bad_commands():
    r = run("/nonexistent/program-e2e")
    o = r.stdout + r.stderr
    check(r.returncode != 0 and "Traceback" not in o, f"rc {r.returncode}: {o[-300:]}")
    r2 = subprocess.run([REFLOP_EXE, "run"], capture_output=True, text=True, env=env_for(), timeout=60)
    check(r2.returncode != 0, "`reflop run` with no command succeeded")
    return f"missing program: rc {r.returncode} {o.strip()[:120]!r}; no command: rc {r2.returncode}"


def main():
    s = Suite("a09_run_shim", __doc__.splitlines()[0])
    s.case("cuda_script_fp32", cuda_script, False, timeout=900)
    s.case("cuda_script_amp_fp16", cuda_script, True, timeout=900)
    s.case("cuda_api_map", cuda_api_map, timeout=600)
    s.case("shim_autocast_cuda_effective", shim_autocast_effective, timeout=300)
    s.case("exit_status", exit_status, timeout=300)
    s.case("signals_128_plus_n", signals, timeout=300)
    s.case("env_defaults_omp_faulthandler", env_defaults, timeout=300)
    s.case("sigterm_forwarded", sigterm_forwarded, timeout=120)
    s.case("ctrl_c_reaches_program", ctrl_c_reaches_program, timeout=120)
    s.case("debug_pid_and_sigusr1_stack", debug_pid_and_usr1, timeout=120)
    s.case("bad_commands", bad_commands, timeout=120)
    s.finish()


if __name__ == "__main__":
    main()
