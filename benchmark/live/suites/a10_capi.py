"""a10 C and C++ API (libreflop): key file, build, `make test` (per-thread last_error, no server), `make check`
(examples), the clib/README nested C++ builder pattern (old N1), matmul C on GPU and server CPU, edge cases (fetch
size query / ERANGE / NULL nbytes / nbytes mismatch / open node / unknown op), macOS install name + relocation,
C API latency."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from live.harness import E2E, IS_MAC, SCRATCH, SRC, Skip, Suite, check, cli, env_for, json_lines, q, tail

CLIB = SRC / "clib"
BUILD = SCRATCH / "capi"
LIB = CLIB / "build" / "libreflop.so"


def sh(cmd, cwd=None, timeout=900, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env or env_for(),
                          shell=isinstance(cmd, str))


def key_file():
    r = cli("key", "--write-file")
    check(r.returncode == 0, r.stderr)
    p = Path.home() / ".config" / "reflop" / "id_ed25519"
    check(p.exists(), f"{p} missing after --write-file")
    mode = oct(p.stat().st_mode & 0o777)
    check(mode == "0o600", f"key file mode {mode}")
    return f"{p} ({mode})"


def build_lib():
    if not shutil.which("go"):
        raise Skip("Go is not installed")
    r = sh(["make", "-C", str(CLIB), "build/libreflop.so"], timeout=1200)
    check(r.returncode == 0 and LIB.exists(), tail(r.stdout + r.stderr))
    out = f"{LIB.stat().st_size / 2**20:.1f} MiB"
    if IS_MAC:
        o = sh(["otool", "-D", str(LIB)]).stdout
        check("@rpath/libreflop.so" in o, f"install name: {o}")
        out += ", install name @rpath/libreflop.so"
    return out


def make_test():
    r = sh(["make", "-C", str(CLIB), "test"], timeout=600)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    return tail(r.stdout.strip(), 200)


def make_check():
    r = sh(["make", "-C", str(CLIB), "check"], timeout=900)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    return tail(r.stdout.strip(), 300)


def compile_prog(src: Path, out: Path, cxx=False):
    out.parent.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CXX", "c++") if cxx else os.environ.get("CC", "cc")
    flags = ["-O2", "-Wall", "-std=c++17"] if cxx else ["-O2", "-Wall"]
    rpath = "@loader_path" if IS_MAC else "$ORIGIN"
    shutil.copy(LIB, out.parent / "libreflop.so")
    r = sh([cc, *flags, f"-I{CLIB / 'include'}", "-o", str(out), str(src), f"-L{out.parent}", f"-Wl,-rpath,{rpath}",
            "-lreflop", "-lm"])
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    return out


def readme_cpp_pattern():
    exe = compile_prog(E2E / "capi" / "mlp.cpp", BUILD / "mlp", cxx=True)
    res = []
    for dev in ("auto", "cpu"):
        r = sh([str(exe), dev], timeout=300)
        j = (json_lines(r.stdout) or [{}])[-1]
        check(r.returncode == 0 and j.get("verdict") == "pass", f"{dev}: rc {r.returncode} {r.stdout[-300:]}")
        res.append(f"{dev}: err {j.get('max_abs_err')}")
    return "README nested g.op(...) pattern works (N1 fixed): " + ", ".join(res)


def c_matmul():
    exe = compile_prog(E2E / "capi" / "matmul.c", BUILD / "matmul")
    res = []
    for dev in ("auto", "cuda", "cpu"):
        r = sh([str(exe), dev], timeout=300)
        j = (json_lines(r.stdout) or [{}])[-1]
        check(r.returncode == 0 and j.get("verdict") == "pass", f"{dev}: rc {r.returncode} {r.stdout[-300:]}")
        res.append(f"{dev}: rel {j.get('max_rel_err')}")
    return ", ".join(res)


def api_edges(s):
    exe = compile_prog(E2E / "capi" / "api_edges.c", BUILD / "api_edges")
    out = {}
    for dev in ("auto", "cpu"):
        r = sh([str(exe), dev, str(q(100, 10))], timeout=600)
        js = json_lines(r.stdout)
        bad = [j for j in js if "check" in j and not j["ok"]]
        for j in js:
            if j.get("bench"):
                s.metric("c_api_latency", **j)
        out[dev] = {j["check"]: j["ok"] for j in js if "check" in j}
        check(js and not bad, f"{dev}: failed checks {bad} (rc {r.returncode}) {r.stderr[-300:]}")
    return {"summary": f"{len(out['auto'])} checks pass on GPU and server CPU", **out}


def relocated_program():
    d = Path(tempfile.mkdtemp(dir=SCRATCH))
    for f in ("matmul", "libreflop.so"):
        shutil.copy(BUILD / f, d / f)
    r = sh([str(d / "matmul"), "auto"], timeout=300, env=env_for(DYLD_LIBRARY_PATH=None, LD_LIBRARY_PATH=None))
    check(r.returncode == 0, f"relocated program failed: {r.stdout[-200:]} {r.stderr[-300:]}")
    return f"runs from {d.name} with the library next to it"


def threads_last_error():
    r = sh(["make", "-C", str(CLIB), "build/test_lasterror"], timeout=600)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    r2 = sh([str(CLIB / "build" / "test_lasterror")], timeout=300)
    check(r2.returncode == 0, tail(r2.stdout + r2.stderr))
    return tail(r2.stdout.strip(), 200) or "ok"


def main():
    s = Suite("a10_capi", __doc__.splitlines()[0])
    s.case("key_write_file", key_file)
    built = s.case("build_libreflop", build_lib, timeout=1500)
    if built["status"] != "pass":
        s.finish()
    s.case("make_test", make_test, timeout=900)
    s.case("per_thread_last_error", threads_last_error, timeout=600)
    s.case("make_check_examples", make_check, timeout=900)
    s.case("readme_cpp_nested_builder", readme_cpp_pattern, timeout=600)
    s.case("c_matmul_auto_cuda_cpu", c_matmul, timeout=600)
    s.case("api_edge_cases", api_edges, s, timeout=900)
    s.case("relocated_program", relocated_program, timeout=300)
    s.finish()


if __name__ == "__main__":
    main()
