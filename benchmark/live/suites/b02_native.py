# SPDX-License-Identifier: Apache-2.0
"""b02 native baseline on the Owner: the same workloads.py, batches, optimizer and timing as b01, with local CUDA
(reflopd's own runtime Python, outside the sandbox), plus a matmul TFLOP/s sweep. Retained throughput
(reflop / native) is computed in the summary. Guarded: waits until no other identity has a session."""

from __future__ import annotations

from live import owner
from live.harness import E2E, Suite, check, json_lines, q


def native(s):
    owner.wait_quiet(300)
    pyr = owner.runtime_python()
    owner.push(str(E2E / "workloads.py"))
    remote = owner.push(str(E2E / "native_bench.py"))
    batches = ",".join(map(str, q((8, 32, 128), (32,))))
    args = f"--batches {batches} --warmup {q(5, 3)} --steps {q(20, 4)} --repeats {q(3, 1)} --matmul"
    r = owner.run(f"{pyr} -I {remote} {args}", timeout=3600)
    js = json_lines(r.stdout)
    check(js, f"no output: {r.stderr[-800:]}")
    env = next(j for j in js if j["kind"] == "env")
    s.meta["native_env"] = env
    n_ok = 0
    for j in js:
        if j["kind"] == "case":
            s.metric("native", **{k: v for k, v in j.items() if k != "kind"})
            n_ok += j.get("status") == "ok"
        elif j["kind"] == "matmul":
            s.metric("native_matmul", **{k: v for k, v in j.items() if k != "kind"})
    return f"{n_ok} native cases on {env['gpu']} (torch {env['torch']}, tf32 matmul {env['tf32_matmul']})"


def main():
    s = Suite("b02_native", __doc__.splitlines()[0])
    s.case("native_cuda_matrix", native, s, timeout=4000)
    s.finish()


if __name__ == "__main__":
    main()
