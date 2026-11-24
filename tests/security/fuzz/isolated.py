# SPDX-License-Identifier: Apache-2.0
"""Fuzz batches in child processes, for CUDA (#25 B, ADR-049).

A device-side assert or an illegal memory access kills the CUDA context
for the whole process, and an abort kills the process itself. So on CUDA
each seed runs in a fresh child process (this file run as a script). It
stops at its first failure; the parent records the failure and starts a
new child at the next case. The child writes the index of the case it is
running before running it, so a crash or a hang is pinned to its case.
Replay files go to REFLOP_FUZZ_OUT as on the CPU, prefixed with the device."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PATHS = [HERE, HERE.parent / "corpus", ROOT / "client/python", ROOT / "server/worker"]
CASE_TIMEOUT = 300  # seconds one case (or the child's start-up) may take
MAX_FAILURES = 20
PREFIX = "FINDING "


def run_isolated(kind: str, device: str, seed: int, cases: int) -> list[str]:
    """Cases 0..cases of a seed ("graph" or "seq"), in child processes; the
    failures, up to MAX_FAILURES."""
    from fuzzrun import artifact_dir

    out = artifact_dir()
    current = None
    if out is not None:
        current = out / (f"{device}-current.bin" if kind == "graph" else f"seq-{device}-current.bin")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([*map(str, PATHS), os.environ.get("PYTHONPATH", "")]),
               PYTHONWARNINGS="ignore")  # keep the stderr tail of a crash readable
    failures: list[str] = []
    start = 0
    with tempfile.TemporaryDirectory() as tmp:
        progress = Path(tmp) / "progress"
        while start < cases and len(failures) < MAX_FAILURES:
            _write(progress, start)
            rc, stdout, stderr = _child(
                [sys.executable, __file__, kind, device, str(seed), str(start), str(cases), str(progress), tmp],
                env, progress, Path(tmp))
            failures += [json.loads(line[len(PREFIX):]) for line in stdout.splitlines() if line.startswith(PREFIX)]
            if rc == 0:
                break
            i = int(progress.read_text())
            if rc != 3:
                what = f"hung for {CASE_TIMEOUT} s" if rc is None else f"died (exit {rc})"
                failures.append(f"seed {seed} case {i}: the worker process {what}: {stderr[-600:]}")
                if current is not None and current.exists():
                    shutil.copy(current, current.with_name(current.name.replace("current", f"seed{seed}-case{i}")))
            start = i + 1
    return failures


def _child(cmd: list[str], env: dict, progress: Path, tmp: Path) -> tuple[int | None, str, str]:
    so, se = tmp / "stdout", tmp / "stderr"
    with open(so, "w") as fo, open(se, "w") as fe:
        proc = subprocess.Popen(cmd, env=env, stdout=fo, stderr=fe)
        seen, since = None, time.monotonic()
        while True:
            try:
                rc = proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass
            now = progress.read_text()
            if now != seen:
                seen, since = now, time.monotonic()
            elif time.monotonic() - since > CASE_TIMEOUT:
                proc.kill()
                proc.wait()
                rc = None
                break
    return rc, so.read_text(), se.read_text()


def _write(progress: Path, i: int) -> None:
    tmp = progress.with_suffix(".tmp")
    tmp.write_text(str(i))
    os.replace(tmp, progress)


def main(kind: str, device: str, seed: str, start: str, end: str, progress: str, tmp: str) -> int:
    def report(i: int) -> None:
        _write(Path(progress), i)

    if kind == "graph":
        from fuzzrun import fuzz_graphs, still_serving
        from graphgen import make_session

        sess = make_session(device)
        failures = fuzz_graphs(sess, int(seed), int(start), int(end), first=True, progress=report)
        if not failures:
            gone = still_serving(sess)
            failures = [f"seed {seed} after case {int(end) - 1}: {gone}"] if gone else []
    else:
        from seqrun import fuzz_sequences

        work = Path(tempfile.mkdtemp(dir=tmp))
        failures = fuzz_sequences(device, int(seed), int(start), int(end), work, first=True, progress=report)
        shutil.rmtree(work, ignore_errors=True)
    for f in failures:
        print(PREFIX + json.dumps(f), flush=True)
    return 3 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
