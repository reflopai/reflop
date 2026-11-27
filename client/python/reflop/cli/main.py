# SPDX-License-Identifier: Apache-2.0
"""`reflop` command line (PLAN.md §16.2)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from .. import _debug
from .._pb import compute_pb2 as cpb
from .._pb import data_pb2 as dpb
from .._pb import graph_pb2 as gpb
from ..errors import sanitize

EPILOG = """\
Your programs run on this machine. Only operator graphs and tensors go to
the server, which runs whitelisted tensor operators and nothing else.

  reflop pair <invite-code>         add a GPU pairing
  reflop unpair [name]              forget a pairing and give up its access
  reflop status                     list pairings
  reflop rename <old> <new>         change a pairing's local name
  reflop use <name>                 choose the default pairing
  reflop usage                      compute you have used on the server
  reflop data push ./pack --name d  upload a dataset once
  reflop run python train.py        run locally with reflop enabled

Set REFLOP_DEBUG=1 for full tracebacks and a log of each connection step.
"""


def _p(*a) -> None:
    print(*a)


def cmd_pair(a) -> int:
    from ..client import TLSConnectionRefused, pair

    try:
        e = pair(a.code, a.name)
    except TLSConnectionRefused as exc:
        if _debug.enabled():
            raise
        sys.stderr.write(f"reflop: {sanitize(str(exc))}\n")
        sys.stderr.write("Warning: Check that reflopd is listening on the Owner's address and port and that "
                         "the firewall allows it. With UFW, run on the Owner's machine:\n")
        for port in exc.ports:
            sys.stderr.write(f"  sudo ufw allow {port}/tcp\n")
        return 1
    card = f" ({e.gpu_name or e.gpu_uuid})" if e.gpu_uuid else ""
    _p(f"paired with {e.name}{card} as {e.identity!r} ({e.transport}, {sanitize(', '.join(e.addresses))})")
    _p(f"server key pinned: {e.fingerprint[:16]}…")
    _p("next: reflop status")
    return 0


_UNPAIR_DONE = {
    "revoked": "the server revoked this machine's key for {identity!r}. The Owner still has to delete it on "
               "their side (`reflopd unpair {identity}`); reflopd shows them that.",
    "already": "the server had already stopped accepting this machine's key, so nothing is left to do there.",
    "unreachable": "could not reach the server to tell it ({detail}). Ask the Owner to run "
                   "`reflopd unpair {identity}`; until then, a new invite from that server cannot pair this machine.",
    "mismatch": "the server presented a different key than the one pinned at pairing, so nothing was sent to it. "
                "Ask the Owner to run `reflopd unpair {identity}`.",
    "unsupported": "its reflopd is too old to be told. Ask the Owner to run `reflopd revoke {identity}` and "
                   "`reflopd users delete {identity}`.",
    "skipped": "the server was not told (--local). Ask the Owner to run `reflopd unpair {identity}`.",
    "failed": "telling the server failed ({detail}). Ask the Owner to run `reflopd unpair {identity}`.",
}


def cmd_unpair(a) -> int:
    from ..client import unpair
    from ..config import ClientConfig

    name = a.name or a.server
    entry = ClientConfig.load().get(name)
    if not a.yes and sys.stdin.isatty():
        how = "" if a.local else " and tell it to stop accepting this machine's key"
        sys.stdout.write(f"Forget {entry.name} (paired as {entry.identity!r}){how}? Pairing again needs a new "
                         "invite. [y/N] ")
        sys.stdout.flush()
        if not sys.stdin.readline().strip().lower().startswith("y"):
            _p("not changed")
            return 1
    entry, outcome, detail = unpair(entry.name, local_only=a.local)
    _p(f"forgot {entry.name}: " + _UNPAIR_DONE[outcome].format(identity=entry.identity, detail=detail or "?"))
    _p("to pair again: reflop pair <new-invite-code>")
    return 0


def cmd_servers(a) -> int:
    from ..config import ClientConfig

    if a.action == "default":
        return cmd_use(a)
    if a.action == "remove":
        # Removing only the local entry left this machine's key registered,
        # so it could never pair with that server again (issue #6).
        if not a.name:
            _p("usage: reflop servers remove <name> (the same as reflop unpair <name>)")
            return 2
        return cmd_unpair(argparse.Namespace(name=a.name, server=None, local=False, yes=False))
    cfg = ClientConfig.load()
    if not cfg.servers:
        _p("no servers paired (reflop pair <invite-code>)")
    for n, s in cfg.servers.items():
        mark = "*" if n == cfg.default else " "
        _p(f"{mark} {n:16} {s.transport:4} {sanitize(', '.join(s.addresses)):30} as {s.identity}  key {s.fingerprint[:16]}…")
    return 0


def cmd_use(a) -> int:
    from ..config import ClientConfig, config_lock

    with config_lock():
        cfg = ClientConfig.load()
        if a.name not in cfg.servers:
            raise ValueError(f"no pairing named {a.name!r} (see `reflop status`)")
        cfg.default = a.name
        cfg.save()
    _p(f"default pairing: {a.name}")
    return 0


def cmd_rename(a) -> int:
    from ..config import ClientConfig, config_lock, validate_pairing_name

    validate_pairing_name(a.new_name)
    with config_lock():
        cfg = ClientConfig.load()
        if a.old_name not in cfg.servers:
            raise ValueError(f"no pairing named {a.old_name!r} (see `reflop status`)")
        if a.old_name == a.new_name:
            return 0
        if a.new_name in cfg.servers:
            raise ValueError(f"the pairing name {a.new_name!r} is already in use")
        cfg.servers = {
            (a.new_name if name == a.old_name else name):
            (replace(server, name=a.new_name) if name == a.old_name else server)
            for name, server in cfg.servers.items()
        }
        if cfg.default == a.old_name:
            cfg.default = a.new_name
        cfg.save()
    _p(f"renamed {a.old_name!r} to {a.new_name!r}")
    return 0


def cmd_key(a) -> int:
    from .. import keys

    if a.new:
        if not a.ssh or a.write_file:
            raise ValueError("--new requires --ssh and stores the new pairing key automatically")
        k = keys.new_ssh_pairing_key()
    else:
        k = keys.load_or_create()
    if a.write_file:
        f = keys.write_key_file(k)
        _p(f"wrote {f} (mode 0600) for libreflop and SSH pipe mode")
        return 0
    if a.ssh:
        _p(keys.openssh_public(k, comment=f"reflop:{os.environ.get('USER', 'guest')}"))
    else:
        _p(keys.public_bytes(k).hex())
    return 0


def _caps_summary(caps) -> list[str]:
    out = []
    for d in caps.devices:
        kind = "cuda" if d.device.kind == gpb.DEVICE_CUDA else "cpu"
        mem = f", {d.total_memory_bytes / 2**30:.1f} GiB" if d.total_memory_bytes else ""
        cc = f", cc {d.compute_capability_major}.{d.compute_capability_minor}" if kind == "cuda" else ""
        out.append(f"{kind}:{d.device.index} {sanitize(d.name, 80)}{mem}{cc}")
    return out


def _char_width(char: str) -> int:
    if unicodedata.category(char) in ("Mn", "Me"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1


def _display_width(text: str) -> int:
    return sum(map(_char_width, text))


def _table_cell(text: str, width: int) -> str:
    used = _display_width(text)
    if used > width:
        used = 0
        for i, char in enumerate(text):
            n = _char_width(char)
            if used + n >= width:
                text = text[:i] + "…"
                used += 1
                break
            used += n
    return text + " " * (width - used)


def _print_status_table(rows: list[tuple[str, str, str, str, str]]) -> None:
    header = (" ", "NAME", "STATUS", "GPU", "ADDRESS")
    minimums = [len(label) for label in header[1:]]
    widths = [max(len(label), *(_display_width(row[i]) for row in rows))
              for i, label in enumerate(header) if i]
    interactive = sys.stdout.isatty()
    if interactive:
        # Reserve two columns for the default marker and one per gap.
        available = max(sum(minimums), shutil.get_terminal_size().columns - 5)
        while sum(widths) > available:
            i = max(range(len(widths)), key=lambda i: widths[i] - minimums[i])
            widths[i] -= 1
    color = interactive and not os.environ.get("NO_COLOR")
    for mark, *values in (header, *rows):
        cells = [_table_cell(value, width) for value, width in zip(values, widths, strict=True)]
        if color and values[1] in ("online", "offline"):
            status_color = 32 if values[1] == "online" else 31
            cells[1] = f"\x1b[{status_color}m{values[1]}\x1b[0m" + cells[1][len(values[1]):]
        _p(f"{mark} " + " ".join(cells).rstrip())


def cmd_status(a) -> int:
    if a.server is None:
        from ..client import pairing_online
        from ..config import ClientConfig

        cfg = ClientConfig.load()
        if not cfg.servers:
            _p("no pairings (reflop pair <invite-code>)")
            return 0
        rows = []
        with ThreadPoolExecutor(max_workers=8) as probes:
            for (name, entry), online in zip(cfg.servers.items(), probes.map(pairing_online, cfg.servers.values()),
                                             strict=True):
                card = entry.gpu_name or entry.gpu_uuid or "host-wide"
                label = "".join(char if char.isprintable() else " " for char in sanitize(name))
                rows.append(("*" if name == cfg.default else " ", label,
                             "online" if online else "offline", " ".join(sanitize(card).split()),
                             " ".join(sanitize(", ".join(entry.addresses)).split())))
        _print_status_table(rows)
        if not cfg.default:
            _p("choose a default with reflop use <name>")
        return 0

    from ..client import admin

    c = admin(a.server)
    try:
        caps = c.caps
        e = c.entry
        _p(f"server:   {e.name} ({e.transport}, {sanitize(', '.join(e.addresses))}), you are {sanitize(c.auth.identity_name, 80)!r}")
        if e.gpu_uuid:
            _p(f"GPU:      {e.gpu_name} {e.gpu_uuid}")
            if not any(d.device.kind == gpb.DEVICE_CUDA for d in caps.devices):
                _p("          paired GPU unavailable")
        for line in _caps_summary(caps):
            _p(f"device:   {line}")
        _p(f"opset:    v1, {len(caps.enabled_op_ids)} operators enabled; capabilities: "
           f"{', '.join(sanitize(x, 40) for x in caps.capabilities)}")
        lim = caps.limits
        quotas = []
        for f, label in (("quota_device_memory_bytes", "device memory"), ("quota_max_sessions", "sessions"),
                         ("quota_execute_wall_ms", "ms/execute"), ("quota_storage_bytes", "storage")):
            v = getattr(lim, f)
            if v:
                quotas.append(f"{label} {v}")
        _p(f"quotas:   {', '.join(quotas) or 'none (usage is agreed between people)'}")
        from .._pb import compute_pb2 as cpb

        ds = c.call(cpb.Envelope(list_datasets=dpb.ListDatasets()), timeout=30).dataset_list.datasets
        _p(f"datasets: {len(ds)}")
        if "usage" in caps.capabilities:
            from ..api import _usage_on

            u = _usage_on(c)
            _p(f"usage:    {_dur(u['device_seconds'])} device time, {_flops(u['flops'])} est., "
               f"{u['executes']} executes (reflop usage)")
        ri = caps.runtime_info
        _p(f"runtime:  torch {sanitize(ri.torch_version, 40)}, CUDA {sanitize(ri.cuda_version, 20) or 'none'} "
           f"(informational), reflopd {sanitize(ri.reflopd_version, 40)}")
    finally:
        c.close()
    return 0


def _dur(sec: float) -> str:
    if 0 < sec < 1:
        return f"{sec * 1000:.0f}ms" if sec >= 0.001 else f"{sec * 1e6:.0f}µs"
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    if m < 60:
        return f"{m}m{int(sec) % 60:02d}s"
    h = m // 60
    if h < 48:
        return f"{h}h{m % 60:02d}m"
    return f"{h // 24}d{h % 24:02d}h"


def _size(n: int) -> str:
    v, units = float(n), ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{n} B" if i == 0 else f"{v:.1f} {units[i]}"


def _flops(n: float) -> str:
    for unit, v in (("PFLOP", 1e15), ("TFLOP", 1e12), ("GFLOP", 1e9), ("MFLOP", 1e6)):
        if n >= v:
            return f"{n / v:.1f} {unit}"
    return f"{n:.0f} FLOP"


def _usage_lines(u: dict, now: float | None = None) -> list[str]:
    now = time.time() if now is None else now
    dev = u["device_seconds"]
    devs = ", ".join(f"{d} {_dur(s)}" for d, s in u["by_device"].items())
    out = [f"device time:  {_dur(dev)}" + (f" ({devs})" if devs else "")]
    busy = f" (devices busy {100 * dev / u['session_seconds']:.0f}% of it)" if u["session_seconds"] else ""
    out.append(f"session time: {_dur(u['session_seconds'])}{busy}")
    avg = f", {1000 * dev / u['executes']:.1f} ms average" if u["executes"] else ""
    out.append(f"executes:     {u['executes']}{avg}")
    rate = f" ({_flops(u['flops'] / dev)}/s while running)" if dev else ""
    out.append(f"est. FLOPs:   {_flops(u['flops'])}{rate}")
    if u["peak_memory_bytes"]:
        out.append(f"peak memory:  {_size(u['peak_memory_bytes'])} (largest execute)")
    out.append(f"transfer:     {_size(u['bytes_up'])} up, {_size(u['bytes_down'])} down")
    last = ""
    if u["last"] is not None:
        ago = now - u["last"].timestamp()
        last = ", last active " + ("just now" if ago < 60 else _dur(ago) + " ago")
    out.append(f"sessions:     {u['sessions']}{last}")
    return out


def cmd_usage(a) -> int:
    from ..api import _usage_on
    from ..client import admin

    c = admin(a.server)
    try:
        u = _usage_on(c)
        name = c.entry.name
    finally:
        c.close()
    if a.json:
        import json

        _p(json.dumps({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in u.items()}, indent=2))
        return 0
    since = u["since"].astimezone().strftime("%Y-%m-%d %H:%M") if u["since"] else "?"
    _p(f"compute used by {u['identity']!r} on {name} since {since}")
    for line in _usage_lines(u):
        _p(line)
    return 0


def cmd_init(a) -> int:
    """Measure the link and write .reflop.toml (capability cache, §14.2)."""
    import torch

    from ..client import admin

    c = admin(a.server)
    try:
        t = []
        for _ in range(5):
            t0 = time.perf_counter()
            c.call(cpb.Envelope(ping=cpb.Ping(nonce=1)), timeout=10)
            t.append(time.perf_counter() - t0)
        rtt = sorted(t)[len(t) // 2]
        caps = c.caps
    finally:
        c.close()
    x = torch.randn(1024, 1024)
    t0 = time.perf_counter()
    for _ in range(5):
        x @ x
    local = 5 * 2 * 1024 ** 3 / (time.perf_counter() - t0)
    lines = ["# Generated by `reflop init`. Safe to commit: it contains no secrets.",
             f'server = "{c.entry.name}"', f"rtt_ms = {rtt * 1000:.2f}", f"local_flops = {local:.3e}",
             f"enabled_ops = {len(caps.enabled_op_ids)}",
             "capabilities = [" + ", ".join(f'"{sanitize(x, 40)}"' for x in caps.capabilities) + "]",
             "devices = [" + ", ".join(f'"{x}"' for x in _caps_summary(caps)) + "]"]
    with open(".reflop.toml", "w") as f:
        f.write("\n".join(lines) + "\n")
    _p(f"wrote .reflop.toml (RTT {rtt * 1000:.1f} ms)")
    return 0


def cmd_data(a) -> int:
    from .._pb import data_pb2 as dpb

    if a.action == "build":
        from ..data import build

        out = build(a.source, a.out, int(a.shard_mib) << 20)
        _p(f"wrote TensorPack {out}")
        return 0
    from ..client import admin

    c = admin(a.server)
    try:
        if a.action == "push":
            from ..data.push import push

            info = push(c, a.pack, a.name)
            _p(f"dataset {info.name!r}: {info.num_samples} samples, {info.bytes / 2**20:.1f} MiB, "
               f"tensors {', '.join(t.name for t in info.tensors)}")
        elif a.action == "ls":
            for d in c.call(cpb.Envelope(list_datasets=dpb.ListDatasets()), timeout=30).dataset_list.datasets:
                _p(f"{sanitize(d.name, 64):24} {d.num_samples:>10} samples  {d.bytes / 2**20:10.1f} MiB  "
                   f"{', '.join(sanitize(t.name, 64) for t in d.tensors)}")
        elif a.action == "rm":
            lst = c.call(cpb.Envelope(list_datasets=dpb.ListDatasets()), timeout=30).dataset_list.datasets
            ids = [d.dataset_id for d in lst if d.name == a.name]
            if not ids:
                _p(f"no dataset named {a.name!r}")
                return 1
            c.call(cpb.Envelope(delete_dataset=dpb.DeleteDataset(dataset_id=ids[0])), timeout=60)
            _p(f"removed {a.name!r}")
    finally:
        c.close()
    return 0


def _signal_name(n: int) -> str:
    import signal

    try:
        return signal.Signals(n).name
    except ValueError:
        return f"signal {n}"


RUN_HELP = """\
Runs the command on this machine with 'cuda' mapped to the reflop device.
The exit status is the command's; if a signal kills it, reflop names the
signal and exits with 128 + n, as a shell would.

Python children get PYTHONFAULTHANDLER=1 (unless it is already set), so a
native crash prints the Python stack of every thread.

On macOS the command gets OMP_NUM_THREADS=1 unless it is already set: a
second OpenMP runtime next to torch's (from scikit-learn, xgboost, conda or
Homebrew) crashes or hangs the program otherwise. The math runs on the
server, so only operators that fall back to this machine's CPU run on one
thread. Set OMP_NUM_THREADS yourself to choose another value;
`reflop doctor` lists the OpenMP runtimes installed here.

REFLOP_DEBUG=1 reflop run ... also:
  - logs each connection step (ssh or TLS dial, handshake, session open,
    reconnect) to stderr;
  - prints the child's pid; `kill -USR1 <pid>` or Ctrl-\\ then prints the
    Python stack of every thread without stopping it, to find a hang.
"""

# Signals that mean a native crash, where the macOS OpenMP clash (#7) is the
# usual cause on a Guest's Mac.
_CRASH_SIGNALS = ("SIGSEGV", "SIGBUS", "SIGABRT", "SIGILL")
OPENMP_ISSUE = "https://github.com/reflopai/reflop/issues/7"


def _run_child(command: list[str], env: dict[str, str]) -> int:
    """Start the command and wait for it like a shell does.

    Ctrl-C and Ctrl-\\ reach the whole foreground process group, so the child
    gets them anyway: reflop ignores them while it waits instead of dying (or
    killing the child) before the child has printed its traceback or stack
    dump. A SIGTERM sent to reflop alone is passed on to the child. Signals
    that were already ignored (a background job) stay as they were, and the
    child starts with the dispositions reflop itself was given.
    """
    import signal
    import subprocess

    child: list[subprocess.Popen] = []
    early: list[int] = []  # arrived before the child existed

    def forward(signum, _frame) -> None:
        if not child:
            early.append(signum)
        for p in child:
            try:
                p.send_signal(signum)
            except OSError:
                pass

    saved = {}
    for name, handler in (("SIGINT", lambda *_: None), ("SIGQUIT", lambda *_: None), ("SIGTERM", forward)):
        s = getattr(signal, name, None)
        if s is None:
            continue
        try:
            if signal.getsignal(s) is signal.SIG_IGN:
                continue
            saved[s] = signal.signal(s, handler)
        except (ValueError, OSError):  # not the main thread
            pass
    try:
        # A Python handler is reset to the default at exec, so the child
        # does not inherit these.
        p = subprocess.Popen(command, env=env)
        child.append(p)
        for s in early:
            p.send_signal(s)
        if _debug.enabled():
            sys.stderr.write(f"reflop: {command[0]} is pid {p.pid}; `kill -USR1 {p.pid}` (or Ctrl-\\) prints "
                             "its Python stacks\n")
        return p.wait()
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


def cmd_run(a) -> int:
    """Run a command locally with reflop's shim enabled (best effort, §16.2)."""
    from pathlib import Path

    if not a.command:
        _p("usage: reflop run python train.py [args...]")
        return 2
    shim = Path(__file__).resolve().parent.parent / "_shim"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shim) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["REFLOP_SHIM"] = "1"
    # A native crash (torch, libomp, another C extension) then prints the
    # Python stacks instead of nothing (issue #8). The user's setting wins.
    env.setdefault("PYTHONFAULTHANDLER", "1")
    # On macOS a second OpenMP runtime next to torch's libomp crashes or hangs
    # the program unless OpenMP stays on one thread (issue #7, ADR-027). The
    # math runs on the server, so the local pool is rarely worth the risk.
    omp_default = sys.platform == "darwin" and "OMP_NUM_THREADS" not in env
    if omp_default:
        env["OMP_NUM_THREADS"] = "1"
    if a.server:
        env["REFLOP_SERVER"] = a.server
    sys.stderr.write("reflop: running on this machine; tensor math on 'cuda' goes to the reflop server\n")
    if omp_default:
        sys.stderr.write("reflop: OMP_NUM_THREADS=1 (macOS default, see `reflop run --help`)\n")
    rc = _run_child(a.command, env)
    if rc < 0:
        # Killed by a signal. The shell only sees our own exit status, so say
        # what happened and report 128 + n like the shell would (issue #8).
        # Only the program name goes out: the rest of the argv may carry
        # credentials, and unquoted arguments read ambiguously in a log line.
        # Like a shell, say nothing for Ctrl-C and a closed pipe.
        n = -rc
        name = _signal_name(n)
        if name not in ("SIGINT", "SIGPIPE"):
            sys.stderr.write(f"reflop: {a.command[0]} was killed by {name} (signal {n})\n")
        if name in _CRASH_SIGNALS and sys.platform == "darwin":
            if env.get("OMP_NUM_THREADS") != "1":
                sys.stderr.write("reflop: on macOS this is often two OpenMP runtimes in one process; try "
                                 f"OMP_NUM_THREADS=1 reflop run ... (see {OPENMP_ISSUE})\n")
            else:
                sys.stderr.write("reflop: `reflop doctor` lists the OpenMP runtimes installed here; a second "
                                 f"one next to torch's is a common cause on macOS (see {OPENMP_ISSUE})\n")
        return 128 + n
    return rc


def cmd_verify(a) -> int:
    from ..verify import verify

    ok = verify(a.server)
    return 0 if ok else 1


# `reflop test` architectures, as in testzoo.WORKLOADS (kept here so the parser does not import torch;
# tests/client/test_reflop_test.py checks they agree).
TEST_WORKLOADS = (
    ("mlp", "3-layer MLP on 784 features"),
    ("cnn", "3-block CNN with BatchNorm, 32x32 images"),
    ("resnet", "ResNet-18, 32x32 images"),
    ("unet", "small U-Net, 64x64 segmentation"),
    ("lstm", "2-layer LSTM, 64 steps"),
    ("gru", "2-layer GRU, 64 steps"),
    ("transformer", "2-layer transformer encoder, 64 tokens"),
    ("gpt", "4-layer causal GPT, 128 tokens"),
    ("lora", "LoRA adapters on a frozen 4-layer MLP"),
    ("autoencoder", "convolutional autoencoder, 32x32 images"),
)
TEST_ALL_MINUTES = (2, 5)   # what `reflop test --all` usually takes, and on a slow connection


def cmd_test(a) -> int:
    from .. import speedtest

    names = [n for n, _ in TEST_WORKLOADS] if a.all else [a.workload]
    if a.all:
        lo, hi = TEST_ALL_MINUTES
        print(f"Testing all {len(names)} architectures takes about {lo} minutes, up to {hi} on a slow "
              "connection. Ctrl-C stops it.", file=sys.stderr)
    return speedtest.run(names, batch=a.batch, server=a.server, cpu=a.cpu)


def cmd_doctor(a) -> int:
    from ..doctor import doctor

    return doctor(a.server)


def cmd_update(a) -> int:
    from ..update import update

    return update(check=a.check, version=a.version)


def cmd_version(a) -> int:
    import torch

    from .. import __version__

    _p(f"reflop {__version__} (torch {torch.__version__}, python {sys.version.split()[0]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reflop", description="reFLOP client: use a remote GPU as a calculator.",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", "-s", help="pairing name (default: the selected pairing)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair", help="redeem an invite code")
    p.add_argument("code")
    p.add_argument("--name", help="local name for this pairing")
    p.set_defaults(fn=cmd_pair)
    p = sub.add_parser("unpair", help="forget a paired server and give up this machine's access to it")
    p.add_argument("name", nargs="?", help="local server name (default: the default server)")
    p.add_argument("--local", action="store_true", help="forget the server without contacting it")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    p.set_defaults(fn=cmd_unpair)
    p = sub.add_parser("servers", help="list paired servers")
    p.add_argument("action", nargs="?", choices=["list", "default", "remove"], default="list")
    p.add_argument("name", nargs="?")
    p.set_defaults(fn=cmd_servers)
    p = sub.add_parser("key", help="print this machine's public key")
    p.add_argument("--ssh", action="store_true", help="OpenSSH format, for SSH pipe mode invites")
    p.add_argument("--new", action="store_true", help="with --ssh, generate a new key for a GPU invite")
    p.add_argument("--write-file", action="store_true",
                   help="also store the private key as ~/.config/reflop/id_ed25519 (needed by libreflop on macOS)")
    p.set_defaults(fn=cmd_key)
    sub.add_parser("status", help="list pairings, or show live details with --server").set_defaults(fn=cmd_status)
    p = sub.add_parser("rename", help="change a pairing's local name")
    p.add_argument("old_name")
    p.add_argument("new_name")
    p.set_defaults(fn=cmd_rename)
    p = sub.add_parser("use", help="choose the default pairing")
    p.add_argument("name")
    p.set_defaults(fn=cmd_use)
    p = sub.add_parser("usage", help="compute you have used on the server (device time, FLOPs, traffic)")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(fn=cmd_usage)
    sub.add_parser("init", help="measure the link, write .reflop.toml").set_defaults(fn=cmd_init)
    p = sub.add_parser("data", help="TensorPacks: build, push, ls, rm")
    ds = p.add_subparsers(dest="action", required=True)
    b = ds.add_parser("build", help="run module:function locally and write a TensorPack")
    b.add_argument("--from", dest="source", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--shard-mib", type=int, default=256)
    pu = ds.add_parser("push", help="upload a TensorPack (resumable)")
    pu.add_argument("pack")
    pu.add_argument("--name", required=True)
    ds.add_parser("ls", help="list datasets on the server")
    rm = ds.add_parser("rm", help="delete a dataset")
    rm.add_argument("name")
    p.set_defaults(fn=cmd_data)
    p = sub.add_parser("run", help="run a command locally with 'cuda' mapped to reflop", description=RUN_HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)
    sub.add_parser("verify", help="cross-check small graphs against local CPU").set_defaults(fn=cmd_verify)
    sub.add_parser("doctor", help="diagnose this machine and the connection").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("test", help="measure training speed over this connection against native speed on the server",
                       description="Train a small model of the chosen kind on the server for a few seconds and "
                       "compare it with the same model trained natively on the server's own device. The score is "
                       "the share of native speed you get over this connection (100% = as fast as running there). "
                       "Pick the architecture closest to yours; --all tests every one.",
                       epilog="Nothing runs on this machine's GPU, and no code from this machine runs on the "
                       "server: it trains reflopd's own copy of the model.")
    g = p.add_mutually_exclusive_group(required=True)
    for wname, label in TEST_WORKLOADS:
        g.add_argument(f"--{wname}", dest="workload", action="store_const", const=wname, help=label)
    g.add_argument("--all", action="store_true", help="every architecture above (a few minutes)")
    p.add_argument("--batch", type=int, default=32, choices=(8, 16, 32, 64, 128), help="batch size (default 32)")
    p.add_argument("--cpu", action="store_true", help="test the server's CPU instead of its GPU")
    p.set_defaults(fn=cmd_test)
    p = sub.add_parser("update", help="update the reflop client")
    p.add_argument("--check", action="store_true")
    p.add_argument("--version")
    p.set_defaults(fn=cmd_update)
    sub.add_parser("version").set_defaults(fn=cmd_version)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - one clean line; REFLOP_DEBUG=1 for the traceback
        if _debug.enabled():
            raise
        sys.stderr.write(f"reflop: {sanitize(str(e))}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
