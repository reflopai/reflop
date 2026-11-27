# SPDX-License-Identifier: Apache-2.0
"""`reflop test`: how fast a model trains through this connection, as a share of native speed on the server's
own device (ADR-064).

For each workload it measures, on this connection, a training step with `reflop.capture` and the data on the
server, the same step uploading its batch every time, and an eager step. reflopd then reports the same
workload trained natively on the same device by its own copy of the model (testzoo.py, from its cache when it
has measured it before). Score = native ms / reFLOP ms: 100% is as fast as running on that GPU locally.
Nothing runs on this machine's GPU, and no code from this machine runs on the server.
"""

from __future__ import annotations

import itertools
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field

import torch

from . import testzoo

# ----------------------------------------------------------------------------- progress

class Progress:
    """A spinner with a status text on a terminal; plain lines otherwise."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.text = ""
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

    def __enter__(self):
        if self.tty:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
            with self._lock:
                self.stream.write("\r\033[K")
                self.stream.flush()

    def update(self, text: str):
        with self._lock:
            self.text = text
            if not self.tty:
                self.stream.write(f"  {text}\n")
                self.stream.flush()

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.wait(0.1):
                return
            with self._lock:
                el = time.monotonic() - self._t0
                self.stream.write(f"\r\033[K{frame} {self.text} ({el:.0f} s)")
                self.stream.flush()


# ----------------------------------------------------------------------------- measurements

@dataclass
class Network:
    rtt_ms: float
    upload_mib_s: float | None


@dataclass
class Result:
    workload: str
    batch: int
    batch_bytes: int = 0
    captured: dict | None = None     # data on the server, reflop.capture
    upload: dict | None = None       # reflop.capture, the batch uploaded every step
    eager: dict | None = None        # data on the server, no capture
    native: dict | None = None       # NativeBaselineResult as a dict
    native_note: str = ""            # why there is no native baseline
    error: str = ""
    notes: list = field(default_factory=list)

    def score(self, mode: str) -> float | None:
        r = getattr(self, mode)
        if not (r and self.native):
            return None
        return min(100.0, self.native["ms_per_step"] / r["ms_per_step"] * 100)


def measure_network(dev) -> Network:
    import reflop

    x = torch.ones(1, device=dev)
    (x + 1).item()
    ts = []
    for _ in range(15):
        t = time.perf_counter()
        (x + 1).item()
        ts.append(time.perf_counter() - t)
    rtt = statistics.median(ts) * 1000
    bw = None
    for mib in (1, 4):
        buf = torch.rand(mib * 2**18)          # mib MiB of float32
        t = time.perf_counter()
        y = buf.to(dev)
        reflop.synchronize()
        el = time.perf_counter() - t - rtt / 1000
        del y
        bw = mib / max(el, 1e-6)
        if el > 1.0:                            # slow link: 1 MiB is enough
            break
    return Network(rtt_ms=rtt, upload_mib_s=bw)


def _native(dev, name: str, batch: int) -> tuple[dict | None, str]:
    from . import session
    from ._pb import compute_pb2 as cpb
    from .errors import ReflopRemoteError

    sess = session.by_index(dev.index)
    if "native_baseline" not in sess.caps.capabilities:
        return None, "this server's reflopd does not measure native baselines (0.4.2 or newer does); ask the Owner to update it"
    try:
        r = sess.conn.call(cpb.Envelope(native_baseline=cpb.NativeBaseline(workload=name, batch=batch)))
    except ReflopRemoteError as e:
        if e.code == cpb.ERROR_BUSY:
            return None, e.message
        raise
    b = r.native_baseline_result
    return {"ms_per_step": b.ms_per_step, "ms_reps": list(b.ms_reps), "steps": b.steps, "cached": b.cached,
            "measured_unix": b.measured_unix}, ""


def measure(dev, name: str, batch: int, progress: Progress, budget_s: float = 1.5) -> Result:
    import reflop

    w = testzoo.WORKLOADS[name]
    res = Result(workload=name, batch=batch)
    try:
        progress.update(f"{name}: building the model")
        model = testzoo.build(name).to(dev)
        opt = testzoo.optimizer(model)
        step = testzoo.make_step(model, opt, w.loss)
        x_cpu, y_cpu = w.batch(batch)
        res.batch_bytes = x_cpu.numel() * x_cpu.element_size() + y_cpu.numel() * y_cpu.element_size()
        x, y = x_cpu.to(dev), y_cpu.to(dev)

        def sync(loss):
            return loss.item()

        captured = reflop.capture(step)
        progress.update(f"{name}: reflop.capture, data on the server")
        res.captured = testzoo.time_steps(lambda: captured(x, y), sync, budget_s)
        progress.update(f"{name}: reflop.capture, uploading the batch every step")
        res.upload = testzoo.time_steps(lambda: captured(x_cpu.to(dev), y_cpu.to(dev)), sync, budget_s,
                                        warmup=1, min_steps=1)

        def eager():
            loss = step(x, y)
            reflop.step_boundary()
            return loss

        progress.update(f"{name}: eager, data on the server")
        res.eager = testzoo.time_steps(eager, sync, budget_s, warmup=2, min_steps=1)
        progress.update(f"{name}: native baseline on the server's device")
        res.native, res.native_note = _native(dev, name, batch)
    except Exception as e:  # noqa: BLE001 - one workload failing does not stop the others
        res.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
    return res


# ----------------------------------------------------------------------------- output

class _Style:
    """ANSI colors on a terminal, nothing otherwise (NO_COLOR, pipes, tests)."""

    def __init__(self, color: bool):
        self.color = color

    def _w(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, text: str) -> str:
        return self._w("1", text)

    def dim(self, text: str) -> str:
        return self._w("2", text)

    def red(self, text: str) -> str:
        return self._w("31", text)

    def rated(self, score: float, text: str) -> str:
        return self._w("32" if score >= GREAT else "33" if score >= OKAY else "31", text)


GREAT, OKAY = 70, 30   # score bands: great, okay, slow


def _word(score: float) -> str:
    return "great" if score >= GREAT else "okay" if score >= OKAY else "slow"


def _ms(r: dict | None) -> str:
    return "" if not r else f"{r['ms_per_step']:.2f}" if r["ms_per_step"] < 100 else f"{r['ms_per_step']:.0f}"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"


def _when(unix: int) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(unix)) if unix else "earlier"


def _header(title: str, net: Network, where: str, st: _Style) -> list[str]:
    bw = f"{net.upload_mib_s:.1f} MiB/s" if net.upload_mib_s else "unknown"
    return [f"{st.bold('reflop test')}: {title}", f"  server   {where}",
            f"  network  round trip {net.rtt_ms:.1f} ms, upload {bw}", ""]


def _mark(score: float | None, st: _Style) -> str:
    """A colored dot on a terminal; the number itself without color, where a dot would say nothing."""
    if score is None:
        return "—"
    return st.rated(score, "●") if st.color else _pct(score)


def render_one(res: Result, net: Network, where: str, color: bool = False) -> str:
    st = _Style(color)
    w = testzoo.WORKLOADS[res.workload]
    lines = _header(f"{res.workload} ({w.label}), batch {res.batch}", net, where, st)
    if res.error:
        return "\n".join(lines + [f"  {st.red('failed')}: {res.error}"]) + "\n"
    s = res.score("captured")
    cap = res.captured["ms_per_step"]
    if s is None:
        lines += [f"  No score: {res.native_note or 'no native baseline'}.", ""]
        for mode, label in (("captured", "reflop.capture"), ("upload", "uploading each batch"),
                            ("eager", "eager, without capture")):
            lines.append(f"  {_ms(getattr(res, mode)):>8} ms/step  {label}")
        return "\n".join(lines) + "\n"

    lines += [f"  {st.rated(s, st.bold('● ' + _word(s)))}  {st.bold(_pct(s))} of native speed with reflop.capture", ""]
    upload, eager = res.upload["ms_per_step"], res.eager["ms_per_step"]
    for mode, label, tip in (
            ("upload", "uploading each batch", "keep data on the server" if upload - cap > 0.5 * cap else ""),
            ("eager", "eager, without capture", "wrap the step in reflop.capture" if eager > 1.5 * cap else "")):
        m = res.score(mode)
        row = f"  {st.rated(m, '●')} {_pct(m):>4}  {label:<24}"
        lines.append((row + st.dim("→ " + tip)) if tip else row.rstrip())
    src = "measured now" if not res.native["cached"] else f"from {_when(res.native['measured_unix'])}"
    lines += ["", st.dim(f"  {_ms(res.captured)} ms/step here, {_ms(res.native)} ms/step native ({src})")]
    if s < 20 and net.rtt_ms < 0.1 * cap:
        lines.append("  Not the network: this model runs as many small operations, each with a fixed cost on the server.")
    if net.rtt_ms > 20:
        lines.append(f"  Slow round trip: every value read back (.item(), print) waits {net.rtt_ms:.0f} ms.")
    return "\n".join(lines) + "\n"


def render_all(results: list[Result], net: Network, where: str, batch: int, seconds: float,
               color: bool = False) -> str:
    st = _Style(color)
    lines = _header(f"all workloads, batch {batch}", net, where, st)
    nw = max(len(r.workload) for r in results) + 2
    lines.append(st.dim(f"  {'':<{nw}}{'capture':<15}{'upload':>6} {'eager':>6}"))

    def key(r: Result):
        s = r.score("captured")
        return (bool(r.error), s is None, -(s or 0))

    for r in sorted(results, key=key):
        name = f"  {r.workload:<{nw}}"
        if r.error:
            lines.append(name + st.red(f"failed: {r.error[:60]}"))
            continue
        s = r.score("captured")
        if s is None:
            lines.append(name + st.dim(f"— no score, {_ms(r.captured)} ms/step"))
            continue
        if color:   # the dots under the middle of "upload" and "eager"; pad the visible text, not the codes
            rest = f"  {_mark(r.score('upload'), st)}      {_mark(r.score('eager'), st)}"
        else:
            rest = f"{_mark(r.score('upload'), st):>6} {_mark(r.score('eager'), st):>6}"
        lines.append(name + st.rated(s, f"● {_pct(s):>4} {_word(s):<5}") + "   " + rest)
    lines.append("")
    foot = ("  " + "  ".join(st.rated(v, "●") + st.dim(t) for v, t in ((GREAT, " ≥70%"), (OKAY, " 30–70%"), (0, " <30%")))
            + st.dim(" of native speed")) if color else "  % of native speed on the server's device"
    if any(r.workload in ("lstm", "gru") and (r.score("captured") or 100) < OKAY for r in results):
        foot += st.dim(" · RNNs are slow on reFLOP")
    lines.append(foot + st.dim(f" · {seconds:.0f} s"))
    if net.rtt_ms > 20:
        lines.append(f"  Slow round trip: every value read back (.item(), print) waits {net.rtt_ms:.0f} ms.")
    lines += [f"  No score for some workloads: {n}." for n in sorted({r.native_note for r in results if r.native_note})]
    return "\n".join(lines) + "\n"


def where_of(dev, server: str | None = None) -> str:
    from . import session

    sess = session.by_index(dev.index)
    name = ""
    for d in sess.caps.devices:
        if d.device.kind == sess.gdevice.kind and d.device.index == sess.gdevice.index:
            name = d.name
    kind = "cuda" if sess.gdevice.kind == 1 else "cpu"
    return " ".join(x for x in (server, f"{kind}:{sess.gdevice.index}", name) if x)


def run(names: list[str], batch: int = testzoo.DEFAULT_BATCH, server: str | None = None, cpu: bool = False,
        out=None, err=None, dev=None) -> int:
    """Measure `names`; print the result. Returns 0, or 1 when a workload failed."""
    import reflop

    out = out or sys.stdout
    t0 = time.monotonic()
    with Progress(err) as progress:
        progress.update("connecting")
        if dev is None:
            dev = reflop.device(server=server, kind="cpu" if cpu else "auto")
        progress.update("measuring the network")
        net = measure_network(dev)
        results = [measure(dev, n, batch, progress) for n in names]
    if server is None:
        try:
            from .config import ClientConfig

            server = ClientConfig.load().default
        except Exception:  # noqa: BLE001 - the name is only shown
            server = None
    where = where_of(dev, server)
    color = hasattr(out, "isatty") and out.isatty() and not os.environ.get("NO_COLOR")
    if len(names) == 1:
        out.write(render_one(results[0], net, where, color))
    else:
        out.write(render_all(results, net, where, batch, time.monotonic() - t0, color))
    out.flush()
    return 1 if any(r.error for r in results) else 0
