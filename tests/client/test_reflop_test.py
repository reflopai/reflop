# SPDX-License-Identifier: Apache-2.0
"""`reflop test` (ADR-064): the client side, against the in-process worker on the CPU."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLIENT_ZOO = ROOT / "client/python/reflop/testzoo.py"
WORKER_ZOO = ROOT / "server/worker/reflop_worker/testzoo.py"


@pytest.fixture
def local():
    import reflop

    reflop.close()
    yield reflop.device(local="cpu")
    reflop.close()


def test_zoo_copies_are_identical():
    # The worker's copy is the native baseline; the client's is what runs on reFLOP. They must be one model.
    assert CLIENT_ZOO.read_bytes() == WORKER_ZOO.read_bytes()


def test_cli_lists_the_zoo():
    from reflop import testzoo
    from reflop.cli.main import TEST_WORKLOADS

    assert TEST_WORKLOADS == tuple((w.name, w.label) for w in testzoo.WORKLOADS.values())


def test_never_touches_cuda_on_this_machine():
    # The Guest may have no CUDA, and the baseline is the server's device, never this machine's.
    for f in (CLIENT_ZOO, ROOT / "client/python/reflop/speedtest.py"):
        assert "torch.cuda" not in f.read_text(), f


def test_cli_needs_exactly_one_architecture(capsys):
    from reflop.cli.main import main

    for argv in (["test"], ["test", "--mlp", "--unet"], ["test", "--mlp", "--all"], ["test", "--mlp", "--batch", "7"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2, argv
    assert "--transformer" in capsys.readouterr().err


def test_all_warns_first(monkeypatch, capsys):
    from reflop import speedtest
    from reflop.cli.main import TEST_ALL_MINUTES, TEST_WORKLOADS, main

    seen = {}
    monkeypatch.setattr(speedtest, "run", lambda names, **kw: seen.update(names=names, **kw) or 0)
    assert main(["test", "--all", "--batch", "64"]) == 0
    err = capsys.readouterr().err
    assert f"up to {TEST_ALL_MINUTES[1]}" in err and "Ctrl-C" in err
    assert seen["names"] == [n for n, _ in TEST_WORKLOADS] and seen["batch"] == 64
    assert main(["test", "--gru"]) == 0
    assert seen["names"] == ["gru"] and "minutes" not in capsys.readouterr().err


def test_one_workload_end_to_end(local):
    from reflop import speedtest

    out, err = io.StringIO(), io.StringIO()
    assert speedtest.run(["mlp"], dev=local, server="local", out=out, err=err) == 0
    text = out.getvalue()
    for want in ("reflop test: mlp", "round trip", "of native speed with reflop.capture", "uploading each batch",
                 "eager, without capture", "ms/step native (measured now)"):
        assert want in text, (want, text)
    assert "\033" not in text   # not a terminal: no colors
    # Not a terminal: plain progress lines, no spinner control codes.
    assert "mlp: native baseline" in err.getvalue() and "\r" not in err.getvalue()


def test_scores(local):
    from reflop import speedtest

    r = speedtest.measure(local, "cnn", 8, speedtest.Progress(io.StringIO()), budget_s=0.3)
    assert not r.error and r.native and not r.native["cached"]
    for mode in ("captured", "upload", "eager"):
        s = r.score(mode)
        assert 0 < s <= 100, (mode, s)
    assert r.batch_bytes == 8 * 3 * 32 * 32 * 4 + 8 * 8


def test_server_without_native_baselines(local, monkeypatch):
    from reflop import session, speedtest

    sess = session.by_index(local.index)
    old = type(sess.caps)()
    old.CopyFrom(sess.caps)
    del old.capabilities[:]
    old.capabilities.extend(c for c in sess.caps.capabilities if c != "native_baseline")
    monkeypatch.setattr(sess, "caps", old)
    r = speedtest.measure(local, "mlp", 8, speedtest.Progress(io.StringIO()), budget_s=0.2)
    assert r.native is None and "reflopd" in r.native_note and r.score("captured") is None
    text = speedtest.render_one(r, speedtest.Network(3.0, 10.0), "srv")
    assert "No score" in text and "reflopd" in text


def test_busy_server(local, monkeypatch):
    from reflop import session, speedtest
    from reflop._pb import compute_pb2 as cpb
    from reflop.errors import ReflopRemoteError

    conn = session.by_index(local.index).conn
    real = conn.call

    def call(env, *a, **kw):
        if env.WhichOneof("body") == "native_baseline":
            raise ReflopRemoteError(cpb.ERROR_BUSY, "another session is using cpu; try again later")
        return real(env, *a, **kw)

    monkeypatch.setattr(conn, "call", call)
    r = speedtest.measure(local, "mlp", 8, speedtest.Progress(io.StringIO()), budget_s=0.2)
    assert r.native is None and "try again" in r.native_note


def test_all_table():
    from reflop import speedtest

    def res(name, native, cap, err=""):
        r = speedtest.Result(workload=name, batch=32, error=err)
        if not err:
            r.native = {"ms_per_step": native, "cached": True, "measured_unix": 0}
            r.captured = r.upload = r.eager = {"ms_per_step": cap}
        return r

    text = speedtest.render_all([res("mlp", 1.0, 4.0), res("lstm", 1.0, 40.0), res("gpt", 0, 0, "RuntimeError: x")],
                                speedtest.Network(4.1, 10.2), "srv cuda:0 GPU", 32, 95.0)
    assert "round trip 4.1 ms, upload 10.2 MiB/s" in text and "· 95 s" in text
    assert "25% slow" in text and "2.5% slow" in text and "failed: RuntimeError" in text
    assert text.index("mlp") < text.index("lstm") < text.index("gpt")   # best first, failures last
    assert "RNNs are slow" in text and "\033" not in text
    colored = speedtest.render_all([res("mlp", 3.0, 4.0)], speedtest.Network(4.1, 10.2), "srv", 32, 95.0, color=True)
    assert "\033[32m● " in colored and "75% great" in colored


def test_spinner_on_a_terminal():
    import time

    from reflop import speedtest

    class Tty(io.StringIO):
        def isatty(self):
            return True

    s = Tty()
    with speedtest.Progress(s) as p:
        p.update("unet: eager")
        time.sleep(0.35)
    out = s.getvalue()
    assert "unet: eager" in out and "\r" in out and out.endswith("\r\033[K")
