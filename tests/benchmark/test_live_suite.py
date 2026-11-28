# SPDX-License-Identifier: Apache-2.0
"""Offline checks of the live suite in benchmark/ (it runs against a real reflopd, never in CI): what it publishes
carries nothing private, the report reproduces the committed tables, and its Owner guards hold."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmark"
# report dir, NEW run, OLD run, generated rows that pin the tables
REPORTS = [
    ("2026-11-28", "2026-11-28_00-00-27", "2026-11-21_16-09-26",
     ("| TOTAL | 662 | 6 | 14 | 0 | 4 nonzero | 3921 | 657 | 17 | 22 | 0 | 8 nonzero | 3976 |",
      "| capture | resident | 30 | -21.2% |")),
    ("2026-11-28-0.4.4", "2026-11-28_18-17-58", "2026-11-28_00-00-27",
     ("| TOTAL | 683 | 0 | 14 | 20 | 0 nonzero | 3564 | 662 | 6 | 14 | 0 | 4 nonzero | 3921 |",
      "| capture | resident | 30 | -2.3% |")),
    ("2026-11-28-0.4.5", "2026-11-28_23-59-43", "2026-11-28_18-17-58",
     ("| TOTAL | 683 | 0 | 14 | 20 | 0 nonzero | 3202 | 683 | 0 | 14 | 20 | 0 nonzero | 3564 |",
      "| capture | resident | 30 | -4.8% |")),
]


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Import benchmark/live with its results and scratch directories under tmp_path."""
    monkeypatch.setenv("E2E_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("E2E_SCRATCH", str(tmp_path / "scratch"))
    for k in ("E2E_OWNER_SSH", "E2E_LOCAL", "E2E_OWNER_HOME", "E2E_GUEST_IDENTITY", "E2E_NO_OWNER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(str(BENCH))
    for m in [m for m in sys.modules if m == "live" or m.startswith("live.")]:
        del sys.modules[m]
    import live.harness
    import live.owner
    import live.report
    import live.sanitize

    yield live
    for m in [m for m in sys.modules if m == "live" or m.startswith("live.")]:
        del sys.modules[m]


PRIVATE = {"server": "myowner-box", "ssh": "myowner", "ip4": "100.64.12.34", "lan": "192.168.7.9",
           "ip6": "fd7a:115c:a1e0::12", "uuid": "gpu-0a1b2c3d-1111-2222-3333-444455556666",
           "mac_home": "/Users/somebody/proj", "linux_home": "/home/someone/.local/share/reflop/runtime/cu1/python",
           "tmp": "/tmp/build-1000/-home-someone-proj/x"}


def fake_run(d: Path):
    d.mkdir(parents=True)
    run = {"run_id": "r1", "versions": {"reflop_client": "9.9.9", "torch": "2.15.0"},
           "settings": {"server": PRIVATE["server"], "owner_ssh": PRIVATE["ssh"]},
           "reflop_status": f"* {PRIVATE['server']} online at {PRIVATE['ip4']}:7460",
           "reflopd_status_start": "anything", "reflopd_status_end": None,
           "owner_gpu_start": {"gpus": [{"name": "Card"}], "processes": [
               {"pid": 7, "user": "someone", "cmd": PRIVATE["linux_home"] + "/bin/python3", "used_memory_mib": 9.0,
                "reflop_worker": True}]},
           "network_start": {"ping_ms": {"avg": 3.2}, "interface": "en0", "tailscale": {"peer": "peer-host", "path": "direct",
                                                         "addr": PRIVATE["lan"] + ":41641"}},
           "suites": [{"suite": "a01_cli", "exit_code": 0, "seconds": 1.0, "counts": {"pass": 1}}]}
    (d / "run.json").write_text(json.dumps(run))
    detail = (f"paired with {PRIVATE['ssh']} via {PRIVATE['ip6']} on {PRIVATE['uuid']} at 12:34:56; "
              f"File \"{PRIVATE['mac_home']}/x.py\"; key in {PRIVATE['tmp']}; e2e-alice ok; torch 2.15.0")
    (d / "a01_cli.json").write_text(json.dumps({"suite": "a01_cli", "description": "a01", "meta": {},
                                                "cases": [{"id": "c", "status": "pass", "detail": detail}],
                                                "metrics": []}))
    (d / "logs").mkdir()
    (d / "logs" / "owner_commands.log").write_text(PRIVATE["ip4"])


def test_sanitize_removes_what_a_run_records(live, tmp_path):
    fake_run(tmp_path / "raw")
    live.sanitize.sanitize_run(tmp_path / "raw", tmp_path / "pub")
    text = "".join(f.read_text() for f in (tmp_path / "pub").rglob("*") if f.is_file())
    for what, v in PRIVATE.items():
        assert v not in text, what
    assert "peer-host" not in text and "someone" not in text and "somebody" not in text
    assert not (tmp_path / "pub" / "logs").exists()
    assert live.sanitize.check([tmp_path / "pub"]) == 0
    # What is not private stays readable.
    for keep in ("e2e-alice", "12:34:56", "2.15.0", "9.9.9", '"tailscale_path":"direct"', '"ping_ms":{"avg":3.2}',
                 '"interface":"en0"'):
        assert keep in text, keep
    assert live.sanitize.check([tmp_path / "raw"]) == 1


@pytest.mark.parametrize("name", [r[0] for r in REPORTS])
def test_published_report_is_clean(live, name):
    assert live.sanitize.check([BENCH / "report" / name]) == 0


@pytest.mark.parametrize("name, new_run, old_run, rows", REPORTS)
def test_report_reproduces_the_published_tables(live, name, new_run, old_run, rows):
    report = BENCH / "report" / name
    data = report / "data"
    new, old = (live.report.Run(data / n) for n in (new_run, old_run))
    text = live.report.report(new, old, None)
    published = (report / "README.md").read_text()
    for row in rows:
        assert row in text
    # Everything but the hand-written section 2 is generated.
    start, end = "## 3. Suite totals", "## Appendix A."
    assert text[text.index(start):text.index(end)] == published[published.index(start):published.index(end)]


def test_owner_guards(live, monkeypatch, tmp_path):
    o = live.owner
    assert o.is_ours("guest") and o.is_ours("guest@gpu-0a1b2c3d-1111-2222-3333-444455556666")
    assert o.is_ours("e2e-alice@gpu-0a1b2c3d-1111-2222-3333-444455556666")
    assert not o.is_ours("alice") and not o.is_ours("guestx")
    with pytest.raises(RuntimeError, match="not one of our sessions"):
        monkeypatch.setattr(o, "sessions", lambda: [{"session": "abcd1234", "identity": "alice@gpu-x"}])
        o.check_target("abcd1234")
    # No Owner configured: nothing is reached, Owner suites skip.
    assert o.available() is False
    with pytest.raises(o.Skip):
        o.run("reflopd version")
    # Local mode never runs against the default reflopd home.
    monkeypatch.setenv("E2E_OWNER_HOME", str(Path.home() / ".local/share/reflop"))
    with pytest.raises(RuntimeError, match="throwaway"):
        o.local_home()
    monkeypatch.setenv("E2E_OWNER_HOME", str(tmp_path / "owner"))
    assert o.local_home() == str((tmp_path / "owner").resolve())


def test_guest_identity_is_configurable(live, monkeypatch):
    monkeypatch.setenv("E2E_GUEST_IDENTITY", "mac")
    o = importlib.reload(live.owner)
    assert o.is_ours("mac@gpu-x") and not o.is_ours("guest")
