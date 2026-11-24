# SPDX-License-Identifier: Apache-2.0

import base64
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from reflop import client, config, keys
from reflop._pb import auth_pb2 as apb
from reflop._pb import capabilities_pb2 as capb
from reflop._pb import compute_pb2 as cpb
from reflop._pb import graph_pb2 as gpb
from reflop.cli.main import main
from reflop.errors import ReflopError

GPU_A = "gpu-00000000-1111-2222-3333-000000000000"
GPU_B = "gpu-00000000-1111-2222-3333-111111111111"
FP = bytes(range(32))
ADDRESS = "lab.example:7460"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    monkeypatch.setenv("REFLOP_NO_KEYCHAIN", "1")
    monkeypatch.delenv("REFLOP_SERVER", raising=False)


def invite(uuid, *, fp=FP, address=ADDRESS, ssh_key=b"", gpu_name="NVIDIA RTX 5070"):
    inv = apb.InviteCode(version=1, transport=apb.TRANSPORT_SSH if ssh_key else apb.TRANSPORT_TLS,
                         addresses=[address], server_key_fingerprint=fp, invite_id=b"x" * 16,
                         secret=b"y" * 32, identity_name="alice", gpu_uuid=uuid, gpu_name=gpu_name,
                         ssh_guest_public_key=ssh_key)
    return client.INVITE_PREFIX + base64.b32encode(inv.SerializeToString()).decode().rstrip("=").lower()


class PairReplies:
    def __init__(self, uuid, identity_id, *, ssh=False, unpair=False, status=False, device=gpb.DEVICE_CUDA, fp=FP):
        self.sent = []
        self.closed = False
        replies = [cpb.Envelope(hello_response=cpb.HelloResponse(protocol_version=1, server_nonce=b"n" * 32,
                                                                  server_key_fingerprint=fp))]
        if not (ssh or unpair or status):
            replies.append(cpb.Envelope(pair_result=apb.PairResult(identity_id=identity_id)))
        caps = capb.ServerCapabilities(devices=[capb.DeviceInfo(device=gpb.Device(kind=device))])
        replies.append(cpb.Envelope(auth_result=apb.AuthResult(gpu_uuid=uuid, identity_name=f"alice@{uuid}",
                                                               identity_id=identity_id, capabilities=caps)))
        if unpair:
            replies.append(cpb.Envelope(unpaired=cpb.Unpaired()))
        self.replies = iter(replies)

    def read(self):
        return next(self.replies)

    def write(self, env):
        self.sent.append(env)

    def close(self):
        self.closed = True


def test_pair_select_and_unpair_gpus_at_one_address(monkeypatch, capsys, tmp_path):
    a = PairReplies(GPU_A, b"a" * 16)
    b = PairReplies(GPU_B, b"b" * 16)
    peers = iter((a, b))
    monkeypatch.setattr(client, "dial_tls", lambda *_: next(peers))
    assert main(["pair", invite(GPU_A), "--name", "lab-a"]) == 0
    assert main(["pair", invite(GPU_B), "--name", "lab-b"]) == 0
    capsys.readouterr()
    assert a.sent[1].pair.gpu_uuid == GPU_A and b.sent[1].pair.gpu_uuid == GPU_B
    assert a.sent[1].pair.public_key != b.sent[1].pair.public_key
    assert json.loads((tmp_path / "servers.json").read_text())["servers"] == {}
    monkeypatch.setattr(client, "pairing_online", lambda entry: entry.gpu_uuid == GPU_A)
    assert main(["status"]) == 0
    status = capsys.readouterr().out
    assert status.splitlines()[0].split() == ["NAME", "STATUS", "GPU", "ADDRESS"]
    assert re.search(r"^\* lab-a +online ", status, re.MULTILINE)
    assert re.search(r"^  lab-b +offline ", status, re.MULTILINE)
    assert status.count(ADDRESS) == 2

    assert main(["use", "lab-b"]) == 0
    assert config.ClientConfig.load().get(None).gpu_uuid == GPU_B
    assert config.ClientConfig.load().get("lab-a").gpu_uuid == GPU_A
    assert main(["status"]) == 0
    assert "* lab-b" in capsys.readouterr().out

    unpair = PairReplies(GPU_A, b"a" * 16, unpair=True)
    monkeypatch.setattr(client, "dial_tls", lambda *_: unpair)
    entry, outcome, _ = client.unpair("lab-a")
    assert entry.gpu_uuid == GPU_A and outcome == client.UNPAIR_REVOKED
    assert unpair.sent[1].authenticate.public_key == a.sent[1].pair.public_key
    assert config.ClientConfig.load().get(None).gpu_uuid == GPU_B
    client.unpair("lab-b", local_only=True)
    assert config.ClientConfig.load().servers == {}


def test_duplicate_uuid_and_alias_leave_invite_unused(monkeypatch):
    monkeypatch.setattr(client, "dial_tls", lambda *_: PairReplies(GPU_A, b"a" * 16))
    client.pair(invite(GPU_A), "lab")
    monkeypatch.setattr(client, "dial_tls", lambda *_: pytest.fail("the invite was sent"))
    with pytest.raises(ReflopError, match="already paired as 'lab'"):
        client.pair(invite(GPU_A, fp=b"d" * 32, address="other.example:8850"))
    with pytest.raises(ReflopError, match="pairing name 'lab' is already in use"):
        client.pair(invite(GPU_B), "lab")
    assert config.ClientConfig.load().get(None).gpu_uuid == GPU_A


def test_legacy_and_gpu_pairings_coexist(monkeypatch, tmp_path):
    legacy = config.ServerEntry("old", "tls", [ADDRESS], FP.hex(), "alice")
    config.ClientConfig("old", {"old": legacy}).save()
    monkeypatch.setattr(client, "dial_tls", lambda *_: PairReplies(GPU_A, b"a" * 16))
    client.pair(invite(GPU_A), "new")
    assert config.ClientConfig.load().get(None) == legacy
    tree = json.loads((tmp_path / "servers.json").read_text())
    assert list(tree["servers"]) == ["old"]
    assert list(tree["gpu_pairings"]) == ["new"]
    assert "gpu_uuid" not in tree["servers"]["old"]


def test_concurrent_pairs_save_both_cards(monkeypatch):
    other = "second.example:7460"
    peers = {ADDRESS: PairReplies(GPU_A, b"a" * 16), other: PairReplies(GPU_B, b"b" * 16)}
    start = threading.Barrier(2)

    def dial(addresses, _fp):
        time.sleep(0.02)
        return peers[addresses[0]]

    def pair(uuid, address):
        start.wait(timeout=5)
        return client.pair(invite(uuid, address=address))

    monkeypatch.setattr(client, "dial_tls", dial)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(pair, GPU_A, ADDRESS)
        second = pool.submit(pair, GPU_B, other)
        a, b = first.result(timeout=5), second.result(timeout=5)
        assert a.gpu_uuid == GPU_A
        assert b.gpu_uuid == GPU_B
        assert {a.name, b.name} == {"nvidia-rtx-5070", "nvidia-rtx-5070-2"}
    assert len(config.ClientConfig.load().servers) == 2


@pytest.mark.parametrize(("description", "name"), [
    ("NVIDIA GeForce MX450", "nvidia-mx450"),
    ("NVIDIA H100 80GB HBM3", "nvidia-h100"),
    ("NVIDIA H100 PCIe", "nvidia-h100"),
    ("NVIDIA GeForce RTX 4090 Ti", "nvidia-rtx-4090-ti"),
    ("Tesla V100-SXM2-16GB", "nvidia-v100"),
    ("", f"gpu-{GPU_A[-8:]}"),
])
def test_gpu_default_names_and_numbering_at_one_address(monkeypatch, description, name):
    peers = iter((PairReplies(GPU_A, b"a" * 16), PairReplies(GPU_B, b"b" * 16)))
    monkeypatch.setattr(client, "dial_tls", lambda *_: next(peers))
    assert client.pair(invite(GPU_A, gpu_name=description)).name == name
    if description:
        assert client.pair(invite(GPU_B, gpu_name=description)).name == f"{name}-2"
    assert config.ClientConfig.load().get(None).addresses == [ADDRESS]


def test_rename_keeps_identity_key_and_default(monkeypatch, capsys):
    legacy = config.ServerEntry("old", "tls", [ADDRESS], FP.hex(), "alice")
    monkeypatch.setattr(client, "dial_tls", lambda *_: PairReplies(GPU_A, b"a" * 16))
    entry = client.pair(invite(GPU_A))
    cfg = config.ClientConfig.load()
    cfg.servers["old"] = legacy
    cfg.save()

    monkeypatch.setattr(client, "dial_tls", lambda *_: pytest.fail("rename made a network connection"))
    assert main(["rename", entry.name, "lab"]) == 0
    cfg = config.ClientConfig.load()
    renamed = cfg.get(None)
    assert cfg.default == renamed.name == "lab"
    assert renamed.__dict__ == {**entry.__dict__, "name": "lab"}
    assert entry.name not in cfg.servers and cfg.get("old") == legacy

    assert main(["rename", "lab", "lab"]) == 0
    for old, new, message in [("lab", "old", "already in use"), ("absent", "lab", "no pairing"),
                              ("", "new", "no pairing"), ("lab", "new\nname", "invalid pairing name")]:
        assert main(["rename", old, new]) == 1
        assert message in capsys.readouterr().err
        assert config.ClientConfig.load() == cfg
    assert main(["rename", "old", "legacy"]) == 0
    assert config.ClientConfig.load().get(None) == renamed
    assert config.ClientConfig.load().get("legacy").fingerprint == FP.hex()


@pytest.mark.parametrize(("device", "fp", "identity_id", "online"), [
    (gpb.DEVICE_CUDA, FP, b"a" * 16, True),
    (gpb.DEVICE_CPU, FP, b"a" * 16, False),
    (gpb.DEVICE_CUDA, FP, b"b" * 16, False),
    (gpb.DEVICE_CUDA, b"x" * 32, b"a" * 16, False),
])
def test_status_authenticates_and_checks_device_without_opening_a_session(monkeypatch, device, fp, identity_id, online):
    monkeypatch.setattr(client, "dial_tls", lambda *_: PairReplies(GPU_A, b"a" * 16))
    entry = client.pair(invite(GPU_A))
    peer = PairReplies(GPU_A, identity_id, status=True, device=device, fp=fp)

    def dial(addresses, fingerprint, timeout, *, read_timeout):
        assert addresses == [ADDRESS] and fingerprint == FP
        assert 0 < timeout <= 3 and 0 < read_timeout <= 3
        return peer

    monkeypatch.setattr(client, "dial_tls", dial)
    assert client.pairing_online(entry) is online
    assert peer.closed
    assert [env.WhichOneof("body") for env in peer.sent] == (
        ["hello", "authenticate"] if fp == FP else ["hello"])


def test_status_table_fits_terminal_and_uses_color(monkeypatch, capsys):
    cfg = config.ClientConfig()
    for name in ("online-alias", "this-alias-is-long-enough-to-wrap"):
        cfg.servers[name] = config.ServerEntry(name, "tls", ["a-very-long-address.example.edu:7460"],
                                               FP.hex(), "alice", gpu_name="A long GPU name\nwith\ttabs")
    cfg.save()
    monkeypatch.setattr(client, "pairing_online", lambda entry: entry.name == "online-alias")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("COLUMNS", "52")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert main(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["NAME", "STATUS", "GPU", "ADDRESS"]
    assert "\x1b[32monline\x1b[0m" in lines[1]
    assert "\x1b[31moffline\x1b[0m" in lines[2]
    assert "…" in lines[2]
    assert all(len(re.sub(r"\x1b\[\d+m", "", line)) <= 52 for line in lines[:3])
    monkeypatch.setenv("NO_COLOR", "1")
    assert main(["status"]) == 0
    assert "\x1b" not in capsys.readouterr().out


def test_legacy_status_without_a_key_does_not_create_one(tmp_path):
    entry = config.ServerEntry("old", "tls", [ADDRESS], FP.hex(), "alice")
    assert client.pairing_online(entry) is False
    assert not (tmp_path / "id_ed25519").exists()


def test_ssh_uses_the_scoped_key(monkeypatch, capsys, tmp_path):
    assert main(["key", "--ssh", "--new"]) == 0
    _, blob, _ = capsys.readouterr().out.split()
    raw_pub = base64.b64decode(blob)[-32:]
    key_id = raw_pub.hex()
    peer = PairReplies(GPU_B, b"b" * 16, ssh=True)

    def dial(address, key, fingerprint, host_keys, *, key_id):
        assert address == "user@lab" and fingerprint == FP.hex() and host_keys == []
        assert keys.public_bytes(key) == raw_pub
        assert keys.ssh_key_file(key, key_id) == tmp_path / "gpu-keys" / raw_pub.hex()
        return peer

    monkeypatch.setattr(client, "dial_ssh", dial)
    entry = client.pair(invite(GPU_B, address="user@lab", ssh_key=raw_pub))
    assert entry.transport == "ssh" and entry.key_id == key_id
    assert [env.WhichOneof("body") for env in peer.sent] == ["hello", "authenticate"]
    assert not (tmp_path / "id_ed25519").exists()


def test_auto_for_scoped_gpu_never_uses_cpu():
    conn = client.RemoteConnection.__new__(client.RemoteConnection)
    conn.entry = config.ServerEntry("lab", "tls", [ADDRESS], FP.hex(), "alice", gpu_uuid=GPU_A)
    conn.caps = capb.ServerCapabilities(devices=[capb.DeviceInfo(device=gpb.Device(kind=gpb.DEVICE_CPU))])
    with pytest.raises(ReflopError, match="paired GPU 'lab' is unavailable"):
        conn._choose_device("auto")
    assert conn._choose_device("cpu").kind == gpb.DEVICE_CPU
