# SPDX-License-Identifier: Apache-2.0

import pytest
from reflop import client, config, keys
from reflop._pb import auth_pb2 as apb
from reflop._pb import compute_pb2 as cpb
from reflop.errors import ReflopError

GPU = "gpu-00000000-1111-2222-3333-000000000000"
OTHER = "gpu-00000000-1111-2222-3333-111111111111"


@pytest.mark.parametrize("bad_scope,bad_id", [("", b"a" * 16), (OTHER, b"a" * 16), (GPU, b"b" * 16)])
def test_unpair_requires_the_original_gpu_and_identity(monkeypatch, tmp_path, bad_scope, bad_id):
    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    fp = b"p" * 32
    key_id = "a" * 64
    keys.pairing_key(key_id, create=True)
    entry = config.ServerEntry("gpu-a", "tls", ["localhost:7460"], fp.hex(), "alice", (b"a" * 16).hex(),
                               gpu_uuid=GPU, key_id=key_id)

    class WrongIdentity:
        def write(self, env):
            assert env.WhichOneof("body") != "unpair"

        def close(self):
            pass

    monkeypatch.setattr(client, "dial_tls", lambda *_: WrongIdentity())
    monkeypatch.setattr(client, "_hello", lambda _: cpb.HelloResponse(server_key_fingerprint=fp))
    monkeypatch.setattr(client, "_authenticate", lambda *_: apb.AuthResult(gpu_uuid=bad_scope, identity_id=bad_id))
    outcome, detail = client._tell_unpair(entry)
    assert outcome == client.UNPAIR_FAILED
    assert "does not match the pairing" in detail

    with pytest.raises(ReflopError, match="does not match the pairing"):
        client.RemoteConnection(entry)
