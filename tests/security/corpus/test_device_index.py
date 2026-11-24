# SPDX-License-Identifier: Apache-2.0
"""Explicit device indexes must identify the session that was requested."""

import numpy as np
import pytest
import reflop
import torch
from reflop import client, connection, session


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch):
    monkeypatch.setattr(session, "_sessions", {})
    monkeypatch.setattr(session, "_default", [])
    monkeypatch.setattr(session, "_seed", None)
    monkeypatch.delenv("REFLOP_LOCAL", raising=False)
    yield
    reflop.close()


@pytest.fixture
def open_session_without_more_connections(monkeypatch):
    reflop.device(local="cpu")
    opened = session.by_index(0)

    def unexpected_connection(*args, **kwargs):
        pytest.fail("invalid device index reached connection setup")

    monkeypatch.setattr(client, "connect", unexpected_connection)
    monkeypatch.setattr(connection, "LocalConnection", unexpected_connection)
    return opened


@pytest.mark.parametrize("local", [None, "cpu"], ids=["remote", "in-process"])
@pytest.mark.parametrize("index", [-1, 128, 255, 256, 257, 1 << 80])
def test_out_of_range_index_is_rejected_before_connecting(open_session_without_more_connections, index, local):
    with pytest.raises(ValueError, match="device index must be between 0 and 127"):
        reflop.device(index, local=local)

    assert session._sessions == {0: open_session_without_more_connections}
    assert session.current() is open_session_without_more_connections


@pytest.mark.parametrize("index", [None, False, 0.0, "0"])
def test_noninteger_index_is_rejected_before_connecting(open_session_without_more_connections, index):
    with pytest.raises(TypeError, match="device index must be an integer"):
        reflop.device(index, checkpoint=True)

    assert session._sessions == {0: open_session_without_more_connections}
    assert open_session_without_more_connections.checkpoints is None


def test_supported_indexes_keep_their_sessions():
    lower = reflop.device(local="cpu")
    upper = reflop.device(np.int64(127), local="cpu")
    opened = session.by_index(127)

    assert (lower.index, upper.index) == (0, 127)
    assert reflop.device(127, local="cpu") == upper
    assert session.by_index(127) is opened
    assert session.current() is opened
    tensor = torch.arange(3, device=upper)
    assert tensor.device == upper
    assert torch.equal(tensor.cpu(), torch.arange(3))
