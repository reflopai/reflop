# SPDX-License-Identifier: Apache-2.0
"""Backward on PyTorch's autograd thread (ADR-053).

PyTorch runs this device's backward on a thread of its own (`Dummy-N`, C++
`pt_autograd_0`) while the thread that called `.backward()` waits. That
thread sees none of the caller's thread-local state, and an exception that
leaves reFLOP's `__torch_dispatch__` there aborts the process inside
PyTorch's device guard.

- `carried()`: `backward()` and `autograd.grad()` on a reflop tensor carry the
  caller's placement mode and the root's session to the autograd thread
  while they run. Two calls in progress at once share that thread: the most
  recent one's context applies.
- `keep()` / `claim()`: an op that fails on the autograd thread keeps its
  error in its session and returns placeholders (see tensor._failed_in_backward).
  The backward call raises the error when it returns; otherwise the next
  read in that session does.
"""

from __future__ import annotations

import contextlib
import threading

import torch

_lock = threading.Lock()
_active: list[Context] = []  # backward calls in progress, most recent last


class Context:
    __slots__ = ("mode", "session")

    def __init__(self, mode: str, session) -> None:
        self.mode = mode
        self.session = session


current_node = torch._C._current_autograd_node


def in_backward() -> bool:
    """True while this thread runs an autograd node."""
    return current_node() is not None


def context() -> Context | None:
    """The context of the backward call this thread runs nodes for, if any."""
    if not _active or current_node() is None:
        return None
    try:
        return _active[-1]
    except IndexError:  # the last call ended meanwhile
        return None


@contextlib.contextmanager
def carried(mode: str, session):
    c = Context(mode, session)
    with _lock:
        _active.append(c)
    try:
        yield
    finally:
        with _lock:
            _active.remove(c)


def keep(sess, e: BaseException) -> None:
    """Keep an error raised on the autograd thread until the backward call
    returns (or the session's next read)."""
    if sess.backward_error is None:
        sess.backward_error = e


def claim() -> BaseException | None:
    """Take a kept backward error, if any session has one."""
    from . import session

    for s in list(session._sessions.values()):
        e = s.backward_error
        if e is not None:
            s.backward_error = None
            return e
    return None


def run(func, types, args, kwargs, sess):
    """Run backward() or autograd.grad() for a reflop root in session sess."""
    from . import _placement

    with carried(_placement.mode(), sess):
        try:
            out = torch._C._disabled_torch_function_impl(func, types, args, kwargs)
        except Exception:
            e = claim()
            if e is not None:
                raise e from None
            raise
    e = claim()
    if e is not None:
        raise e
    return out
