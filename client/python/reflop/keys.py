# SPDX-License-Identifier: Apache-2.0
"""The client's Ed25519 identity key (PLAN.md §12: key storage).

macOS: the private key lives in the login Keychain (one entry per
REFLOP_CLIENT_HOME). Elsewhere (or when the
Keychain is unavailable): ~/.config/reflop/id_ed25519, mode 0600, OpenSSH
format, which SSH pipe mode can use directly.

SSH pipe mode needs a key file for ssh(1), so on macOS the first SSH
connection also writes ~/.config/reflop/id_ed25519 (mode 0600): from then on
the key is protected by file permissions, not only by the Keychain.
"""

from __future__ import annotations

import base64
import os
import platform
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .config import config_dir

SERVICE = "reflop-client-key"


def _key_file() -> Path:
    return config_dir() / "id_ed25519"


def _scoped_path(key_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", key_id):
        raise ValueError("invalid GPU pairing key identifier")
    return config_dir() / "gpu-keys" / key_id


def pairing_key(key_id: str, *, create: bool = False) -> Ed25519PrivateKey:
    if not key_id:
        return load_or_create()
    path = _scoped_path(key_id)
    if create and not path.exists():
        _store_pairing_key(path, Ed25519PrivateKey.generate())
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"{path} is readable by other users; run: chmod 600 {path}")
    key = serialization.load_ssh_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("GPU pairing key is not Ed25519")
    return key


def new_ssh_pairing_key() -> Ed25519PrivateKey:
    key = Ed25519PrivateKey.generate()
    _store_pairing_key(_scoped_path(public_bytes(key).hex()), key)
    return key


def _store_pairing_key(path: Path, key: Ed25519PrivateKey) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
                             serialization.NoEncryption())
    with tempfile.NamedTemporaryFile(dir=path.parent) as temp:
        temp.write(data)
        temp.flush()
        try:
            os.link(temp.name, path)
        except FileExistsError:
            pass


def _keychain_ok() -> bool:
    return (platform.system() == "Darwin" and shutil.which("security") is not None
            and os.environ.get("REFLOP_NO_KEYCHAIN") != "1")


def _account() -> str:
    """Keychain account: one key per client home, so Guests simulated with
    REFLOP_CLIENT_HOME get separate identities."""
    user = os.environ.get("USER", "reflop")
    home = os.environ.get("REFLOP_CLIENT_HOME")
    return f"{user}:{Path(home).expanduser().resolve()}" if home else user


def _keychain_get() -> bytes | None:
    r = subprocess.run(["security", "find-generic-password", "-s", SERVICE, "-a", _account(), "-w"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return base64.b64decode(r.stdout.strip(), validate=True)
    except ValueError:
        return None


def _keychain_put(seed: bytes) -> bool:
    """Store the seed. The secret goes to `security -i` on stdin, never on a
    command line where `ps` would show it (SEC_SWEEP L7)."""
    acct = _account()
    if any(c in acct for c in '"\\\n\r'):
        return False  # cannot be quoted for `security -i`; use the key file
    line = f'add-generic-password -U -s {SERVICE} -a "{acct}" -w {base64.b64encode(seed).decode()}\n'
    try:
        subprocess.run(["security", "-i"], input=line, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    # `security -i` does not report a failed command in its exit status:
    # read the entry back to know whether it was stored.
    return _keychain_get() == seed


def load_or_create() -> Ed25519PrivateKey:
    if _keychain_ok():
        seed = _keychain_get()
        if seed is not None:
            return Ed25519PrivateKey.from_private_bytes(seed)
    f = _key_file()
    if f.exists():
        st = f.stat()
        if st.st_mode & 0o077:
            raise PermissionError(f"{f} is readable by other users; run: chmod 600 {f}")
        return serialization.load_ssh_private_key(f.read_bytes(), password=None)
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    if _keychain_ok() and _keychain_put(seed):
        return key
    write_key_file(key)
    return key


def write_key_file(key: Ed25519PrivateKey) -> Path:
    f = _key_file()
    f.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption())
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return f


def ssh_key_file(key: Ed25519PrivateKey, key_id: str = "") -> Path:
    """A key file ssh(1) can use (written on demand for SSH pipe mode)."""
    if key_id:
        return _scoped_path(key_id)
    f = _key_file()
    if not f.exists():
        write_key_file(key)
    return f


def public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def openssh_public(key: Ed25519PrivateKey, comment: str = "reflop") -> str:
    pub = public_bytes(key)
    blob = struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32) + pub
    return f"ssh-ed25519 {base64.b64encode(blob).decode()} {comment}"
