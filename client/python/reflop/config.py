# SPDX-License-Identifier: Apache-2.0
"""Client configuration: paired servers and the default server.

Stored in ~/.config/reflop/servers.json (mode 0600). Nothing in here is
ever sent to a server (invariant I12).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import ReflopError, sanitize

GPU_UUID = re.compile(r"gpu-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def validate_pairing_name(name: str) -> None:
    if not name.strip() or name.startswith("-") or not name.isprintable() or sanitize(name, 80) != name:
        raise ReflopError("invalid pairing name")


def config_dir() -> Path:
    if h := os.environ.get("REFLOP_CLIENT_HOME"):
        return Path(h)
    base = os.environ.get("XDG_CONFIG_HOME")
    return Path(base) / "reflop" if base else Path.home() / ".config" / "reflop"


@contextlib.contextmanager
def config_lock():
    """Serialize changes to pairings and the default across CLI processes."""
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / "servers.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


@dataclass
class ServerEntry:
    name: str
    transport: str                     # "tls" or "ssh"
    addresses: list[str]
    fingerprint: str                   # hex SHA-256 of the server SPKI (pinned)
    identity: str                      # our identity name on that server
    identity_id: str = ""
    ssh_port: int | None = None
    extra: dict = field(default_factory=dict)
    ssh_host_keys: list[str] = field(default_factory=list)  # SSH mode: pinned sshd host keys
    gpu_uuid: str = ""
    gpu_name: str = ""
    key_id: str = ""


@dataclass
class ClientConfig:
    default: str = ""
    servers: dict[str, ServerEntry] = field(default_factory=dict)

    @classmethod
    def load(cls) -> ClientConfig:
        p = config_dir() / "servers.json"
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        servers = d.get("servers", {})
        scoped = d.get("gpu_pairings", {})
        if servers.keys() & scoped.keys():
            raise ValueError("duplicate pairing aliases in servers.json")
        for v in scoped.values():
            if (not GPU_UUID.fullmatch(v.get("gpu_uuid", ""))
                    or not re.fullmatch(r"[0-9a-f]{64}", v.get("key_id", ""))):
                raise ValueError("invalid GPU pairing in servers.json")
        return cls(default=d.get("default", ""), servers={k: ServerEntry(**v) for k, v in (servers | scoped).items()})

    def save(self) -> None:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        legacy, scoped = {}, {}
        for name, entry in self.servers.items():
            fields = asdict(entry)
            if entry.gpu_uuid:
                scoped[name] = fields
            else:
                for field_name in ("gpu_uuid", "gpu_name", "key_id"):
                    del fields[field_name]
                legacy[name] = fields
        data = json.dumps({"default": self.default, "servers": legacy, "gpu_pairings": scoped}, indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".servers-")
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, d / "servers.json")

    def get(self, name: str | None) -> ServerEntry:
        name = name or os.environ.get("REFLOP_SERVER") or self.default
        if not name:
            if self.servers:
                raise RuntimeError("no default pairing; run `reflop status` and `reflop use <name>`")
            raise RuntimeError("no reflop server is paired; run `reflop pair <invite-code>` first")
        if name not in self.servers:
            raise RuntimeError(f"no paired server named {name!r} (see `reflop servers`)")
        return self.servers[name]
