"""Structured audit logging for authentication events.

Writes JSON-lines to a dedicated audit log file. Each line is a
self-contained event with timestamp, event type, username, and result.
Thread-safe and rotate-friendly (append-only, no state).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

log = logging.getLogger(__name__)


class AuditEvent(StrEnum):
    AUTH_ATTEMPT = "auth_attempt"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAIL = "auth_fail"
    ENROLL = "enroll"
    DELETE = "delete"
    SPOOF_DETECTED = "spoof_detected"
    DAEMON_START = "daemon_start"
    DAEMON_STOP = "daemon_stop"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: str
    event: str
    username: str = ""
    peer_uid: int = -1
    result: bool = True
    details: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        # Remove empty optional fields for cleaner output
        if not d["username"]:
            del d["username"]
        if d["peer_uid"] == -1:
            del d["peer_uid"]
        if not d["details"]:
            del d["details"]
        return json.dumps(d, separators=(",", ":"))


class AuditLogger:
    """Thread-safe JSON-lines audit logger."""

    def __init__(self, path: Path | None = None):
        if path is None:
            path = _default_audit_path()
        self._path = path
        self._lock = threading.Lock()
        self._ensure_directory()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_directory(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self._path.parent, 0o700)
        except OSError as e:
            log.warning("Cannot create audit log directory %s: %s", self._path.parent, e)

    def log_event(
        self,
        event: AuditEvent,
        *,
        username: str = "",
        peer_uid: int = -1,
        result: bool = True,
        details: dict | None = None,
    ) -> None:
        """Write an audit event to the log file."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event=event.value,
            username=username,
            peer_uid=peer_uid,
            result=result,
            details=details or {},
        )
        line = entry.to_json() + "\n"

        with self._lock:
            try:
                with open(self._path, "a") as f:
                    f.write(line)
                # Set restrictive permissions on first write
                if self._path.exists():
                    os.chmod(self._path, 0o600)
            except OSError as e:
                log.warning("Failed to write audit event: %s", e)


def _default_audit_path() -> Path:
    """Default audit log path based on run mode."""
    if os.geteuid() == 0:
        return Path("/var/log/faceauth/audit.jsonl")
    return (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        / "faceauth"
        / "audit.jsonl"
    )


_singleton: AuditLogger | None = None
_singleton_lock = threading.Lock()


def get_audit_logger(path: Path | None = None) -> AuditLogger:
    """Get or create the singleton AuditLogger."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = AuditLogger(path)
        return _singleton


def reset_audit_logger() -> None:
    """Reset the singleton (for testing only)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
