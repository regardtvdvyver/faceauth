"""Tests for faceauth.audit module."""

import json
import os
import stat
from datetime import datetime
from unittest.mock import patch

import pytest

from faceauth.audit import (
    AuditEntry,
    AuditEvent,
    AuditLogger,
    _default_audit_path,
    get_audit_logger,
    reset_audit_logger,
)


# ============================================================================
# AuditEntry
# ============================================================================


@pytest.mark.unit
def test_entry_to_json_all_fields():
    """Entry with all fields serializes correctly."""
    entry = AuditEntry(
        timestamp="2026-01-15T10:30:00+00:00",
        event="auth_success",
        username="alice",
        peer_uid=1000,
        result=True,
        details={"score": 0.87},
    )
    data = json.loads(entry.to_json())
    assert data["timestamp"] == "2026-01-15T10:30:00+00:00"
    assert data["event"] == "auth_success"
    assert data["username"] == "alice"
    assert data["peer_uid"] == 1000
    assert data["result"] is True
    assert data["details"]["score"] == 0.87


@pytest.mark.unit
def test_entry_to_json_minimal():
    """Entry with optional fields empty omits them."""
    entry = AuditEntry(
        timestamp="2026-01-15T10:30:00+00:00",
        event="daemon_start",
    )
    data = json.loads(entry.to_json())
    assert "username" not in data
    assert "peer_uid" not in data
    assert "details" not in data
    assert data["result"] is True


@pytest.mark.unit
def test_entry_to_json_is_compact():
    """JSON output has no extra whitespace."""
    entry = AuditEntry(timestamp="t", event="e", username="u", peer_uid=1)
    j = entry.to_json()
    assert " " not in j


# ============================================================================
# AuditEvent enum
# ============================================================================


@pytest.mark.unit
def test_audit_event_values():
    """All expected event types exist."""
    assert AuditEvent.AUTH_ATTEMPT == "auth_attempt"
    assert AuditEvent.AUTH_SUCCESS == "auth_success"
    assert AuditEvent.AUTH_FAIL == "auth_fail"
    assert AuditEvent.ENROLL == "enroll"
    assert AuditEvent.DELETE == "delete"
    assert AuditEvent.SPOOF_DETECTED == "spoof_detected"
    assert AuditEvent.DAEMON_START == "daemon_start"
    assert AuditEvent.DAEMON_STOP == "daemon_stop"


# ============================================================================
# AuditLogger
# ============================================================================


@pytest.mark.unit
def test_logger_creates_file(tmp_path):
    """Logger creates the audit log file on first event."""
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path)
    logger.log_event(AuditEvent.DAEMON_START)

    assert log_path.exists()


@pytest.mark.unit
def test_logger_file_permissions(tmp_path):
    """Audit log file has restrictive permissions."""
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path)
    logger.log_event(AuditEvent.DAEMON_START)

    mode = stat.S_IMODE(os.stat(log_path).st_mode)
    assert mode == 0o600


@pytest.mark.unit
def test_logger_directory_permissions(tmp_path):
    """Audit log directory has restrictive permissions."""
    log_dir = tmp_path / "subdir"
    log_path = log_dir / "audit.jsonl"
    AuditLogger(path=log_path)

    mode = stat.S_IMODE(os.stat(log_dir).st_mode)
    assert mode == 0o700


@pytest.mark.unit
def test_logger_appends_events(tmp_path):
    """Multiple events are appended as separate lines."""
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path)

    logger.log_event(AuditEvent.AUTH_ATTEMPT, username="alice")
    logger.log_event(AuditEvent.AUTH_SUCCESS, username="alice", details={"score": 0.9})
    logger.log_event(AuditEvent.AUTH_FAIL, username="bob", result=False)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 3

    first = json.loads(lines[0])
    assert first["event"] == "auth_attempt"
    assert first["username"] == "alice"

    second = json.loads(lines[1])
    assert second["event"] == "auth_success"
    assert second["details"]["score"] == 0.9

    third = json.loads(lines[2])
    assert third["event"] == "auth_fail"
    assert third["result"] is False


@pytest.mark.unit
def test_logger_timestamp_is_iso8601(tmp_path):
    """Events have valid ISO 8601 timestamps."""
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path)
    logger.log_event(AuditEvent.DAEMON_START)

    data = json.loads(log_path.read_text().strip())
    ts = datetime.fromisoformat(data["timestamp"])
    assert ts.tzinfo is not None  # timezone-aware


@pytest.mark.unit
def test_logger_event_fields(tmp_path):
    """Event has all specified fields."""
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path)
    logger.log_event(
        AuditEvent.ENROLL,
        username="alice",
        peer_uid=1000,
        result=True,
        details={"samples": 5},
    )

    data = json.loads(log_path.read_text().strip())
    assert data["event"] == "enroll"
    assert data["username"] == "alice"
    assert data["peer_uid"] == 1000
    assert data["result"] is True
    assert data["details"]["samples"] == 5


@pytest.mark.unit
def test_logger_handles_write_failure_gracefully(tmp_path):
    """Write failure logs warning but doesn't raise."""
    # Create a read-only directory
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir(mode=0o700)
    log_path = readonly_dir / "audit.jsonl"

    logger = AuditLogger(path=log_path)

    # Make directory read-only AFTER logger init (which creates the dir)
    os.chmod(readonly_dir, 0o500)

    try:
        # Should not raise
        logger.log_event(AuditEvent.DAEMON_START)
    finally:
        # Restore permissions for cleanup
        os.chmod(readonly_dir, 0o700)


@pytest.mark.unit
def test_logger_path_property(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path)
    assert logger.path == log_path


# ============================================================================
# Default path
# ============================================================================


@pytest.mark.unit
def test_default_path_system_mode():
    """System mode (root) uses /var/log/faceauth/."""
    with patch("faceauth.audit.os.geteuid", return_value=0):
        path = _default_audit_path()
    assert str(path) == "/var/log/faceauth/audit.jsonl"


@pytest.mark.unit
def test_default_path_user_mode():
    """User mode uses XDG data home."""
    with patch("faceauth.audit.os.geteuid", return_value=1000):
        path = _default_audit_path()
    assert "faceauth/audit.jsonl" in str(path)
    assert "/var/log" not in str(path)


# ============================================================================
# Singleton
# ============================================================================


@pytest.mark.unit
def test_singleton_returns_same_instance(tmp_path):
    """get_audit_logger returns the same instance."""
    reset_audit_logger()
    try:
        logger1 = get_audit_logger(tmp_path / "audit.jsonl")
        logger2 = get_audit_logger()
        assert logger1 is logger2
    finally:
        reset_audit_logger()


@pytest.mark.unit
def test_reset_clears_singleton(tmp_path):
    """reset_audit_logger clears the singleton."""
    reset_audit_logger()
    try:
        logger1 = get_audit_logger(tmp_path / "a.jsonl")
        reset_audit_logger()
        logger2 = get_audit_logger(tmp_path / "b.jsonl")
        assert logger1 is not logger2
        assert logger1.path != logger2.path
    finally:
        reset_audit_logger()
