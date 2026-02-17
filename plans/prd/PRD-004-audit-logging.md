# PRD-004: Audit Logging
**Priority:** P1
**Effort:** S
**Dependencies:** None

## Problem
faceauth is a security-critical authentication system but has no auditable trail of authentication events. Current issues:

1. **No accountability**: Can't determine who accessed system when
2. **Security leak**: Similarity scores logged to INFO level (should be audit-only)
3. **No forensics**: Can't investigate failed auth attempts or brute force patterns
4. **Compliance gap**: Security tools require audit trails for enterprise use

Example of bad current logging:
```
INFO: Authentication successful for user 'alice' (similarity: 0.87)
```

Similarity scores are sensitive data (reveal biometric match quality) and shouldn't be in general logs.

## Requirements

### Must Have
- **Structured JSON-lines audit log** with fields:
  - `timestamp` (ISO 8601 UTC)
  - `event` (enum: auth_attempt, auth_success, auth_fail, enroll, delete, spoof_detected)
  - `username` (target user)
  - `peer_uid` (requesting user ID from PAM)
  - `result` (success/fail/spoof)
  - `reason` (why failed: no_embedding, spoof_detected, similarity_low, etc.)
  - `similarity` (float, only for auth events)
  - `session_id` (track related events)
- **Sensitive data isolation**: Remove similarity scores from INFO logs
- **Restrictive permissions**: Audit log readable only by root (0o600)
- **Configurable path**: Default `~/.local/share/faceauth/audit.log` (user) or `/var/log/faceauth/audit.log` (system)

### Should Have
- **Log rotation**: Integrate with logrotate or internal rotation (max size/age)
- **CLI viewer**: `faceauth audit` command to view/tail audit log with formatting
- **Filters**: `faceauth audit --user alice --since 2024-01-15`
- **Statistics**: `faceauth audit stats` (success rate, top users, spoof attempts)
- **Alert markers**: Flag suspicious patterns (10+ fails from same peer)

### Nice to Have
- Syslog forwarding option (for centralized logging)
- Rate limiting audit logs (prevent flooding)
- Audit log signing/integrity verification
- Export to CSV/JSON for analysis tools
- Real-time tail with color coding

## Technical Approach

### Implementation

New module: `faceauth/audit.py`

```python
import json
import time
from pathlib import Path
from typing import Optional
from enum import Enum

class AuditEvent(Enum):
    AUTH_ATTEMPT = "auth_attempt"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAIL = "auth_fail"
    ENROLL = "enroll"
    DELETE = "delete"
    SPOOF_DETECTED = "spoof_detected"

class AuditLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._ensure_secure_file()

    def log_event(
        self,
        event: AuditEvent,
        username: str,
        peer_uid: int,
        result: str,
        reason: Optional[str] = None,
        similarity: Optional[float] = None,
        session_id: Optional[str] = None,
    ):
        """Write structured audit event to log."""
        entry = {
            "timestamp": time.time(),
            "event": event.value,
            "username": username,
            "peer_uid": peer_uid,
            "result": result,
            "reason": reason,
            "similarity": similarity,
            "session_id": session_id,
        }
        # Remove None values
        entry = {k: v for k, v in entry.items() if v is not None}

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

### Integration Points

Daemon modifications (`faceauth/daemon.py`):

```python
# Initialize audit logger
audit = AuditLogger(config.audit_log_path)

# Before authentication
audit.log_event(
    AuditEvent.AUTH_ATTEMPT,
    username=username,
    peer_uid=peer_uid,
    result="pending",
    session_id=session_id,
)

# On success
audit.log_event(
    AuditEvent.AUTH_SUCCESS,
    username=username,
    peer_uid=peer_uid,
    result="success",
    similarity=similarity,
    session_id=session_id,
)

# On spoof detection
audit.log_event(
    AuditEvent.SPOOF_DETECTED,
    username=username,
    peer_uid=peer_uid,
    result="fail",
    reason="antispoof_score_low",
    similarity=similarity,
    session_id=session_id,
)

# Remove from INFO log:
# logger.info(f"Auth success (similarity: {similarity})")
# Replace with:
# logger.info(f"Auth success for {username}")
```

### CLI Viewer

Add to `cli/faceauth_cli.py`:

```python
@click.command()
@click.option('--user', help='Filter by username')
@click.option('--since', help='Events since date (ISO format)')
@click.option('--follow', '-f', is_flag=True, help='Tail mode')
@click.option('--stats', is_flag=True, help='Show statistics')
def audit(user, since, follow, stats):
    """View audit log."""
    if stats:
        show_audit_stats()
    else:
        show_audit_log(user=user, since=since, follow=follow)
```

Example output:
```
2024-01-15T10:23:45Z [AUTH_SUCCESS] alice by uid:1000 similarity:0.89
2024-01-15T10:24:12Z [AUTH_FAIL] bob by uid:1001 reason:no_embedding
2024-01-15T10:25:33Z [SPOOF_DETECTED] alice by uid:1002 reason:antispoof_score_low
```

### Configuration

Add to `config.toml`:

```toml
[audit]
enabled = true
log_path = "~/.local/share/faceauth/audit.log"
max_size_mb = 100
rotate_days = 90
```

### Log Rotation

Create `/etc/logrotate.d/faceauth`:

```
/var/log/faceauth/audit.log {
    daily
    rotate 90
    compress
    missingok
    notifempty
    create 0600 root root
}
```

## Success Criteria
- Every authentication event is logged to audit.log
- Similarity scores removed from INFO logs
- Audit log is valid JSON-lines (parseable with `jq`)
- `faceauth audit` shows formatted event history
- `faceauth audit stats` shows success rate and patterns
- Log file has 0o600 permissions (root only)
- No performance impact (<10ms per event)

Example queries:
```bash
# Show all auth events for alice
jq 'select(.username == "alice")' audit.log

# Count spoof attempts
jq 'select(.event == "spoof_detected") | .username' audit.log | sort | uniq -c

# Calculate success rate
jq -s 'group_by(.result) | map({result: .[0].result, count: length})' audit.log
```

## Out of Scope
- SIEM integration (Splunk, ELK, etc.)
- Real-time alerting (email/Slack on spoof detection)
- Web UI for audit log viewing
- Log encryption at rest
- Network transport (syslog TLS)
- Audit log backups
- Multi-node log aggregation
- Compliance reporting (SOC2, ISO27001)
