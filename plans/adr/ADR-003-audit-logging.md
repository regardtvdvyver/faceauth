# ADR-003: Security Audit Logging

**Status:** Accepted
**Date:** 2026-02-17

## Context

faceauth is a security-critical authentication system. Like all authentication mechanisms, it must maintain an auditable trail of authentication events for:

1. **Security monitoring**: Detect brute-force attempts, unusual patterns, potential compromises
2. **Compliance**: Many regulatory frameworks (SOC 2, ISO 27001, PCI DSS) require authentication audit logs
3. **Forensics**: Investigate security incidents after the fact
4. **Debugging**: Troubleshoot authentication failures without exposing sensitive data

### Current State Issues

The current logging implementation has several problems:

1. **Similarity scores leak to INFO logs**: The daemon logs similarity scores at INFO level. This is a security risk because:
   - Scores reveal how close an attacker is to success
   - Repeated attempts allow score-based model inversion attacks
   - Logs may be exposed via syslog forwarding to less-secure systems

2. **No structured audit trail**: Authentication events are logged as unstructured text mixed with operational logs. This makes:
   - Automated analysis difficult (can't easily query "all failed auth attempts last week")
   - SIEM integration problematic
   - Compliance audits manual and time-consuming

3. **Missing critical events**: Some security-relevant events aren't logged at all:
   - Enrollment operations (who added faces, when)
   - Face deletion (who removed faces, when)
   - Spoof detection triggers (attempted photo/mask attacks)
   - PAM client connection failures

4. **No peer identity tracking**: The daemon doesn't log which user/process initiated an authentication attempt, making it hard to correlate with system logs.

## Decision

Implement a dedicated security audit logging system with these characteristics:

### 1. Separate Audit Log File
- **Format**: JSON Lines (newline-delimited JSON objects)
- **Location**:
  - System mode: `/var/log/faceauth/audit.jsonl`
  - User mode: `~/.local/share/faceauth/audit.jsonl`
- **Permissions**:
  - System mode: root:root, 0600 (root-only access)
  - User mode: user:user, 0600 (user-only access)
- **Rotation**: Integrate with logrotate (system mode) or manual rotation (user mode)

### 2. Audit Events

Each event is a JSON object with these common fields:

```json
{
  "timestamp": "2026-02-17T14:32:15.123456+02:00",  // ISO 8601 with microseconds
  "event": "auth_attempt",                           // Event type
  "username": "regardt",                              // Target user
  "peer_uid": 1000,                                   // Requesting user UID (from SO_PEERCRED)
  "peer_pid": 12345,                                  // Requesting process PID
  "session_id": "a1b2c3d4",                          // Unique ID for this auth session
  "result": "success",                                // success/failure/error
  "details": {}                                       // Event-specific data
}
```

**Event Types:**

1. **auth_attempt**: Authentication attempt started
   - details: `{"camera": "/dev/video2", "timeout": 5.0}`

2. **auth_success**: Authentication succeeded
   - details: `{"duration_ms": 245, "confidence": "high"}`
   - Note: No similarity score, no embedding data

3. **auth_failure**: Authentication failed
   - details: `{"reason": "no_face_detected", "duration_ms": 5002}`
   - Reasons: no_face_detected, confidence_too_low, timeout, unknown_user, spoof_detected, camera_error

4. **spoof_detected**: Anti-spoofing check failed
   - details: `{"spoof_score": 0.89, "threshold": 0.7, "method": "minifasnet"}`
   - This is the ONLY place spoof scores are logged, and it's in a protected audit log

5. **enroll**: Face enrolled
   - details: `{"samples": 5, "initiated_by_uid": 1000}`

6. **delete**: Face deleted
   - details: `{"initiated_by_uid": 0, "reason": "manual"}`

7. **daemon_start**: Daemon started
   - details: `{"mode": "system", "version": "1.0.0", "device": "/dev/video2"}`

8. **daemon_stop**: Daemon stopped
   - details: `{"uptime_seconds": 86400, "auth_count": 142}`

9. **config_change**: Configuration modified
   - details: `{"changed_by_uid": 1000, "setting": "antispoof.enabled", "old": false, "new": true}`

### 3. Implementation

Create `faceauth/audit.py`:

```python
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class AuditLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event: str,
        username: Optional[str],
        result: str,
        peer_uid: Optional[int] = None,
        peer_pid: Optional[int] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            "username": username,
            "peer_uid": peer_uid,
            "peer_pid": peer_pid,
            "session_id": session_id,
            "result": result,
            "details": details or {}
        }

        # Atomic append
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

### 4. Regular Logs Changes

Remove sensitive data from regular INFO/DEBUG logs:
- **Remove**: Similarity scores, embedding values, spoof scores
- **Keep**: High-level status ("Face detected", "Authentication succeeded")
- **Add**: Session IDs to correlate with audit log

### 5. Query Tool

Add `faceauth audit` subcommand for querying audit logs:

```bash
faceauth audit list                        # Last 50 events
faceauth audit list --event auth_failure   # Filter by event type
faceauth audit list --user regardt         # Filter by username
faceauth audit list --since "1 hour ago"   # Time range
faceauth audit stats                       # Summary statistics
faceauth audit export --format csv         # Export for analysis
```

### 6. Monitoring Integration

Provide documentation and examples for:
- Fail2ban integration (ban IPs with excessive failures)
- SIEM integration (Splunk, Elasticsearch, Graylog)
- Prometheus metrics export (future enhancement)
- Systemd journal integration (optional forwarding)

## Consequences

**Positive:**
- Meets security audit requirements for enterprise/government deployments
- Enables automated security monitoring and alerting
- Protects against information leakage via logs
- Makes debugging authentication issues easier (structured data)
- Provides data for usage analytics and optimization

**Negative:**
- Adds I/O overhead on every authentication (mitigated by async logging)
- Audit logs can grow large (requires rotation strategy)
- More code to maintain (audit.py module)
- Requires documentation for interpretation

**Security Benefits:**
- Attackers can't use logs to guide attacks (no scores leaked)
- Administrators can detect attack patterns
- Compliance auditors have clear evidence
- Incident response teams have forensic data

**Performance Impact:**
- JSON serialization: ~0.1ms per event (negligible)
- File append: 1-5ms with buffering (acceptable)
- No impact on authentication latency (async write)

## Alternatives Considered

### 1. Use syslog only
**Rejected.** Syslog mixes audit events with operational logs, making analysis difficult. Also, syslog may forward to less-secure systems, risking information leakage. However, we can optionally ALSO log to syslog for integration purposes.

### 2. SQLite database for audit logs
**Rejected.** Overkill for append-only logs. SQLite adds complexity (locking, corruption recovery) without significant benefits. JSON Lines is simpler, grep-able, and SIEM-friendly.

### 3. Binary log format (protobuf, messagepack)
**Rejected.** Harder to troubleshoot (requires special tools), harder to integrate with existing log tooling. JSON is human-readable and universally supported.

### 4. Keep similarity scores in logs but at DEBUG level
**Rejected.** DEBUG logs are often captured in production for troubleshooting. Better to remove sensitive data entirely from regular logs.

### 5. No audit logging (continue current approach)
**Rejected.** Unacceptable for a security tool. Audit logging is industry standard for authentication systems (PAM, SSH, sudo all have it).

### 6. Linux audit subsystem (auditd)
**Rejected for primary implementation.** Auditd is complex, requires root, and is designed for kernel-level events. However, we could optionally integrate with auditd for defense-in-depth (future enhancement).

## Implementation Plan

### Phase 1: Core audit logging
- Implement AuditLogger class
- Add audit events for auth_attempt, auth_success, auth_failure
- Remove similarity scores from INFO logs
- Add session IDs to correlate logs

### Phase 2: Comprehensive events
- Add enroll, delete, daemon lifecycle events
- Add spoof_detected events
- Extract peer credentials (SO_PEERCRED) from socket

### Phase 3: Query tool
- Implement `faceauth audit` subcommand
- Add filtering, time ranges, statistics

### Phase 4: Integration
- Write fail2ban filter rules
- Document SIEM integration
- Create example Splunk/ELK dashboards

## Security Considerations

- **Log injection**: Sanitize all fields to prevent JSON injection (use json.dumps)
- **PII**: Audit logs contain usernames. Document retention policies.
- **Access control**: Ensure audit logs are readable only by authorized users
- **Tampering**: Consider signing audit logs (future enhancement with HMAC)
- **Retention**: Default 90 days, configurable via config.toml

## Documentation Impact

- Add SECURITY.md: Explain audit logging, what's logged, retention
- Add MONITORING.md: Integration guides for fail2ban, SIEM, Prometheus
- Update README.md: Mention audit logging as a security feature
- Add examples/ directory with fail2ban rules, logrotate config, SIEM parsers
