"""faceauth PAM module for pam_python.

This module is loaded by pam_python.so and called during PAM authentication.
It connects to the faceauth daemon via Unix socket and requests face verification.

PAM configuration (e.g. /etc/pam.d/sudo):
    auth sufficient pam_python.so /path/to/faceauth_pam.py

This module is intentionally self-contained - it uses only stdlib to avoid
depending on the project's virtualenv. Communication with the daemon uses
the same JSON-over-socket protocol defined in faceauth.protocol.
"""

import json
import os
import socket
import syslog

# PAM constants (from pam_python)
PAM_SUCCESS = 0
PAM_AUTH_ERR = 7
PAM_IGNORE = 25

# Configuration
SYSTEM_SOCKET = "/run/faceauth/faceauth.sock"
USER_SOCKET_TEMPLATE = "/run/user/{uid}/faceauth.sock"
TIMEOUT = 10  # seconds


def _log(priority, msg):
    """Log via syslog for PAM context."""
    syslog.syslog(priority, f"faceauth_pam: {msg}")


def _get_socket_path(uid):
    """Try system socket first, then per-user socket."""
    if os.path.exists(SYSTEM_SOCKET):
        return SYSTEM_SOCKET
    return USER_SOCKET_TEMPLATE.format(uid=uid)


def _verify_face(username, uid):
    """Send verify request to daemon. Returns (success, score, error)."""
    sock_path = _get_socket_path(uid)

    if not os.path.exists(sock_path):
        return False, 0.0, f"daemon socket not found: {sock_path}"

    request = json.dumps({"action": "verify", "username": username}) + "\n"

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)

    try:
        sock.connect(sock_path)
        sock.sendall(request.encode())

        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break

        data = b"".join(chunks)
        if not data:
            return False, 0.0, "empty response from daemon"

        resp = json.loads(data.strip())
        ok = resp.get("ok", False)
        score = resp.get("score", 0.0)
        error = resp.get("error", "")
        return ok, score, error

    except socket.timeout:
        return False, 0.0, f"timeout after {TIMEOUT}s"
    except ConnectionRefusedError:
        return False, 0.0, "daemon not running"
    except Exception as e:
        return False, 0.0, str(e)
    finally:
        sock.close()


def pam_sm_authenticate(pamh, flags, argv):
    """Called by PAM to authenticate the user."""
    try:
        username = pamh.get_user(None)
    except Exception:
        _log(syslog.LOG_ERR, "failed to get username from PAM")
        return PAM_IGNORE

    if not username:
        _log(syslog.LOG_ERR, "empty username")
        return PAM_IGNORE

    # Get the uid of the user being authenticated
    try:
        import pwd
        pw = pwd.getpwnam(username)
        uid = pw.pw_uid
    except KeyError:
        _log(syslog.LOG_ERR, f"unknown user: {username}")
        return PAM_IGNORE

    _log(syslog.LOG_INFO, f"attempting face auth for '{username}' (uid={uid})")

    ok, score, error = _verify_face(username, uid)

    if ok:
        _log(syslog.LOG_INFO, f"face auth SUCCESS for '{username}' (score={score:.3f})")
        return PAM_SUCCESS
    else:
        reason = error if error else f"score={score:.3f}"
        _log(syslog.LOG_INFO, f"face auth FAIL for '{username}': {reason}")
        return PAM_IGNORE  # Fall through to next auth method


def pam_sm_setcred(pamh, flags, argv):
    """Called by PAM to set credentials. No-op for us."""
    return PAM_SUCCESS


def pam_sm_acct_mgmt(pamh, flags, argv):
    """Called by PAM for account management. No-op for us."""
    return PAM_SUCCESS


def pam_sm_open_session(pamh, flags, argv):
    return PAM_SUCCESS


def pam_sm_close_session(pamh, flags, argv):
    return PAM_SUCCESS


def pam_sm_chauthtok(pamh, flags, argv):
    return PAM_SUCCESS
