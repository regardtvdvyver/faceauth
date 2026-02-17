"""Client for communicating with the faceauth daemon.

Used by both the PAM module and CLI to send requests to the daemon
over Unix socket.
"""

import logging
import os
import socket
from pathlib import Path

from .protocol import Request, Response

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0  # seconds

SYSTEM_SOCKET = Path("/run/faceauth/faceauth.sock")


def get_socket_path(uid: int | None = None) -> Path:
    """Get the daemon socket path.

    Tries system socket first (/run/faceauth/faceauth.sock),
    then falls back to user socket (/run/user/{uid}/faceauth.sock).
    """
    if SYSTEM_SOCKET.exists():
        return SYSTEM_SOCKET
    if uid is None:
        uid = os.getuid()
    return Path(f"/run/user/{uid}/faceauth.sock")


def send_request(
    request: Request,
    socket_path: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Response:
    """Send a request to the daemon and return the response.

    Args:
        request: The request to send.
        socket_path: Override socket path (default: /run/user/{uid}/faceauth.sock).
        timeout: Timeout in seconds.

    Returns:
        Response from daemon.

    Raises:
        ConnectionError: If daemon is not running or unreachable.
        TimeoutError: If request times out.
    """
    if socket_path is None:
        socket_path = get_socket_path()

    if not socket_path.exists():
        raise ConnectionError(f"Daemon socket not found: {socket_path}")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(str(socket_path))
        sock.sendall(request.to_json())

        # Read response (newline-terminated JSON)
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
            raise ConnectionError("Empty response from daemon")

        return Response.from_json(data)

    except socket.timeout:
        raise TimeoutError(f"Daemon request timed out after {timeout}s")
    except FileNotFoundError:
        raise ConnectionError(f"Daemon socket not found: {socket_path}")
    except ConnectionRefusedError:
        raise ConnectionError("Daemon not running (connection refused)")
    finally:
        sock.close()


def verify(
    username: str,
    timeout: float = DEFAULT_TIMEOUT,
    socket_path: Path | None = None,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """Convenience: verify a face via daemon.

    Returns (match, score).
    Raises ConnectionError if daemon unavailable.
    """
    req = Request(action="verify", username=username, threshold=threshold)
    resp = send_request(req, socket_path=socket_path, timeout=timeout)
    return resp.ok, resp.score


def daemon_status(socket_path: Path | None = None) -> dict | None:
    """Check daemon status. Returns status dict or None if unreachable."""
    try:
        req = Request(action="status")
        resp = send_request(req, socket_path=socket_path, timeout=3.0)
        if resp.ok:
            return resp.data
    except (ConnectionError, TimeoutError):
        pass
    return None
