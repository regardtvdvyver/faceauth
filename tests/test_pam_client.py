"""Unit tests for faceauth.pam_client module."""

import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from faceauth.pam_client import (
    DEFAULT_TIMEOUT,
    daemon_status,
    get_socket_path,
    send_request,
    verify,
)
from faceauth.protocol import Request, Response


@pytest.mark.unit
class TestGetSocketPath:
    """Tests for get_socket_path function."""

    @patch("faceauth.pam_client.SYSTEM_SOCKET")
    def test_returns_correct_path_with_explicit_uid(self, mock_sys_sock):
        """Test that get_socket_path returns correct path when uid is provided."""
        mock_sys_sock.exists.return_value = False
        uid = 1000
        expected_path = Path("/run/user/1000/faceauth.sock")
        assert get_socket_path(uid) == expected_path

    @patch("faceauth.pam_client.SYSTEM_SOCKET")
    def test_returns_correct_path_with_different_uid(self, mock_sys_sock):
        """Test with a different uid value."""
        mock_sys_sock.exists.return_value = False
        uid = 1001
        expected_path = Path("/run/user/1001/faceauth.sock")
        assert get_socket_path(uid) == expected_path

    @patch("faceauth.pam_client.os.getuid")
    @patch("faceauth.pam_client.SYSTEM_SOCKET")
    def test_returns_path_using_getuid_when_none(self, mock_sys_sock, mock_getuid):
        """Test that get_socket_path uses os.getuid() when uid is None."""
        mock_sys_sock.exists.return_value = False
        mock_getuid.return_value = 1000
        expected_path = Path("/run/user/1000/faceauth.sock")
        assert get_socket_path(None) == expected_path
        mock_getuid.assert_called_once()

    @patch("faceauth.pam_client.os.getuid")
    @patch("faceauth.pam_client.SYSTEM_SOCKET")
    def test_default_uid_parameter_uses_getuid(self, mock_sys_sock, mock_getuid):
        """Test that default uid parameter (no args) uses os.getuid()."""
        mock_sys_sock.exists.return_value = False
        mock_getuid.return_value = 1234
        expected_path = Path("/run/user/1234/faceauth.sock")
        assert get_socket_path() == expected_path
        mock_getuid.assert_called_once()


@pytest.mark.unit
class TestSendRequest:
    """Tests for send_request function."""

    def test_socket_not_found_raises_connection_error(self):
        """Test that non-existent socket raises ConnectionError."""
        request = Request(action="status")
        socket_path = Path("/run/user/9999/nonexistent.sock")

        with pytest.raises(ConnectionError, match="Daemon socket not found"):
            send_request(request, socket_path=socket_path)

    @patch("faceauth.pam_client.socket.socket")
    def test_successful_request_response_roundtrip(self, mock_socket_class):
        """Test successful request/response communication."""
        # Setup mock socket
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock

        # Mock response data
        response_data = Response(ok=True, score=0.95)
        mock_sock.recv.side_effect = [
            response_data.to_json(),
            b"",  # End of stream
        ]

        # Create request
        request = Request(action="verify", username="testuser")
        socket_path = Path("/tmp/test.sock")

        # Mock socket path exists
        with patch.object(Path, "exists", return_value=True):
            result = send_request(request, socket_path=socket_path)

        # Assertions
        assert result.ok is True
        assert result.score == 0.95
        mock_sock.connect.assert_called_once_with(str(socket_path))
        mock_sock.sendall.assert_called_once_with(request.to_json())
        mock_sock.close.assert_called_once()

    @patch("faceauth.pam_client.socket.socket")
    def test_socket_timeout_raises_timeout_error(self, mock_socket_class):
        """Test that socket timeout raises TimeoutError."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.recv.side_effect = socket.timeout("timeout")

        request = Request(action="verify", username="testuser")
        socket_path = Path("/tmp/test.sock")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(TimeoutError, match="timed out after"):
                send_request(request, socket_path=socket_path, timeout=5.0)

        mock_sock.close.assert_called_once()

    @patch("faceauth.pam_client.socket.socket")
    def test_connection_refused_raises_connection_error(self, mock_socket_class):
        """Test that ConnectionRefusedError is wrapped as ConnectionError."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.side_effect = ConnectionRefusedError("refused")

        request = Request(action="verify", username="testuser")
        socket_path = Path("/tmp/test.sock")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ConnectionError, match="Daemon not running"):
                send_request(request, socket_path=socket_path)

        mock_sock.close.assert_called_once()

    @patch("faceauth.pam_client.socket.socket")
    def test_empty_response_raises_connection_error(self, mock_socket_class):
        """Test that empty response raises ConnectionError."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.recv.return_value = b""

        request = Request(action="status")
        socket_path = Path("/tmp/test.sock")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ConnectionError, match="Empty response from daemon"):
                send_request(request, socket_path=socket_path)

        mock_sock.close.assert_called_once()

    @patch("faceauth.pam_client.socket.socket")
    def test_socket_always_closed_on_success(self, mock_socket_class):
        """Test that socket is closed even on successful operation."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock

        response_data = Response(ok=True)
        mock_sock.recv.side_effect = [response_data.to_json(), b""]

        request = Request(action="status")
        socket_path = Path("/tmp/test.sock")

        with patch.object(Path, "exists", return_value=True):
            send_request(request, socket_path=socket_path)

        mock_sock.close.assert_called_once()

    @patch("faceauth.pam_client.socket.socket")
    def test_socket_always_closed_on_exception(self, mock_socket_class):
        """Test that socket is closed even when exception occurs."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.side_effect = ConnectionRefusedError()

        request = Request(action="status")
        socket_path = Path("/tmp/test.sock")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ConnectionError):
                send_request(request, socket_path=socket_path)

        mock_sock.close.assert_called_once()

    @patch("faceauth.pam_client.socket.socket")
    def test_uses_default_socket_path_when_none(self, mock_socket_class):
        """Test that get_socket_path is called when socket_path is None."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock

        response_data = Response(ok=True)
        mock_sock.recv.side_effect = [response_data.to_json(), b""]

        request = Request(action="status")

        # Mock SYSTEM_SOCKET.exists() -> False so it falls back to user socket
        with patch("faceauth.pam_client.SYSTEM_SOCKET") as mock_sys_sock:
            mock_sys_sock.exists.return_value = False
            with patch("faceauth.pam_client.os.getuid", return_value=1000):
                expected_path = Path("/run/user/1000/faceauth.sock")
                with patch.object(Path, "exists", return_value=True):
                    send_request(request, socket_path=None)

        mock_sock.connect.assert_called_once_with(str(expected_path))

    @patch("faceauth.pam_client.socket.socket")
    def test_handles_chunked_response(self, mock_socket_class):
        """Test that multi-chunk responses are properly assembled."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock

        # Split response across multiple chunks
        response_data = Response(ok=True, data={"info": "Long message" * 100})
        full_json = response_data.to_json()
        chunk1 = full_json[: len(full_json) // 2]
        chunk2 = full_json[len(full_json) // 2 :]

        mock_sock.recv.side_effect = [chunk1, chunk2, b""]

        request = Request(action="status")
        socket_path = Path("/tmp/test.sock")

        with patch.object(Path, "exists", return_value=True):
            result = send_request(request, socket_path=socket_path)

        assert result.ok is True
        assert result.data.get("info") == "Long message" * 100


@pytest.mark.unit
class TestVerify:
    """Tests for verify function."""

    @patch("faceauth.pam_client.send_request")
    def test_returns_true_and_score_on_successful_verify(self, mock_send_request):
        """Test that verify returns (True, score) on successful verification."""
        mock_send_request.return_value = Response(ok=True, score=0.92)

        ok, score = verify("testuser")

        assert ok is True
        assert score == 0.92
        mock_send_request.assert_called_once()
        call_args = mock_send_request.call_args
        assert call_args[0][0].action == "verify"
        assert call_args[0][0].username == "testuser"

    @patch("faceauth.pam_client.send_request")
    def test_returns_false_and_score_on_failed_verify(self, mock_send_request):
        """Test that verify returns (False, 0.0) on failed verification."""
        mock_send_request.return_value = Response(ok=False, score=0.0)

        ok, score = verify("testuser")

        assert ok is False
        assert score == 0.0

    @patch("faceauth.pam_client.send_request")
    def test_passes_timeout_parameter(self, mock_send_request):
        """Test that verify passes timeout to send_request."""
        mock_send_request.return_value = Response(ok=True, score=0.85)

        verify("testuser", timeout=10.0)

        call_kwargs = mock_send_request.call_args[1]
        assert call_kwargs["timeout"] == 10.0

    @patch("faceauth.pam_client.send_request")
    def test_passes_socket_path_parameter(self, mock_send_request):
        """Test that verify passes socket_path to send_request."""
        mock_send_request.return_value = Response(ok=True, score=0.85)
        custom_path = Path("/tmp/custom.sock")

        verify("testuser", socket_path=custom_path)

        call_kwargs = mock_send_request.call_args[1]
        assert call_kwargs["socket_path"] == custom_path

    @patch("faceauth.pam_client.send_request")
    def test_passes_threshold_parameter(self, mock_send_request):
        """Test that verify passes threshold to Request."""
        mock_send_request.return_value = Response(ok=True, score=0.85)

        verify("testuser", threshold=0.7)

        call_args = mock_send_request.call_args[0]
        assert call_args[0].threshold == 0.7

    @patch("faceauth.pam_client.send_request")
    def test_uses_default_timeout_when_not_specified(self, mock_send_request):
        """Test that verify uses DEFAULT_TIMEOUT when not specified."""
        mock_send_request.return_value = Response(ok=True, score=0.85)

        verify("testuser")

        call_kwargs = mock_send_request.call_args[1]
        assert call_kwargs["timeout"] == DEFAULT_TIMEOUT


@pytest.mark.unit
class TestDaemonStatus:
    """Tests for daemon_status function."""

    @patch("faceauth.pam_client.send_request")
    def test_returns_data_dict_when_daemon_running(self, mock_send_request):
        """Test that daemon_status returns data dict when daemon responds."""
        status_data = {
            "pid": 1234,
            "uptime": 3600,
            "enrolled_users": ["user1", "user2"],
        }
        mock_send_request.return_value = Response(ok=True, data=status_data)

        result = daemon_status()

        assert result == status_data
        assert result["pid"] == 1234
        assert result["enrolled_users"] == ["user1", "user2"]

    @patch("faceauth.pam_client.send_request")
    def test_returns_none_on_connection_error(self, mock_send_request):
        """Test that daemon_status returns None when ConnectionError occurs."""
        mock_send_request.side_effect = ConnectionError("Daemon not running")

        result = daemon_status()

        assert result is None

    @patch("faceauth.pam_client.send_request")
    def test_returns_none_on_timeout_error(self, mock_send_request):
        """Test that daemon_status returns None when TimeoutError occurs."""
        mock_send_request.side_effect = TimeoutError("Timeout")

        result = daemon_status()

        assert result is None

    @patch("faceauth.pam_client.send_request")
    def test_uses_short_timeout(self, mock_send_request):
        """Test that daemon_status uses a short timeout (3.0s)."""
        mock_send_request.return_value = Response(ok=True, data={})

        daemon_status()

        call_kwargs = mock_send_request.call_args[1]
        assert call_kwargs["timeout"] == 3.0

    @patch("faceauth.pam_client.send_request")
    def test_passes_socket_path_parameter(self, mock_send_request):
        """Test that daemon_status passes socket_path to send_request."""
        mock_send_request.return_value = Response(ok=True, data={})
        custom_path = Path("/tmp/custom.sock")

        daemon_status(socket_path=custom_path)

        call_kwargs = mock_send_request.call_args[1]
        assert call_kwargs["socket_path"] == custom_path

    @patch("faceauth.pam_client.send_request")
    def test_sends_status_action_request(self, mock_send_request):
        """Test that daemon_status sends a status action request."""
        mock_send_request.return_value = Response(ok=True, data={})

        daemon_status()

        call_args = mock_send_request.call_args[0]
        assert call_args[0].action == "status"

    @patch("faceauth.pam_client.send_request")
    def test_returns_none_when_response_not_ok(self, mock_send_request):
        """Test that daemon_status returns None when response.ok is False."""
        mock_send_request.return_value = Response(ok=False, error="Error")

        result = daemon_status()

        assert result is None
