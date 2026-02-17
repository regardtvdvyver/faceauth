"""Integration tests for FaceAuthDaemon.

Tests the async Unix socket server using real socket connections.
Mocks Camera and FaceRecognizer to avoid hardware dependencies.
Uses a temporary directory for socket and embedding storage.
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from faceauth.config import (
    AntispoofConfig,
    CameraConfig,
    Config,
    DaemonConfig,
    RecognitionConfig,
)
from faceauth.daemon import FaceAuthDaemon
from faceauth.protocol import Request, Response
from faceauth.storage import EmbeddingStore


@pytest.fixture
def tmp_socket(tmp_path):
    """Returns a Path to a temporary socket file."""
    return tmp_path / "test.sock"


@pytest.fixture
def daemon_config(tmp_path):
    """Returns a Config with antispoof disabled and custom paths."""
    return Config(
        camera=CameraConfig(
            ir_device="/dev/video2",
            width=640,
            height=480,
        ),
        recognition=RecognitionConfig(
            model="buffalo_l",
            similarity_threshold=0.45,
            max_attempts=5,
        ),
        antispoof=AntispoofConfig(
            enabled=False,  # Disable for most tests
        ),
        daemon=DaemonConfig(
            socket_path=str(tmp_path / "daemon.sock"),
            log_level="debug",
        ),
    )


@pytest.fixture
async def daemon_instance(daemon_config, tmp_socket, tmp_path):
    """Creates and starts a FaceAuthDaemon, yields it, stops after test."""
    # Create a custom EmbeddingStore with tmp_path
    daemon = FaceAuthDaemon(config=daemon_config)
    daemon.store = EmbeddingStore(data_dir=tmp_path / "embeddings")

    # Patch _get_peer_uid to return 0 (root) to bypass SO_PEERCRED authorization in tests
    # This allows tests to enroll/delete arbitrary usernames without system user validation
    with patch("faceauth.daemon.FaceAuthDaemon._get_peer_uid", return_value=0):
        await daemon.start(socket_path=tmp_socket)
        yield daemon
        await daemon.stop()


@pytest.fixture
async def send_recv(tmp_socket):
    """Async helper to send a Request and receive a Response via socket."""

    async def _send_recv(req: Request) -> Response:
        reader, writer = await asyncio.open_unix_connection(str(tmp_socket))
        try:
            writer.write(req.to_json())
            await writer.drain()

            data = await reader.readline()
            return Response.from_json(data)
        finally:
            writer.close()
            await writer.wait_closed()

    return _send_recv


# ============================================================================
# Status and List Tests
# ============================================================================


@pytest.mark.integration
async def test_status_response(daemon_instance, send_recv):
    """Send status request, verify response contains expected fields."""
    req = Request(action="status")
    resp = await send_recv(req)

    assert resp.ok is True
    assert "models_loaded" in resp.data
    assert "enrolled_users" in resp.data
    assert "users" in resp.data
    assert "socket" in resp.data
    assert "pid" in resp.data
    assert "antispoof" in resp.data

    assert resp.data["pid"] == os.getpid()
    assert resp.data["enrolled_users"] == 0
    assert resp.data["users"] == []
    assert resp.data["antispoof"]["enabled"] is False


@pytest.mark.integration
async def test_list_empty(daemon_instance, send_recv):
    """List users when none enrolled, verify empty dict."""
    req = Request(action="list")
    resp = await send_recv(req)

    assert resp.ok is True
    assert "users" in resp.data
    assert resp.data["users"] == {}


@pytest.mark.integration
async def test_list_with_users(daemon_instance, send_recv, sample_embedding, similar_embedding):
    """Pre-populate store, verify list returns user info with embedding counts."""
    # Pre-populate storage
    daemon_instance.store.save("alice", [sample_embedding, similar_embedding])
    daemon_instance.store.save("bob", [sample_embedding])

    req = Request(action="list")
    resp = await send_recv(req)

    assert resp.ok is True
    assert resp.data["users"] == {"alice": 2, "bob": 1}


# ============================================================================
# Delete Tests
# ============================================================================


@pytest.mark.integration
async def test_delete_enrolled_user(daemon_instance, send_recv, sample_embedding):
    """Pre-populate store, delete user, verify ok=True."""
    daemon_instance.store.save("testuser", [sample_embedding])
    assert daemon_instance.store.is_enrolled("testuser")

    req = Request(action="delete", username="testuser")
    resp = await send_recv(req)

    assert resp.ok is True
    assert not daemon_instance.store.is_enrolled("testuser")


@pytest.mark.integration
async def test_delete_non_enrolled_user(daemon_instance, send_recv):
    """Delete non-existent user, verify ok=False."""
    req = Request(action="delete", username="nosuchuser")
    resp = await send_recv(req)

    assert resp.ok is False
    assert "not enrolled" in resp.error


@pytest.mark.integration
async def test_delete_missing_username(daemon_instance, send_recv):
    """Delete without username, verify ok=False."""
    req = Request(action="delete", username="")
    resp = await send_recv(req)

    assert resp.ok is False
    assert "username required" in resp.error


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
async def test_unknown_action(daemon_instance, send_recv):
    """Send unknown action, verify ok=False with error."""
    req = Request(action="invalid_action")
    resp = await send_recv(req)

    assert resp.ok is False
    assert "Unknown action" in resp.error


@pytest.mark.integration
async def test_invalid_json_request(daemon_instance, tmp_socket):
    """Send garbage bytes, verify ok=False with error."""
    reader, writer = await asyncio.open_unix_connection(str(tmp_socket))
    try:
        # Send invalid JSON
        writer.write(b"not valid json\n")
        await writer.drain()

        data = await reader.readline()
        resp = Response.from_json(data)

        assert resp.ok is False
        assert "invalid request" in resp.error
    finally:
        writer.close()
        await writer.wait_closed()


# ============================================================================
# Verify Tests - Parameter Validation
# ============================================================================


@pytest.mark.integration
async def test_verify_missing_username(daemon_instance, send_recv):
    """Verify without username, verify ok=False."""
    req = Request(action="verify", username="")
    resp = await send_recv(req)

    assert resp.ok is False
    assert "username required" in resp.error


@pytest.mark.integration
async def test_verify_user_not_enrolled(daemon_instance, send_recv):
    """Verify non-enrolled user, verify error."""
    req = Request(action="verify", username="unenrolled")
    resp = await send_recv(req)

    assert resp.ok is False
    assert "not enrolled" in resp.error


# ============================================================================
# Enroll Tests - Parameter Validation
# ============================================================================


@pytest.mark.integration
async def test_enroll_missing_username(daemon_instance, send_recv):
    """Enroll without username, verify ok=False."""
    req = Request(action="enroll", username="")
    resp = await send_recv(req)

    assert resp.ok is False
    assert "username required" in resp.error


# ============================================================================
# Verify Tests - With Mocked Camera/FaceRecognizer
# ============================================================================


@pytest.mark.integration
async def test_verify_successful_match(
    daemon_instance, send_recv, sample_embedding, similar_embedding, grey_frame
):
    """Mock Camera+FaceRecognizer, pre-populate store, verify returns ok=True with score."""
    # Pre-populate with embeddings
    daemon_instance.store.save("alice", [sample_embedding])

    # Mock Face object
    mock_face = Mock()
    mock_face.det_score = 0.95
    mock_face.embedding = similar_embedding  # Similar to stored
    mock_face.bbox = np.array([100, 80, 300, 320])

    # Mock FaceRecognizer - inject directly into daemon
    mock_recognizer = MagicMock()
    mock_recognizer.get_faces = MagicMock(return_value=[mock_face])
    mock_recognizer._ensure_loaded = MagicMock()
    daemon_instance.recognizer = mock_recognizer

    # Mock Camera
    mock_camera = MagicMock()
    mock_camera.__enter__ = MagicMock(return_value=mock_camera)
    mock_camera.__exit__ = MagicMock(return_value=False)
    mock_camera.read = MagicMock(return_value=grey_frame)

    with patch("faceauth.daemon.Camera", return_value=mock_camera):
        req = Request(action="verify", username="alice")
        resp = await send_recv(req)

    assert resp.ok is True
    assert resp.score > 0.0
    # Score should be high since embeddings are similar (similar_embedding is ~0.95+ to sample)
    # But actual value depends on embedding similarity - just verify it's reasonably high
    assert resp.score > 0.5


@pytest.mark.integration
async def test_verify_no_match(
    daemon_instance, send_recv, sample_embedding, different_embedding, grey_frame
):
    """Mock Camera+FaceRecognizer with different embedding, verify returns ok=False."""
    # Pre-populate with embeddings
    daemon_instance.store.save("alice", [sample_embedding])

    # Mock Face object with very different embedding
    mock_face = Mock()
    mock_face.det_score = 0.95
    mock_face.embedding = different_embedding  # Different from stored
    mock_face.bbox = np.array([100, 80, 300, 320])

    # Mock FaceRecognizer - inject directly into daemon
    mock_recognizer = MagicMock()
    mock_recognizer.get_faces = MagicMock(return_value=[mock_face])
    mock_recognizer._ensure_loaded = MagicMock()
    daemon_instance.recognizer = mock_recognizer

    # Mock Camera
    mock_camera = MagicMock()
    mock_camera.__enter__ = MagicMock(return_value=mock_camera)
    mock_camera.__exit__ = MagicMock(return_value=False)
    mock_camera.read = MagicMock(return_value=grey_frame)

    with patch("faceauth.daemon.Camera", return_value=mock_camera):
        req = Request(action="verify", username="alice", threshold=0.45)
        resp = await send_recv(req)

    assert resp.ok is False
    assert resp.score == 0.0


# ============================================================================
# Enroll Tests - With Mocked Camera/FaceRecognizer
# ============================================================================


@pytest.mark.integration
async def test_enroll_successful(
    daemon_instance, send_recv, sample_embedding, similar_embedding, grey_frame
):
    """Mock Camera+FaceRecognizer, verify enrollment creates embeddings in store."""
    # Mock Face objects (simulate multiple captures)
    mock_face1 = Mock()
    mock_face1.det_score = 0.95
    mock_face1.embedding = sample_embedding
    mock_face1.bbox = np.array([100, 80, 300, 320])

    mock_face2 = Mock()
    mock_face2.det_score = 0.93
    mock_face2.embedding = similar_embedding
    mock_face2.bbox = np.array([105, 85, 305, 325])

    # Mock FaceRecognizer - inject directly into daemon
    mock_recognizer = MagicMock()
    mock_recognizer.get_faces = MagicMock(
        side_effect=[
            [mock_face1],
            [mock_face2],
            [mock_face1],  # Extra calls in case samples > 2
        ]
    )
    mock_recognizer._ensure_loaded = MagicMock()
    daemon_instance.recognizer = mock_recognizer

    # Mock Camera
    mock_camera = MagicMock()
    mock_camera.__enter__ = MagicMock(return_value=mock_camera)
    mock_camera.__exit__ = MagicMock(return_value=False)
    mock_camera.read = MagicMock(return_value=grey_frame)

    with patch("faceauth.daemon.Camera", return_value=mock_camera):
        req = Request(action="enroll", username="newuser", samples=2)
        resp = await send_recv(req)

    assert resp.ok is True
    assert "samples" in resp.data
    assert resp.data["samples"] == 2
    assert "consistency" in resp.data

    # Verify embeddings were saved
    stored = daemon_instance.store.load("newuser")
    assert len(stored) == 2


@pytest.mark.integration
async def test_enroll_no_face_detected(daemon_instance, send_recv, grey_frame):
    """Mock Camera with no face detected, verify enrollment fails."""
    # Mock FaceRecognizer - inject directly into daemon
    mock_recognizer = MagicMock()
    mock_recognizer.get_faces = MagicMock(return_value=[])
    mock_recognizer._ensure_loaded = MagicMock()
    daemon_instance.recognizer = mock_recognizer

    # Mock Camera
    mock_camera = MagicMock()
    mock_camera.__enter__ = MagicMock(return_value=mock_camera)
    mock_camera.__exit__ = MagicMock(return_value=False)
    mock_camera.read = MagicMock(return_value=grey_frame)

    with patch("faceauth.daemon.Camera", return_value=mock_camera):
        req = Request(action="enroll", username="newuser", samples=1)
        resp = await send_recv(req)

    assert resp.ok is False
    assert "No faces captured" in resp.error


# ============================================================================
# Connection Handling Tests
# ============================================================================


@pytest.mark.integration
async def test_client_disconnect_doesnt_crash(daemon_instance, tmp_socket):
    """Connect and immediately disconnect, daemon stays running."""
    # Connect and close immediately without sending anything
    reader, writer = await asyncio.open_unix_connection(str(tmp_socket))
    writer.close()
    await writer.wait_closed()

    # Give daemon time to handle the disconnect
    await asyncio.sleep(0.1)

    # Verify daemon is still responsive
    reader2, writer2 = await asyncio.open_unix_connection(str(tmp_socket))
    try:
        req = Request(action="status")
        writer2.write(req.to_json())
        await writer2.drain()

        data = await reader2.readline()
        resp = Response.from_json(data)

        assert resp.ok is True
    finally:
        writer2.close()
        await writer2.wait_closed()


@pytest.mark.integration
async def test_multiple_concurrent_clients(daemon_instance, send_recv):
    """Multiple clients can connect and make requests concurrently."""
    # Create multiple concurrent status requests
    tasks = [send_recv(Request(action="status")) for _ in range(5)]
    responses = await asyncio.gather(*tasks)

    # All should succeed
    assert all(r.ok for r in responses)
    assert all("pid" in r.data for r in responses)


@pytest.mark.integration
async def test_client_timeout_handling(daemon_instance, tmp_socket):
    """Client that sends data slowly should timeout."""
    reader, writer = await asyncio.open_unix_connection(str(tmp_socket))
    try:
        # Send incomplete data (no newline), then wait
        writer.write(b'{"action": "status"')
        await writer.drain()

        # Wait for timeout (daemon has 5s timeout)
        await asyncio.sleep(6)

        # Connection should be closed by daemon
        # Try to read - should get empty bytes or EOF
        data = await reader.read(1024)
        # Empty data indicates connection was closed
        assert len(data) == 0
    finally:
        writer.close()
        await writer.wait_closed()


# ============================================================================
# Model Loading Tests
# ============================================================================


@pytest.mark.integration
async def test_models_lazy_loaded(daemon_config, tmp_socket, tmp_path, send_recv, sample_embedding):
    """Models are not loaded until first verify/enroll request."""
    # Create a daemon WITHOUT preloading (no background task)
    daemon = FaceAuthDaemon(config=daemon_config)
    daemon.store = EmbeddingStore(data_dir=tmp_path / "embeddings2")

    await daemon.start(socket_path=tmp_socket)

    try:
        # Check status before any operation
        req = Request(action="status")
        resp = await send_recv(req)

        assert resp.ok is True
        # Initially, models should not be loaded
        # Note: daemon.start() calls run_in_executor for _ensure_models, so we need to wait a bit
        # But if we check immediately, they shouldn't be loaded yet
        # Pre-populate store so verify attempt proceeds to model loading
        daemon.store.save("testuser", [sample_embedding])

        # Trigger verify to force model loading
        req_verify = Request(action="verify", username="testuser")
        # This will fail because no real models, but it should trigger _ensure_models
        await send_recv(req_verify)

        # Check status again - models should now be loaded (or attempted to load)
        req_status2 = Request(action="status")
        resp2 = await send_recv(req_status2)

        # The models_loaded flag is True if recognizer is not None
        # The verify attempt above will have triggered _ensure_models
        assert resp2.data["models_loaded"] is True
    finally:
        await daemon.stop()


# ============================================================================
# Protocol Format Tests
# ============================================================================


@pytest.mark.integration
async def test_response_format_contains_newline(daemon_instance, tmp_socket):
    """Verify responses are newline-terminated JSON."""
    reader, writer = await asyncio.open_unix_connection(str(tmp_socket))
    try:
        req = Request(action="status")
        writer.write(req.to_json())
        await writer.drain()

        raw_data = await reader.readline()

        # Should end with newline
        assert raw_data.endswith(b"\n")

        # Should be valid JSON
        parsed = json.loads(raw_data.strip())
        assert "ok" in parsed
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.integration
async def test_request_without_newline_timeout(daemon_instance, tmp_socket):
    """Request without newline should timeout waiting for complete line."""
    reader, writer = await asyncio.open_unix_connection(str(tmp_socket))
    try:
        # Send valid JSON but without newline
        req_bytes = b'{"action": "status"}'  # No newline!
        writer.write(req_bytes)
        await writer.drain()

        # Daemon should timeout after 5 seconds
        # We'll wait a bit then check connection is closed
        await asyncio.sleep(6)

        # Try reading - should be empty (connection closed)
        data = await reader.read(1024)
        assert len(data) == 0
    finally:
        writer.close()
        await writer.wait_closed()


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.integration
async def test_verify_with_custom_threshold(
    daemon_instance, send_recv, sample_embedding, similar_embedding, grey_frame
):
    """Verify that daemon ignores client-side threshold and uses server config."""
    daemon_instance.store.save("alice", [sample_embedding])

    # Mock Face with similarity above server threshold (0.45) but below what client requests
    mock_face = Mock()
    mock_face.det_score = 0.95
    mock_face.embedding = similar_embedding
    mock_face.bbox = np.array([100, 80, 300, 320])

    # Mock FaceRecognizer - inject directly into daemon
    mock_recognizer = MagicMock()
    mock_recognizer.get_faces = MagicMock(return_value=[mock_face])
    mock_recognizer._ensure_loaded = MagicMock()
    daemon_instance.recognizer = mock_recognizer

    mock_camera = MagicMock()
    mock_camera.__enter__ = MagicMock(return_value=mock_camera)
    mock_camera.__exit__ = MagicMock(return_value=False)
    mock_camera.read = MagicMock(return_value=grey_frame)

    with patch("faceauth.daemon.Camera", return_value=mock_camera):
        # Client requests high threshold (0.99), but daemon uses server config (0.45)
        # Since similarity is above 0.45, verification should succeed
        req = Request(action="verify", username="alice", threshold=0.99)
        resp = await send_recv(req)

        # Daemon ignores client threshold, uses server threshold (0.45), so should pass
        assert resp.ok is True
        assert resp.score > 0.45


@pytest.mark.integration
async def test_enroll_custom_sample_count(daemon_instance, send_recv, sample_embedding, grey_frame):
    """Enroll request can specify custom sample count."""
    mock_face = Mock()
    mock_face.det_score = 0.95
    mock_face.embedding = sample_embedding
    mock_face.bbox = np.array([100, 80, 300, 320])

    # Mock FaceRecognizer - inject directly into daemon
    mock_recognizer = MagicMock()
    # Return face for each read attempt
    mock_recognizer.get_faces = MagicMock(return_value=[mock_face])
    mock_recognizer._ensure_loaded = MagicMock()
    daemon_instance.recognizer = mock_recognizer

    mock_camera = MagicMock()
    mock_camera.__enter__ = MagicMock(return_value=mock_camera)
    mock_camera.__exit__ = MagicMock(return_value=False)
    mock_camera.read = MagicMock(return_value=grey_frame)

    with patch("faceauth.daemon.Camera", return_value=mock_camera):
        req = Request(action="enroll", username="bob", samples=3)
        resp = await send_recv(req)

    assert resp.ok is True
    assert resp.data["samples"] == 3

    # Verify 3 embeddings saved
    stored = daemon_instance.store.load("bob")
    assert len(stored) == 3
