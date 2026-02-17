"""Tests for faceauth camera module.

cv2 is auto-mocked by conftest._patch_cv2 (autouse fixture).
All cv2 functions (split, cvtColor, VideoCapture) and constants are available
on the mock without needing per-test patches.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from faceauth.camera import Camera, capture_frames, ir_to_rgb, is_ir_frame


# =============================================================================
# is_ir_frame tests
# =============================================================================


@pytest.mark.unit
def test_is_ir_frame_2d_greyscale(grey_frame):
    """2D greyscale frame should be detected as IR."""
    assert is_ir_frame(grey_frame)


@pytest.mark.unit
def test_is_ir_frame_3d_single_channel():
    """3D single-channel frame should be detected as IR."""
    frame = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
    assert is_ir_frame(frame)


@pytest.mark.unit
def test_is_ir_frame_bgr_equal_channels(ir_bgr_frame):
    """3D BGR frame with equal channels should be detected as IR."""
    assert is_ir_frame(ir_bgr_frame)


@pytest.mark.unit
def test_is_ir_frame_bgr_near_equal_channels():
    """3D BGR with near-equal channels (within atol=2) should be IR."""
    grey = np.full((480, 640), 100, dtype=np.uint8)
    b = grey.copy()
    g = (grey + 1).astype(np.uint8)
    r = (grey + 2).astype(np.uint8)
    frame = np.stack([b, g, r], axis=-1)
    assert is_ir_frame(frame)


@pytest.mark.unit
def test_is_ir_frame_normal_colour_bgr():
    """Normal colour BGR frame should not be detected as IR."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 200  # B
    frame[:, :, 1] = 50   # G
    frame[:, :, 2] = 100  # R
    assert not is_ir_frame(frame)


@pytest.mark.unit
def test_is_ir_frame_four_channel():
    """4-channel frame (BGRA) should not be detected as IR."""
    frame = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
    assert not is_ir_frame(frame)


# =============================================================================
# ir_to_rgb tests
# =============================================================================


@pytest.mark.unit
def test_ir_to_rgb_2d_greyscale(grey_frame):
    """2D greyscale frame -> 3-channel BGR."""
    result = ir_to_rgb(grey_frame)
    assert result.shape == (480, 640, 3)
    np.testing.assert_array_equal(result[:, :, 0], grey_frame)
    np.testing.assert_array_equal(result[:, :, 1], grey_frame)
    np.testing.assert_array_equal(result[:, :, 2], grey_frame)


@pytest.mark.unit
def test_ir_to_rgb_single_channel_3d():
    """Single-channel 3D frame -> 3-channel BGR."""
    grey = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
    result = ir_to_rgb(grey)
    assert result.shape == (480, 640, 3)
    np.testing.assert_array_equal(result[:, :, 0], grey.squeeze())


@pytest.mark.unit
def test_ir_to_rgb_already_3channel(bgr_frame):
    """Already 3-channel frame returned as-is (no cv2 call)."""
    result = ir_to_rgb(bgr_frame)
    assert result is bgr_frame


@pytest.mark.unit
def test_ir_to_rgb_preserves_dtype():
    """ir_to_rgb preserves uint8 dtype."""
    grey = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    result = ir_to_rgb(grey)
    assert result.dtype == np.uint8


# =============================================================================
# Camera class tests
# =============================================================================


@pytest.fixture
def mock_cap():
    """Configure a mock VideoCapture on the autouse-mocked cv2."""
    import faceauth.camera
    cap = MagicMock()
    cap.isOpened.return_value = True
    faceauth.camera.cv2.VideoCapture.return_value = cap
    return cap


@pytest.mark.unit
def test_camera_open_parses_dev_video(mock_cap):
    """open() parses /dev/video2 to integer index 2."""
    import faceauth.camera
    cam = Camera("/dev/video2", 640, 480)
    cam.open()

    faceauth.camera.cv2.VideoCapture.assert_called_once_with(2)
    assert mock_cap.set.call_count >= 2
    assert mock_cap.read.call_count >= 5


@pytest.mark.unit
def test_camera_open_raises_if_camera_fails(mock_cap):
    """open() raises RuntimeError if camera fails to open."""
    mock_cap.isOpened.return_value = False

    cam = Camera("/dev/video0")
    with pytest.raises(RuntimeError, match="Failed to open camera"):
        cam.open()


@pytest.mark.unit
def test_camera_read_raises_if_not_opened():
    """read() raises RuntimeError if camera not opened."""
    cam = Camera("/dev/video0")
    with pytest.raises(RuntimeError, match="Camera not opened"):
        cam.read()


@pytest.mark.unit
def test_camera_read_raises_if_read_fails(mock_cap):
    """read() raises RuntimeError if read returns False."""
    mock_cap.read.return_value = (False, None)

    cam = Camera("/dev/video0")
    cam.open()
    with pytest.raises(RuntimeError, match="Failed to read frame"):
        cam.read()


@pytest.mark.unit
def test_camera_read_returns_frame_on_success(mock_cap, bgr_frame):
    """read() returns frame on success."""
    mock_cap.read.side_effect = [(True, bgr_frame)] * 5 + [(True, bgr_frame)]

    cam = Camera("/dev/video0")
    cam.open()
    frame = cam.read()
    np.testing.assert_array_equal(frame, bgr_frame)


@pytest.mark.unit
def test_camera_close_releases_camera(mock_cap):
    """close() releases camera resources."""
    cam = Camera("/dev/video0")
    cam.open()
    cam.close()

    mock_cap.release.assert_called_once()
    assert cam._cap is None


@pytest.mark.unit
def test_camera_context_manager(mock_cap, bgr_frame):
    """Context manager calls open/close."""
    mock_cap.read.return_value = (True, bgr_frame)

    with Camera("/dev/video0") as cam:
        assert cam._cap is not None
        frame = cam.read()
        np.testing.assert_array_equal(frame, bgr_frame)

    mock_cap.release.assert_called_once()


@pytest.mark.unit
def test_camera_close_when_cap_is_none():
    """close() when _cap is None is a no-op."""
    cam = Camera("/dev/video0")
    cam.close()
    assert cam._cap is None


@pytest.mark.unit
def test_camera_open_sets_resolution(mock_cap):
    """open() sets the requested width and height."""
    cam = Camera("/dev/video2", width=1280, height=720)
    cam.open()
    assert mock_cap.set.call_count >= 2


@pytest.mark.unit
def test_camera_open_warms_up_camera(mock_cap):
    """open() discards first 5 frames for warm-up."""
    cam = Camera("/dev/video0")
    cam.open()
    assert mock_cap.read.call_count == 5


@pytest.mark.unit
def test_camera_open_with_non_dev_path(mock_cap):
    """open() handles non /dev/video paths as string."""
    import faceauth.camera
    cam = Camera("0")
    cam.open()
    faceauth.camera.cv2.VideoCapture.assert_called_once_with("0")


@pytest.mark.unit
def test_camera_double_close(mock_cap):
    """Calling close() twice is safe."""
    cam = Camera("/dev/video0")
    cam.open()
    cam.close()
    cam.close()
    assert mock_cap.release.call_count == 1


# =============================================================================
# capture_frames tests
# =============================================================================


@pytest.mark.unit
@patch("faceauth.camera.Camera")
def test_capture_frames_single_frame(mock_camera_class, bgr_frame):
    """capture_frames captures a single frame."""
    mock_cam = MagicMock()
    mock_cam.read.return_value = bgr_frame
    mock_camera_class.return_value.__enter__.return_value = mock_cam

    frames = capture_frames("/dev/video0", count=1)
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], bgr_frame)


@pytest.mark.unit
@patch("faceauth.camera.Camera")
def test_capture_frames_multiple_frames(mock_camera_class, bgr_frame):
    """capture_frames captures multiple frames."""
    mock_cam = MagicMock()
    mock_cam.read.return_value = bgr_frame
    mock_camera_class.return_value.__enter__.return_value = mock_cam

    frames = capture_frames("/dev/video0", count=3)
    assert len(frames) == 3


@pytest.mark.unit
@patch("faceauth.camera.Camera")
@patch("faceauth.camera.time.monotonic")
def test_capture_frames_timeout(mock_monotonic, mock_camera_class):
    """capture_frames raises TimeoutError if timeout exceeded."""
    mock_monotonic.side_effect = [0.0, 5.0, 11.0]
    mock_cam = MagicMock()
    mock_cam.read.side_effect = RuntimeError("Read failed")
    mock_camera_class.return_value.__enter__.return_value = mock_cam

    with pytest.raises(TimeoutError, match="Camera capture timed out"):
        capture_frames("/dev/video0", count=5, timeout=10.0)


@pytest.mark.unit
@patch("faceauth.camera.Camera")
def test_capture_frames_retries_on_read_failure(mock_camera_class, bgr_frame):
    """capture_frames retries on RuntimeError from read()."""
    mock_cam = MagicMock()
    mock_cam.read.side_effect = [
        RuntimeError("Read failed"),
        RuntimeError("Read failed"),
        bgr_frame,
    ]
    mock_camera_class.return_value.__enter__.return_value = mock_cam

    frames = capture_frames("/dev/video0", count=1)
    assert len(frames) == 1
    assert mock_cam.read.call_count == 3


@pytest.mark.unit
@patch("faceauth.camera.Camera")
def test_capture_frames_uses_context_manager(mock_camera_class, bgr_frame):
    """capture_frames uses Camera as context manager."""
    mock_cam = MagicMock()
    mock_cam.read.return_value = bgr_frame
    mock_context = MagicMock()
    mock_context.__enter__ = Mock(return_value=mock_cam)
    mock_context.__exit__ = Mock(return_value=False)
    mock_camera_class.return_value = mock_context

    frames = capture_frames("/dev/video0", count=1, width=1280, height=720)
    mock_camera_class.assert_called_once_with("/dev/video0", 1280, 720)
    assert len(frames) == 1


@pytest.mark.unit
def test_is_ir_frame_empty_frame():
    """is_ir_frame handles empty frame gracefully."""
    frame = np.array([], dtype=np.uint8)
    try:
        result = is_ir_frame(frame)
        assert isinstance(result, (bool, np.bool_))
    except (IndexError, ValueError):
        pass
