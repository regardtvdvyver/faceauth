"""Shared test fixtures for faceauth tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest


def _mock_cvtcolor(frame, code):
    """Simulate cv2.cvtColor for GRAY2BGR and BGR2RGB."""
    if len(frame.shape) == 2:
        return np.stack([frame, frame, frame], axis=-1)
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        s = frame[:, :, 0]
        return np.stack([s, s, s], axis=-1)
    # BGR2RGB: swap channels
    return frame[:, :, ::-1].copy() if len(frame.shape) == 3 else frame


def _mock_cv2_split(frame):
    """Simulate cv2.split for multi-channel frames."""
    if len(frame.shape) == 3:
        return tuple(frame[:, :, i] for i in range(frame.shape[2]))
    return (frame,)


def _mock_cv2_resize(img, size):
    """Simulate cv2.resize (nearest-neighbour approximation)."""
    w, h = size
    # Simple nearest-neighbour resize
    from numpy import linspace
    rows = np.round(linspace(0, img.shape[0] - 1, h)).astype(int)
    cols = np.round(linspace(0, img.shape[1] - 1, w)).astype(int)
    return img[np.ix_(rows, cols)] if len(img.shape) == 2 else img[np.ix_(rows, cols)]


def make_mock_cv2():
    """Create a mock cv2 module with essential constants and functions."""
    mock = MagicMock()
    mock.cvtColor = MagicMock(side_effect=_mock_cvtcolor)
    mock.split = MagicMock(side_effect=_mock_cv2_split)
    mock.resize = MagicMock(side_effect=_mock_cv2_resize)
    mock.COLOR_BGR2RGB = 4
    mock.COLOR_GRAY2BGR = 8
    mock.CAP_PROP_FRAME_WIDTH = 3
    mock.CAP_PROP_FRAME_HEIGHT = 4
    # VideoCapture is left as MagicMock (tests configure it per-test)
    return mock


@pytest.fixture(autouse=True)
def _patch_cv2(monkeypatch):
    """Auto-patch cv2 on all faceauth modules since OpenCV binaries aren't available."""
    mock_cv2 = make_mock_cv2()
    import faceauth.camera
    import faceauth.detector
    monkeypatch.setattr(faceauth.camera, "cv2", mock_cv2)
    monkeypatch.setattr(faceauth.detector, "cv2", mock_cv2)


@pytest.fixture
def sample_embedding():
    """A random 512-dim normalized embedding."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


@pytest.fixture
def similar_embedding(sample_embedding):
    """An embedding similar to sample_embedding (cosine sim ~0.95+)."""
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(512).astype(np.float32) * 0.05
    emb = sample_embedding + noise
    emb /= np.linalg.norm(emb)
    return emb


@pytest.fixture
def different_embedding():
    """An embedding very different from sample_embedding."""
    rng = np.random.default_rng(7)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


@pytest.fixture
def grey_frame():
    """A 480x640 greyscale frame (simulates IR camera)."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (480, 640), dtype=np.uint8)


@pytest.fixture
def bgr_frame():
    """A 480x640x3 BGR frame."""
    rng = np.random.default_rng(2)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def ir_bgr_frame():
    """A 480x640x3 BGR frame where all channels are equal (IR camera in BGR mode)."""
    rng = np.random.default_rng(3)
    grey = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    return np.stack([grey, grey, grey], axis=-1)


@pytest.fixture
def sample_bbox():
    """A bounding box array [x1, y1, x2, y2] in InsightFace format."""
    return np.array([100, 80, 300, 320], dtype=np.float32)


@pytest.fixture
def bright_ir_frame():
    """An IR frame with high brightness in the face region (simulates real face)."""
    frame = np.full((480, 640), 120, dtype=np.uint8)
    return frame


@pytest.fixture
def dark_ir_frame():
    """An IR frame with low brightness in the face region (simulates spoof)."""
    frame = np.full((480, 640), 5, dtype=np.uint8)
    return frame
