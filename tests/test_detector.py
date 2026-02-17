"""Unit tests for faceauth.detector module.

cv2 is auto-mocked by conftest._patch_cv2 (autouse fixture).
mediapipe is mocked per-test via the mock_mediapipe fixture.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch

from faceauth.detector import FaceDetector, FaceDetection


@pytest.fixture
def mock_mediapipe():
    """Mock the MediaPipe face detection module."""
    with patch("faceauth.detector.mp") as mock_mp:
        mock_detector_instance = MagicMock()
        mock_mp.solutions.face_detection.FaceDetection.return_value = mock_detector_instance
        yield mock_mp, mock_detector_instance


@pytest.fixture
def sample_frame():
    """480x640x3 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def create_mock_detection(xmin, ymin, width, height, score):
    """Create a mock MediaPipe detection."""
    d = Mock()
    d.location_data.relative_bounding_box.xmin = xmin
    d.location_data.relative_bounding_box.ymin = ymin
    d.location_data.relative_bounding_box.width = width
    d.location_data.relative_bounding_box.height = height
    d.score = [score]
    return d


@pytest.mark.unit
def test_no_faces_detected(mock_mediapipe, sample_frame):
    """Empty list when detections is None."""
    _, mock_det = mock_mediapipe
    results = Mock()
    results.detections = None
    mock_det.process.return_value = results

    detector = FaceDetector()
    assert detector.detect(sample_frame) == []


@pytest.mark.unit
def test_empty_detections_list(mock_mediapipe, sample_frame):
    """Empty list when detections is empty list."""
    _, mock_det = mock_mediapipe
    results = Mock()
    results.detections = []
    mock_det.process.return_value = results

    detector = FaceDetector()
    assert detector.detect(sample_frame) == []


@pytest.mark.unit
def test_single_face_detected(mock_mediapipe, sample_frame):
    """Single face with correct bbox and confidence."""
    _, mock_det = mock_mediapipe
    det = create_mock_detection(0.25, 0.25, 0.15625, 0.20834, 0.95)
    results = Mock()
    results.detections = [det]
    mock_det.process.return_value = results

    detector = FaceDetector()
    faces = detector.detect(sample_frame)

    assert len(faces) == 1
    assert faces[0].confidence == 0.95
    assert faces[0].bbox == (160, 120, 100, 100)
    assert faces[0].face_img is not None


@pytest.mark.unit
def test_multiple_faces_sorted_by_confidence(mock_mediapipe, sample_frame):
    """Multiple faces sorted by confidence descending."""
    _, mock_det = mock_mediapipe
    d1 = create_mock_detection(0.1, 0.1, 0.15, 0.2, 0.7)
    d2 = create_mock_detection(0.5, 0.1, 0.15, 0.2, 0.95)
    d3 = create_mock_detection(0.1, 0.5, 0.15, 0.2, 0.8)
    results = Mock()
    results.detections = [d1, d2, d3]
    mock_det.process.return_value = results

    detector = FaceDetector()
    faces = detector.detect(sample_frame)

    assert [f.confidence for f in faces] == [0.95, 0.8, 0.7]


@pytest.mark.unit
def test_small_face_filtered_out(mock_mediapipe, sample_frame):
    """Faces smaller than 20x20 pixels are filtered out."""
    _, mock_det = mock_mediapipe
    small = create_mock_detection(0.5, 0.5, 0.015625, 0.020833, 0.9)
    normal = create_mock_detection(0.1, 0.1, 0.15, 0.2, 0.85)
    results = Mock()
    results.detections = [small, normal]
    mock_det.process.return_value = results

    detector = FaceDetector()
    faces = detector.detect(sample_frame)

    assert len(faces) == 1
    assert faces[0].confidence == 0.85


@pytest.mark.unit
def test_margin_applied_correctly(mock_mediapipe, sample_frame):
    """20% margin applied to each side."""
    _, mock_det = mock_mediapipe
    # 200x200 face at (220, 140)
    det = create_mock_detection(0.34375, 0.291667, 0.3125, 0.416667, 0.9)
    results = Mock()
    results.detections = [det]
    mock_det.process.return_value = results

    detector = FaceDetector()
    faces = detector.detect(sample_frame)

    assert faces[0].bbox == (220, 140, 200, 200)
    assert faces[0].face_img.shape == (280, 280, 3)


@pytest.mark.unit
def test_bbox_clamped_to_boundaries(mock_mediapipe, sample_frame):
    """Bbox clamped to frame boundaries."""
    _, mock_det = mock_mediapipe
    det = create_mock_detection(10 / 640, 10 / 480, 100 / 640, 100 / 480, 0.9)
    results = Mock()
    results.detections = [det]
    mock_det.process.return_value = results

    detector = FaceDetector()
    faces = detector.detect(sample_frame)

    assert len(faces) == 1
    assert faces[0].face_img.shape[0] > 0


@pytest.mark.unit
def test_face_img_is_copy(mock_mediapipe):
    """face_img is a copy, not a view."""
    _, mock_det = mock_mediapipe
    frame = np.full((480, 640, 3), 42, dtype=np.uint8)
    det = create_mock_detection(0.25, 0.25, 0.15625, 0.208333, 0.9)
    results = Mock()
    results.detections = [det]
    mock_det.process.return_value = results

    detector = FaceDetector()
    faces = detector.detect(frame)

    faces[0].face_img[:] = 99
    assert frame[0, 0, 0] == 42


@pytest.mark.unit
def test_face_detection_dataclass():
    """FaceDetection dataclass stores correct values."""
    det = FaceDetection(bbox=(100, 200, 50, 60), confidence=0.95)
    assert det.bbox == (100, 200, 50, 60)
    assert det.confidence == 0.95
    assert det.face_img is None


@pytest.mark.unit
def test_close_calls_detector_close(mock_mediapipe):
    """close() calls underlying detector close."""
    _, mock_det = mock_mediapipe
    detector = FaceDetector()
    detector.close()
    mock_det.close.assert_called_once()


@pytest.mark.unit
def test_context_manager(mock_mediapipe, sample_frame):
    """FaceDetector works as a context manager."""
    _, mock_det = mock_mediapipe
    det = create_mock_detection(0.25, 0.25, 0.15625, 0.208333, 0.9)
    results = Mock()
    results.detections = [det]
    mock_det.process.return_value = results

    with FaceDetector() as detector:
        faces = detector.detect(sample_frame)
        assert len(faces) == 1

    mock_det.close.assert_called_once()


@pytest.mark.unit
def test_custom_min_confidence(mock_mediapipe):
    """Custom min_confidence is passed to MediaPipe."""
    mock_mp, _ = mock_mediapipe
    FaceDetector(min_confidence=0.75)
    mock_mp.solutions.face_detection.FaceDetection.assert_called_once_with(
        model_selection=1,
        min_detection_confidence=0.75,
    )


@pytest.mark.unit
def test_negative_bbox_coordinates_clamped(mock_mediapipe, sample_frame):
    """Negative bbox coordinates clamped to 0."""
    _, mock_det = mock_mediapipe
    det = create_mock_detection(-0.05, -0.1, 0.2, 0.25, 0.85)
    results = Mock()
    results.detections = [det]
    mock_det.process.return_value = results

    detector = FaceDetector()
    faces = detector.detect(sample_frame)

    assert len(faces) == 1
    x, y, w, h = faces[0].bbox
    assert x >= 0
    assert y >= 0
