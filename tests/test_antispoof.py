"""Comprehensive tests for the faceauth antispoof module.

Tests cover:
- IRBrightnessChecker: brightness thresholds, edge cases
- MiniFASNetChecker: model availability, liveness scoring
- AntispoofChecker: coordination logic, fallback behavior
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from faceauth.antispoof import (
    AntispoofChecker,
    AntispoofResult,
    IRBrightnessChecker,
    MiniFASNetChecker,
)


# ============================================================================
# IRBrightnessChecker Tests
# ============================================================================


@pytest.mark.unit
def test_ir_brightness_bright_frame_passes(bright_ir_frame, sample_bbox):
    """Test that a bright IR frame passes the brightness check."""
    checker = IRBrightnessChecker(brightness_min=15.0)
    passed, brightness = checker.check(bright_ir_frame, sample_bbox)

    assert passed is True
    assert brightness == 120.0  # frame is filled with 120


@pytest.mark.unit
def test_ir_brightness_dark_frame_fails(dark_ir_frame, sample_bbox):
    """Test that a dark IR frame fails the brightness check."""
    checker = IRBrightnessChecker(brightness_min=15.0)
    passed, brightness = checker.check(dark_ir_frame, sample_bbox)

    assert passed is False
    assert brightness == 5.0  # frame is filled with 5


@pytest.mark.unit
def test_ir_brightness_exact_threshold_passes(sample_bbox):
    """Test that brightness exactly at threshold passes."""
    threshold = 50.0
    ir_frame = np.full((480, 640), int(threshold), dtype=np.uint8)
    checker = IRBrightnessChecker(brightness_min=threshold)

    passed, brightness = checker.check(ir_frame, sample_bbox)

    assert passed is True
    assert brightness == threshold


@pytest.mark.unit
def test_ir_brightness_bgr_extracts_first_channel(ir_bgr_frame, sample_bbox):
    """Test that BGR IR frame correctly extracts first channel."""
    # ir_bgr_frame has all channels equal, filled with random values
    # Extract the face region to get expected brightness
    x1, y1, x2, y2 = (
        int(sample_bbox[0]),
        int(sample_bbox[1]),
        int(sample_bbox[2]),
        int(sample_bbox[3]),
    )
    expected_brightness = float(np.mean(ir_bgr_frame[y1:y2, x1:x2, 0]))

    checker = IRBrightnessChecker(brightness_min=15.0)
    passed, brightness = checker.check(ir_bgr_frame, sample_bbox)

    assert brightness == expected_brightness


@pytest.mark.unit
def test_ir_brightness_invalid_bbox_returns_false(bright_ir_frame):
    """Test that invalid bbox returns (False, 0.0)."""
    checker = IRBrightnessChecker(brightness_min=15.0)

    # Bbox with x2 <= x1
    invalid_bbox_1 = np.array([300, 80, 100, 320], dtype=np.float32)
    passed, brightness = checker.check(bright_ir_frame, invalid_bbox_1)
    assert passed is False
    assert brightness == 0.0

    # Bbox with y2 <= y1
    invalid_bbox_2 = np.array([100, 320, 300, 80], dtype=np.float32)
    passed, brightness = checker.check(bright_ir_frame, invalid_bbox_2)
    assert passed is False
    assert brightness == 0.0


@pytest.mark.unit
def test_ir_brightness_custom_threshold_respected(bright_ir_frame, sample_bbox):
    """Test that custom brightness_min threshold is respected."""
    # Frame brightness is 120, should fail with high threshold
    checker_high = IRBrightnessChecker(brightness_min=150.0)
    passed, brightness = checker_high.check(bright_ir_frame, sample_bbox)
    assert passed is False
    assert brightness == 120.0

    # Should pass with low threshold
    checker_low = IRBrightnessChecker(brightness_min=10.0)
    passed, brightness = checker_low.check(bright_ir_frame, sample_bbox)
    assert passed is True
    assert brightness == 120.0


# ============================================================================
# MiniFASNetChecker Tests
# ============================================================================


@pytest.mark.unit
def test_minifasnet_model_not_available_returns_none(bgr_frame, sample_bbox, tmp_path):
    """Test that check() returns None when model is not available."""
    non_existent_path = tmp_path / "nonexistent_model.onnx"
    checker = MiniFASNetChecker(model_path=non_existent_path)

    assert checker.available is False
    result = checker.check(bgr_frame, sample_bbox)
    assert result is None


@pytest.mark.unit
def test_minifasnet_high_liveness_score_passes(bgr_frame, sample_bbox, tmp_path):
    """Test that high liveness score passes the check."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()  # Create dummy file to satisfy existence check

    # Mock ONNX runtime
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]

    # Return logits that result in high liveness score
    # logits = [0.1, 2.0] -> probs ≈ [0.15, 0.85] -> liveness_score = 0.85
    mock_session.run.return_value = [np.array([[0.1, 2.0]], dtype=np.float32)]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = MiniFASNetChecker(model_path=model_path, threshold=0.8)
            passed, score = checker.check(bgr_frame, sample_bbox)

    assert passed is True
    assert score > 0.8


@pytest.mark.unit
def test_minifasnet_low_liveness_score_fails(bgr_frame, sample_bbox, tmp_path):
    """Test that low liveness score fails the check."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]

    # Return logits that result in low liveness score
    # logits = [2.0, 0.1] -> probs ≈ [0.85, 0.15] -> liveness_score = 0.15
    mock_session.run.return_value = [np.array([[2.0, 0.1]], dtype=np.float32)]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = MiniFASNetChecker(model_path=model_path, threshold=0.8)
            passed, score = checker.check(bgr_frame, sample_bbox)

    assert passed is False
    assert score < 0.8


@pytest.mark.unit
def test_minifasnet_invalid_bbox_returns_false(bgr_frame, tmp_path):
    """Test that invalid bbox returns (False, 0.0)."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        checker = MiniFASNetChecker(model_path=model_path)

        # Invalid bbox with x2 <= x1
        invalid_bbox = np.array([300, 80, 100, 320], dtype=np.float32)
        result = checker.check(bgr_frame, invalid_bbox)

    assert result is not None
    passed, score = result
    assert passed is False
    assert score == 0.0


# ============================================================================
# AntispoofChecker Coordination Tests
# ============================================================================


@pytest.mark.unit
def test_antispoof_both_pass_require_both_true(bright_ir_frame, bgr_frame, sample_bbox, tmp_path):
    """Test that both checks passing with require_both=True results in passed=True."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[0.1, 2.0]], dtype=np.float32)]  # High liveness

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = AntispoofChecker(
                ir_brightness_min=15.0,
                minifasnet_threshold=0.8,
                minifasnet_model_path=model_path,
                require_both=True,
            )
            result = checker.check(bright_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is True
    assert result.ir_passed is True
    assert result.liveness_passed is True


@pytest.mark.unit
def test_antispoof_ir_pass_liveness_fail_require_both_true(
    bright_ir_frame, bgr_frame, sample_bbox, tmp_path
):
    """Test IR pass + liveness fail with require_both=True results in passed=False."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[2.0, 0.1]], dtype=np.float32)]  # Low liveness

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = AntispoofChecker(
                ir_brightness_min=15.0,
                minifasnet_threshold=0.8,
                minifasnet_model_path=model_path,
                require_both=True,
            )
            result = checker.check(bright_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is False
    assert result.ir_passed is True
    assert result.liveness_passed is False


@pytest.mark.unit
def test_antispoof_ir_fail_liveness_pass_require_both_true(
    dark_ir_frame, bgr_frame, sample_bbox, tmp_path
):
    """Test IR fail + liveness pass with require_both=True results in passed=False."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[0.1, 2.0]], dtype=np.float32)]  # High liveness

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = AntispoofChecker(
                ir_brightness_min=15.0,
                minifasnet_threshold=0.8,
                minifasnet_model_path=model_path,
                require_both=True,
            )
            result = checker.check(dark_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is False
    assert result.ir_passed is False
    assert result.liveness_passed is True


@pytest.mark.unit
def test_antispoof_ir_pass_liveness_fail_require_both_false(
    bright_ir_frame, bgr_frame, sample_bbox, tmp_path
):
    """Test IR pass + liveness fail with require_both=False uses OR logic (passed=True)."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[2.0, 0.1]], dtype=np.float32)]  # Low liveness

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = AntispoofChecker(
                ir_brightness_min=15.0,
                minifasnet_threshold=0.8,
                minifasnet_model_path=model_path,
                require_both=False,
            )
            result = checker.check(bright_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is True  # OR logic: IR passed
    assert result.ir_passed is True
    assert result.liveness_passed is False


@pytest.mark.unit
def test_antispoof_minifasnet_unavailable_ir_fallback_pass(
    bright_ir_frame, bgr_frame, sample_bbox, tmp_path
):
    """Test MiniFASNet unavailable + ir_only_fallback=True with IR pass results in passed=True."""
    non_existent_path = tmp_path / "nonexistent.onnx"

    checker = AntispoofChecker(
        ir_brightness_min=15.0,
        minifasnet_model_path=non_existent_path,
        ir_only_fallback=True,
    )
    result = checker.check(bright_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is True
    assert result.ir_passed is True
    assert result.liveness_score == 0.0  # No liveness check performed


@pytest.mark.unit
def test_antispoof_minifasnet_unavailable_ir_fallback_fail(
    dark_ir_frame, bgr_frame, sample_bbox, tmp_path
):
    """Test MiniFASNet unavailable + ir_only_fallback=True with IR fail results in passed=False."""
    non_existent_path = tmp_path / "nonexistent.onnx"

    checker = AntispoofChecker(
        ir_brightness_min=15.0,
        minifasnet_model_path=non_existent_path,
        ir_only_fallback=True,
    )
    result = checker.check(dark_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is False
    assert result.ir_passed is False


@pytest.mark.unit
def test_antispoof_minifasnet_unavailable_no_fallback_fails(
    bright_ir_frame, bgr_frame, sample_bbox, tmp_path
):
    """Test MiniFASNet unavailable + ir_only_fallback=False results in passed=False with reason."""
    non_existent_path = tmp_path / "nonexistent.onnx"

    checker = AntispoofChecker(
        ir_brightness_min=15.0,
        minifasnet_model_path=non_existent_path,
        ir_only_fallback=False,
    )
    result = checker.check(bright_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is False
    assert "MiniFASNet model not available" in result.reason
    assert "ir_only_fallback=False" in result.reason


@pytest.mark.unit
def test_antispoof_failure_reason_includes_details(dark_ir_frame, bgr_frame, sample_bbox, tmp_path):
    """Test that failure reason string includes correct details."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[2.0, 0.1]], dtype=np.float32)]  # Low liveness

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = AntispoofChecker(
                ir_brightness_min=15.0,
                minifasnet_threshold=0.8,
                minifasnet_model_path=model_path,
                require_both=True,
            )
            result = checker.check(dark_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is False
    # Both checks should fail, reason should include both
    assert "IR brightness" in result.reason
    assert "5.0" in result.reason  # dark_ir_frame brightness
    assert "15" in result.reason  # threshold
    assert "liveness score" in result.reason


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


@pytest.mark.unit
def test_antispoof_result_dataclass_defaults():
    """Test AntispoofResult has correct default values."""
    result = AntispoofResult(passed=False)

    assert result.passed is False
    assert result.ir_brightness == 0.0
    assert result.ir_passed is False
    assert result.liveness_score == 0.0
    assert result.liveness_passed is False
    assert result.reason == ""


@pytest.mark.unit
def test_minifasnet_checker_available_property_caching(tmp_path):
    """Test that MiniFASNetChecker.available property caches its result."""
    model_path = tmp_path / "test_model.onnx"
    checker = MiniFASNetChecker(model_path=model_path)

    # First call checks filesystem
    assert checker.available is False

    # Create file after first check
    model_path.touch()

    # Second call uses cached value (still False)
    assert checker.available is False

    # New checker instance sees the file
    checker2 = MiniFASNetChecker(model_path=model_path)
    assert checker2.available is True


@pytest.mark.unit
def test_ir_brightness_bbox_boundary_clamping(bright_ir_frame):
    """Test that bbox coordinates are properly clamped to frame boundaries."""
    checker = IRBrightnessChecker(brightness_min=15.0)

    # Bbox extends beyond frame boundaries
    out_of_bounds_bbox = np.array([-10, -10, 700, 500], dtype=np.float32)
    passed, brightness = checker.check(bright_ir_frame, out_of_bounds_bbox)

    # Should clamp to [0, 0, 640, 480] and still work
    assert passed is True
    assert brightness == 120.0


@pytest.mark.unit
def test_antispoof_both_checks_fail_reason_combines(
    dark_ir_frame, bgr_frame, sample_bbox, tmp_path
):
    """Test that when both checks fail, reason combines both failures."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[2.0, 0.1]], dtype=np.float32)]  # Low liveness

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            checker = AntispoofChecker(
                ir_brightness_min=15.0,
                minifasnet_threshold=0.8,
                minifasnet_model_path=model_path,
                require_both=True,
            )
            result = checker.check(dark_ir_frame, bgr_frame, sample_bbox)

    assert result.passed is False
    # Reason should contain both failure messages separated by semicolon
    assert "IR brightness" in result.reason
    assert "liveness score" in result.reason
    assert ";" in result.reason


@pytest.mark.unit
def test_minifasnet_model_loads_lazily(bgr_frame, sample_bbox, tmp_path):
    """Test that MiniFASNet model is loaded lazily on first check() call."""
    model_path = tmp_path / "test_model.onnx"
    model_path.touch()

    checker = MiniFASNetChecker(model_path=model_path)

    # Model should not be loaded yet
    assert checker._session is None

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([[0.1, 2.0]], dtype=np.float32)]

    with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_ort:
        with patch(
            "cv2.resize", return_value=np.zeros((128, 128, 3), dtype=np.float32), create=True
        ):
            # First check loads the model
            checker.check(bgr_frame, sample_bbox)
            assert mock_ort.call_count == 1

            # Second check reuses loaded model
            checker.check(bgr_frame, sample_bbox)
            assert mock_ort.call_count == 1  # Still only called once


@pytest.mark.unit
def test_antispoof_checker_minifasnet_available_property(tmp_path):
    """Test that AntispoofChecker exposes minifasnet_available property."""
    non_existent_path = tmp_path / "nonexistent.onnx"
    checker_unavailable = AntispoofChecker(minifasnet_model_path=non_existent_path)
    assert checker_unavailable.minifasnet_available is False

    existing_path = tmp_path / "existing.onnx"
    existing_path.touch()
    checker_available = AntispoofChecker(minifasnet_model_path=existing_path)
    assert checker_available.minifasnet_available is True
