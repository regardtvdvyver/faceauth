"""Tests for faceauth.pipeline shared functions and constants."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from faceauth.antispoof import AntispoofChecker, AntispoofResult
from faceauth.config import Config
from faceauth.pipeline import (
    ATTEMPT_MULTIPLIER,
    FRAME_DELAY_ENROLL,
    FRAME_DELAY_VERIFY,
    MIN_DET_SCORE,
    MIN_SELF_CONSISTENCY,
    check_antispoof,
    compute_self_consistency,
    make_antispoof,
    match_embedding,
    process_frame,
    select_best_face,
)
from faceauth.recognizer import FaceRecognizer


# ============================================================================
# Constants Tests
# ============================================================================


@pytest.mark.unit
class TestConstants:
    def test_min_det_score(self):
        assert MIN_DET_SCORE == 0.5

    def test_min_self_consistency(self):
        assert MIN_SELF_CONSISTENCY == 0.3

    def test_attempt_multiplier(self):
        assert ATTEMPT_MULTIPLIER == 10

    def test_frame_delay_verify(self):
        assert FRAME_DELAY_VERIFY == 0.15

    def test_frame_delay_enroll(self):
        assert FRAME_DELAY_ENROLL == 0.3


# ============================================================================
# make_antispoof Tests
# ============================================================================


@pytest.mark.unit
class TestMakeAntispoof:
    def test_returns_none_when_disabled(self):
        cfg = Config()
        cfg.antispoof.enabled = False
        assert make_antispoof(cfg) is None

    def test_returns_checker_when_enabled(self, tmp_path):
        cfg = Config()
        cfg.antispoof.enabled = True
        cfg.antispoof.ir_brightness_min = 20.0
        cfg.antispoof.minifasnet_threshold = 0.9
        cfg.antispoof.require_both = False
        cfg.antispoof.ir_only_fallback = True
        cfg.antispoof.minifasnet_model_path = str(tmp_path / "model.onnx")

        checker = make_antispoof(cfg)
        assert isinstance(checker, AntispoofChecker)
        assert checker.ir_checker.brightness_min == 20.0
        assert checker.fas_checker.threshold == 0.9
        assert checker.require_both is False
        assert checker.ir_only_fallback is True


# ============================================================================
# process_frame Tests
# ============================================================================


@pytest.mark.unit
class TestProcessFrame:
    def test_ir_greyscale_frame(self, grey_frame):
        raw_ir, bgr_frame = process_frame(grey_frame)
        assert raw_ir is not None
        assert bgr_frame.shape[2] == 3  # converted to BGR

    def test_bgr_frame(self, bgr_frame):
        raw_ir, result_frame = process_frame(bgr_frame)
        assert raw_ir is None
        assert result_frame is bgr_frame  # unchanged

    def test_ir_bgr_frame(self, ir_bgr_frame):
        raw_ir, bgr_frame = process_frame(ir_bgr_frame)
        assert raw_ir is not None
        assert bgr_frame.shape[2] == 3

    def test_raw_ir_is_copy(self, grey_frame):
        raw_ir, _ = process_frame(grey_frame)
        # Modifying raw_ir should not modify the original
        raw_ir[0, 0] = 255
        assert not np.array_equal(raw_ir, grey_frame) or grey_frame[0, 0] == 255


# ============================================================================
# select_best_face Tests
# ============================================================================


@pytest.mark.unit
class TestSelectBestFace:
    def test_empty_list_returns_none(self):
        assert select_best_face([]) is None

    def test_single_face_above_threshold(self):
        face = Mock(det_score=0.8)
        assert select_best_face([face]) is face

    def test_single_face_below_threshold(self):
        face = Mock(det_score=0.3)
        assert select_best_face([face]) is None

    def test_single_face_at_exact_threshold(self):
        face = Mock(det_score=MIN_DET_SCORE)
        # det_score < MIN_DET_SCORE returns None, == threshold passes
        assert select_best_face([face]) is face

    def test_multiple_faces_picks_best(self):
        face1 = Mock(det_score=0.6)
        face2 = Mock(det_score=0.9)
        face3 = Mock(det_score=0.7)
        assert select_best_face([face1, face2, face3]) is face2

    def test_all_below_threshold(self):
        face1 = Mock(det_score=0.2)
        face2 = Mock(det_score=0.4)
        assert select_best_face([face1, face2]) is None


# ============================================================================
# check_antispoof Tests
# ============================================================================


@pytest.mark.unit
class TestCheckAntispoof:
    def test_none_antispoof_returns_none(self, grey_frame, bgr_frame, sample_bbox):
        assert check_antispoof(None, grey_frame, bgr_frame, sample_bbox) is None

    def test_none_raw_ir_returns_none(self, bgr_frame, sample_bbox):
        checker = MagicMock(spec=AntispoofChecker)
        assert check_antispoof(checker, None, bgr_frame, sample_bbox) is None

    def test_both_none_returns_none(self, bgr_frame, sample_bbox):
        assert check_antispoof(None, None, bgr_frame, sample_bbox) is None

    def test_runs_check_when_both_present(self, grey_frame, bgr_frame, sample_bbox):
        expected = AntispoofResult(passed=True, ir_brightness=100.0)
        checker = MagicMock(spec=AntispoofChecker)
        checker.check.return_value = expected

        result = check_antispoof(checker, grey_frame, bgr_frame, sample_bbox)
        assert result is expected
        checker.check.assert_called_once_with(grey_frame, bgr_frame, sample_bbox)


# ============================================================================
# match_embedding Tests
# ============================================================================


@pytest.mark.unit
class TestMatchEmbedding:
    def test_identical_embedding(self, sample_embedding):
        score = match_embedding(sample_embedding, [sample_embedding])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_similar_embedding(self, sample_embedding, similar_embedding):
        score = match_embedding(sample_embedding, [similar_embedding])
        assert score > 0.5

    def test_different_embedding(self, sample_embedding, different_embedding):
        score = match_embedding(sample_embedding, [different_embedding])
        assert score < 0.5

    def test_best_of_multiple(self, sample_embedding, similar_embedding, different_embedding):
        score = match_embedding(sample_embedding, [different_embedding, similar_embedding])
        # Should pick the best score (similar_embedding)
        expected = FaceRecognizer.compute_similarity(sample_embedding, similar_embedding)
        assert score == pytest.approx(expected, abs=0.001)

    def test_empty_stored_returns_zero(self, sample_embedding):
        score = match_embedding(sample_embedding, [])
        assert score == 0.0


# ============================================================================
# compute_self_consistency Tests
# ============================================================================


@pytest.mark.unit
class TestComputeSelfConsistency:
    def test_fewer_than_two_returns_zeros(self, sample_embedding):
        assert compute_self_consistency([]) == (0.0, 0.0)
        assert compute_self_consistency([sample_embedding]) == (0.0, 0.0)

    def test_identical_embeddings(self, sample_embedding):
        avg, min_score = compute_self_consistency([sample_embedding, sample_embedding])
        assert avg == pytest.approx(1.0, abs=0.01)
        assert min_score == pytest.approx(1.0, abs=0.01)

    def test_similar_embeddings(self, sample_embedding, similar_embedding):
        avg, min_score = compute_self_consistency([sample_embedding, similar_embedding])
        assert avg > 0.5
        assert min_score > 0.5

    def test_mixed_embeddings(self, sample_embedding, similar_embedding, different_embedding):
        avg, min_score = compute_self_consistency(
            [sample_embedding, similar_embedding, different_embedding]
        )
        # avg should be pulled down by the different embedding
        assert avg < 0.9
        # min should be the score against the different embedding
        assert min_score < 0.5

    def test_three_identical(self, sample_embedding):
        avg, min_score = compute_self_consistency(
            [sample_embedding, sample_embedding, sample_embedding]
        )
        assert avg == pytest.approx(1.0, abs=0.01)
        assert min_score == pytest.approx(1.0, abs=0.01)
