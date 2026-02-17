"""Unit tests for faceauth.recognizer module."""

import sys

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch

from faceauth.recognizer import FaceRecognizer
import faceauth.recognizer


@pytest.fixture
def reset_singleton():
    """Reset the module-level singleton before each test to avoid cross-test pollution."""
    faceauth.recognizer._app = None
    yield
    faceauth.recognizer._app = None


@pytest.fixture
def mock_insightface():
    """Mock insightface by injecting it into sys.modules and patching get_ort_providers."""
    mock_if = MagicMock()
    mock_app_instance = MagicMock()
    mock_if.app.FaceAnalysis.return_value = mock_app_instance

    # Temporarily inject a fake insightface module into sys.modules
    old_module = sys.modules.get("insightface")
    sys.modules["insightface"] = mock_if

    with patch("faceauth.providers.get_ort_providers", return_value=["CPUExecutionProvider"]):
        yield mock_if, mock_app_instance

    # Restore original state
    if old_module is None:
        sys.modules.pop("insightface", None)
    else:
        sys.modules["insightface"] = old_module


def create_mock_face(embedding: np.ndarray, det_score: float, bbox: np.ndarray = None):
    """Helper to create a mock InsightFace face object."""
    face = Mock()
    face.embedding = embedding
    face.det_score = det_score
    face.bbox = bbox if bbox is not None else np.array([100, 80, 300, 320], dtype=np.float32)
    return face


# ============================================================================
# compute_similarity tests (pure math, no mocking needed)
# ============================================================================


@pytest.mark.unit
def test_compute_similarity_identical_embeddings(sample_embedding):
    """Test that identical embeddings have similarity = 1.0."""
    similarity = FaceRecognizer.compute_similarity(sample_embedding, sample_embedding)
    assert np.isclose(similarity, 1.0, atol=1e-6)


@pytest.mark.unit
def test_compute_similarity_similar_embeddings(sample_embedding, similar_embedding):
    """Test that similar embeddings have higher similarity than different ones."""
    similarity = FaceRecognizer.compute_similarity(sample_embedding, similar_embedding)
    # The similar_embedding fixture adds small noise, so similarity should be positive and reasonable
    # but not necessarily >0.9 (actual value is ~0.64 based on the fixture implementation)
    assert similarity > 0.5
    assert similarity < 1.0


@pytest.mark.unit
def test_compute_similarity_different_embeddings(sample_embedding, different_embedding):
    """Test that different embeddings have low similarity."""
    similarity = FaceRecognizer.compute_similarity(sample_embedding, different_embedding)
    assert similarity < 0.5


@pytest.mark.unit
def test_compute_similarity_opposite_embeddings(sample_embedding):
    """Test that opposite embeddings have similarity = -1.0."""
    opposite_embedding = -sample_embedding
    similarity = FaceRecognizer.compute_similarity(sample_embedding, opposite_embedding)
    assert np.isclose(similarity, -1.0, atol=1e-6)


@pytest.mark.unit
def test_compute_similarity_handles_non_normalized_inputs():
    """Test that compute_similarity normalizes inputs internally."""
    # Create non-normalized embeddings (just double the magnitude)
    rng = np.random.default_rng(42)
    emb1 = rng.standard_normal(512).astype(np.float32) * 10.0
    emb2 = rng.standard_normal(512).astype(np.float32) * 5.0

    # Compute similarity with non-normalized
    sim_non_norm = FaceRecognizer.compute_similarity(emb1, emb2)

    # Compute similarity with normalized versions
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    sim_norm = FaceRecognizer.compute_similarity(emb1_norm, emb2_norm)

    # Both should give the same result
    assert np.isclose(sim_non_norm, sim_norm, atol=1e-6)


# ============================================================================
# get_faces tests
# ============================================================================


@pytest.mark.unit
def test_get_faces_delegates_to_insightface(
    reset_singleton, mock_insightface, bgr_frame, sample_embedding
):
    """Test that get_faces delegates to insightface app.get()."""
    mock_if, mock_app_instance = mock_insightface

    # Create mock faces
    face1 = create_mock_face(sample_embedding, 0.95)
    face2 = create_mock_face(sample_embedding, 0.87)
    mock_app_instance.get.return_value = [face1, face2]

    recognizer = FaceRecognizer()
    faces = recognizer.get_faces(bgr_frame)

    assert len(faces) == 2
    assert faces[0] == face1
    assert faces[1] == face2
    mock_app_instance.get.assert_called_once_with(bgr_frame)


@pytest.mark.unit
def test_get_faces_singleton_initialization(reset_singleton, mock_insightface, bgr_frame):
    """Test that _ensure_loaded calls _get_app with correct parameters."""
    mock_if, mock_app_instance = mock_insightface
    mock_app_instance.get.return_value = []

    recognizer = FaceRecognizer(model_name="buffalo_s", det_size=320)
    recognizer.get_faces(bgr_frame)

    # Verify FaceAnalysis was created with correct parameters
    mock_if.app.FaceAnalysis.assert_called_once_with(
        name="buffalo_s",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    # Verify prepare was called with correct parameters
    mock_app_instance.prepare.assert_called_once_with(ctx_id=-1, det_size=(320, 320))


# ============================================================================
# get_embedding tests
# ============================================================================


@pytest.mark.unit
def test_get_embedding_no_faces_detected(reset_singleton, mock_insightface, bgr_frame):
    """Test that get_embedding returns None when no faces are detected."""
    mock_if, mock_app_instance = mock_insightface
    mock_app_instance.get.return_value = []

    recognizer = FaceRecognizer()
    embedding = recognizer.get_embedding(bgr_frame)

    assert embedding is None


@pytest.mark.unit
def test_get_embedding_best_face_below_threshold(
    reset_singleton, mock_insightface, bgr_frame, sample_embedding
):
    """Test that get_embedding returns None when best face score is below 0.5."""
    mock_if, mock_app_instance = mock_insightface

    # Create faces with low detection scores
    face1 = create_mock_face(sample_embedding, 0.3)
    face2 = create_mock_face(sample_embedding, 0.45)
    mock_app_instance.get.return_value = [face1, face2]

    recognizer = FaceRecognizer()
    embedding = recognizer.get_embedding(bgr_frame)

    assert embedding is None


@pytest.mark.unit
def test_get_embedding_returns_highest_scoring_face(
    reset_singleton,
    mock_insightface,
    bgr_frame,
    sample_embedding,
    similar_embedding,
    different_embedding,
):
    """Test that get_embedding returns the embedding of the highest-scoring face."""
    mock_if, mock_app_instance = mock_insightface

    # Create faces with different detection scores and embeddings
    face1 = create_mock_face(sample_embedding, 0.75)
    face2 = create_mock_face(similar_embedding, 0.95)  # Highest score
    face3 = create_mock_face(different_embedding, 0.82)
    mock_app_instance.get.return_value = [face1, face2, face3]

    recognizer = FaceRecognizer()
    embedding = recognizer.get_embedding(bgr_frame)

    # Should return the embedding from face2 (highest score)
    assert embedding is not None
    assert np.array_equal(embedding, similar_embedding)


@pytest.mark.unit
def test_get_embedding_single_valid_face(
    reset_singleton, mock_insightface, bgr_frame, sample_embedding
):
    """Test that get_embedding returns embedding when single face with score >= 0.5."""
    mock_if, mock_app_instance = mock_insightface

    face = create_mock_face(sample_embedding, 0.87)
    mock_app_instance.get.return_value = [face]

    recognizer = FaceRecognizer()
    embedding = recognizer.get_embedding(bgr_frame)

    assert embedding is not None
    assert np.array_equal(embedding, sample_embedding)


# ============================================================================
# get_all_embeddings tests
# ============================================================================


@pytest.mark.unit
def test_get_all_embeddings_returns_only_valid_faces(
    reset_singleton,
    mock_insightface,
    bgr_frame,
    sample_embedding,
    similar_embedding,
    different_embedding,
):
    """Test that get_all_embeddings returns embeddings only for faces with det_score >= 0.5."""
    mock_if, mock_app_instance = mock_insightface

    # Create faces with various detection scores
    face1 = create_mock_face(sample_embedding, 0.75)  # Valid
    face2 = create_mock_face(similar_embedding, 0.45)  # Invalid (below threshold)
    face3 = create_mock_face(different_embedding, 0.92)  # Valid
    face4 = create_mock_face(sample_embedding, 0.50)  # Valid (exactly at threshold)
    face5 = create_mock_face(similar_embedding, 0.49)  # Invalid (just below threshold)

    mock_app_instance.get.return_value = [face1, face2, face3, face4, face5]

    recognizer = FaceRecognizer()
    embeddings = recognizer.get_all_embeddings(bgr_frame)

    # Should return 3 embeddings (face1, face3, face4)
    assert len(embeddings) == 3
    assert np.array_equal(embeddings[0], sample_embedding)
    assert np.array_equal(embeddings[1], different_embedding)
    assert np.array_equal(embeddings[2], sample_embedding)


@pytest.mark.unit
def test_get_all_embeddings_no_valid_faces(
    reset_singleton, mock_insightface, bgr_frame, sample_embedding
):
    """Test that get_all_embeddings returns empty list when no faces meet threshold."""
    mock_if, mock_app_instance = mock_insightface

    # Create faces all below threshold
    face1 = create_mock_face(sample_embedding, 0.3)
    face2 = create_mock_face(sample_embedding, 0.45)
    mock_app_instance.get.return_value = [face1, face2]

    recognizer = FaceRecognizer()
    embeddings = recognizer.get_all_embeddings(bgr_frame)

    assert embeddings == []


@pytest.mark.unit
def test_get_all_embeddings_no_faces_detected(reset_singleton, mock_insightface, bgr_frame):
    """Test that get_all_embeddings returns empty list when no faces detected."""
    mock_if, mock_app_instance = mock_insightface
    mock_app_instance.get.return_value = []

    recognizer = FaceRecognizer()
    embeddings = recognizer.get_all_embeddings(bgr_frame)

    assert embeddings == []


# ============================================================================
# verify tests
# ============================================================================


@pytest.mark.unit
def test_verify_match_found(
    reset_singleton, mock_insightface, bgr_frame, sample_embedding, similar_embedding
):
    """Test verify returns (True, score) when a match is found."""
    mock_if, mock_app_instance = mock_insightface

    # Mock get() to return a face with similar embedding
    face = create_mock_face(similar_embedding, 0.92)
    mock_app_instance.get.return_value = [face]

    recognizer = FaceRecognizer()
    stored_embeddings = [sample_embedding]
    match, score = recognizer.verify(bgr_frame, stored_embeddings, threshold=0.45)

    assert match is True
    assert score > 0.5  # similar_embedding has ~0.64 similarity


@pytest.mark.unit
def test_verify_no_match_below_threshold(
    reset_singleton, mock_insightface, bgr_frame, sample_embedding, different_embedding
):
    """Test verify returns (False, score) when score is below threshold."""
    mock_if, mock_app_instance = mock_insightface

    # Mock get() to return a face with different embedding
    face = create_mock_face(different_embedding, 0.87)
    mock_app_instance.get.return_value = [face]

    recognizer = FaceRecognizer()
    stored_embeddings = [sample_embedding]
    match, score = recognizer.verify(bgr_frame, stored_embeddings, threshold=0.45)

    assert match is False
    assert 0.0 <= score < 0.45


@pytest.mark.unit
def test_verify_no_face_in_frame(reset_singleton, mock_insightface, bgr_frame, sample_embedding):
    """Test verify returns (False, 0.0) when no face is detected in frame."""
    mock_if, mock_app_instance = mock_insightface
    mock_app_instance.get.return_value = []

    recognizer = FaceRecognizer()
    stored_embeddings = [sample_embedding]
    match, score = recognizer.verify(bgr_frame, stored_embeddings, threshold=0.45)

    assert match is False
    assert score == 0.0


@pytest.mark.unit
def test_verify_multiple_stored_embeddings_returns_best_score(
    reset_singleton,
    mock_insightface,
    bgr_frame,
    sample_embedding,
    similar_embedding,
    different_embedding,
):
    """Test verify returns best score when comparing against multiple stored embeddings."""
    mock_if, mock_app_instance = mock_insightface

    # Create embedding that's very similar to similar_embedding
    rng = np.random.default_rng(999)
    noise = rng.standard_normal(512).astype(np.float32) * 0.01
    test_embedding = similar_embedding + noise
    test_embedding /= np.linalg.norm(test_embedding)

    face = create_mock_face(test_embedding, 0.89)
    mock_app_instance.get.return_value = [face]

    recognizer = FaceRecognizer()
    # Store three embeddings: one very different, one somewhat similar, one very similar
    stored_embeddings = [different_embedding, sample_embedding, similar_embedding]
    match, score = recognizer.verify(bgr_frame, stored_embeddings, threshold=0.45)

    assert match is True
    # Score should be highest against similar_embedding (>0.5)
    assert score > 0.5


@pytest.mark.unit
def test_verify_custom_threshold(
    reset_singleton, mock_insightface, bgr_frame, sample_embedding, similar_embedding
):
    """Test verify respects custom threshold parameter."""
    mock_if, mock_app_instance = mock_insightface

    face = create_mock_face(similar_embedding, 0.87)
    mock_app_instance.get.return_value = [face]

    recognizer = FaceRecognizer()
    stored_embeddings = [sample_embedding]

    # With high threshold, should not match
    match, score = recognizer.verify(bgr_frame, stored_embeddings, threshold=0.98)
    assert match is False
    assert score < 0.98

    # With low threshold, should match
    match2, score2 = recognizer.verify(bgr_frame, stored_embeddings, threshold=0.40)
    assert match2 is True
    assert score2 > 0.40
    assert score == score2  # Score should be the same


# ============================================================================
# Singleton behavior tests
# ============================================================================


@pytest.mark.unit
def test_ensure_loaded_only_calls_get_app_once(reset_singleton, mock_insightface, bgr_frame):
    """Test that _ensure_loaded only initializes the app once (singleton pattern)."""
    mock_if, mock_app_instance = mock_insightface
    mock_app_instance.get.return_value = []

    recognizer = FaceRecognizer()

    # Call multiple methods that trigger _ensure_loaded
    recognizer.get_faces(bgr_frame)
    recognizer.get_faces(bgr_frame)
    recognizer.get_embedding(bgr_frame)
    recognizer.get_all_embeddings(bgr_frame)

    # FaceAnalysis should only be instantiated once
    assert mock_if.app.FaceAnalysis.call_count == 1
    # prepare should only be called once
    assert mock_app_instance.prepare.call_count == 1


@pytest.mark.unit
def test_multiple_recognizer_instances_share_singleton(
    reset_singleton, mock_insightface, bgr_frame
):
    """Test that multiple FaceRecognizer instances share the same underlying app."""
    mock_if, mock_app_instance = mock_insightface
    mock_app_instance.get.return_value = []

    recognizer1 = FaceRecognizer()
    recognizer1.get_faces(bgr_frame)

    recognizer2 = FaceRecognizer()
    recognizer2.get_faces(bgr_frame)

    # Should only create one FaceAnalysis instance despite two FaceRecognizer instances
    assert mock_if.app.FaceAnalysis.call_count == 1
    assert mock_app_instance.prepare.call_count == 1


@pytest.mark.unit
def test_recognizer_with_different_params_uses_same_singleton(
    reset_singleton, mock_insightface, bgr_frame
):
    """Test that recognizer instances with different params still share singleton (first wins)."""
    mock_if, mock_app_instance = mock_insightface
    mock_app_instance.get.return_value = []

    # First recognizer with specific params
    recognizer1 = FaceRecognizer(model_name="buffalo_l", det_size=640)
    recognizer1.get_faces(bgr_frame)

    # Second recognizer with different params (should still use first instance)
    recognizer2 = FaceRecognizer(model_name="buffalo_s", det_size=320)
    recognizer2.get_faces(bgr_frame)

    # Should only create one FaceAnalysis with first params
    assert mock_if.app.FaceAnalysis.call_count == 1
    mock_if.app.FaceAnalysis.assert_called_once_with(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    mock_app_instance.prepare.assert_called_once_with(ctx_id=-1, det_size=(640, 640))
