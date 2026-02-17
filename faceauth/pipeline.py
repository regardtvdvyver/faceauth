"""Shared face-processing pipeline functions.

Eliminates duplication between CLI and daemon by extracting the common
frame-processing, face-selection, anti-spoof, and embedding-comparison logic.
"""

import numpy as np

from .antispoof import AntispoofChecker, AntispoofResult
from .camera import ir_to_rgb, is_ir_frame
from .recognizer import FaceRecognizer

# ---------------------------------------------------------------------------
# Named constants replacing magic numbers scattered across CLI and daemon
# ---------------------------------------------------------------------------
MIN_DET_SCORE = 0.5
"""Minimum face detection confidence to consider a face valid."""

MIN_SELF_CONSISTENCY = 0.3
"""Minimum pairwise cosine similarity during enrollment consistency check."""

ATTEMPT_MULTIPLIER = 10
"""max_attempts = samples * ATTEMPT_MULTIPLIER during enrollment."""

FRAME_DELAY_VERIFY = 0.15
"""Seconds to sleep between verify attempts."""

FRAME_DELAY_ENROLL = 0.3
"""Seconds to sleep between successful enrollment captures."""


def make_antispoof(cfg) -> AntispoofChecker | None:
    """Create an AntispoofChecker from config, or None if disabled.

    Args:
        cfg: A Config object with antispoof and antispoof_model_path attributes.
    """
    if not cfg.antispoof.enabled:
        return None
    asc = cfg.antispoof
    return AntispoofChecker(
        ir_brightness_min=asc.ir_brightness_min,
        minifasnet_threshold=asc.minifasnet_threshold,
        require_both=asc.require_both,
        ir_only_fallback=asc.ir_only_fallback,
        minifasnet_model_path=cfg.antispoof_model_path,
    )


def process_frame(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Detect IR and convert frame for model consumption.

    Args:
        frame: Raw camera frame (greyscale or BGR).

    Returns:
        (raw_ir, bgr_frame) where raw_ir is the original frame if IR, else None,
        and bgr_frame is the BGR-converted frame suitable for face models.
    """
    raw_ir = frame.copy() if is_ir_frame(frame) else None
    bgr_frame = ir_to_rgb(frame) if raw_ir is not None else frame
    return raw_ir, bgr_frame


def select_best_face(faces: list) -> object | None:
    """Select the face with the highest detection score above threshold.

    Args:
        faces: List of InsightFace Face objects with .det_score attribute.

    Returns:
        The best face, or None if no face meets the minimum score.
    """
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score)
    if best.det_score < MIN_DET_SCORE:
        return None
    return best


def check_antispoof(
    antispoof: AntispoofChecker | None,
    raw_ir: np.ndarray | None,
    bgr_frame: np.ndarray,
    bbox: np.ndarray,
) -> AntispoofResult | None:
    """Run anti-spoof check if both checker and IR frame are available.

    Args:
        antispoof: AntispoofChecker instance, or None if disabled.
        raw_ir: Raw IR frame, or None if not an IR camera.
        bgr_frame: BGR frame for liveness model.
        bbox: Face bounding box [x1, y1, x2, y2].

    Returns:
        AntispoofResult if a check was performed, None if skipped.
    """
    if antispoof is None or raw_ir is None:
        return None
    return antispoof.check(raw_ir, bgr_frame, bbox)


def match_embedding(
    embedding: np.ndarray, stored: list[np.ndarray]
) -> float:
    """Compare a live embedding against stored embeddings.

    Args:
        embedding: The live face embedding.
        stored: List of enrolled reference embeddings.

    Returns:
        The highest cosine similarity score found.
    """
    best_score = 0.0
    for s in stored:
        score = FaceRecognizer.compute_similarity(embedding, s)
        best_score = max(best_score, score)
    return best_score


def compute_self_consistency(embeddings: list[np.ndarray]) -> tuple[float, float]:
    """Compute pairwise consistency of enrollment embeddings.

    Uses the first embedding as the reference and compares all others to it.

    Args:
        embeddings: List of face embeddings (must have at least 2).

    Returns:
        (average_score, minimum_score) tuple. Returns (0.0, 0.0) if fewer
        than 2 embeddings are provided.
    """
    if len(embeddings) < 2:
        return 0.0, 0.0
    base = embeddings[0]
    scores = [FaceRecognizer.compute_similarity(base, e) for e in embeddings[1:]]
    return sum(scores) / len(scores), min(scores)
