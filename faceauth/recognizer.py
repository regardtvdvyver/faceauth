"""Face recognition using InsightFace (ArcFace/buffalo_l).

Uses InsightFace's full pipeline: RetinaFace detection + ArcFace embedding.
This ensures faces are properly aligned before embedding extraction.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Lazy-loaded singleton to avoid slow import at module level
_app = None


def _get_app(model_name: str = "buffalo_l", det_size: int = 640):
    """Get or create the InsightFace FaceAnalysis instance (singleton)."""
    global _app
    if _app is None:
        import insightface

        from .providers import get_ort_providers

        providers = get_ort_providers()
        log.info("Loading InsightFace model '%s' (providers=%s)...", model_name, providers)
        try:
            _app = insightface.app.FaceAnalysis(
                name=model_name,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
        except Exception as e:
            log.warning("InsightFace failed with %s, falling back to CPU: %s", providers, e)
            _app = insightface.app.FaceAnalysis(
                name=model_name,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
        _app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        log.info("InsightFace model loaded")
    return _app


class FaceRecognizer:
    """ArcFace-based face recognition via InsightFace."""

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640):
        self.model_name = model_name
        self.det_size = det_size
        self._app = None

    def _ensure_loaded(self):
        if self._app is None:
            self._app = _get_app(self.model_name, self.det_size)

    def ensure_loaded(self):
        """Ensure the underlying model is loaded. Public wrapper around _ensure_loaded."""
        self._ensure_loaded()

    def get_faces(self, frame: np.ndarray) -> list:
        """Detect faces and compute embeddings.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of insightface Face objects, each with .embedding (512-dim),
            .bbox, .det_score, .landmark, etc.
        """
        self._ensure_loaded()
        faces = self._app.get(frame)
        log.debug("Found %d face(s)", len(faces))
        return faces

    def get_embedding(self, frame: np.ndarray) -> np.ndarray | None:
        """Get the embedding for the largest/best face in frame.

        Returns 512-dim numpy array, or None if no face detected.
        """
        faces = self.get_faces(frame)
        if not faces:
            return None

        # Pick face with highest detection score
        best = max(faces, key=lambda f: f.det_score)
        if best.det_score < 0.5:
            log.debug("Best face score too low: %.2f", best.det_score)
            return None

        return best.embedding

    def get_all_embeddings(self, frame: np.ndarray) -> list[np.ndarray]:
        """Get embeddings for all detected faces."""
        faces = self.get_faces(frame)
        return [f.embedding for f in faces if f.det_score >= 0.5]

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Returns value in [-1, 1]. Higher = more similar.
        Typical thresholds: 0.3-0.5 for same person.
        """
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1_norm, emb2_norm))

    def verify(
        self,
        frame: np.ndarray,
        stored_embeddings: list[np.ndarray],
        threshold: float = 0.45,
    ) -> tuple[bool, float]:
        """Verify a face against stored embeddings.

        Args:
            frame: BGR image to verify.
            stored_embeddings: List of reference embeddings for the user.
            threshold: Cosine similarity threshold (higher = stricter).

        Returns:
            (match: bool, best_score: float)
        """
        embedding = self.get_embedding(frame)
        if embedding is None:
            return False, 0.0

        best_score = 0.0
        for stored in stored_embeddings:
            score = self.compute_similarity(embedding, stored)
            best_score = max(best_score, score)

        match = best_score >= threshold
        log.info("Verify: score=%.3f threshold=%.3f match=%s", best_score, threshold, match)
        return match, best_score
