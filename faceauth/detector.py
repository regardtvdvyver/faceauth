"""Face detection using MediaPipe BlazeFace."""

import logging
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """A detected face with bounding box and confidence."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    # Cropped and aligned face image (BGR, suitable for recognition)
    face_img: np.ndarray | None = None


class FaceDetector:
    """MediaPipe-based face detector."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 = full range model (better for varying distance)
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR frame. Returns list of FaceDetection."""
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return []

        faces = []
        for detection in results.detections:
            bbox_rel = detection.location_data.relative_bounding_box
            x = max(0, int(bbox_rel.xmin * w))
            y = max(0, int(bbox_rel.ymin * h))
            bw = min(int(bbox_rel.width * w), w - x)
            bh = min(int(bbox_rel.height * h), h - y)

            if bw < 20 or bh < 20:
                continue

            # Add margin for better recognition (20% each side)
            margin_x = int(bw * 0.2)
            margin_y = int(bh * 0.2)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + bw + margin_x)
            y2 = min(h, y + bh + margin_y)

            face_img = frame[y1:y2, x1:x2].copy()

            confidence = detection.score[0]
            faces.append(FaceDetection(
                bbox=(x, y, bw, bh),
                confidence=confidence,
                face_img=face_img,
            ))

        # Sort by confidence descending
        faces.sort(key=lambda f: f.confidence, reverse=True)
        log.debug("Detected %d face(s)", len(faces))
        return faces

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
