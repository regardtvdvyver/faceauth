"""Anti-spoofing checks for face authentication.

Two independent layers:
1. IR Brightness - real faces reflect IR light and appear bright; photos/screens appear dark.
2. MiniFASNet Liveness - ONNX model that detects print/screen texture artifacts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _default_model_path() -> Path:
    """Compute model path at runtime (not import time) based on effective UID."""
    import os

    if os.geteuid() == 0:
        return Path("/var/lib/faceauth/models/antispoof_minifasnet.onnx")
    xdg = os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")
    return Path(xdg) / "faceauth/models/antispoof_minifasnet.onnx"


DEFAULT_MODEL_PATH = None  # Computed at runtime; use _default_model_path()


@dataclass
class AntispoofResult:
    """Result of anti-spoofing checks."""

    passed: bool
    ir_brightness: float = 0.0
    ir_passed: bool = False
    liveness_score: float = 0.0
    liveness_passed: bool = False
    reason: str = ""


class IRBrightnessChecker:
    """Check that the face region in the raw IR frame is sufficiently bright.

    Real faces reflect IR light from the IR emitter and appear bright.
    Photos and phone screens absorb/don't reflect IR, appearing dark.
    """

    def __init__(self, brightness_min: float = 15.0):
        self.brightness_min = brightness_min

    def check(self, ir_frame: np.ndarray, bbox: np.ndarray) -> tuple[bool, float]:
        """Check IR brightness of the face bounding box region.

        Args:
            ir_frame: Raw IR frame (greyscale or BGR with equal channels).
            bbox: InsightFace bbox array [x1, y1, x2, y2].

        Returns:
            (passed, brightness) tuple.
        """
        h, w = ir_frame.shape[:2]

        # Extract face region, clamped to frame bounds
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            log.warning("IR check: invalid bbox %s", bbox)
            return False, 0.0

        face_region = ir_frame[y1:y2, x1:x2]

        # Convert to single channel if needed
        if len(face_region.shape) == 3:
            face_region = face_region[:, :, 0]

        brightness = float(np.mean(face_region))
        passed = brightness >= self.brightness_min
        log.debug(
            "IR brightness: %.1f (min=%.1f, passed=%s)", brightness, self.brightness_min, passed
        )
        return passed, brightness


class MiniFASNetChecker:
    """MiniFASNet liveness detection via ONNX.

    Analyses face texture to detect print/screen artifacts. Model is lazy-loaded
    on first use. If the model file doesn't exist, check() returns None to signal
    unavailability (caller handles fallback).
    """

    def __init__(self, model_path: Path | None = None, threshold: float = 0.8):
        self.model_path = model_path or _default_model_path()
        self.threshold = threshold
        self._session = None
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self.model_path.exists()
            if not self._available:
                log.info("MiniFASNet model not found at %s (IR-only mode)", self.model_path)
        return self._available

    def _ensure_loaded(self):
        if self._session is not None:
            return
        if not self.available:
            return

        import onnxruntime as ort

        from .providers import get_ort_providers

        providers = get_ort_providers()
        log.info("Loading MiniFASNet model from %s (providers=%s)", self.model_path, providers)
        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )
        except Exception as e:
            # Some models don't work on GPU/NPU — fall back to CPU
            log.warning("MiniFASNet failed with %s, falling back to CPU: %s", providers, e)
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        actual = self._session.get_providers()
        log.info("MiniFASNet model loaded (active providers: %s)", actual)

    def check(self, bgr_frame: np.ndarray, bbox: np.ndarray) -> tuple[bool, float] | None:
        """Run liveness check on a face crop.

        Args:
            bgr_frame: BGR image (the converted frame InsightFace processes).
            bbox: InsightFace bbox array [x1, y1, x2, y2].

        Returns:
            (passed, score) tuple, or None if model not available.
        """
        if not self.available:
            return None

        self._ensure_loaded()
        if self._session is None:
            return None

        h, w = bgr_frame.shape[:2]

        # Extract and resize face crop to 128x128
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            log.warning("MiniFASNet: invalid bbox %s", bbox)
            return False, 0.0

        import cv2

        face_crop = bgr_frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (128, 128))

        # Preprocess: HWC -> CHW, float32, normalise to [0, 1]
        blob = face_resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # CHW
        blob = np.expand_dims(blob, axis=0)  # NCHW

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})
        logits = outputs[0][0]  # shape: (2,) - [fake, real]

        # Softmax to get probability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        liveness_score = float(probs[1])  # probability of "real"

        passed = liveness_score >= self.threshold
        log.debug(
            "MiniFASNet: score=%.3f (threshold=%.2f, passed=%s)",
            liveness_score,
            self.threshold,
            passed,
        )
        return passed, liveness_score


class AntispoofChecker:
    """Coordinator that runs available anti-spoof checks and applies policy.

    If require_both is True and both checks are available, both must pass.
    If MiniFASNet is unavailable and ir_only_fallback is True, only IR is used.
    """

    def __init__(
        self,
        ir_brightness_min: float = 15.0,
        minifasnet_threshold: float = 0.8,
        minifasnet_model_path: Path | None = None,
        require_both: bool = True,
        ir_only_fallback: bool = True,
    ):
        self.ir_checker = IRBrightnessChecker(brightness_min=ir_brightness_min)
        self.fas_checker = MiniFASNetChecker(
            model_path=minifasnet_model_path, threshold=minifasnet_threshold
        )
        self.require_both = require_both
        self.ir_only_fallback = ir_only_fallback

    @property
    def minifasnet_available(self) -> bool:
        return self.fas_checker.available

    @property
    def model_path(self) -> Path:
        """Public access to the MiniFASNet model path."""
        return self.fas_checker.model_path

    def check(
        self, ir_frame: np.ndarray, bgr_frame: np.ndarray, bbox: np.ndarray
    ) -> AntispoofResult:
        """Run all available anti-spoof checks.

        Args:
            ir_frame: Raw IR frame (before conversion).
            bgr_frame: BGR frame (after ir_to_rgb conversion).
            bbox: InsightFace bbox [x1, y1, x2, y2].

        Returns:
            AntispoofResult with individual and combined results.
        """
        result = AntispoofResult(passed=False)

        # 1. IR brightness check (always available)
        result.ir_passed, result.ir_brightness = self.ir_checker.check(ir_frame, bbox)

        # 2. MiniFASNet liveness check (optional)
        fas_result = self.fas_checker.check(bgr_frame, bbox)
        if fas_result is not None:
            result.liveness_passed, result.liveness_score = fas_result
        else:
            # Model not available
            if not self.ir_only_fallback:
                result.reason = "MiniFASNet model not available and ir_only_fallback=False"
                return result

        # Apply policy
        if fas_result is not None:
            # Both checks available
            if self.require_both:
                result.passed = result.ir_passed and result.liveness_passed
            else:
                result.passed = result.ir_passed or result.liveness_passed
        else:
            # IR-only fallback
            result.passed = result.ir_passed

        # Build reason string for failures
        if not result.passed:
            reasons = []
            if not result.ir_passed:
                reasons.append(
                    f"IR brightness {result.ir_brightness:.1f} < {self.ir_checker.brightness_min}"
                )
            if fas_result is not None and not result.liveness_passed:
                reasons.append(
                    f"liveness score {result.liveness_score:.3f} < {self.fas_checker.threshold}"
                )
            result.reason = "; ".join(reasons)

        return result
