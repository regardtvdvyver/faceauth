"""Camera capture for RGB and IR cameras."""

import logging
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Camera:
    """OpenCV-based camera capture supporting RGB and IR devices."""

    def __init__(self, device: str, width: int = 640, height: int = 480):
        self.device = device
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the camera device."""
        # Parse /dev/videoN -> integer index, or use string path
        if self.device.startswith("/dev/video"):
            try:
                idx = int(self.device.replace("/dev/video", ""))
            except ValueError:
                idx = self.device
        else:
            idx = self.device

        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Warm up - discard first few frames
        for _ in range(5):
            self._cap.read()

        log.info("Opened camera %s (%dx%d)", self.device, self.width, self.height)

    def read(self) -> np.ndarray:
        """Read a single frame. Returns BGR or greyscale numpy array."""
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from {self.device}")
        return frame

    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Closed camera %s", self.device)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def capture_frames(
    device: str,
    count: int = 1,
    width: int = 640,
    height: int = 480,
    timeout: float = 10.0,
) -> list[np.ndarray]:
    """Capture N frames from a camera device with timeout.

    Returns list of BGR/greyscale numpy arrays.
    """
    frames = []
    deadline = time.monotonic() + timeout

    with Camera(device, width, height) as cam:
        while len(frames) < count:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Camera capture timed out after {timeout}s ({len(frames)}/{count} frames)"
                )
            try:
                frame = cam.read()
                frames.append(frame)
            except RuntimeError:
                time.sleep(0.05)

    return frames


def is_ir_frame(frame: np.ndarray) -> bool:
    """Heuristic: check if a frame is from an IR camera (greyscale)."""
    if len(frame.shape) == 2:
        return True
    if frame.shape[2] == 1:
        return True
    # IR cameras often return BGR where all channels are equal
    if frame.shape[2] == 3:
        b, g, r = cv2.split(frame)
        return np.allclose(b, g, atol=2) and np.allclose(g, r, atol=2)
    return False


def ir_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert IR greyscale frame to 3-channel BGR for models expecting colour input."""
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame
