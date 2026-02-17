"""Auto-detection of camera hardware using V4L2 ioctls.

Scans /dev/video* devices to identify IR and RGB cameras without
requiring v4l2-ctl or any external dependencies. Uses pure Python
fcntl to query V4L2 device capabilities.
"""

from __future__ import annotations

import fcntl
import glob
import logging
import struct
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# V4L2 ioctl numbers (Linux x86_64)
_VIDIOC_QUERYCAP = 0x80685600  # _IOR('V', 0, 104)
_VIDIOC_ENUM_FMT = 0xC0405602  # _IOWR('V', 2, 64)
_VIDIOC_ENUM_FRAMESIZES = 0xC02C564A  # _IOWR('V', 74, 44)

# V4L2 constants
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_V4L2_FRMSIZE_TYPE_DISCRETE = 1
_V4L2_FRMSIZE_TYPE_STEPWISE = 3


def _fourcc(code: bytes) -> int:
    """Convert a 4-byte fourcc code to a uint32."""
    return struct.unpack("<I", code.ljust(4, b" ")[:4])[0]


def _fourcc_str(value: int) -> str:
    """Convert a uint32 fourcc value back to a readable string."""
    return struct.pack("<I", value).decode("ascii", errors="replace").rstrip()


# Pixel format sets (must be after _fourcc definition)
IR_FORMATS = frozenset(
    {
        _fourcc(b"GREY"),  # 8-bit greyscale
        _fourcc(b"Y10 "),  # 10-bit greyscale
        _fourcc(b"Y12 "),  # 12-bit greyscale
        _fourcc(b"Y16 "),  # 16-bit greyscale
        _fourcc(b"Y8  "),  # 8-bit greyscale (alias)
        _fourcc(b"Y10B"),  # 10-bit greyscale packed
    }
)

RGB_FORMATS = frozenset(
    {
        _fourcc(b"YUYV"),
        _fourcc(b"MJPG"),
        _fourcc(b"NV12"),
        _fourcc(b"YU12"),
        _fourcc(b"RGB3"),
        _fourcc(b"BGR3"),
        _fourcc(b"H264"),
    }
)


@dataclass
class Resolution:
    """A supported camera resolution."""

    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class PixelFormat:
    """A supported pixel format with its resolutions."""

    fourcc: int
    name: str
    description: str
    resolutions: list[Resolution] = field(default_factory=list)

    @property
    def is_ir(self) -> bool:
        return self.fourcc in IR_FORMATS

    @property
    def is_rgb(self) -> bool:
        return self.fourcc in RGB_FORMATS


@dataclass
class CameraDevice:
    """A detected camera device."""

    path: str
    driver: str
    card: str
    bus_info: str
    formats: list[PixelFormat] = field(default_factory=list)

    @property
    def is_ir(self) -> bool:
        """Device has at least one IR format and no RGB formats."""
        return any(f.is_ir for f in self.formats) and not any(f.is_rgb for f in self.formats)

    @property
    def is_rgb(self) -> bool:
        """Device has at least one RGB/color format."""
        return any(f.is_rgb for f in self.formats)

    @property
    def best_resolution(self) -> Resolution | None:
        """Highest resolution available across all formats."""
        all_res = [r for f in self.formats for r in f.resolutions]
        if not all_res:
            return None
        return max(all_res, key=lambda r: r.width * r.height)

    def __str__(self) -> str:
        kind = "IR" if self.is_ir else "RGB" if self.is_rgb else "unknown"
        res = self.best_resolution
        res_str = f" {res}" if res else ""
        return f"{self.path} [{kind}] {self.card}{res_str}"


@dataclass
class DetectionResult:
    """Result of scanning for cameras."""

    ir_cameras: list[CameraDevice] = field(default_factory=list)
    rgb_cameras: list[CameraDevice] = field(default_factory=list)
    other_devices: list[CameraDevice] = field(default_factory=list)

    @property
    def has_ir(self) -> bool:
        return len(self.ir_cameras) > 0

    @property
    def has_rgb(self) -> bool:
        return len(self.rgb_cameras) > 0

    @property
    def best_ir(self) -> CameraDevice | None:
        return _best_by_resolution(self.ir_cameras)

    @property
    def best_rgb(self) -> CameraDevice | None:
        return _best_by_resolution(self.rgb_cameras)


def _best_by_resolution(cameras: list[CameraDevice]) -> CameraDevice | None:
    """Return the camera with the highest resolution."""
    if not cameras:
        return None
    return max(
        cameras,
        key=lambda c: (
            (c.best_resolution.width * c.best_resolution.height) if c.best_resolution else 0
        ),
    )


def _query_capabilities(fd: int) -> tuple[str, str, str] | None:
    """Query device capabilities via VIDIOC_QUERYCAP."""
    buf = bytearray(104)
    try:
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
    except OSError:
        return None
    driver = buf[0:16].split(b"\x00")[0].decode("ascii", errors="replace")
    card = buf[16:48].split(b"\x00")[0].decode("ascii", errors="replace")
    bus_info = buf[48:80].split(b"\x00")[0].decode("ascii", errors="replace")
    return driver, card, bus_info


def _enumerate_formats(fd: int) -> list[PixelFormat]:
    """Enumerate supported pixel formats via VIDIOC_ENUM_FMT."""
    formats = []
    idx = 0
    while True:
        buf = bytearray(64)
        struct.pack_into("<II", buf, 0, idx, _V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, buf)
        except OSError:
            break
        description = buf[12:44].split(b"\x00")[0].decode("ascii", errors="replace")
        pixfmt = struct.unpack_from("<I", buf, 44)[0]
        resolutions = _enumerate_frame_sizes(fd, pixfmt)
        formats.append(
            PixelFormat(
                fourcc=pixfmt,
                name=_fourcc_str(pixfmt),
                description=description,
                resolutions=resolutions,
            )
        )
        idx += 1
    return formats


def _enumerate_frame_sizes(fd: int, pixfmt: int) -> list[Resolution]:
    """Enumerate supported frame sizes for a pixel format."""
    resolutions = []
    idx = 0
    while True:
        buf = bytearray(44)
        struct.pack_into("<II", buf, 0, idx, pixfmt)
        try:
            fcntl.ioctl(fd, _VIDIOC_ENUM_FRAMESIZES, buf)
        except OSError:
            break
        ftype = struct.unpack_from("<I", buf, 8)[0]
        if ftype == _V4L2_FRMSIZE_TYPE_DISCRETE:
            w, h = struct.unpack_from("<II", buf, 12)
            resolutions.append(Resolution(width=w, height=h))
        elif ftype == _V4L2_FRMSIZE_TYPE_STEPWISE:
            _, max_w, _, _, max_h, _ = struct.unpack_from("<6I", buf, 12)
            resolutions.append(Resolution(width=max_w, height=max_h))
        idx += 1
    return resolutions


def _probe_device(path: str) -> CameraDevice | None:
    """Probe a single video device for capabilities and formats."""
    try:
        with open(path, "rb") as f:
            caps = _query_capabilities(f.fileno())
            if caps is None:
                return None
            driver, card, bus_info = caps
            formats = _enumerate_formats(f.fileno())
            if not formats:
                return None
            return CameraDevice(
                path=path, driver=driver, card=card, bus_info=bus_info, formats=formats
            )
    except (OSError, PermissionError) as e:
        log.debug("Cannot probe %s: %s", path, e)
        return None


def detect_cameras() -> DetectionResult:
    """Scan /dev/video* devices and classify as IR, RGB, or other.

    Returns a DetectionResult with categorized cameras. Devices that
    have no video capture formats (e.g. metadata nodes) are skipped.
    """
    result = DetectionResult()
    devices = sorted(glob.glob("/dev/video*"))

    for path in devices:
        device = _probe_device(path)
        if device is None:
            continue
        if device.is_ir:
            result.ir_cameras.append(device)
            log.info("Found IR camera: %s", device)
        elif device.is_rgb:
            result.rgb_cameras.append(device)
            log.info("Found RGB camera: %s", device)
        else:
            result.other_devices.append(device)
            log.debug("Found device with unknown formats: %s", device)

    return result
