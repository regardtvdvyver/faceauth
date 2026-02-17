"""Tests for faceauth.hardware module (V4L2 camera auto-detection)."""

import struct
from unittest.mock import MagicMock, patch

import pytest

from faceauth.hardware import (
    IR_FORMATS,
    RGB_FORMATS,
    CameraDevice,
    DetectionResult,
    PixelFormat,
    Resolution,
    _fourcc,
    _fourcc_str,
    detect_cameras,
)


# ============================================================================
# Fourcc Helpers
# ============================================================================


@pytest.mark.unit
def test_fourcc_grey():
    """GREY fourcc converts correctly."""
    result = _fourcc(b"GREY")
    assert result == struct.unpack("<I", b"GREY")[0]


@pytest.mark.unit
def test_fourcc_mjpg():
    """MJPG fourcc converts correctly."""
    result = _fourcc(b"MJPG")
    assert result == struct.unpack("<I", b"MJPG")[0]


@pytest.mark.unit
def test_fourcc_short_code_padded():
    """Short codes are right-padded with spaces."""
    result = _fourcc(b"Y8")
    assert result == struct.unpack("<I", b"Y8  ")[0]


@pytest.mark.unit
def test_fourcc_str_roundtrip():
    """Fourcc value converts back to readable string."""
    value = _fourcc(b"GREY")
    assert _fourcc_str(value) == "GREY"


@pytest.mark.unit
def test_fourcc_str_mjpg_roundtrip():
    value = _fourcc(b"MJPG")
    assert _fourcc_str(value) == "MJPG"


# ============================================================================
# Format Classification
# ============================================================================


@pytest.mark.unit
def test_ir_formats_contain_grey():
    assert _fourcc(b"GREY") in IR_FORMATS


@pytest.mark.unit
def test_ir_formats_contain_y16():
    assert _fourcc(b"Y16 ") in IR_FORMATS


@pytest.mark.unit
def test_rgb_formats_contain_mjpg():
    assert _fourcc(b"MJPG") in RGB_FORMATS


@pytest.mark.unit
def test_rgb_formats_contain_yuyv():
    assert _fourcc(b"YUYV") in RGB_FORMATS


@pytest.mark.unit
def test_ir_and_rgb_formats_disjoint():
    """IR and RGB format sets must not overlap."""
    assert IR_FORMATS.isdisjoint(RGB_FORMATS)


# ============================================================================
# Resolution
# ============================================================================


@pytest.mark.unit
def test_resolution_str():
    r = Resolution(640, 360)
    assert str(r) == "640x360"


# ============================================================================
# PixelFormat
# ============================================================================


@pytest.mark.unit
def test_pixel_format_is_ir():
    fmt = PixelFormat(fourcc=_fourcc(b"GREY"), name="GREY", description="8-bit Greyscale")
    assert fmt.is_ir is True
    assert fmt.is_rgb is False


@pytest.mark.unit
def test_pixel_format_is_rgb():
    fmt = PixelFormat(fourcc=_fourcc(b"MJPG"), name="MJPG", description="Motion-JPEG")
    assert fmt.is_ir is False
    assert fmt.is_rgb is True


@pytest.mark.unit
def test_pixel_format_unknown():
    """Unknown fourcc is neither IR nor RGB."""
    fmt = PixelFormat(fourcc=_fourcc(b"ZZZZ"), name="ZZZZ", description="Unknown")
    assert fmt.is_ir is False
    assert fmt.is_rgb is False


# ============================================================================
# CameraDevice
# ============================================================================


@pytest.fixture
def ir_device():
    return CameraDevice(
        path="/dev/video2",
        driver="uvcvideo",
        card="Integrated Camera: Integrated I",
        bus_info="usb-0000:00:14.0-4",
        formats=[
            PixelFormat(
                fourcc=_fourcc(b"GREY"),
                name="GREY",
                description="8-bit Greyscale",
                resolutions=[Resolution(640, 360)],
            )
        ],
    )


@pytest.fixture
def rgb_device():
    return CameraDevice(
        path="/dev/video0",
        driver="uvcvideo",
        card="Integrated Camera: Integrated C",
        bus_info="usb-0000:00:14.0-4",
        formats=[
            PixelFormat(
                fourcc=_fourcc(b"MJPG"),
                name="MJPG",
                description="Motion-JPEG",
                resolutions=[Resolution(1920, 1080), Resolution(640, 480)],
            ),
            PixelFormat(
                fourcc=_fourcc(b"YUYV"),
                name="YUYV",
                description="YUYV 4:2:2",
                resolutions=[Resolution(640, 480)],
            ),
        ],
    )


@pytest.mark.unit
def test_camera_device_is_ir(ir_device):
    assert ir_device.is_ir is True
    assert ir_device.is_rgb is False


@pytest.mark.unit
def test_camera_device_is_rgb(rgb_device):
    assert rgb_device.is_ir is False
    assert rgb_device.is_rgb is True


@pytest.mark.unit
def test_camera_device_best_resolution_ir(ir_device):
    res = ir_device.best_resolution
    assert res.width == 640
    assert res.height == 360


@pytest.mark.unit
def test_camera_device_best_resolution_rgb(rgb_device):
    """Best resolution picks highest pixel count."""
    res = rgb_device.best_resolution
    assert res.width == 1920
    assert res.height == 1080


@pytest.mark.unit
def test_camera_device_no_formats():
    """Device with no formats has no resolution."""
    dev = CameraDevice(path="/dev/video9", driver="x", card="x", bus_info="x", formats=[])
    assert dev.best_resolution is None
    assert dev.is_ir is False
    assert dev.is_rgb is False


@pytest.mark.unit
def test_camera_device_str_ir(ir_device):
    s = str(ir_device)
    assert "/dev/video2" in s
    assert "IR" in s
    assert "640x360" in s


@pytest.mark.unit
def test_camera_device_str_rgb(rgb_device):
    s = str(rgb_device)
    assert "/dev/video0" in s
    assert "RGB" in s


@pytest.mark.unit
def test_camera_mixed_formats_classified_as_rgb():
    """A device with both IR and RGB formats is classified as RGB (has color capability)."""
    dev = CameraDevice(
        path="/dev/video5",
        driver="x",
        card="x",
        bus_info="x",
        formats=[
            PixelFormat(fourcc=_fourcc(b"GREY"), name="GREY", description="", resolutions=[]),
            PixelFormat(fourcc=_fourcc(b"MJPG"), name="MJPG", description="", resolutions=[]),
        ],
    )
    # Has RGB formats, so is_ir is False (even though it also has GREY)
    assert dev.is_ir is False
    assert dev.is_rgb is True


# ============================================================================
# DetectionResult
# ============================================================================


@pytest.mark.unit
def test_detection_result_empty():
    result = DetectionResult()
    assert result.has_ir is False
    assert result.has_rgb is False
    assert result.best_ir is None
    assert result.best_rgb is None


@pytest.mark.unit
def test_detection_result_with_cameras(ir_device, rgb_device):
    result = DetectionResult(ir_cameras=[ir_device], rgb_cameras=[rgb_device])
    assert result.has_ir is True
    assert result.has_rgb is True
    assert result.best_ir is ir_device
    assert result.best_rgb is rgb_device


@pytest.mark.unit
def test_detection_result_best_picks_highest_resolution():
    """When multiple IR cameras, best_ir picks highest resolution."""
    low_res = CameraDevice(
        path="/dev/video2",
        driver="x",
        card="Low",
        bus_info="x",
        formats=[
            PixelFormat(
                fourcc=_fourcc(b"GREY"),
                name="GREY",
                description="",
                resolutions=[Resolution(320, 240)],
            )
        ],
    )
    high_res = CameraDevice(
        path="/dev/video4",
        driver="x",
        card="High",
        bus_info="x",
        formats=[
            PixelFormat(
                fourcc=_fourcc(b"GREY"),
                name="GREY",
                description="",
                resolutions=[Resolution(640, 480)],
            )
        ],
    )
    result = DetectionResult(ir_cameras=[low_res, high_res])
    assert result.best_ir is high_res


# ============================================================================
# detect_cameras (with mocked V4L2 ioctls)
# ============================================================================


def _make_querycap_response(driver: str, card: str, bus: str) -> bytearray:
    """Build a VIDIOC_QUERYCAP response buffer."""
    buf = bytearray(104)
    buf[0 : len(driver)] = driver.encode()
    buf[16 : 16 + len(card)] = card.encode()
    buf[48 : 48 + len(bus)] = bus.encode()
    # capabilities at offset 84, device_caps at 88
    struct.pack_into("<II", buf, 84, 0x04200001, 0x04200001)
    return buf


def _make_fmtdesc_response(index: int, description: str, fourcc: int) -> bytearray:
    """Build a VIDIOC_ENUM_FMT response buffer."""
    buf = bytearray(64)
    struct.pack_into("<II", buf, 0, index, 1)
    buf[12 : 12 + len(description)] = description.encode()
    struct.pack_into("<I", buf, 44, fourcc)
    return buf


def _make_framesize_response(index: int, pixfmt: int, width: int, height: int) -> bytearray:
    """Build a VIDIOC_ENUM_FRAMESIZES response (discrete type)."""
    buf = bytearray(44)
    struct.pack_into("<III", buf, 0, index, pixfmt, 1)  # type=1 (discrete)
    struct.pack_into("<II", buf, 12, width, height)
    return buf


@pytest.mark.unit
def test_detect_cameras_mocked():
    """Test full detection flow with mocked V4L2 ioctls."""
    grey_fourcc = _fourcc(b"GREY")
    mjpg_fourcc = _fourcc(b"MJPG")

    # Define ioctl responses per device
    # video0: RGB camera with MJPG 1920x1080
    # video2: IR camera with GREY 640x360
    ioctl_calls = {}

    def mock_ioctl(fd, request, buf):
        # We track which file descriptor maps to which device
        device = ioctl_calls.get(fd)
        if device is None:
            raise OSError("unknown fd")

        from faceauth.hardware import (
            _VIDIOC_ENUM_FMT,
            _VIDIOC_ENUM_FRAMESIZES,
            _VIDIOC_QUERYCAP,
        )

        if request == _VIDIOC_QUERYCAP:
            if device == "video0":
                resp = _make_querycap_response("uvcvideo", "Integrated C", "usb-0")
            else:
                resp = _make_querycap_response("uvcvideo", "Integrated I", "usb-0")
            buf[:] = resp

        elif request == _VIDIOC_ENUM_FMT:
            idx = struct.unpack_from("<I", buf, 0)[0]
            if device == "video0" and idx == 0:
                buf[:] = _make_fmtdesc_response(0, "Motion-JPEG", mjpg_fourcc)
            elif device == "video2" and idx == 0:
                buf[:] = _make_fmtdesc_response(0, "8-bit Greyscale", grey_fourcc)
            else:
                raise OSError("EINVAL")

        elif request == _VIDIOC_ENUM_FRAMESIZES:
            idx = struct.unpack_from("<I", buf, 0)[0]
            pixfmt = struct.unpack_from("<I", buf, 4)[0]
            if device == "video0" and idx == 0 and pixfmt == mjpg_fourcc:
                buf[:] = _make_framesize_response(0, mjpg_fourcc, 1920, 1080)
            elif device == "video2" and idx == 0 and pixfmt == grey_fourcc:
                buf[:] = _make_framesize_response(0, grey_fourcc, 640, 360)
            else:
                raise OSError("EINVAL")
        else:
            raise OSError("unknown ioctl")

    mock_fd_counter = [10]  # mutable counter for fake fds

    def mock_open_func(path, mode="r"):
        fd = mock_fd_counter[0]
        mock_fd_counter[0] += 1
        device_name = path.split("/")[-1]
        ioctl_calls[fd] = device_name
        mock_file = MagicMock()
        mock_file.fileno.return_value = fd
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        return mock_file

    with (
        patch("faceauth.hardware.glob.glob", return_value=["/dev/video0", "/dev/video2"]),
        patch("faceauth.hardware.fcntl.ioctl", side_effect=mock_ioctl),
        patch("builtins.open", side_effect=mock_open_func),
    ):
        result = detect_cameras()

    assert len(result.ir_cameras) == 1
    assert len(result.rgb_cameras) == 1
    assert result.ir_cameras[0].path == "/dev/video2"
    assert result.ir_cameras[0].card == "Integrated I"
    assert result.rgb_cameras[0].path == "/dev/video0"
    assert result.best_ir.best_resolution.width == 640
    assert result.best_rgb.best_resolution.width == 1920


@pytest.mark.unit
def test_detect_cameras_no_devices():
    """No video devices returns empty result."""
    with patch("faceauth.hardware.glob.glob", return_value=[]):
        result = detect_cameras()
    assert result.has_ir is False
    assert result.has_rgb is False


@pytest.mark.unit
def test_detect_cameras_permission_denied():
    """Inaccessible devices are skipped gracefully."""
    with (
        patch("faceauth.hardware.glob.glob", return_value=["/dev/video0"]),
        patch("builtins.open", side_effect=PermissionError("access denied")),
    ):
        result = detect_cameras()
    assert len(result.ir_cameras) == 0
    assert len(result.rgb_cameras) == 0
