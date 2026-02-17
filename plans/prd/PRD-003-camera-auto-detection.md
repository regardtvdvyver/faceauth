# PRD-003: Camera Auto-Detection
**Priority:** P0
**Effort:** M
**Dependencies:** None

## Problem
Users must manually discover IR camera device path (`/dev/videoN`) and resolution. Current process:
1. Run `v4l2-ctl --list-devices` (requires v4l2-utils installed)
2. Test each device with `v4l2-ctl -d /dev/videoN --list-formats-ext`
3. Look for GREY/Y8/Y16 formats (IR indicators)
4. Trial and error with different resolutions
5. Manually edit config.toml

This is error-prone and a barrier to adoption. Setup wizard (PRD-001) cannot be implemented without automatic camera detection.

## Requirements

### Must Have
- Detect IR cameras by scanning `/dev/video*` for greyscale pixel formats:
  - GREY (8-bit greyscale)
  - Y8, Y10, Y12, Y16 (various bit depths)
- Return list of IR cameras with:
  - Device path (`/dev/video2`)
  - Supported resolutions (width x height)
  - Pixel format
- Work without `v4l2-ctl` or any external tools installed
- Detect RGB cameras (for fallback) by scanning for:
  - YUYV, MJPG, RGB3, BGR3, NV12
- Handle permissions errors gracefully (camera access denied)

### Should Have
- Return device name/model from V4L2 capabilities (e.g., "Integrated IR Camera")
- Return bus info (e.g., "usb-0000:00:14.0-6")
- Prioritize IR over RGB in results
- Cache detection results (avoid rescanning on every call)
- Validate camera is accessible (can open device)

### Nice to Have
- Test frame capture from detected cameras
- Return camera capabilities (has_flash, has_focus, etc.)
- Detect if camera is already in use by another process
- Estimate frame rate capabilities
- Detect USB vs built-in cameras

## Technical Approach

### Implementation

New module: `faceauth/hardware.py`

```python
from dataclasses import dataclass
from typing import List, Optional
import os
import ctypes
from pathlib import Path

@dataclass
class CameraInfo:
    device_path: str          # /dev/video2
    name: str                 # "Integrated IR Camera"
    resolutions: List[tuple]  # [(640, 360), (640, 480)]
    pixel_format: str         # "GREY", "YUYV"
    is_infrared: bool         # True/False
    bus_info: str            # "usb-0000:00:14.0-6"

def detect_cameras() -> List[CameraInfo]:
    """Scan /dev/video* and return all cameras with metadata."""
    pass

def find_ir_camera() -> Optional[CameraInfo]:
    """Return best IR camera (highest resolution)."""
    pass

def find_rgb_camera() -> Optional[CameraInfo]:
    """Return best RGB camera (fallback)."""
    pass
```

### V4L2 Implementation
Use ctypes for V4L2 ioctls (no subprocess calls):

```python
import fcntl

# V4L2 constants
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xc0405602
VIDIOC_ENUM_FRAMESIZES = 0xc02c564a

# Structures (ctypes)
class v4l2_capability(ctypes.Structure):
    _fields_ = [
        ('driver', ctypes.c_char * 16),
        ('card', ctypes.c_char * 32),
        ('bus_info', ctypes.c_char * 32),
        # ... more fields
    ]

# Detection logic
1. Scan /dev/video* (glob /dev/video[0-9]*)
2. For each device:
   - Open device with os.open()
   - IOCTL: VIDIOC_QUERYCAP -> get name, bus_info
   - IOCTL: VIDIOC_ENUM_FMT -> enumerate pixel formats
   - For each format:
     - IOCTL: VIDIOC_ENUM_FRAMESIZES -> get resolutions
   - Classify as IR (GREY/Y*) or RGB (YUYV/MJPG/RGB*)
   - Close device
3. Return sorted list (IR first, then by resolution desc)
```

### IR Format Detection
Pixel formats indicating IR camera:
- `GREY` - 8-bit greyscale (most common, ThinkPad)
- `Y8` - 8-bit luminance
- `Y10` - 10-bit greyscale
- `Y12` - 12-bit greyscale
- `Y16` - 16-bit greyscale

### Error Handling
- Permission denied: Return empty list with warning (not error)
- Device busy: Mark camera as "in use", still return info
- Invalid device: Skip and continue scanning
- No cameras found: Return empty list (caller handles)

### Testing
- Unit tests with mock V4L2 ioctls
- Integration test on real hardware (ThinkPad P14s Gen 5)
- Test with USB camera connected/disconnected
- Test with no cameras present
- Test with permission denied

## Success Criteria
- `faceauth.hardware.detect_cameras()` correctly identifies IR camera on ThinkPad P14s Gen 5 as `/dev/video2` with 640x360 resolution
- Works without v4l2-utils installed
- Runs in <500ms for typical system (4 cameras)
- No false positives (RGB cameras marked as IR)
- Clean error messages when camera access denied
- Setup wizard (PRD-001) uses this for zero-config camera detection

## Out of Scope
- Camera configuration/control (brightness, exposure, etc.)
- USB hotplug detection with udev
- Camera enumeration on non-Linux platforms
- Testing/validation of camera image quality
- Automatic selection of "best" camera (caller decides)
- Persistent caching across reboots
- Support for non-V4L2 cameras (network cameras, etc.)
