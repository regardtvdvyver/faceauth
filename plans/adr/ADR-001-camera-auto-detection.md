# ADR-001: Camera Auto-Detection

**Status:** Accepted
**Date:** 2026-02-17

## Context

The current setup process requires users to manually identify their IR camera device path and supported resolution using v4l2-ctl. This involves:

1. Running `v4l2-ctl --list-devices` to find candidate video devices
2. Running `v4l2-ctl -d /dev/videoN --list-formats-ext` for each device
3. Identifying which device exposes greyscale formats (GREY, Y8, Y10, Y16)
4. Finding a supported resolution (typically 640x360 or 640x480)
5. Manually editing config.toml with the correct values

This is the single biggest friction point in the installation process. Users report confusion about which camera is which (most laptops have multiple video devices for RGB camera, IR camera, and sometimes virtual devices). A typical ThinkPad P14s Gen 5 exposes 4-6 /dev/video* devices.

Additionally, v4l2-ctl may not be installed on all systems (it's part of v4l-utils package), and subprocess-based detection is fragile and harder to error-handle than native system calls.

## Decision

Create `faceauth/hardware.py` module that implements native V4L2 device scanning using Python's fcntl and ctypes to make ioctl calls directly to video devices. This module will:

1. **Enumerate devices**: Scan /dev/video* devices (0-63 range is reasonable)
2. **Query capabilities**: Use VIDIOC_QUERYCAP to get device name and capabilities
3. **Detect IR cameras**: Use VIDIOC_ENUM_FMT to list supported pixel formats, filter for IR-indicative formats:
   - V4L2_PIX_FMT_GREY (8-bit greyscale)
   - V4L2_PIX_FMT_Y10 (10-bit greyscale)
   - V4L2_PIX_FMT_Y12 (12-bit greyscale)
   - V4L2_PIX_FMT_Y16 (16-bit greyscale)
4. **Query resolutions**: Use VIDIOC_ENUM_FRAMESIZES to get supported frame sizes for each IR-capable format
5. **Return structured data**: List of detected cameras with device path, name, formats, and resolutions

The module will provide both programmatic API (for setup wizard) and CLI interface (`faceauth detect-cameras` command for troubleshooting).

Implementation approach:
```python
# Key V4L2 constants and structures
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xC0405602
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A

class v4l2_capability(ctypes.Structure):
    _fields_ = [
        ('driver', ctypes.c_char * 16),
        ('card', ctypes.c_char * 32),
        # ... full structure
    ]
```

Error handling will be robust: skip devices that fail to open (permission issues), skip non-camera devices, handle devices that disappear during enumeration.

## Consequences

**Positive:**
- Eliminates manual camera detection for 90% of users
- Setup wizard can auto-select IR camera with high confidence
- More reliable than subprocess-based v4l2-ctl parsing
- No external dependencies (v4l-utils not required)
- Same code works across all Linux distributions
- Provides troubleshooting tool for edge cases

**Negative:**
- Adds complexity with low-level V4L2 ioctl handling
- ctypes code is more verbose and requires careful structure definitions
- Must maintain V4L2 constant definitions (though these are stable)
- Edge cases: some IR cameras might use uncommon pixel formats we don't detect

**Risks:**
- V4L2 API changes (very low risk, stable since 2.6.x kernel era)
- Camera manufacturers using non-standard format codes (mitigated by allowing manual override)

**Mitigations:**
- Keep manual config option as fallback
- Extensive testing on multiple laptop models (ThinkPad P14s Gen 5, Dell XPS, HP Elite)
- Clear error messages when detection fails
- Log detected cameras at DEBUG level for troubleshooting

## Alternatives Considered

### 1. Continue using v4l2-ctl subprocess
**Rejected.** Fragile (depends on output format), requires external package, harder to error-handle, subprocess overhead. The only advantage is simplicity, but the current manual process already shows this isn't working.

### 2. Use udev rules to tag IR cameras
**Rejected for primary solution.** Udev rules would require per-device configuration or complex heuristics. However, we could provide optional udev rules as a supplement for users who want to assign stable device names.

### 3. Python v4l2 binding libraries (python-v4l2, v4l2py)
**Rejected.** Adds external dependency. Our needs are simple enough that direct fcntl/ctypes is justified. We only need 3-4 ioctls total.

### 4. Manual configuration only
**Rejected.** Current state is unacceptable for user experience. Power users can still manually override if needed.

### 5. Attempt to open each camera and analyze first frame
**Rejected.** Too slow (requires initializing camera), invasive (turns on camera LED), unreliable (requires inference about image content). Format-based detection is instant and deterministic.

## Implementation Notes

- Place V4L2 constants in a separate module (faceauth/v4l2_defs.py) for clarity
- Support both blocking and non-blocking detection (timeout option)
- Cache detection results for 60 seconds to avoid repeated hardware queries
- Add `--force-detect` flag to bypass cache
- Include camera index in returned data (useful for logging: "Using IR camera at /dev/video2")

## Future Enhancements

- Detect RGB cameras similarly (could enable RGB fallback mode)
- Query camera capabilities (autofocus, exposure control) for future optimization features
- Monitor for camera hot-plug events (udev integration)
- Provide camera quality scoring (prefer higher resolution, prefer known-good models)
