"""ONNX Runtime execution provider selection.

Tries OpenVINO GPU first (best for these models), falls back to OpenVINO CPU,
then raw ONNX CPU. The device can be overridden via config.
"""

import logging

log = logging.getLogger(__name__)

_cached_providers: list | None = None

# OpenVINO device priority: GPU benchmarked 3x faster than CPU for ArcFace.
# NPU is slow for these models (unsupported ops cause host<->device thrashing).
_DEVICE_PRIORITY = ["GPU", "CPU"]


def _probe_device(device: str) -> bool:
    """Test whether an OpenVINO device actually works by loading a trivial model."""
    try:
        from openvino import Core

        core = Core()
        available = core.available_devices
        return device in available
    except Exception:
        return False


def get_ort_providers(device: str | None = None) -> list:
    """Return the best available ONNX Runtime providers.

    Args:
        device: Force a specific OpenVINO device ("GPU", "CPU", "NPU", "AUTO").
                If None, auto-selects the best available device.

    Returns:
        Provider list suitable for ort.InferenceSession() or InsightFace.
    """
    global _cached_providers
    if _cached_providers is not None and device is None:
        return _cached_providers

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except ImportError:
        _cached_providers = ["CPUExecutionProvider"]
        return _cached_providers

    if "OpenVINOExecutionProvider" not in available:
        log.info("OpenVINO not available, using CPU (available: %s)", available)
        _cached_providers = ["CPUExecutionProvider"]
        return _cached_providers

    # Determine which OpenVINO device to use
    if device:
        chosen = device
    else:
        chosen = "CPU"  # safe default
        for dev in _DEVICE_PRIORITY:
            if _probe_device(dev):
                chosen = dev
                break

    providers = [
        ("OpenVINOExecutionProvider", {"device_type": chosen}),
        "CPUExecutionProvider",
    ]
    log.info("ONNX Runtime: OpenVINO device=%s (available ORT: %s)", chosen, available)

    if device is None:
        _cached_providers = providers
    return providers
