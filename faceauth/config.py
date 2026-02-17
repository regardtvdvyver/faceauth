"""Configuration management for faceauth."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class CameraConfig:
    ir_device: str = "/dev/video2"
    rgb_device: str = "/dev/video0"
    capture_timeout: int = 10
    width: int = 640
    height: int = 480


@dataclass
class RecognitionConfig:
    model: str = "buffalo_l"
    similarity_threshold: float = 0.45
    max_attempts: int = 5
    openvino_device: str = ""  # OpenVINO device: "GPU", "CPU", "NPU", "AUTO", or "" for auto-select


@dataclass
class AntispoofConfig:
    enabled: bool = False
    ir_brightness_min: float = 15.0
    minifasnet_threshold: float = 0.8
    require_both: bool = True
    ir_only_fallback: bool = True  # Use IR-only if MiniFASNet model not downloaded


@dataclass
class DaemonConfig:
    socket_path: str = "/run/user/{uid}/faceauth.sock"
    system_socket_path: str = "/run/faceauth/faceauth.sock"
    system_mode: bool = False
    log_level: str = "info"
    log_file: str = "/tmp/faceauth.log"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    antispoof: AntispoofConfig = field(default_factory=AntispoofConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)

    @property
    def data_dir(self) -> Path:
        """Embedding storage directory. Uses XDG for dev, /var/lib for production."""
        if os.geteuid() == 0:
            return Path("/var/lib/faceauth")
        return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "faceauth"

    @property
    def config_dir(self) -> Path:
        if os.geteuid() == 0:
            return Path("/etc/faceauth")
        return (
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "faceauth"
        )


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file, falling back to defaults."""
    cfg = Config()

    if path is None:
        # Try user config, then system config
        candidates = [
            cfg.config_dir / "config.toml",
            Path("/etc/faceauth/config.toml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if path is not None and path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)

        if "camera" in data:
            for k, v in data["camera"].items():
                if hasattr(cfg.camera, k):
                    setattr(cfg.camera, k, v)

        if "recognition" in data:
            for k, v in data["recognition"].items():
                if hasattr(cfg.recognition, k):
                    setattr(cfg.recognition, k, v)

        if "antispoof" in data:
            for k, v in data["antispoof"].items():
                if hasattr(cfg.antispoof, k):
                    setattr(cfg.antispoof, k, v)

        if "daemon" in data:
            for k, v in data["daemon"].items():
                if hasattr(cfg.daemon, k):
                    setattr(cfg.daemon, k, v)

    return cfg
