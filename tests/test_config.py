"""Comprehensive tests for faceauth configuration management."""

import os
from pathlib import Path

import pytest

from faceauth.config import (
    AntispoofConfig,
    CameraConfig,
    Config,
    DaemonConfig,
    RecognitionConfig,
    load_config,
)


@pytest.mark.unit
class TestCameraConfigDefaults:
    """Test default values for CameraConfig."""

    def test_default_ir_device(self):
        cfg = CameraConfig()
        assert cfg.ir_device == "/dev/video2"

    def test_default_rgb_device(self):
        cfg = CameraConfig()
        assert cfg.rgb_device == "/dev/video0"

    def test_default_capture_timeout(self):
        cfg = CameraConfig()
        assert cfg.capture_timeout == 10

    def test_default_width(self):
        cfg = CameraConfig()
        assert cfg.width == 640

    def test_default_height(self):
        cfg = CameraConfig()
        assert cfg.height == 480


@pytest.mark.unit
class TestRecognitionConfigDefaults:
    """Test default values for RecognitionConfig."""

    def test_default_model(self):
        cfg = RecognitionConfig()
        assert cfg.model == "buffalo_l"

    def test_default_similarity_threshold(self):
        cfg = RecognitionConfig()
        assert cfg.similarity_threshold == 0.45

    def test_default_max_attempts(self):
        cfg = RecognitionConfig()
        assert cfg.max_attempts == 5


@pytest.mark.unit
class TestAntispoofConfigDefaults:
    """Test default values for AntispoofConfig."""

    def test_default_enabled(self):
        cfg = AntispoofConfig()
        assert cfg.enabled is False

    def test_default_ir_brightness_min(self):
        cfg = AntispoofConfig()
        assert cfg.ir_brightness_min == 15.0

    def test_default_minifasnet_threshold(self):
        cfg = AntispoofConfig()
        assert cfg.minifasnet_threshold == 0.8

    def test_default_require_both(self):
        cfg = AntispoofConfig()
        assert cfg.require_both is True

    def test_default_ir_only_fallback(self):
        cfg = AntispoofConfig()
        assert cfg.ir_only_fallback is True


@pytest.mark.unit
class TestDaemonConfigDefaults:
    """Test default values for DaemonConfig."""

    def test_default_socket_path(self):
        cfg = DaemonConfig()
        assert cfg.socket_path == "/run/user/{uid}/faceauth.sock"

    def test_default_log_level(self):
        cfg = DaemonConfig()
        assert cfg.log_level == "info"

    def test_default_log_file(self):
        cfg = DaemonConfig()
        assert cfg.log_file == "/tmp/faceauth.log"


@pytest.mark.unit
class TestConfigDefaults:
    """Test default values for top-level Config."""

    def test_config_has_camera_defaults(self):
        cfg = Config()
        assert isinstance(cfg.camera, CameraConfig)
        assert cfg.camera.ir_device == "/dev/video2"

    def test_config_has_recognition_defaults(self):
        cfg = Config()
        assert isinstance(cfg.recognition, RecognitionConfig)
        assert cfg.recognition.model == "buffalo_l"

    def test_config_has_antispoof_defaults(self):
        cfg = Config()
        assert isinstance(cfg.antispoof, AntispoofConfig)
        assert cfg.antispoof.enabled is False

    def test_config_has_daemon_defaults(self):
        cfg = Config()
        assert isinstance(cfg.daemon, DaemonConfig)
        assert cfg.daemon.log_level == "info"


@pytest.mark.unit
class TestConfigDataDir:
    """Test Config.data_dir property for different user contexts."""

    def test_data_dir_for_non_root_user(self, monkeypatch):
        """Non-root users get XDG_DATA_HOME/faceauth or ~/.local/share/faceauth."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        cfg = Config()
        expected = Path.home() / ".local/share/faceauth"
        assert cfg.data_dir == expected

    def test_data_dir_for_root_user(self, monkeypatch):
        """Root user gets /var/lib/faceauth."""
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        cfg = Config()
        assert cfg.data_dir == Path("/var/lib/faceauth")

    def test_data_dir_respects_xdg_data_home(self, monkeypatch, tmp_path):
        """XDG_DATA_HOME environment variable is respected."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        custom_data = tmp_path / "custom_data"
        monkeypatch.setenv("XDG_DATA_HOME", str(custom_data))

        cfg = Config()
        assert cfg.data_dir == custom_data / "faceauth"


@pytest.mark.unit
class TestConfigConfigDir:
    """Test Config.config_dir property for different user contexts."""

    def test_config_dir_for_non_root_user(self, monkeypatch):
        """Non-root users get XDG_CONFIG_HOME/faceauth or ~/.config/faceauth."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        cfg = Config()
        expected = Path.home() / ".config/faceauth"
        assert cfg.config_dir == expected

    def test_config_dir_for_root_user(self, monkeypatch):
        """Root user gets /etc/faceauth."""
        monkeypatch.setattr(os, "geteuid", lambda: 0)

        cfg = Config()
        assert cfg.config_dir == Path("/etc/faceauth")

    def test_config_dir_respects_xdg_config_home(self, monkeypatch, tmp_path):
        """XDG_CONFIG_HOME environment variable is respected."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        custom_config = tmp_path / "custom_config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(custom_config))

        cfg = Config()
        assert cfg.config_dir == custom_config / "faceauth"


@pytest.mark.unit
class TestLoadConfigDefaults:
    """Test load_config with no file or non-existent paths."""

    def test_load_config_no_file_returns_defaults(self, monkeypatch, tmp_path):
        """load_config with no file returns default Config."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        # Point config_dir to non-existent location
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nonexistent"))

        cfg = load_config()

        assert cfg.camera.ir_device == "/dev/video2"
        assert cfg.recognition.model == "buffalo_l"
        assert cfg.antispoof.enabled is False
        assert cfg.daemon.log_level == "info"

    def test_load_config_nonexistent_path(self, tmp_path):
        """load_config with non-existent path returns defaults."""
        nonexistent = tmp_path / "does_not_exist.toml"

        cfg = load_config(nonexistent)

        assert cfg.camera.ir_device == "/dev/video2"
        assert cfg.recognition.similarity_threshold == 0.45


@pytest.mark.unit
class TestLoadConfigFromFile:
    """Test load_config from TOML files."""

    def test_load_full_config(self, tmp_path):
        """Load complete config from TOML file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[camera]
ir_device = "/dev/video4"
rgb_device = "/dev/video1"
capture_timeout = 15
width = 1280
height = 720

[recognition]
model = "buffalo_sc"
similarity_threshold = 0.55
max_attempts = 3

[antispoof]
enabled = true
ir_brightness_min = 20.0
minifasnet_threshold = 0.9
require_both = false
ir_only_fallback = false

[daemon]
socket_path = "/tmp/custom.sock"
log_level = "debug"
log_file = "/var/log/faceauth.log"
""")

        cfg = load_config(config_file)

        # Verify camera config
        assert cfg.camera.ir_device == "/dev/video4"
        assert cfg.camera.rgb_device == "/dev/video1"
        assert cfg.camera.capture_timeout == 15
        assert cfg.camera.width == 1280
        assert cfg.camera.height == 720

        # Verify recognition config
        assert cfg.recognition.model == "buffalo_sc"
        assert cfg.recognition.similarity_threshold == 0.55
        assert cfg.recognition.max_attempts == 3

        # Verify antispoof config
        assert cfg.antispoof.enabled is True
        assert cfg.antispoof.ir_brightness_min == 20.0
        assert cfg.antispoof.minifasnet_threshold == 0.9
        assert cfg.antispoof.require_both is False
        assert cfg.antispoof.ir_only_fallback is False

        # Verify daemon config
        assert cfg.daemon.socket_path == "/tmp/custom.sock"
        assert cfg.daemon.log_level == "debug"
        assert cfg.daemon.log_file == "/var/log/faceauth.log"

    def test_load_partial_config_camera_only(self, tmp_path):
        """Load partial config with only [camera] section."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[camera]
ir_device = "/dev/video6"
width = 1920
height = 1080
""")

        cfg = load_config(config_file)

        # Camera values are overridden
        assert cfg.camera.ir_device == "/dev/video6"
        assert cfg.camera.width == 1920
        assert cfg.camera.height == 1080
        # But capture_timeout and rgb_device remain default
        assert cfg.camera.capture_timeout == 10
        assert cfg.camera.rgb_device == "/dev/video0"

        # Other sections remain default
        assert cfg.recognition.model == "buffalo_l"
        assert cfg.antispoof.enabled is False
        assert cfg.daemon.log_level == "info"

    def test_load_partial_config_mixed_sections(self, tmp_path):
        """Load config with some sections present, others absent."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[recognition]
similarity_threshold = 0.6

[daemon]
log_level = "error"
""")

        cfg = load_config(config_file)

        # Camera config is all defaults
        assert cfg.camera.ir_device == "/dev/video2"
        assert cfg.camera.width == 640

        # Recognition has one override
        assert cfg.recognition.similarity_threshold == 0.6
        assert cfg.recognition.model == "buffalo_l"  # still default

        # Antispoof is all defaults
        assert cfg.antispoof.enabled is False

        # Daemon has one override
        assert cfg.daemon.log_level == "error"
        assert cfg.daemon.socket_path == "/run/user/{uid}/faceauth.sock"  # still default

    def test_load_config_ignores_unknown_keys(self, tmp_path):
        """Unknown keys in TOML are silently ignored."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[camera]
ir_device = "/dev/video3"
unknown_key = "should be ignored"
another_unknown = 999

[unknown_section]
foo = "bar"
""")

        cfg = load_config(config_file)

        # Known key is applied
        assert cfg.camera.ir_device == "/dev/video3"
        # Unknown keys don't cause errors
        assert not hasattr(cfg.camera, "unknown_key")
        assert not hasattr(cfg, "unknown_section")

    def test_load_config_empty_toml(self, tmp_path):
        """Empty TOML file loads defaults."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")

        cfg = load_config(config_file)

        assert cfg.camera.ir_device == "/dev/video2"
        assert cfg.recognition.similarity_threshold == 0.45
        assert cfg.antispoof.enabled is False
        assert cfg.daemon.log_level == "info"


@pytest.mark.unit
class TestLoadConfigFallback:
    """Test config file discovery fallback mechanism."""

    def test_load_config_finds_user_config(self, monkeypatch, tmp_path):
        """load_config finds user config in XDG_CONFIG_HOME."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        config_home = tmp_path / "config"
        config_home.mkdir()
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

        # Create user config file
        user_config_dir = config_home / "faceauth"
        user_config_dir.mkdir()
        user_config = user_config_dir / "config.toml"
        user_config.write_text("""
[camera]
ir_device = "/dev/video9"
""")

        cfg = load_config()

        assert cfg.camera.ir_device == "/dev/video9"

    def test_load_config_explicit_path_overrides_discovery(self, monkeypatch, tmp_path):
        """Explicit path parameter overrides config discovery."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)

        # Create two config files
        discovered_config = tmp_path / "discovered.toml"
        discovered_config.write_text("""
[camera]
ir_device = "/dev/video1"
""")

        explicit_config = tmp_path / "explicit.toml"
        explicit_config.write_text("""
[camera]
ir_device = "/dev/video2"
""")

        # Load with explicit path
        cfg = load_config(explicit_config)

        assert cfg.camera.ir_device == "/dev/video2"


@pytest.mark.unit
class TestConfigDataTypes:
    """Test that config values maintain correct types."""

    def test_camera_config_types(self, tmp_path):
        """Verify camera config field types are preserved."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[camera]
ir_device = "/dev/video5"
capture_timeout = 20
width = 800
height = 600
""")

        cfg = load_config(config_file)

        assert isinstance(cfg.camera.ir_device, str)
        assert isinstance(cfg.camera.capture_timeout, int)
        assert isinstance(cfg.camera.width, int)
        assert isinstance(cfg.camera.height, int)

    def test_recognition_config_types(self, tmp_path):
        """Verify recognition config field types are preserved."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[recognition]
model = "custom_model"
similarity_threshold = 0.75
max_attempts = 10
""")

        cfg = load_config(config_file)

        assert isinstance(cfg.recognition.model, str)
        assert isinstance(cfg.recognition.similarity_threshold, float)
        assert isinstance(cfg.recognition.max_attempts, int)

    def test_antispoof_config_types(self, tmp_path):
        """Verify antispoof config field types are preserved."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[antispoof]
enabled = true
ir_brightness_min = 25.5
minifasnet_threshold = 0.85
require_both = false
ir_only_fallback = true
""")

        cfg = load_config(config_file)

        assert isinstance(cfg.antispoof.enabled, bool)
        assert isinstance(cfg.antispoof.ir_brightness_min, float)
        assert isinstance(cfg.antispoof.minifasnet_threshold, float)
        assert isinstance(cfg.antispoof.require_both, bool)
        assert isinstance(cfg.antispoof.ir_only_fallback, bool)


@pytest.mark.unit
class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_load_config_with_comments(self, tmp_path):
        """TOML comments are properly ignored."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
# This is a comment
[camera]
ir_device = "/dev/video3"  # inline comment
# Another comment
capture_timeout = 25
""")

        cfg = load_config(config_file)

        assert cfg.camera.ir_device == "/dev/video3"
        assert cfg.camera.capture_timeout == 25

    def test_config_partial_section_override(self, tmp_path):
        """Overriding one field doesn't affect others in same section."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[camera]
ir_device = "/dev/video7"
""")

        cfg = load_config(config_file)

        # Only ir_device is overridden
        assert cfg.camera.ir_device == "/dev/video7"
        # All other camera fields remain default
        assert cfg.camera.rgb_device == "/dev/video0"
        assert cfg.camera.capture_timeout == 10
        assert cfg.camera.width == 640
        assert cfg.camera.height == 480

    def test_config_zero_and_negative_values(self, tmp_path):
        """Config can handle zero and negative numeric values."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[camera]
capture_timeout = 0

[recognition]
similarity_threshold = 0.0
max_attempts = 1

[antispoof]
ir_brightness_min = 0.0
""")

        cfg = load_config(config_file)

        assert cfg.camera.capture_timeout == 0
        assert cfg.recognition.similarity_threshold == 0.0
        assert cfg.recognition.max_attempts == 1
        assert cfg.antispoof.ir_brightness_min == 0.0
