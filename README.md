# faceauth

PAM-based face authentication for Linux using IR camera, ArcFace embeddings, and anti-spoofing. Works with any USB IR camera; optional OpenVINO GPU acceleration for Intel hardware.

Developed and tested on ThinkPad P14s Gen 5 (Intel Core Ultra 9 185H, Intel Arc iGPU).

## How it Works

1. **Face detection** -- MediaPipe BlazeFace locates faces in the camera frame
2. **Embedding extraction** -- InsightFace ArcFace (buffalo_l) computes a 512-dimensional face embedding
3. **Comparison** -- Cosine similarity between the live embedding and stored enrollment embeddings
4. **Anti-spoofing** -- IR brightness check (photos/screens don't reflect IR) plus optional MiniFASNet liveness detection

Authentication takes ~200ms per frame on Intel GPU. The daemon keeps models loaded in RAM so subsequent requests skip model loading.

## Architecture

```
PAM (login/sudo/GDM)
  |
  v
pam_exec -> faceauth-pam-helper (stdlib-only Python)
  |
  v (Unix socket)
faceauth daemon (user or system mode)
  |
  +-- MediaPipe BlazeFace (face detection)
  +-- InsightFace ArcFace (embedding extraction, buffalo_l model)
  +-- MiniFASNet (anti-spoof liveness, optional)
  +-- IR Brightness Check (anti-spoof)
```

## Key Paths

| Purpose | User Mode | System Mode |
|---------|-----------|-------------|
| Embeddings | `~/.local/share/faceauth/` | `/var/lib/faceauth/` |
| Config | `~/.config/faceauth/config.toml` | `/etc/faceauth/config.toml` |
| Socket | `/run/user/{uid}/faceauth.sock` | `/run/faceauth/faceauth.sock` |

## Hardware Requirements

- **IR camera** -- common on business laptops (ThinkPad, Latitude, EliteBook). RGB-only cameras work for enrollment/verification but without anti-spoof protection.
- **x86_64 Linux** with Python 3.11+
- **Intel GPU** recommended for OpenVINO acceleration (optional; CPU fallback works on any hardware)

## Installation

```bash
git clone https://github.com/regardtvdvyver/faceauth.git
cd faceauth
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### OpenVINO (optional, recommended)

For Intel GPU acceleration (~1.4x faster ArcFace inference):

```bash
pip install openvino onnxruntime-openvino
```

The provider is auto-detected: GPU > CPU with automatic fallback.

## Quick Start

### 1. Configure

Create `~/.config/faceauth/config.toml`:

```toml
[camera]
ir_device = "/dev/video2"
width = 640
height = 360              # Match your IR camera resolution

[recognition]
model = "buffalo_l"
similarity_threshold = 0.45
# openvino_device = ""    # Auto-select (default), or force "GPU", "CPU", "NPU"

[antispoof]
enabled = true
ir_brightness_min = 15.0
ir_only_fallback = true   # Use IR-only if MiniFASNet model not downloaded

[daemon]
log_level = "info"
```

Check your IR camera resolution with:
```bash
v4l2-ctl -d /dev/video2 --list-formats-ext
```

### 2. Test Camera

```bash
faceauth test
```

### 3. Enroll and Verify

```bash
faceauth enroll USERNAME
faceauth verify USERNAME
faceauth list
```

## Daemon

### User-Level Daemon (Development)

```bash
# Install systemd user service
faceauth daemon install

# Start daemon
faceauth daemon start
# or run in foreground
faceauth daemon start --foreground

# Check status
faceauth status
```

### System-Level Daemon (Production / GDM)

Required for GDM login since the daemon must run before any user session.

```bash
# Install system daemon service (templates Python path into service file)
sudo .venv/bin/faceauth daemon install --system

# Start and enable
sudo systemctl enable --now faceauth

# Migrate user embeddings to system store
sudo .venv/bin/faceauth migrate-to-system
```

## PAM / GDM Integration

### Install PAM Helper

```bash
sudo .venv/bin/faceauth install-pam
```

This installs:
- `/usr/local/lib/faceauth/faceauth-pam-helper` (stdlib-only script for pam_exec)
- `/usr/share/pam-configs/faceauth` (pam-auth-update profile)

### Enable for sudo (test first)

```bash
sudo pam-auth-update
# Select "Face authentication (faceauth)"
```

Then test:
```bash
sudo -k whoami
```

Face auth is attempted first. On failure it falls through to password (`default=ignore`).

### Enable for GDM

The same pam-auth-update profile applies via `common-auth`, so enabling it for sudo also enables it for GDM. The system daemon must be running.

## Safety Notes

1. Always keep a root TTY open during GDM testing
2. The PAM profile defaults to `Default: no` (explicit opt-in via `pam-auth-update`)
3. Face auth failure always falls through to password prompt
4. Test with `sudo` before relying on it for GDM login

## Configuration Reference

```toml
[camera]
ir_device = "/dev/video2"     # IR camera device
rgb_device = "/dev/video0"    # RGB camera (optional)
capture_timeout = 10          # Seconds before giving up
width = 640
height = 360

[recognition]
model = "buffalo_l"           # InsightFace model name
similarity_threshold = 0.45   # Cosine similarity (higher = stricter)
max_attempts = 5              # Frames to try before failing
openvino_device = ""          # "" = auto (GPU>CPU), or "GPU", "CPU", "NPU", "AUTO"

[antispoof]
enabled = true
ir_brightness_min = 15.0      # Min IR brightness (rejects photos/screens)
minifasnet_threshold = 0.8    # Liveness score threshold
require_both = true           # Must pass both IR AND MiniFASNet
ir_only_fallback = true       # Use IR-only if MiniFASNet model unavailable

[daemon]
socket_path = "/run/user/{uid}/faceauth.sock"
system_socket_path = "/run/faceauth/faceauth.sock"
log_level = "info"
log_file = "/tmp/faceauth.log"
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=faceauth
```

231 tests across 9 test files covering protocol, storage, config, camera, anti-spoof, detection, recognition, PAM client, and daemon integration.

## Performance

GPU is auto-selected for best overall performance. Models that fail on GPU (e.g., MiniFASNet) automatically fall back to CPU.

Benchmarks on Intel Core Ultra 9 185H (timings are hardware-specific):

| Model | GPU | CPU (OpenVINO) | NPU |
|-------|-----|----------------|-----|
| ArcFace recognition | **17.6ms** | 25ms | 1778ms |
| Face detection | 58ms | **47ms** | 1035ms |

## Supported Hardware

faceauth should work with any Linux-compatible IR camera. Tested on:

| Laptop | IR Camera | GPU Accel | Notes |
|--------|-----------|-----------|-------|
| ThinkPad P14s Gen 5 | `/dev/video2` (640x360) | Intel Arc iGPU | Primary development hardware |

If you test faceauth on other hardware, please [open an issue](https://github.com/regardtvdvyver/faceauth/issues) to add it to this list.

## Model Licensing

**Important**: The faceauth source code is MIT-licensed, but the ML models it uses have their own licenses:

| Model | Purpose | License | Commercial Use |
|-------|---------|---------|----------------|
| [InsightFace buffalo_l](https://github.com/deepinsight/insightface) | Face detection + recognition | **Non-commercial research only** | Requires [commercial license](https://www.insightface.ai/services/models-commercial-licensing) |
| [MiniFASNet](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | Anti-spoof liveness (optional) | Apache 2.0 | Allowed |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Face detection | Apache 2.0 | Allowed |

The buffalo_l model weights are auto-downloaded by InsightFace on first use. If you intend to use faceauth commercially, you must obtain a license from InsightFace ([contact](mailto:recognition-oss-pack@insightface.ai)) or substitute an alternatively-licensed recognition model.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

MIT (source code). See [Model Licensing](#model-licensing) for ML model terms.

## Author

Regardt van de Vyver (regardt@vdvyver.net)
