# faceauth

**Face authentication for Linux. Like Windows Hello, but open source.**

Unlock your laptop, `sudo`, and GDM login with your face. Uses your laptop's IR camera for hardware-backed anti-spoofing that rejects photos and screens. Authentication takes ~200ms.

```bash
pip install faceauth
faceauth setup
```

## Why faceauth?

| | [Howdy](https://github.com/boltgolt/howdy) | faceauth |
|---|---|---|
| Anti-spoofing | None (RGB only) | IR + liveness detection |
| Speed | Cold start every auth | ~200ms (daemon keeps models loaded) |
| Install | dlib/cmake build pain | `pip install` |
| Security model | Runs as root | Minimal-privilege daemon, stdlib-only PAM helper |
| Camera setup | Manual | Auto-detects IR cameras via V4L2 |
| Status | Last release 2022 | Active development |

## How it Works

1. **Face detection** -- MediaPipe BlazeFace locates faces in the camera frame
2. **Embedding extraction** -- ArcFace computes a 512-dimensional face embedding
3. **Comparison** -- Cosine similarity against stored enrollment embeddings
4. **Anti-spoofing** -- IR brightness check (photos/screens don't reflect IR) + optional MiniFASNet liveness detection

The daemon keeps ML models loaded in memory. First auth loads models (~2s), subsequent auths take ~200ms on Intel GPU.

## Quick Start

### One-command setup

```bash
faceauth setup
```

This will:
1. Auto-detect your IR and RGB cameras
2. Write the configuration
3. Enroll your face (5 samples)
4. Install and start the systemd service

### Manual setup

If you prefer step-by-step:

```bash
# 1. Install
pip install faceauth

# 2. Configure (auto-detection writes this for you during setup)
faceauth setup --skip-enroll --skip-pam

# 3. Enroll your face
faceauth enroll $USER

# 4. Verify it works
faceauth verify $USER

# 5. Enable for sudo/GDM (requires root)
sudo faceauth install-pam
sudo pam-auth-update  # select "Face authentication (faceauth)"
```

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
  +-- InsightFace ArcFace (embedding extraction)
  +-- MiniFASNet (anti-spoof liveness, optional)
  +-- IR Brightness Check (anti-spoof)
```

**Security design:**
- The PAM helper is stdlib-only Python -- no third-party imports, minimal attack surface
- Communication via Unix socket with `SO_PEERCRED` UID verification
- Embeddings stored with restrictive permissions (0o600 files, 0o700 directories)
- Face auth failure always falls through to password prompt

## Daemon

### User mode (development)

```bash
faceauth daemon install
faceauth daemon start
faceauth status
```

### System mode (production / GDM)

Required for GDM login since the daemon must run before any user session.

```bash
sudo faceauth daemon install --system
sudo systemctl enable --now faceauth
sudo faceauth migrate-to-system
```

## Hardware

### Requirements

- **IR camera** -- common on business laptops (ThinkPad, Latitude, EliteBook). RGB-only cameras work but without anti-spoof protection.
- **x86_64 Linux** with Python 3.11+
- **Intel GPU** recommended for OpenVINO acceleration (optional; CPU works on any hardware)

### Tested Hardware

| Laptop | IR Camera | GPU Accel | Notes |
|--------|-----------|-----------|-------|
| ThinkPad P14s Gen 5 | `/dev/video2` (640x360) | Intel Arc iGPU | Primary dev hardware |

Tested on other hardware? [Open a hardware report](https://github.com/regardtvdvyver/faceauth/issues/new?template=hardware_report.md) to add it to this list.

### Camera Auto-Detection

faceauth scans V4L2 devices to automatically find IR and RGB cameras:

```bash
faceauth setup  # auto-detects during setup
```

No need to manually find device paths or check resolutions.

### OpenVINO (optional)

For Intel GPU acceleration (~1.4x faster inference):

```bash
pip install faceauth[openvino]
```

GPU is auto-selected when available, with automatic CPU fallback.

## Performance

Benchmarks on Intel Core Ultra 9 185H (hardware-specific):

| Model | GPU | CPU (OpenVINO) |
|-------|-----|----------------|
| ArcFace recognition | **17.6ms** | 25ms |
| Face detection | 58ms | **47ms** |

## Safety

1. Face auth failure **always** falls through to password prompt
2. The PAM profile defaults to `Default: no` (explicit opt-in via `pam-auth-update`)
3. Always keep a root TTY open during GDM testing
4. Test with `sudo -k whoami` before relying on GDM

## Configuration

Config lives at `~/.config/faceauth/config.toml` (user) or `/etc/faceauth/config.toml` (system). The `faceauth setup` wizard generates this for you.

<details>
<summary>Full configuration reference</summary>

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
openvino_device = ""          # "" = auto, or "GPU", "CPU", "NPU"

[antispoof]
enabled = true
ir_brightness_min = 15.0      # Min IR brightness (rejects photos/screens)
minifasnet_threshold = 0.8    # Liveness score threshold
require_both = true           # Must pass both IR AND MiniFASNet
ir_only_fallback = true       # Use IR-only if MiniFASNet unavailable

[daemon]
socket_path = "/run/user/{uid}/faceauth.sock"
system_socket_path = "/run/faceauth/faceauth.sock"
log_level = "info"
```

</details>

## Model Licensing

The faceauth source code is MIT-licensed. The ML models have their own licenses:

| Model | License | Commercial Use |
|-------|---------|----------------|
| [InsightFace buffalo_l](https://github.com/deepinsight/insightface) | Non-commercial research only | Requires [commercial license](https://www.insightface.ai) |
| [MiniFASNet](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | Apache 2.0 | Allowed |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache 2.0 | Allowed |

The buffalo_l model weights are downloaded automatically on first use. For commercial use, obtain a license from InsightFace or substitute an open-licensed model.

## Development

```bash
git clone https://github.com/regardtvdvyver/faceauth.git
cd faceauth
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v              # 290+ tests
ruff check . && ruff format . # lint + format
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for PR guidelines and [ROADMAP.md](ROADMAP.md) for planned features.

## License

MIT (source code). See [Model Licensing](#model-licensing) for ML model terms.
