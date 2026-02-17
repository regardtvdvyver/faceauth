# faceauth - Modern Linux Face Authentication

## Project Overview
PAM-based face authentication for Linux using IR camera, ArcFace embeddings, and anti-spoofing.
Built for ThinkPad P14s Gen 5 with Intel Core Ultra 9 185H (NPU + iGPU).

## Architecture
- **PAM module** (pam_exec) -> `faceauth-pam-helper` -> daemon via Unix socket
- **Daemon** (user or system mode) keeps ML models loaded in RAM for fast inference
- **ML Pipeline**: MediaPipe (detection) -> InsightFace/ArcFace (recognition) -> MiniFASNet (anti-spoof)
- **Hardware accel**: OpenVINO GPU auto-selected, CPU fallback for incompatible models

## Git Identity
- Name: `Regardt van de Vyver`
- Email: `regardt@vdvyver.net`

## Key Paths
- IR camera: `/dev/video2` (640x360 greyscale)
- RGB camera: `/dev/video0`
- Embeddings: `~/.local/share/faceauth/` (dev) or `/var/lib/faceauth/` (production)
- Config: `~/.config/faceauth/config.toml` or `/etc/faceauth/config.toml`
- Socket (user): `/run/user/{uid}/faceauth.sock`
- Socket (system): `/run/faceauth/faceauth.sock`

## Development
```bash
source .venv/bin/activate
faceauth --help
faceauth test          # Visual camera test
faceauth enroll USER   # Enroll a face
faceauth verify USER   # Verify against stored embedding
pytest tests/ -v       # 231 tests
```

## Phase Status
- [x] Phase 1: ML Pipeline + CLI
- [x] Phase 2: Daemon + PAM Module
- [x] Phase 3: Anti-Spoofing + IR
- [x] Phase 4: GDM Integration (pam_exec + system daemon)
- [x] Phase 5: Optimisation (OpenVINO GPU acceleration)
