# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-15

### Added

- **Phase 1 — ML Pipeline + CLI**
  - Face detection via InsightFace (RetinaFace + ArcFace buffalo_l model)
  - 512-dim embedding extraction and cosine similarity matching
  - Click-based CLI: `enroll`, `verify`, `test`, `list`, `delete`, `status`
  - Secure `.npz` embedding storage with `0o600` permissions and path traversal prevention

- **Phase 2 — Daemon + PAM Module**
  - Async Unix socket daemon keeping ML models loaded in RAM
  - JSON-over-socket wire protocol for verify/enroll/delete/status/list
  - `SO_PEERCRED` authentication on Unix socket connections
  - PAM integration via `pam_exec` with stdlib-only helper script
  - Systemd service files for user and system mode

- **Phase 3 — Anti-Spoofing + IR**
  - IR brightness check (real faces reflect IR; photos/screens appear dark)
  - MiniFASNet ONNX liveness detection (optional)
  - Configurable policy: require both, either, or IR-only fallback
  - IR camera auto-detection and greyscale-to-BGR conversion

- **Phase 4 — GDM Integration**
  - System-level daemon mode for pre-login authentication
  - `pam-auth-update` profile for easy PAM configuration
  - `migrate-to-system` command for copying user embeddings to `/var/lib/faceauth/`
  - Face auth falls through to password on failure

- **Phase 5 — Optimisation**
  - OpenVINO GPU acceleration with automatic device selection
  - GPU > CPU priority with per-model fallback (MiniFASNet falls back to CPU)
  - Shared provider cache across all ONNX sessions
  - 231 tests across 9 test files

[0.1.0]: https://github.com/regardtvdvyver/faceauth/releases/tag/v0.1.0
