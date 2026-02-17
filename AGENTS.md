# faceauth - Project Guidelines

Guidelines for contributors and AI coding agents working on this project.

## Architecture

```
PAM (login/sudo/GDM)
  -> pam_exec -> faceauth-pam-helper (stdlib-only Python)
  -> Unix socket -> faceauth daemon (user or system mode)
     +-- MediaPipe BlazeFace (face detection)
     +-- InsightFace ArcFace buffalo_l (embedding extraction)
     +-- MiniFASNet (anti-spoof liveness, optional)
     +-- IR Brightness Check (anti-spoof)
```

| Module | Role |
|--------|------|
| `pam/faceauth-pam-helper` | PAM helper -- stdlib-only, no third-party imports (security boundary) |
| `faceauth/daemon.py` | Async Unix socket server, keeps models in RAM |
| `faceauth/pipeline.py` | Shared face-processing functions used by CLI and daemon |
| `cli/faceauth_cli.py` | Click-based CLI: enroll, verify, test, daemon management |

## Key Paths

| Purpose | User Mode | System Mode |
|---------|-----------|-------------|
| Embeddings | `~/.local/share/faceauth/` | `/var/lib/faceauth/` |
| Config | `~/.config/faceauth/config.toml` | `/etc/faceauth/config.toml` |
| Socket | `/run/user/{uid}/faceauth.sock` | `/run/faceauth/faceauth.sock` |

## Development

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v           # full test suite
ruff check .               # lint
ruff format --check .      # format check
```

## Coding Standards

### Python Style
- **Python 3.11+** minimum
- **ruff** for linting and formatting (config in `pyproject.toml`)
- **Line length**: 100 characters
- **Type hints**: use on public APIs and function signatures
- **Imports**: stdlib first, then third-party, then local. Let ruff sort them.

### Code Quality
- No unused imports (F401)
- No f-strings without placeholders (F541)
- No unused variables -- prefix with `_` if intentionally unused (F841)
- No bare `except:` -- always catch specific exceptions
- Keep functions focused; extract helpers when exceeding ~50 lines
- Use named constants from `faceauth/pipeline.py` instead of magic numbers (`MIN_DET_SCORE`, `FRAME_DELAY_VERIFY`, etc.)

### Security
- **PAM helper must remain stdlib-only** -- no third-party imports in `pam/faceauth-pam-helper`
- Embedding files use restrictive permissions (0o600 files, 0o700 directories)
- Unix socket auth uses SO_PEERCRED to verify peer UID
- Validate all usernames against path traversal (`/`, `\0`, `..`)
- Never log embedding data or similarity scores at INFO level

### Testing
- All tests in `tests/` directory
- Mark tests: `@pytest.mark.unit` or `@pytest.mark.integration`
- Unit tests mock all hardware (camera, GPU, models)
- Integration tests use real async sockets with temporary paths
- Use `conftest.py` fixtures: `sample_embedding`, `similar_embedding`, `grey_frame`, etc.
- CI runs unit and integration tests separately
- All tests must pass and ruff must be clean before merging

### Commits
- Concise messages in imperative mood; explain "why" not "what"
- Verify before committing: `ruff check . && ruff format --check . && pytest tests/ -v`

## Model Licensing

The source code is MIT-licensed, but ML model weights have separate licenses:

| Model | License | Commercial Use |
|-------|---------|----------------|
| InsightFace buffalo_l | Non-commercial research only | Requires [commercial license](https://www.insightface.ai) |
| MiniFASNet | Apache 2.0 (Minivision) | Allowed |
| MediaPipe | Apache 2.0 (Google) | Allowed |

Any code that downloads or references model weights must include license notices.

## Dependencies

- No upper bounds on dependencies (project is young)
- Core: insightface, onnxruntime, mediapipe, opencv-contrib-python, numpy, click
- Dev: pytest, pytest-asyncio, pytest-cov, pytest-mock, ruff
- Optional: openvino, onnxruntime-openvino (GPU acceleration)
