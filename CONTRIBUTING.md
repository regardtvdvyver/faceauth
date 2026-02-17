# Contributing to faceauth

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/regardtvdvyver/faceauth.git
cd faceauth
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Optional: OpenVINO acceleration

```bash
pip install openvino onnxruntime-openvino
```

## Running Tests

```bash
# All tests (unit + integration)
pytest tests/ -v

# Unit tests only
pytest tests/ -v -m unit

# Integration tests only (async daemon tests)
pytest tests/ -v -m integration

# With coverage
pytest tests/ --cov=faceauth
```

All ML dependencies are mocked in tests -- no GPU or camera hardware is needed to run the test suite.

## Code Style

- **Linter/formatter**: [ruff](https://docs.astral.sh/ruff/)
- **Line length**: 100 characters
- **Python**: 3.11+ (uses `X | Y` union syntax, `tomllib`, etc.)
- **Type hints**: encouraged on public APIs

```bash
# Check lint
ruff check .

# Auto-fix
ruff check --fix .

# Format
ruff format .
```

## Pull Request Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes and add tests
4. Run `pytest tests/ -v` and `ruff check .` to verify
5. Commit with a clear message describing the "why"
6. Open a Pull Request against `main`

## Architecture

See the [README](README.md) for a high-level architecture diagram. Key modules:

| Module | Purpose |
|--------|---------|
| `faceauth/pipeline.py` | Shared face-processing functions (frame handling, face selection, anti-spoof, embedding comparison) |
| `faceauth/daemon.py` | Async Unix socket server, keeps ML models in RAM |
| `faceauth/recognizer.py` | InsightFace/ArcFace face detection and embedding |
| `faceauth/antispoof.py` | IR brightness + MiniFASNet liveness checks |
| `faceauth/camera.py` | OpenCV camera capture, IR detection |
| `faceauth/storage.py` | Secure `.npz` embedding storage |
| `faceauth/protocol.py` | JSON-over-socket wire protocol |
| `faceauth/pam_client.py` | Client for daemon communication |
| `cli/faceauth_cli.py` | Click-based CLI |

## Testing Notes

- **IR camera not required**: All camera and ML interactions are mocked in tests
- **Fixtures**: Common test data (embeddings, frames, bboxes) is in `tests/conftest.py`
- **Integration tests**: Use real Unix sockets with a temporary directory
- **cv2 is auto-mocked**: The `conftest.py` auto-patches OpenCV since CI environments don't have the binary module

## Reporting Issues

- Use [GitHub Issues](https://github.com/regardtvdvyver/faceauth/issues) for bugs and feature requests
- For security vulnerabilities, see [SECURITY.md](SECURITY.md)
