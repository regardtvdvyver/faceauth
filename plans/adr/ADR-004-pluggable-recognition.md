# ADR-004: Pluggable Face Recognition Backend

**Status:** Accepted
**Date:** 2026-02-17

## Context

The current face recognition implementation uses InsightFace's buffalo_l model, which provides excellent accuracy (99.8% on LFW benchmark). However, this creates a critical licensing barrier:

### The Problem

1. **Model License**: Buffalo_l weights are released under a non-commercial research license
2. **Distribution Blocking**: This prevents:
   - Inclusion in Linux distribution repositories (Debian, Fedora, Arch)
   - Corporate/enterprise deployment without legal review
   - Use in any commercial context
   - Redistribution with faceauth package
3. **Uncertainty**: Cautious users/organizations avoid projects with unclear IP status
4. **Upstream Risk**: InsightFace could change licensing terms or remove models

### Impact on Adoption

- **Home users**: Mostly unaffected (personal use likely OK), but uncomfortable with legal ambiguity
- **Enterprises**: Completely blocked without legal sign-off
- **Linux distros**: Cannot package faceauth while it requires non-free model weights
- **Open source purists**: Won't use tools with non-free dependencies

### Current Licensing Landscape

| Component | License | Status |
|-----------|---------|--------|
| faceauth code | MIT | Free ✓ |
| MediaPipe (detection) | Apache 2.0 | Free ✓ |
| MiniFASNet (anti-spoof) | Custom research | Research-only ✗ |
| InsightFace models | Non-commercial | Non-free ✗ |
| ONNX Runtime | MIT | Free ✓ |

Two of our three ML models have problematic licenses. This ADR addresses face recognition; anti-spoofing is a separate concern.

### Why Not Just Switch Models?

We could replace InsightFace with a permissively-licensed model immediately, but:
1. **Risk**: Accuracy drop could make faceauth unusable
2. **Research needed**: Must evaluate multiple alternative models thoroughly
3. **User choice**: Power users who accept the license want the best accuracy

Better approach: Support both, let users choose.

## Decision

Implement a pluggable face recognition backend system that allows runtime selection between different face recognition models while maintaining a consistent interface.

### Architecture

#### 1. Recognition Backend Protocol

Define a protocol (Python Protocol/ABC) for face recognition backends:

```python
# faceauth/recognition/backend.py
from typing import Protocol, Tuple
import numpy as np

class RecognitionBackend(Protocol):
    """Protocol for face recognition implementations."""

    def load_models(self) -> None:
        """Load recognition models into memory."""
        ...

    def generate_embedding(
        self,
        face_image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Generate face embedding vector from image.

        Args:
            face_image: RGB or grayscale image (H x W x C)
            bbox: Bounding box (x, y, w, h) from detector

        Returns:
            Normalized embedding vector (typically 512-dim)
        """
        ...

    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compare two embeddings, return similarity score.

        Returns:
            Similarity in range [0.0, 1.0]
        """
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embedding vectors."""
        ...

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    def license(self) -> str:
        """License of the model weights."""
        ...
```

#### 2. Backend Implementations

**Backend 1: InsightFace (Current)**
```python
# faceauth/recognition/insightface_backend.py
class InsightFaceBackend:
    """InsightFace buffalo_l recognition backend."""

    name = "InsightFace (buffalo_l)"
    license = "Non-commercial research only"
    embedding_dim = 512

    # Current implementation, refactored to match protocol
```

**Backend 2: Open (Future)**
```python
# faceauth/recognition/open_backend.py
class OpenBackend:
    """Open-licensed recognition backend.

    Candidates:
    - AdaFace (MIT license, 99.7% LFW)
    - ElasticFace (Apache 2.0, 99.6% LFW)
    - MagFace (Custom permissive, 99.8% LFW)

    Final choice TBD after benchmarking.
    """

    name = "Open Backend"
    license = "MIT/Apache 2.0"
    embedding_dim = 512

    # To be implemented with permissively-licensed model
```

#### 3. Configuration

Add to config.toml:

```toml
[recognition]
# Backend selection: "insightface" (default) or "open"
backend = "insightface"

# Backend-specific settings
[recognition.insightface]
model = "buffalo_l"
# ... current settings

[recognition.open]
model = "adaface"  # Example, TBD
# ... future settings
```

#### 4. Backend Factory

```python
# faceauth/recognition/__init__.py
def get_recognition_backend(config: Config) -> RecognitionBackend:
    """Factory function to instantiate configured backend."""
    backend_name = config.recognition.backend

    if backend_name == "insightface":
        from .insightface_backend import InsightFaceBackend
        return InsightFaceBackend(config)
    elif backend_name == "open":
        from .open_backend import OpenBackend
        return OpenBackend(config)
    else:
        raise ValueError(f"Unknown recognition backend: {backend_name}")
```

#### 5. License Warnings

At startup, daemon logs:
```
INFO: Using recognition backend: InsightFace (buffalo_l)
WARN: Model license: Non-commercial research only
WARN: See https://github.com/deepinsight/insightface for terms
```

For the open backend:
```
INFO: Using recognition backend: Open (AdaFace)
INFO: Model license: MIT - free for all uses
```

#### 6. Embedding Compatibility

**Critical constraint**: Embeddings from different backends are NOT compatible. A user enrolled with InsightFace cannot be verified with Open backend.

**Solution**: Store backend identifier with each embedding:

```python
# faceauth/storage.py
@dataclass
class UserEmbedding:
    username: str
    embedding: np.ndarray
    backend: str  # NEW: "insightface" or "open"
    timestamp: datetime
    metadata: Dict[str, Any]
```

On verification, daemon checks:
```python
if stored_embedding.backend != current_backend.name:
    raise ValueError(
        f"Embedding created with {stored_embedding.backend}, "
        f"but daemon is using {current_backend.name}. "
        f"Re-enroll user or change backend in config."
    )
```

#### 7. Migration Tool

Provide `faceauth migrate-backend` command:
```bash
faceauth migrate-backend --from insightface --to open
# Prompts user to re-enroll all users
# Backs up old embeddings
# Guides through re-enrollment process
```

### Phased Implementation

This ADR defines the architecture, but implementation is phased:

**Phase 1 (Immediate):**
- Design and implement RecognitionBackend protocol
- Refactor current InsightFace code to match protocol
- Move to faceauth/recognition/ module structure
- Add backend selection to config
- Add license warnings to daemon logs

**Phase 2 (Future):**
- Research permissively-licensed models (AdaFace, ElasticFace, MagFace)
- Benchmark accuracy on test dataset
- Implement OpenBackend with chosen model
- Convert models to ONNX format if needed
- Document accuracy trade-offs

**Phase 3 (Future):**
- Add backend migration tool
- Update setup wizard to offer backend choice
- Add backend info to `faceauth info` command

### Model Candidates for Open Backend

Research needed, but promising options:

1. **AdaFace** (MIT license)
   - Paper: "AdaFace: Quality Adaptive Margin for Face Recognition" (CVPR 2022)
   - Accuracy: 99.7% LFW, 98.5% AgeDB
   - Model size: 120MB
   - Pre-trained weights available

2. **ElasticFace** (Apache 2.0)
   - Paper: "ElasticFace: Elastic Margin Loss for Deep Face Recognition" (CVPRW 2022)
   - Accuracy: 99.6% LFW
   - Model size: 100MB

3. **MagFace** (Custom permissive)
   - Paper: "MagFace: A Universal Representation for Face Recognition and Quality Assessment" (CVPR 2021)
   - Accuracy: 99.8% LFW
   - Need to verify license terms

Selection criteria:
- Permissive license (MIT, Apache 2.0, or similar)
- Competitive accuracy (>99.5% LFW)
- Available ONNX weights or convertible
- Active maintenance/research backing

## Consequences

**Positive:**
- **Enables distribution**: Linux distros can package faceauth with open backend as default
- **Corporate adoption**: Enterprises can deploy without legal concerns
- **User choice**: Power users can opt into higher accuracy with license acceptance
- **Future-proof**: Can add more backends as new models emerge
- **Clear licensing**: Makes IP status transparent to users

**Negative:**
- **More complexity**: Multiple code paths to maintain
- **Embedding incompatibility**: Changing backends requires re-enrollment
- **Documentation burden**: Must explain backend differences, trade-offs
- **Testing overhead**: Must test both backends
- **Initial work**: Significant upfront design and refactoring

**Trade-offs:**
- **Accuracy vs. licensing**: Open backend may have slightly lower accuracy (99.7% vs. 99.8%)
  - Practical impact: Negligible for most users (0.1% difference)
  - Mitigated by anti-spoofing, multi-sample enrollment
- **Default backend**: Which should be default?
  - Proposal: **InsightFace remains default for now**, with prominent license warning
  - Rationale: Don't force accuracy loss on existing users, but make license explicit
  - Future: Switch default to open backend once validated

## Alternatives Considered

### 1. Replace InsightFace entirely with open model
**Rejected.** Too risky - if accuracy drops significantly, faceauth becomes unusable. Better to support both and let users choose. Power users who accept the license want maximum accuracy.

### 2. Keep only InsightFace, document license limitation
**Rejected.** Doesn't solve the distribution problem. Linux distros still can't package it. Enterprises still need legal review.

### 3. Train our own model with open data
**Rejected.** Requires massive dataset (millions of faces), GPU resources, months of training, expertise. Unrealistic for a small project. Better to use existing research.

### 4. Dual-license the faceauth project itself
**Rejected.** Doesn't help - the problem is model weight licenses, not code licenses. faceauth code is already MIT.

### 5. Make backend pluggable via external process (gRPC, REST)
**Rejected.** Massive complexity increase, performance overhead, deployment nightmare. Python Protocol is sufficient for our needs.

### 6. Use cloud API (AWS Rekognition, Azure Face)
**Rejected.** Defeats the purpose of local authentication. Adds latency, cost, privacy concerns, internet dependency. Completely misaligned with project goals.

### 7. Ignore the problem until someone complains
**Rejected.** Irresponsible for a security tool. Better to address proactively than have distro maintainers or corporate legal teams reject faceauth later.

## Risks and Mitigations

### Risk 1: Open backend has significantly worse accuracy
**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Benchmark thoroughly before release
- If accuracy drop is >1%, keep InsightFace as recommended option
- Document accuracy numbers transparently
- Consider ensemble approach (average embeddings from both)

### Risk 2: Embedding incompatibility causes user confusion
**Likelihood:** High
**Impact:** Medium
**Mitigation:**
- Clear error messages when backend mismatch detected
- Migration tool guides users through re-enrollment
- Store backend in embedding metadata (already planned)
- Wizard warns before changing backend

### Risk 3: Maintenance burden of multiple backends
**Likelihood:** High
**Impact:** Medium
**Mitigation:**
- Protocol enforces consistent interface
- Shared test suite for all backends
- Keep backends simple (delegate to ONNX models)
- Only add well-maintained models

### Risk 4: Open model license terms are unclear
**Likelihood:** Low
**Impact:** High
**Mitigation:**
- Thorough license review before selection
- Contact authors if unclear
- Document due diligence in code comments
- Prefer models from reputable institutions (MIT, Stanford, etc.)

## Success Metrics

- **Adoption**: Faceauth packaged in at least 1 major distro (Debian, Arch, Fedora)
- **Accuracy**: Open backend achieves >99.5% LFW (within 0.3% of InsightFace)
- **Performance**: Open backend inference time <100ms on CPU (comparable to InsightFace)
- **Usability**: <5% of users report confusion about backend selection

## Documentation Impact

- **README.md**: Add "Licensing & Model Backends" section
- **BACKENDS.md**: New doc explaining backends, licenses, trade-offs
- **INSTALLATION.md**: Note backend choice during setup
- **FAQ.md**: Add "Why two backends?" and "Which should I use?" entries
- **LICENSE**: Clarify that faceauth code is MIT, but model weights vary

## Future Enhancements

Beyond this ADR's scope, but enabled by this architecture:

1. **Ensemble backend**: Combine multiple models for better accuracy
2. **Custom backend**: Allow users to bring their own ONNX models
3. **Auto-selection**: Profile backends at first run, pick best for hardware
4. **Backend benchmarking**: `faceauth benchmark-backends` command
5. **Accuracy reporting**: Track per-backend false accept/reject rates
