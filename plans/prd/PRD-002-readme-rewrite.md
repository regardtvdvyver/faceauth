# PRD-002: README Rewrite
**Priority:** P0
**Effort:** S
**Dependencies:** PRD-001 (setup wizard for quickstart)

## Problem
Current README doesn't sell the project. No compelling hero section, no comparison to alternatives, assumes reader already decided to use it. First-time visitors don't understand:
- **Why** this exists (what problem it solves)
- **Why** this vs Howdy (the established competitor)
- **Whether** it's production-ready and secure

The README is structured like internal documentation, not a landing page.

## Requirements

### Must Have
- **Hero section** with clear value prop: "Face authentication for Linux, like Windows Hello"
- **Comparison table** vs Howdy highlighting key advantages:
  - IR camera support (Howdy: RGB only)
  - Anti-spoofing (Howdy: none)
  - Daemon architecture (Howdy: loads models per auth)
  - Active maintenance (Howdy: unmaintained since 2022)
  - Clean dependencies (Howdy: dlib compilation pain)
- **Simplified quick start** pointing to `faceauth setup` (1 command)
- **Security/Trust section** addressing:
  - Embeddings stored locally, never transmitted
  - Anti-spoofing methodology (MiniFASNet IR)
  - PAM integration security model
  - Open source, auditable code
- **Hardware requirements** clearly stated upfront

### Should Have
- Badge header showing build status, Python version, license
- Architecture diagram (keep existing mermaid diagram, improve placement)
- Hardware compatibility table (tested devices)
- Link to detailed documentation (INSTALLATION.md, ARCHITECTURE.md)
- FAQ section covering common questions
- Contributing guidelines link

### Nice to Have
- Demo GIF showing face auth in action (requires screen recording setup)
- ASCII art logo for terminal aesthetics
- Performance comparison numbers (auth latency vs Howdy)
- Community section (contributors, Discord/GitHub Discussions)

## Technical Approach

### README Structure
```markdown
# faceauth

[Badges: build status, Python 3.9+, license, stars]

Face authentication for Linux. Like Windows Hello, but open source.

[Demo GIF if available]

## Why faceauth?

- IR camera support with anti-spoofing
- Fast: Daemon keeps models loaded (< 1s auth)
- Secure: Local embeddings, PAM integration
- Modern: Hardware acceleration (OpenVINO GPU/NPU)
- Maintained: Active development, tested on recent hardware

### vs Howdy
| Feature | faceauth | Howdy |
|---------|----------|-------|
| IR cameras | ✓ | ✗ |
| Anti-spoofing | ✓ (MiniFASNet) | ✗ |
| Speed | <1s (daemon) | 2-4s (load per auth) |
| Dependencies | pip install | dlib compile |
| Maintenance | Active | Archived (2022) |

## Quick Start

```bash
pip install faceauth
faceauth setup  # One command setup
```

Detailed: [INSTALLATION.md](docs/INSTALLATION.md)

## Hardware Requirements

- IR camera (preferred) or RGB camera
- Tested: ThinkPad P14s Gen 5 (640x360 IR)
- See [hardware compatibility](docs/HARDWARE.md)

## Security

[Trust section content]

## How It Works

[Architecture diagram]

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [FAQ](docs/FAQ.md)

## Contributing

[Link to CONTRIBUTING.md]

## License

[License info]
```

### Content Changes
- Move technical implementation details to docs/ARCHITECTURE.md
- Move step-by-step install to docs/INSTALLATION.md
- Keep README focused on "why" and "what", not "how"
- Use tables and visual hierarchy for scannability

## Success Criteria
- First-time visitor understands what this is in 5 seconds
- Clear answer to "why not Howdy?" visible above the fold
- Quick start shows path from zero to working in <5 minutes
- Security concerns addressed proactively
- README length < 300 lines (move details to docs/)

## Out of Scope
- Full documentation website (ReadTheDocs/MkDocs)
- Video tutorials
- Blog post or launch announcement
- Translation to other languages
- Marketing materials beyond GitHub README
