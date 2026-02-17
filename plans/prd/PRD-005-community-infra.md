# PRD-005: Community Infrastructure
**Priority:** P1
**Effort:** S
**Dependencies:** None

## Problem
No clear process for community contributions. Issues arrive without:
- Hardware details (camera model, resolution)
- Logs or error messages
- Steps to reproduce
- System information

Feature requests lack context on use case and priority. Contributors don't know:
- Where to start ("good first issue")
- Coding standards
- PR process
- What's planned vs rejected

This leads to:
- Low-quality bug reports requiring back-and-forth
- Duplicate feature requests
- Stale PRs that don't match project direction
- Missed contributions from people unsure where to help

## Requirements

### Must Have
- **GitHub issue templates**:
  - `bug_report.md` - Structured bug report with hardware info, logs, steps to reproduce
  - `feature_request.md` - Feature proposal with use case, alternatives considered
  - `hardware_report.md` - New hardware compatibility report (camera model, test results)
- **Pull request template** (`.github/PULL_REQUEST_TEMPLATE.md`):
  - What problem does this solve?
  - Testing done
  - Checklist (tests pass, docs updated)
- **ROADMAP.md** with:
  - Current phase/milestone
  - Planned features (P0/P1/P2)
  - Completed milestones
  - Long-term vision
- **Label guide** in `CONTRIBUTING.md`:
  - `good-first-issue` - Entry points for new contributors
  - `help-wanted` - Maintainer needs community help
  - `hardware` - Hardware-specific issues
  - `security` - Security-related issues (expedited)

### Should Have
- **CONTRIBUTING.md** with:
  - Development setup (`source .venv/bin/activate`)
  - Code style (black, pylint)
  - Testing expectations (`pytest tests/ -v`)
  - PR process (fork, branch, test, submit)
  - Commit message format
- **"Good first issue" identification**:
  - Tag 5-10 starter issues
  - Document in CONTRIBUTING.md
- **Hardware compatibility page** (docs/HARDWARE.md):
  - Table of tested devices
  - Process for submitting test results
  - Link to hardware_report issue template

### Nice to Have
- GitHub Discussions enabled for Q&A
- Project board for issue tracking
- Automated labels (github-actions/labeler)
- Issue triage bot (stale issue closer)
- PR checklist validation bot
- Community metrics dashboard
- Contributor recognition (all-contributors bot)

## Technical Approach

### File Structure
```
.github/
├── ISSUE_TEMPLATE/
│   ├── bug_report.md
│   ├── feature_request.md
│   └── hardware_report.md
├── PULL_REQUEST_TEMPLATE.md
└── workflows/
    └── label-issues.yml (optional)

ROADMAP.md
CONTRIBUTING.md
docs/HARDWARE.md
```

### Bug Report Template
```markdown
---
name: Bug Report
about: Report a problem with faceauth
labels: bug
---

## Bug Description
[Clear description of the issue]

## Hardware
- Device: [e.g., ThinkPad P14s Gen 5]
- Camera: [output of `v4l2-ctl --list-devices`]
- OS: [e.g., Ubuntu 24.04]
- Python: [output of `python3 --version`]

## Steps to Reproduce
1.
2.
3.

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Logs
```
[Output of `faceauth test` or daemon logs]
```

## Configuration
```toml
[Paste relevant config.toml sections]
```

## Additional Context
[Screenshots, etc.]
```

### Feature Request Template
```markdown
---
name: Feature Request
about: Suggest a new feature
labels: enhancement
---

## Problem
[What user problem does this solve?]

## Proposed Solution
[How should this work?]

## Alternatives Considered
[What other approaches did you consider?]

## Use Case
[When would you use this?]

## Additional Context
[Mockups, examples from other tools, etc.]
```

### Hardware Report Template
```markdown
---
name: Hardware Report
about: Report compatibility with new hardware
labels: hardware
---

## Hardware Details
- Device: [e.g., ThinkPad X1 Carbon Gen 11]
- Camera Model: [from `v4l2-ctl --info`]
- Device Path: [e.g., /dev/video2]
- Resolution: [e.g., 640x480]
- Pixel Format: [e.g., GREY, YUYV]

## Testing Results
- [ ] `faceauth test` - Camera preview works
- [ ] `faceauth enroll` - Enrollment succeeds
- [ ] `faceauth verify` - Verification works
- [ ] Authentication in GDM login works
- [ ] Anti-spoofing tested (photo/video spoof)

## Issues Encountered
[Any problems or quirks]

## Configuration
```toml
[Paste your working config.toml]
```

## Performance
- Auth latency: [time from camera on to result]
- Hardware acceleration: [GPU/NPU/CPU]
```

### Pull Request Template
```markdown
## Description
[What does this PR do?]

## Problem
[What problem does this solve? Link to issue if applicable]

## Changes
- [Bullet point summary of changes]

## Testing
- [ ] Unit tests pass (`pytest tests/`)
- [ ] Manual testing completed
- [ ] New tests added for new functionality
- [ ] Documentation updated

## Checklist
- [ ] Code follows project style (black, pylint clean)
- [ ] Commit messages are clear
- [ ] No debug/print statements left in code
- [ ] Config changes documented in docs/CONFIGURATION.md
```

### ROADMAP.md Structure
```markdown
# faceauth Roadmap

## Current Phase: Phase 5 - Community & Polish

### Completed
- [x] Phase 1: ML Pipeline + CLI
- [x] Phase 2: Daemon + PAM Module
- [x] Phase 3: Anti-Spoofing + IR
- [x] Phase 4: GDM Integration
- [x] Phase 5: OpenVINO GPU Acceleration

### In Progress (P0)
- [ ] PRD-001: Setup Wizard
- [ ] PRD-002: README Rewrite
- [ ] PRD-003: Camera Auto-Detection

### Planned (P1)
- [ ] PRD-004: Audit Logging
- [ ] PRD-005: Community Infrastructure
- [ ] Multi-face enrollment (handle glasses, beard, etc.)
- [ ] Brightness/exposure auto-adjustment

### Future (P2)
- [ ] Web UI for management
- [ ] LDAP/AD integration
- [ ] Multi-factor with face + PIN
- [ ] Mobile app for enrollment QR

### Long-term Vision
- Replace all password auth with face on Linux
- Enterprise-grade security and compliance
- Hardware vendor partnerships (pre-configured on laptops)
- Cross-platform (BSD, macOS)
```

### CONTRIBUTING.md Sections
1. Welcome
2. Code of Conduct link
3. Development Setup
4. Code Style
5. Testing
6. Pull Request Process
7. Issue Labels Guide
8. Good First Issues
9. Hardware Testing
10. Release Process (for maintainers)

## Success Criteria
- New contributor can file a bug report with all needed info using template
- Feature requests include use case and alternatives
- 5+ issues tagged "good first issue" with clear scope
- ROADMAP.md clearly shows what's planned and completed
- PR template ensures consistency (tests, docs, style)
- Community can self-serve answers from Discussions/FAQ
- Hardware compatibility reports follow standard format

Example "good first issue" tags:
- Add support for Y10/Y12 pixel formats (camera.py)
- Add `--version` flag to CLI
- Improve error message when camera not found
- Add unit test for config validation
- Document TOML config options in docs/

## Out of Scope
- Discord/Slack community channel
- Mailing list
- Documentation website (ReadTheDocs)
- Video tutorials or screencasts
- Internationalization (i18n) of templates
- Automated PR size/complexity checks
- Community call/office hours
- Maintainer team expansion process
