# ADR-002: Interactive Setup Wizard

**Status:** Accepted
**Date:** 2026-02-17

## Context

Current installation requires 8+ manual steps spread across documentation:

1. Install dependencies
2. Create virtual environment
3. Install faceauth package
4. Manually detect IR camera with v4l2-ctl
5. Create config directory
6. Write config.toml with correct values
7. Run enrollment for each user
8. Install systemd service (choose user vs system mode)
9. Enable and start service
10. Configure PAM (edit system files)
11. Test authentication

This is a significant barrier to adoption. Users report spending 30-60 minutes on initial setup, with common failure points:
- Wrong camera device selected
- Config file syntax errors
- Incorrect systemd service paths
- PAM misconfiguration leading to lockouts

Other modern Linux security tools (fprintd, pam-u2f) provide interactive setup. faceauth should match this standard.

## Decision

Add `faceauth setup` command that provides an interactive, step-by-step setup wizard. The wizard will:

### Phase 1: Hardware Detection
1. Scan for cameras using hardware.py (ADR-001)
2. Display detected IR cameras with names and resolutions
3. If multiple IR cameras found: prompt user to select
4. If no IR cameras found: offer RGB camera fallback or manual entry
5. Test camera by capturing a frame (show preview if possible)

### Phase 2: Configuration
6. Ask for deployment mode: user (default) or system
7. Write config.toml to appropriate location:
   - User mode: `~/.config/faceauth/config.toml`
   - System mode: `/etc/faceauth/config.toml`
8. Set reasonable defaults for anti-spoofing, confidence threshold
9. Display config path and offer to edit before continuing

### Phase 3: Enrollment
10. Prompt for username (default: $USER for user mode, require input for system mode)
11. Run enrollment with live feedback (face detection status, quality indicators)
12. Confirm embedding saved
13. Offer to enroll additional users (system mode)

### Phase 4: Daemon Installation
14. Install systemd service file:
    - User mode: `~/.config/systemd/user/faceauth.service`
    - System mode: `/etc/systemd/system/faceauth.service`
15. Run `systemctl --user enable faceauth` or `systemctl enable faceauth`
16. Start daemon
17. Verify daemon is running via socket connection

### Phase 5: PAM Integration (Optional)
18. Ask if user wants to configure PAM now
19. If yes: display instructions for editing /etc/pam.d/common-auth or /etc/pam.d/system-auth
20. Offer to create backup of PAM files
21. **WARNING**: Require second terminal/TTY test before closing initial shell
22. If no: provide documentation link

### Output
- Summary of what was configured
- Next steps (test with `faceauth verify USERNAME`)
- Troubleshooting tips
- Config file locations

### Safety Features
- All steps are idempotent (can re-run if interrupted)
- Validate each step before proceeding
- Create backups of any modified system files
- Provide rollback instructions
- Never auto-edit PAM files (too risky for lockout)
- Require explicit confirmation for system-wide installation

### Implementation Structure
```python
# faceauth/setup.py
class SetupWizard:
    def run(self):
        self.detect_hardware()
        self.configure()
        self.enroll()
        self.install_daemon()
        self.configure_pam()

    def detect_hardware(self):
        # Uses hardware.py from ADR-001

    def configure(self):
        # Interactive config generation

    # ... etc
```

CLI interface:
```bash
faceauth setup                    # Interactive (default)
faceauth setup --non-interactive  # Use all defaults, fail if ambiguous
faceauth setup --user             # Force user mode
faceauth setup --system           # Force system mode
faceauth setup --camera /dev/video2  # Skip detection
```

## Consequences

**Positive:**
- Reduces setup time from 30-60 minutes to 5-10 minutes
- Eliminates most common setup errors
- Provides guided experience for non-expert users
- Makes faceauth competitive with other PAM modules
- Reduces support burden (fewer "how do I install" questions)
- Creates proper foundation for future packaging (distro packages can run setup in postinstall)

**Negative:**
- Adds ~500-800 lines of setup-specific code
- Requires terminal interaction (not suitable for automated deployment)
- Must maintain compatibility as system APIs evolve
- More complex testing (requires mocking hardware detection, user input)

**User Experience:**
- Clear, friendly language (avoid jargon)
- Progress indicators for long-running steps
- Colorized output (warnings in yellow, errors in red, success in green)
- Pause points for user to verify (e.g., "Camera LED should be on now")

**Safety:**
- Never make destructive changes without confirmation
- Always provide rollback/undo instructions
- Special care around PAM (provide test procedure, require confirmation)
- Create logs of what was done

## Alternatives Considered

### 1. Shell script installer
**Rejected.** Not portable across distributions, harder to maintain, can't reuse existing Python code (camera detection, config generation), poor error handling.

### 2. Non-interactive only (all via flags)
**Rejected.** Misses the point - we want to guide users through decisions, not make them learn all the flags first. However, we will provide `--non-interactive` mode for automated deployments.

### 3. Web-based setup UI
**Rejected.** Massive scope increase, requires web server, authentication, port management. Wrong tool for system-level security configuration.

### 4. Ansible/Salt playbook
**Rejected.** Not all users use configuration management. Playbooks are complementary (power users can use them), but we need a built-in solution.

### 5. Keep current manual process, improve documentation
**Rejected.** Documentation can only go so far. Interactive setup is industry standard for PAM modules.

### 6. Fully automatic setup (no user interaction)
**Rejected.** Too many decisions require user input (which camera, which mode, which users to enroll). Automatic with no confirmation is dangerous for security tools.

## Implementation Phases

This ADR describes the full vision. Implementation can be phased:

**Phase 1 (MVP):**
- Hardware detection integration
- Config file generation
- Basic enrollment flow
- Systemd service installation

**Phase 2 (Polish):**
- Camera preview/test
- Multiple user enrollment
- Better progress indicators
- Colorized output

**Phase 3 (Advanced):**
- PAM configuration assistance (with strong safety rails)
- Rollback/uninstall wizard
- Setup verification/validation command

## Testing Strategy

- Unit tests for each wizard step (mocked I/O)
- Integration tests in Docker containers (various distros)
- Manual testing on physical hardware (ThinkPad, Dell, HP)
- Failure injection tests (what if camera is unplugged mid-setup?)
- Idempotency tests (run setup twice, verify no corruption)

## Documentation Impact

- README.md: Simplify to "Run `faceauth setup`"
- INSTALL.md: Keep detailed manual steps for reference/troubleshooting
- Add SETUP.md: Wizard walkthrough with screenshots
- FAQ: Add "What if setup fails at step X?" section
