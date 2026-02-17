# PRD-001: Setup Wizard
**Priority:** P0
**Effort:** M
**Dependencies:** PRD-003 (camera auto-detection)

## Problem
Installation currently requires 8+ manual steps (detect camera, edit config, enroll face, create systemd service, enable service, configure PAM, etc.). Users unfamiliar with systemd or PAM give up before completing setup. The barrier to entry is too high for a security tool that should "just work."

## Requirements

### Must Have
- `faceauth setup` command that runs interactive setup wizard
- Auto-detect IR camera device path and resolution (depends on PRD-003)
- Generate config.toml with detected camera settings
- Enroll current user's face with live camera feedback
- Install systemd service (user or system mode based on --system flag)
- Enable and start the daemon service
- Verify enrollment works before finishing

### Should Have
- PAM configuration installation (requires root, skip with warning if not root)
- Test authentication after enrollment (verify round-trip)
- Rollback on failure (remove partial config/service files)
- `--system` flag for system-wide daemon setup
- `--unattended` flag for scripted deployments (accept defaults, skip confirmation)

### Nice to Have
- `faceauth uninstall` command to undo setup (remove service, PAM config, embeddings)
- Setup progress indicator (step X of Y)
- Validate hardware compatibility before starting (check V4L2, IR camera exists)
- Option to import existing embeddings from Howdy

## Technical Approach

### Implementation Plan
1. **New CLI command** in `cli/faceauth_cli.py`:
   - `setup` subcommand with `--system` and `--unattended` flags
   - Use `hardware.py` for camera detection (PRD-003)
   - Call existing `config.py` to write config.toml
   - Call existing `enroll` logic for face enrollment
   - Call existing daemon installation from `daemon.py`

2. **Setup workflow**:
   ```
   1. Check prerequisites (V4L2 available, user has camera access)
   2. Detect IR cameras -> select best (highest resolution)
   3. Generate config.toml with camera settings
   4. Start temporary daemon for enrollment
   5. Run enrollment with live feedback
   6. Install systemd service (user or system mode)
   7. Enable and start service
   8. [Optional] Configure PAM (if root)
   9. Test verification against daemon
   10. Print success message with next steps
   ```

3. **Error handling**:
   - Validate each step before proceeding
   - Rollback on failure (remove created files)
   - Clear error messages with remediation steps

4. **Files modified**:
   - `cli/faceauth_cli.py` - new setup command
   - `faceauth/setup.py` - new module with SetupWizard class
   - `faceauth/hardware.py` - camera detection (PRD-003)

### PAM Installation
- Detect PAM configuration directory (`/etc/pam.d/`)
- For GDM: prepend `auth sufficient pam_exec.so quiet /path/to/faceauth-pam-helper` to `/etc/pam.d/gdm-password`
- For sudo: similar for `/etc/pam.d/sudo`
- Backup original files before modification
- Verify faceauth-pam-helper is installed and executable

## Success Criteria
- User runs `pip install faceauth && faceauth setup`
- Wizard completes in <2 minutes
- Face authentication works on next GDM login
- Zero manual configuration required for standard case (ThinkPad with IR camera)
- Setup can be re-run safely (idempotent)
- Clear error messages if hardware incompatible

## Out of Scope
- GUI installer (GTK/Qt application)
- Web-based configuration interface
- Automatic updates or self-update mechanism
- Multi-user enrollment in one command (each user runs setup individually)
- Windows/macOS support
- Integration with non-PAM authentication systems
