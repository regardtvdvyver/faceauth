# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in faceauth, please report it responsibly.

**Email**: [faceauth@vdvyver.net](mailto:faceauth@vdvyver.net)

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

I will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days.

**Do not** open a public GitHub issue for security vulnerabilities.

## Security Design

faceauth is designed with defence-in-depth for a local authentication system:

### Embedding Storage
- Biometric embeddings stored as `.npz` files with `0o600` permissions (owner-only read/write)
- Storage directory set to `0o700`
- Username validation prevents path traversal attacks

### Daemon Authentication
- `SO_PEERCRED` on the Unix socket verifies the UID of connecting processes
- Only root or the matching user can enroll/delete for a given username
- Socket permissions: `0o600` in user mode, `0o666` in system mode (local access only)

### PAM Helper
- The PAM helper (`faceauth-pam-helper`) uses **stdlib only** -- no third-party dependencies in the authentication path
- Communicates with the daemon over Unix socket (no network exposure)

### Anti-Spoofing
- **IR brightness check**: Real faces reflect IR light from the emitter; photos and screens appear dark
- **MiniFASNet liveness** (optional): ONNX model detecting print/screen texture artifacts
- Configurable policy: require both checks, either, or IR-only fallback

## Threat Model

faceauth protects against:
- **Photo/screen replay attacks** via IR brightness and optional MiniFASNet liveness
- **Unauthorised enrollment/deletion** via SO_PEERCRED UID verification
- **Path traversal** via strict username validation and resolved-path checks

faceauth does **not** protect against:
- **Adversarial ML attacks** (crafted inputs designed to fool the embedding model)
- **Physical 3D masks** (requires depth sensing hardware not present in standard IR cameras)
- **Root-level attackers** (root can access embeddings and the socket directly)

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes      |
