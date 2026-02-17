"""faceauth CLI - enroll, verify, test, list, delete faces."""

import logging
import os
import sys
import time

import click
import cv2

from faceauth.camera import Camera, ir_to_rgb, is_ir_frame
from faceauth.config import load_config
from faceauth.pam_client import daemon_status, get_socket_path
from faceauth.pipeline import (
    ATTEMPT_MULTIPLIER,
    FRAME_DELAY_ENROLL,
    FRAME_DELAY_VERIFY,
    MIN_SELF_CONSISTENCY,
    check_antispoof,
    compute_self_consistency,
    make_antispoof,
    match_embedding,
    process_frame,
    select_best_face,
)
from faceauth.recognizer import FaceRecognizer
from faceauth.storage import EmbeddingStore


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(package_name="faceauth")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Config file path")
@click.pass_context
def cli(ctx, verbose, config_path):
    """faceauth - Modern Linux face authentication."""
    setup_logging(verbose)
    from pathlib import Path

    cfg = load_config(Path(config_path) if config_path else None)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("username")
@click.option("--samples", default=5, help="Number of face samples to capture")
@click.option("--camera", "camera_device", default=None, help="Camera device override")
@click.pass_context
def enroll(ctx, username, samples, camera_device):
    """Enroll a face for USERNAME."""
    cfg = ctx.obj["config"]
    device = camera_device or cfg.camera.ir_device

    click.echo(f"Enrolling face for '{username}' using {device}")
    click.echo(f"Capturing {samples} samples. Look at the camera...")

    recognizer = FaceRecognizer(model_name=cfg.recognition.model)
    store = EmbeddingStore()
    antispoof = make_antispoof(cfg)

    if antispoof:
        fas_label = "MiniFASNet" if antispoof.minifasnet_available else "IR-only"
        click.echo(f"Anti-spoof: enabled ({fas_label})")
    click.echo()

    embeddings = []
    spoof_rejects = 0

    with Camera(device, cfg.camera.width, cfg.camera.height) as cam:
        attempt = 0
        max_attempts = samples * ATTEMPT_MULTIPLIER

        while len(embeddings) < samples and attempt < max_attempts:
            attempt += 1
            try:
                frame = cam.read()
            except RuntimeError:
                continue

            raw_ir, bgr_frame = process_frame(frame)

            faces = recognizer.get_faces(bgr_frame)
            if not faces:
                if attempt % 5 == 0:
                    click.echo("  ... no face detected, keep looking at the camera")
                continue

            best = select_best_face(faces)
            if best is None:
                continue

            spoof_result = check_antispoof(antispoof, raw_ir, bgr_frame, best.bbox)
            if spoof_result is not None and not spoof_result.passed:
                spoof_rejects += 1
                click.echo(f"  [SPOOF] {spoof_result.reason}", err=True)
                continue

            embeddings.append(best.embedding)
            click.echo(f"  [{len(embeddings)}/{samples}] Face captured")
            time.sleep(FRAME_DELAY_ENROLL)

    if not embeddings:
        click.echo("Failed to capture any faces. Check camera and lighting.", err=True)
        if spoof_rejects > 0:
            click.echo(f"  ({spoof_rejects} frames rejected as spoofed)", err=True)
        sys.exit(1)

    if len(embeddings) < samples:
        click.echo(f"Warning: only captured {len(embeddings)}/{samples} samples", err=True)

    if len(embeddings) > 1:
        avg_score, min_score = compute_self_consistency(embeddings)
        click.echo(f"\n  Self-consistency: avg={avg_score:.3f} min={min_score:.3f}")
        if min_score < MIN_SELF_CONSISTENCY:
            click.echo(
                "Warning: low self-consistency. Multiple faces may have been captured.",
                err=True,
            )

    store.save(username, embeddings)
    click.echo(f"\nEnrolled '{username}' with {len(embeddings)} sample(s)")


@cli.command()
@click.argument("username")
@click.option("--camera", "camera_device", default=None, help="Camera device override")
@click.option("--threshold", default=None, type=float, help="Similarity threshold override")
@click.pass_context
def verify(ctx, username, camera_device, threshold):
    """Verify a face against stored enrollment for USERNAME."""
    cfg = ctx.obj["config"]
    device = camera_device or cfg.camera.ir_device
    thresh = threshold if threshold is not None else cfg.recognition.similarity_threshold

    store = EmbeddingStore()
    stored = store.load(username)
    if not stored:
        click.echo(f"User '{username}' is not enrolled.", err=True)
        sys.exit(1)

    click.echo(f"Verifying '{username}' (threshold={thresh:.2f}, device={device})")
    antispoof = make_antispoof(cfg)
    if antispoof:
        fas_label = "MiniFASNet" if antispoof.minifasnet_available else "IR-only"
        click.echo(f"Anti-spoof: enabled ({fas_label})")
    click.echo("Look at the camera...")

    recognizer = FaceRecognizer(model_name=cfg.recognition.model)

    with Camera(device, cfg.camera.width, cfg.camera.height) as cam:
        for attempt in range(cfg.recognition.max_attempts):
            try:
                frame = cam.read()
            except RuntimeError:
                continue

            raw_ir, bgr_frame = process_frame(frame)

            faces = recognizer.get_faces(bgr_frame)
            if not faces:
                time.sleep(FRAME_DELAY_VERIFY)
                continue

            best = select_best_face(faces)
            if best is None:
                time.sleep(FRAME_DELAY_VERIFY)
                continue

            spoof_result = check_antispoof(antispoof, raw_ir, bgr_frame, best.bbox)
            if spoof_result is not None and not spoof_result.passed:
                click.echo(f"  [SPOOF] {spoof_result.reason}")
                click.echo("  FAIL - spoof detected")
                sys.exit(1)

            best_score = match_embedding(best.embedding, stored)

            if best_score >= thresh:
                click.echo(f"  MATCH (score={best_score:.3f})")
                sys.exit(0)
            elif best_score > 0:
                click.echo(f"  attempt {attempt + 1}: score={best_score:.3f} (below threshold)")
            time.sleep(FRAME_DELAY_VERIFY)

    click.echo("  FAIL - face not recognised")
    sys.exit(1)


@cli.command()
@click.option("--camera", "camera_device", default=None, help="Camera device override")
@click.pass_context
def test(ctx, camera_device):
    """Show live camera feed with face detection and anti-spoof overlay.

    Press 'q' to quit.
    """
    cfg = ctx.obj["config"]
    device = camera_device or cfg.camera.ir_device

    click.echo(f"Opening camera test: {device}")
    click.echo("Press 'q' to quit")

    recognizer = FaceRecognizer(model_name=cfg.recognition.model)
    antispoof = make_antispoof(cfg)
    if antispoof:
        fas_label = "MiniFASNet" if antispoof.minifasnet_available else "IR-only"
        click.echo(f"Anti-spoof: enabled ({fas_label})")

    with Camera(device, cfg.camera.width, cfg.camera.height) as cam:
        while True:
            try:
                frame = cam.read()
            except RuntimeError:
                continue

            display = frame.copy()
            is_ir = is_ir_frame(frame)
            raw_ir = frame.copy() if is_ir else None
            detect_frame = ir_to_rgb(frame) if is_ir else frame

            faces = recognizer.get_faces(detect_frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                score = face.det_score

                # Anti-spoof overlay
                spoof_text = ""
                if antispoof is not None and raw_ir is not None:
                    spoof_result = antispoof.check(raw_ir, detect_frame, face.bbox)
                    if spoof_result.passed:
                        colour = (0, 255, 0)  # Green = live
                        spoof_text = f"LIVE IR:{spoof_result.ir_brightness:.0f}"
                        if spoof_result.liveness_score > 0:
                            spoof_text += f" FAS:{spoof_result.liveness_score:.2f}"
                    else:
                        colour = (0, 0, 255)  # Red = spoof
                        spoof_text = f"SPOOF IR:{spoof_result.ir_brightness:.0f}"
                        if spoof_result.liveness_score > 0:
                            spoof_text += f" FAS:{spoof_result.liveness_score:.2f}"
                else:
                    colour = (0, 255, 0) if score > 0.7 else (0, 255, 255)

                cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    display,
                    f"det:{score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    1,
                )

                # Anti-spoof text below bbox
                if spoof_text:
                    cv2.putText(
                        display,
                        spoof_text,
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        1,
                    )

                if face.landmark is not None:
                    for pt in face.landmark:
                        cv2.circle(display, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

            # Info overlay
            ir_label = "IR" if is_ir else "RGB"
            h, w = display.shape[:2]
            info = f"{ir_label} {w}x{h} | {len(faces)} face(s)"
            if antispoof:
                info += " | antispoof ON"
            cv2.putText(
                display,
                info,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("faceauth test", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    click.echo("Done")


@cli.command(name="list")
@click.pass_context
def list_users(ctx):
    """List enrolled users."""
    store = EmbeddingStore()
    users = store.list_users()

    if not users:
        click.echo("No users enrolled")
        return

    click.echo(f"Enrolled users ({len(users)}):")
    for user in users:
        count = store.get_embedding_count(user)
        click.echo(f"  {user} ({count} sample{'s' if count != 1 else ''})")


@cli.command()
@click.argument("username")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx, username, force):
    """Delete stored face data for USERNAME."""
    store = EmbeddingStore()

    if not store.is_enrolled(username):
        click.echo(f"User '{username}' is not enrolled.", err=True)
        sys.exit(1)

    if not force:
        click.confirm(f"Delete face data for '{username}'?", abort=True)

    store.delete(username)
    click.echo(f"Deleted face data for '{username}'")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status - daemon, camera, models, enrolled users."""
    cfg = ctx.obj["config"]

    click.echo("faceauth status")
    click.echo("=" * 40)

    # Check daemon
    ds = daemon_status()
    if ds:
        click.echo(f"  Daemon: RUNNING (pid={ds.get('pid', '?')}, socket={ds.get('socket', '?')})")
        click.echo(f"    Models loaded: {ds.get('models_loaded', False)}")
    else:
        sock = get_socket_path()
        click.echo(f"  Daemon: NOT RUNNING (expected at {sock})")

    # Check cameras
    click.echo()
    for label, device in [("IR", cfg.camera.ir_device), ("RGB", cfg.camera.rgb_device)]:
        try:
            with Camera(device) as cam:
                frame = cam.read()
                h, w = frame.shape[:2]
                is_ir = is_ir_frame(frame)
                click.echo(f"  {label} camera ({device}): OK ({w}x{h}, {'IR' if is_ir else 'RGB'})")
        except Exception as e:
            click.echo(f"  {label} camera ({device}): FAIL ({e})")

    # Check model
    click.echo()
    try:
        recognizer = FaceRecognizer(model_name=cfg.recognition.model)
        recognizer.ensure_loaded()
        click.echo(f"  Model ({cfg.recognition.model}): OK")
    except Exception as e:
        click.echo(f"  Model ({cfg.recognition.model}): FAIL ({e})")

    # Anti-spoof status
    click.echo()
    asc = cfg.antispoof
    if asc.enabled:
        click.echo("  Anti-spoof: ENABLED")
        click.echo(f"    IR brightness min: {asc.ir_brightness_min}")
        click.echo(f"    MiniFASNet threshold: {asc.minifasnet_threshold}")
        click.echo(f"    Require both: {asc.require_both}")
        click.echo(f"    IR-only fallback: {asc.ir_only_fallback}")
        antispoof = make_antispoof(cfg)
        if antispoof.minifasnet_available:
            click.echo(f"    MiniFASNet model: OK ({antispoof.model_path})")
        else:
            click.echo(f"    MiniFASNet model: NOT FOUND ({antispoof.model_path})")
            if asc.ir_only_fallback:
                click.echo("    (using IR-only fallback)")
            else:
                click.echo("    WARNING: anti-spoof will reject all frames!", err=True)
    else:
        click.echo("  Anti-spoof: DISABLED")

    # Enrolled users
    click.echo()
    store = EmbeddingStore()
    users = store.list_users()
    click.echo(f"  Enrolled users: {len(users)}")
    for user in users:
        count = store.get_embedding_count(user)
        click.echo(f"    - {user} ({count} samples)")

    click.echo()
    click.echo(f"  Data dir: {store.data_dir}")
    click.echo(f"  Config: {cfg.config_dir / 'config.toml'}")
    click.echo(f"  Similarity threshold: {cfg.recognition.similarity_threshold}")


@cli.group()
def daemon():
    """Manage the faceauth daemon."""
    pass


@daemon.command(name="start")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't use systemd)")
@click.pass_context
def daemon_start(ctx, foreground):
    """Start the faceauth daemon."""
    if foreground:
        from faceauth.daemon import run_daemon

        click.echo("Starting daemon in foreground (Ctrl+C to stop)...")
        run_daemon()
    else:
        import subprocess

        result = subprocess.run(
            ["systemctl", "--user", "start", "faceauth"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            click.echo("Daemon started via systemd")
        else:
            click.echo(f"Failed to start daemon: {result.stderr.strip()}", err=True)
            click.echo("Try: faceauth daemon start --foreground", err=True)
            sys.exit(1)


@daemon.command(name="stop")
def daemon_stop():
    """Stop the faceauth daemon."""
    import subprocess

    result = subprocess.run(
        ["systemctl", "--user", "stop", "faceauth"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        click.echo("Daemon stopped")
    else:
        click.echo(f"Failed to stop daemon: {result.stderr.strip()}", err=True)
        sys.exit(1)


@daemon.command(name="status")
def daemon_status_cmd():
    """Show daemon status."""
    ds = daemon_status()
    if ds:
        click.echo("Daemon: RUNNING")
        click.echo(f"  PID: {ds.get('pid', '?')}")
        click.echo(f"  Socket: {ds.get('socket', '?')}")
        click.echo(f"  Models loaded: {ds.get('models_loaded', False)}")
        click.echo(f"  Enrolled users: {ds.get('enrolled_users', 0)}")
    else:
        click.echo("Daemon: NOT RUNNING")
        sys.exit(1)


@daemon.command(name="install")
@click.option("--system", is_flag=True, help="Install system-level service (requires root)")
def daemon_install(system):
    """Install the systemd service file."""
    from pathlib import Path

    if system:
        src = Path(__file__).resolve().parent.parent / "systemd" / "faceauth-system.service"
        dest = Path("/etc/systemd/system/faceauth.service")

        if os.geteuid() != 0:
            click.echo("System install requires root. Run with sudo.", err=True)
            sys.exit(1)

        if not src.exists():
            click.echo(f"Service file not found: {src}", err=True)
            sys.exit(1)

        # Template the Python path into the service file
        # Don't .resolve() — venv python is a symlink to system python,
        # but must be invoked via venv path to get the right site-packages
        python_path = Path(sys.executable)
        content = src.read_text()
        content = content.replace("FACEAUTH_PYTHON", str(python_path))
        dest.write_text(content)
        os.chmod(dest, 0o644)
        click.echo(f"Installed: {dest}")
        click.echo(f"  Python: {python_path}")

        import subprocess

        subprocess.run(["systemctl", "daemon-reload"], check=False)
        click.echo("Run 'sudo systemctl start faceauth' to start the daemon")
        click.echo("Run 'sudo systemctl enable faceauth' to start on boot")
    else:
        src = Path(__file__).resolve().parent.parent / "systemd" / "faceauth.service"
        dest_dir = Path.home() / ".config/systemd/user"
        dest = dest_dir / "faceauth.service"

        if not src.exists():
            click.echo(f"Service file not found: {src}", err=True)
            sys.exit(1)

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Template the Python path into the service file
        python_path = Path(sys.executable)
        content = src.read_text()
        content = content.replace("FACEAUTH_PYTHON", str(python_path))
        dest.write_text(content)
        click.echo(f"Installed: {dest}")
        click.echo(f"  Python: {python_path}")

        import subprocess

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        click.echo("Run 'faceauth daemon start' to start the daemon")
        click.echo("Run 'systemctl --user enable faceauth' to start on login")


@cli.command(name="install-pam")
def install_pam():
    """Install PAM helper and pam-auth-update profile (requires root)."""
    import shutil
    from pathlib import Path

    if os.geteuid() != 0:
        click.echo("PAM install requires root. Run with sudo.", err=True)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent

    # Install PAM helper
    helper_src = project_root / "pam" / "faceauth-pam-helper"
    helper_dest = Path("/usr/local/lib/faceauth/faceauth-pam-helper")

    if not helper_src.exists():
        click.echo(f"PAM helper not found: {helper_src}", err=True)
        sys.exit(1)

    helper_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(helper_src, helper_dest)
    os.chmod(helper_dest, 0o755)
    click.echo(f"Installed: {helper_dest}")

    # Install pam-auth-update profile
    profile_src = project_root / "pam" / "pam-configs-faceauth"
    profile_dest = Path("/usr/share/pam-configs/faceauth")

    if not profile_src.exists():
        click.echo(f"PAM profile not found: {profile_src}", err=True)
        sys.exit(1)

    shutil.copy2(profile_src, profile_dest)
    click.echo(f"Installed: {profile_dest}")

    click.echo()
    click.echo("To enable face authentication:")
    click.echo("  sudo pam-auth-update")
    click.echo("  (select 'Face authentication (faceauth)')")


@cli.command(name="setup-models")
@click.pass_context
def setup_models(ctx):
    """Download required ML models."""
    click.echo("Setting up faceauth models...")
    click.echo()

    # InsightFace models auto-download on first use
    click.echo("[1/2] InsightFace (buffalo_l):")
    click.echo("  NOTE: buffalo_l model weights are for non-commercial research only.")
    click.echo("  Commercial use requires a license from InsightFace.")
    click.echo("  See: https://github.com/deepinsight/insightface")
    click.echo("  Auto-downloads on first use. Triggering load...")
    try:
        from faceauth.recognizer import FaceRecognizer

        cfg = ctx.obj["config"]
        recognizer = FaceRecognizer(model_name=cfg.recognition.model)
        recognizer.ensure_loaded()
        click.echo("  OK - model loaded")
    except Exception as e:
        click.echo(f"  WARN - {e}")
        click.echo("  (will retry on first use)")

    # MiniFASNet needs manual download
    click.echo()
    click.echo("[2/2] MiniFASNet anti-spoof (optional):")
    from faceauth.antispoof import _default_model_path

    model_path = _default_model_path()
    if model_path.exists():
        click.echo(f"  OK - already at {model_path}")
    else:
        click.echo(f"  NOT FOUND at {model_path}")
        click.echo("  To enable liveness detection, place a MiniFASNet ONNX model at:")
        click.echo(f"    {model_path}")
        click.echo("  Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing")
        click.echo("  License: Apache 2.0 (Minivision)")
        click.echo("  Without it, anti-spoof falls back to IR brightness checks only.")

    click.echo()
    click.echo("Setup complete.")


@cli.command(name="migrate-to-system")
@click.option("--force", is_flag=True, help="Overwrite existing system embeddings")
def migrate_to_system(force):
    """Copy user embeddings to system location for GDM auth."""
    import shutil
    from pathlib import Path

    # Under sudo, use the original user's home (not /root)
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        import pwd

        user_home = Path(pwd.getpwnam(sudo_user).pw_dir)
    else:
        user_home = Path.home()

    user_dir = Path(os.environ.get("XDG_DATA_HOME", user_home / ".local/share")) / "faceauth"
    system_dir = Path("/var/lib/faceauth")

    if not user_dir.exists():
        click.echo(f"No user embeddings found at {user_dir}", err=True)
        sys.exit(1)

    npz_files = list(user_dir.glob("*.npz"))
    if not npz_files:
        click.echo("No enrolled users found", err=True)
        sys.exit(1)

    if os.geteuid() != 0:
        click.echo("Migration requires root. Run with sudo.", err=True)
        sys.exit(1)

    system_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(system_dir, 0o700)

    copied = 0
    for src in npz_files:
        dest = system_dir / src.name
        if dest.exists() and not force:
            click.echo(f"  SKIP {src.stem} (already exists, use --force to overwrite)")
            continue
        shutil.copy2(src, dest)
        os.chmod(dest, 0o600)
        click.echo(f"  Copied {src.stem}")
        copied += 1

    click.echo(f"\nMigrated {copied} user(s) to {system_dir}")


if __name__ == "__main__":
    cli()
