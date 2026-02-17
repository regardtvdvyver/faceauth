"""faceauth daemon - persistent process with models loaded in RAM.

Listens on a Unix domain socket for verify/enroll/delete/status/list requests.
Designed to be run as a systemd user service.
"""

import asyncio
import logging
import os
import signal
import socket as socket_module
import struct
import time
from pathlib import Path

import numpy as np

from .antispoof import AntispoofChecker
from .camera import Camera
from .config import Config, load_config
from .pipeline import (
    ATTEMPT_MULTIPLIER,
    FRAME_DELAY_ENROLL,
    FRAME_DELAY_VERIFY,
    check_antispoof,
    compute_self_consistency,
    make_antispoof,
    match_embedding,
    process_frame,
    select_best_face,
)
from .protocol import Request, Response
from .recognizer import FaceRecognizer
from .storage import EmbeddingStore

log = logging.getLogger(__name__)


class FaceAuthDaemon:
    """Async Unix socket server for face authentication."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.recognizer: FaceRecognizer | None = None
        self.antispoof: AntispoofChecker | None = None
        self.store = EmbeddingStore()
        self._server: asyncio.Server | None = None
        self._socket_path: Path | None = None
        self._lock = asyncio.Lock()  # Serialize camera access

    @property
    def socket_path(self) -> Path:
        if self._socket_path is not None:
            return self._socket_path
        if self.config.daemon.system_mode:
            return Path(self.config.daemon.system_socket_path)
        uid = os.getuid()
        raw = self.config.daemon.socket_path.replace("{uid}", str(uid))
        return Path(raw)

    def _ensure_models(self):
        """Lazy-load models on first request."""
        if self.recognizer is None:
            # Prime the provider cache with the configured device before any model loads
            from .providers import get_ort_providers

            device = self.config.recognition.openvino_device or None
            get_ort_providers(device=device)

            log.info("Loading ML models...")
            self.recognizer = FaceRecognizer(model_name=self.config.recognition.model)
            self.recognizer.ensure_loaded()
            log.info("InsightFace model loaded")

        if self.antispoof is None and self.config.antispoof.enabled:
            self.antispoof = make_antispoof(self.config)
            fas_status = "available" if self.antispoof.minifasnet_available else "not found (IR-only)"
            log.info("Anti-spoof enabled (MiniFASNet: %s)", fas_status)

    @staticmethod
    def _get_peer_uid(writer: asyncio.StreamWriter) -> int | None:
        """Get the UID of the connecting process via SO_PEERCRED (Linux only)."""
        try:
            sock = writer.get_extra_info('socket')
            if sock is None:
                return None
            creds = sock.getsockopt(
                socket_module.SOL_SOCKET,
                socket_module.SO_PEERCRED,
                struct.calcsize('3i')
            )
            _pid, uid, _gid = struct.unpack('3i', creds)
            return uid
        except (OSError, AttributeError, struct.error):
            return None

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single client connection."""
        peer = writer.get_extra_info("peername") or "unknown"
        log.debug("Client connected: %s", peer)

        try:
            data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not data:
                return

            peer_uid = self._get_peer_uid(writer)

            try:
                req = Request.from_json(data)
            except Exception as e:
                log.warning("Invalid request from peer_uid=%s: %s", peer_uid, e)
                resp = Response(ok=False, error="invalid request")
                writer.write(resp.to_json())
                await writer.drain()
                return

            log.info("Request: action=%s username=%s peer_uid=%s", req.action, req.username, peer_uid)
            resp = await self._dispatch(req, peer_uid=peer_uid)

            writer.write(resp.to_json())
            await writer.drain()
            log.info("Response: ok=%s score=%.3f", resp.ok, resp.score)

        except asyncio.TimeoutError:
            log.warning("Client timed out")
        except ConnectionResetError:
            log.debug("Client disconnected")
        except Exception as e:
            log.exception("Error handling client: %s", e)
            try:
                resp = Response(ok=False, error="internal error")
                writer.write(resp.to_json())
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch(self, req: Request, peer_uid: int | None = None) -> Response:
        """Route request to the appropriate handler."""
        # Authorization: enroll/delete require root or matching UID
        if req.action in ("enroll", "delete") and peer_uid is not None:
            if peer_uid != 0:
                try:
                    import pwd
                    target_uid = pwd.getpwnam(req.username).pw_uid
                except KeyError:
                    return Response(ok=False, error="unknown user")
                if peer_uid != target_uid:
                    log.warning(
                        "Authorization denied: uid=%d tried to %s user '%s' (uid=%d)",
                        peer_uid, req.action, req.username, target_uid,
                    )
                    return Response(ok=False, error="permission denied")

        handlers = {
            "verify": self._handle_verify,
            "enroll": self._handle_enroll,
            "delete": self._handle_delete,
            "status": self._handle_status,
            "list": self._handle_list,
        }

        handler = handlers.get(req.action)
        if handler is None:
            return Response(ok=False, error=f"Unknown action: {req.action}")

        try:
            return await handler(req)
        except Exception as e:
            log.exception("Handler error for action=%s: %s", req.action, e)
            return Response(ok=False, error="internal error")

    async def _handle_verify(self, req: Request) -> Response:
        """Verify a face against stored embeddings."""
        if not req.username:
            return Response(ok=False, error="username required")

        stored = self.store.load(req.username)
        if not stored:
            return Response(ok=False, error=f"User '{req.username}' not enrolled")

        self._ensure_models()
        # Server-side security policy: always use configured values
        threshold = self.config.recognition.similarity_threshold
        device = self.config.camera.ir_device

        async with self._lock:
            match, score = await asyncio.get_event_loop().run_in_executor(
                None, self._verify_sync, device, stored, threshold
            )

        return Response(ok=match, score=score)

    def _verify_sync(
        self, device: str, stored: list[np.ndarray], threshold: float
    ) -> tuple[bool, float]:
        """Synchronous verification (runs in executor)."""
        with Camera(device, self.config.camera.width, self.config.camera.height) as cam:
            for attempt in range(self.config.recognition.max_attempts):
                try:
                    frame = cam.read()
                except RuntimeError:
                    continue

                raw_ir, bgr_frame = process_frame(frame)

                faces = self.recognizer.get_faces(bgr_frame)
                if not faces:
                    time.sleep(FRAME_DELAY_VERIFY)
                    continue

                best = select_best_face(faces)
                if best is None:
                    time.sleep(FRAME_DELAY_VERIFY)
                    continue

                spoof_result = check_antispoof(self.antispoof, raw_ir, bgr_frame, best.bbox)
                if spoof_result is not None and not spoof_result.passed:
                    log.warning("Spoof rejected: %s", spoof_result.reason)
                    return False, 0.0

                best_score = match_embedding(best.embedding, stored)

                if best_score >= threshold:
                    return True, best_score
                time.sleep(FRAME_DELAY_VERIFY)

        return False, 0.0

    async def _handle_enroll(self, req: Request) -> Response:
        """Enroll a face."""
        if not req.username:
            return Response(ok=False, error="username required")

        self._ensure_models()
        device = self.config.camera.ir_device
        samples = req.samples

        async with self._lock:
            embeddings, consistency = await asyncio.get_event_loop().run_in_executor(
                None, self._enroll_sync, device, samples
            )

        if not embeddings:
            return Response(ok=False, error="No faces captured")

        self.store.save(req.username, embeddings)
        return Response(
            ok=True,
            data={
                "samples": len(embeddings),
                "consistency": round(consistency, 3),
            },
        )

    def _enroll_sync(self, device: str, samples: int) -> tuple[list[np.ndarray], float]:
        """Synchronous enrollment (runs in executor)."""
        embeddings = []
        max_attempts = samples * ATTEMPT_MULTIPLIER

        with Camera(device, self.config.camera.width, self.config.camera.height) as cam:
            attempt = 0
            while len(embeddings) < samples and attempt < max_attempts:
                attempt += 1
                try:
                    frame = cam.read()
                except RuntimeError:
                    continue

                raw_ir, bgr_frame = process_frame(frame)

                faces = self.recognizer.get_faces(bgr_frame)
                if not faces:
                    continue

                best = select_best_face(faces)
                if best is None:
                    continue

                spoof_result = check_antispoof(self.antispoof, raw_ir, bgr_frame, best.bbox)
                if spoof_result is not None and not spoof_result.passed:
                    log.warning("Enroll: spoof rejected frame %d: %s", attempt, spoof_result.reason)
                    continue

                embeddings.append(best.embedding)
                time.sleep(FRAME_DELAY_ENROLL)

        avg_consistency, _ = compute_self_consistency(embeddings)
        return embeddings, avg_consistency

    async def _handle_delete(self, req: Request) -> Response:
        if not req.username:
            return Response(ok=False, error="username required")
        deleted = self.store.delete(req.username)
        if deleted:
            return Response(ok=True)
        return Response(ok=False, error=f"User '{req.username}' not enrolled")

    async def _handle_status(self, req: Request) -> Response:
        models_loaded = self.recognizer is not None
        users = self.store.list_users()
        antispoof_data = {
            "enabled": self.config.antispoof.enabled,
        }
        if self.antispoof is not None:
            antispoof_data["minifasnet_available"] = self.antispoof.minifasnet_available
            antispoof_data["ir_brightness_min"] = self.config.antispoof.ir_brightness_min
        return Response(
            ok=True,
            data={
                "models_loaded": models_loaded,
                "enrolled_users": len(users),
                "users": users,
                "socket": str(self.socket_path),
                "pid": os.getpid(),
                "antispoof": antispoof_data,
            },
        )

    async def _handle_list(self, req: Request) -> Response:
        users = self.store.list_users()
        user_info = {}
        for u in users:
            user_info[u] = self.store.get_embedding_count(u)
        return Response(ok=True, data={"users": user_info})

    async def start(self, socket_path: Path | None = None):
        """Start the daemon."""
        self._socket_path = socket_path or self.socket_path
        sock = self._socket_path

        # Clean up stale socket
        if sock.exists():
            sock.unlink()

        # Ensure parent directory exists
        sock.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self.handle_client, path=str(sock)
        )

        # System mode: 0o666 so any local user can connect
        # User mode: 0o600 restricted to owning user only
        os.chmod(sock, 0o666 if self.config.daemon.system_mode else 0o600)

        log.info("Daemon listening on %s (pid=%d)", sock, os.getpid())

        # Pre-load models in background
        asyncio.get_event_loop().run_in_executor(None, self._ensure_models)

    async def stop(self):
        """Stop the daemon."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            log.info("Server stopped")

        if self._socket_path and self._socket_path.exists():
            self._socket_path.unlink()
            log.info("Socket removed")

    async def run_forever(self):
        """Start and run until signalled."""
        await self.start()

        stop_event = asyncio.Event()

        def _signal_handler():
            log.info("Received shutdown signal")
            stop_event.set()

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

        await stop_event.wait()
        await self.stop()


def run_daemon(
    config_path: str | None = None,
    socket_path: str | None = None,
    system: bool = False,
):
    """Entry point for running the daemon."""
    from pathlib import Path as P

    cfg = load_config(P(config_path) if config_path else None)

    if system:
        cfg.daemon.system_mode = True

    logging.basicConfig(
        level=getattr(logging, cfg.daemon.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    daemon = FaceAuthDaemon(config=cfg)
    sock = P(socket_path) if socket_path else None

    asyncio.run(daemon.run_forever() if sock is None else _run_with_path(daemon, sock))


async def _run_with_path(daemon: FaceAuthDaemon, socket_path: Path):
    await daemon.start(socket_path)
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    await daemon.stop()
