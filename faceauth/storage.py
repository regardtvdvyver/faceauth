"""Embedding storage - save/load face embeddings as .npz files."""

import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Valid Linux username: starts with lowercase letter or underscore,
# followed by lowercase alphanumeric, underscore, or hyphen. Max 32 chars.
_USERNAME_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


class EmbeddingStore:
    """Stores face embeddings per user as .npz files.

    Each user gets a single file: {data_dir}/{username}.npz
    containing an array of 512-dim embeddings (multiple enrollment captures).
    """

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            if os.geteuid() == 0:
                data_dir = Path("/var/lib/faceauth")
            else:
                xdg = os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")
                data_dir = Path(xdg) / "faceauth"
        self.data_dir = Path(data_dir)

    def _ensure_dir(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Restrictive permissions - embeddings are biometric data
        self.data_dir.chmod(0o700)

    def _user_path(self, username: str) -> Path:
        """Get the storage path for a username, with validation."""
        if not _USERNAME_RE.match(username):
            raise ValueError(f"Invalid username: {username!r}")
        path = self.data_dir / f"{username}.npz"
        # Defense in depth: verify the resolved path is within data_dir
        if not path.resolve().is_relative_to(self.data_dir.resolve()):
            raise ValueError(f"Path traversal detected for username: {username!r}")
        return path

    def save(self, username: str, embeddings: list[np.ndarray]) -> Path:
        """Save embeddings for a user. Overwrites existing."""
        self._ensure_dir()
        path = self._user_path(username)
        stacked = np.stack(embeddings)
        np.savez_compressed(path, embeddings=stacked)
        os.chmod(path, 0o600)
        log.info("Saved %d embedding(s) for '%s' -> %s", len(embeddings), username, path)
        return path

    def load(self, username: str) -> list[np.ndarray]:
        """Load embeddings for a user. Returns empty list if not enrolled."""
        path = self._user_path(username)
        if not path.exists():
            log.debug("No embeddings found for '%s'", username)
            return []
        data = np.load(path)
        embeddings = list(data["embeddings"])
        log.debug("Loaded %d embedding(s) for '%s'", len(embeddings), username)
        return embeddings

    def delete(self, username: str) -> bool:
        """Delete stored embeddings for a user."""
        path = self._user_path(username)
        if path.exists():
            path.unlink()
            log.info("Deleted embeddings for '%s'", username)
            return True
        return False

    def list_users(self) -> list[str]:
        """List all enrolled usernames."""
        if not self.data_dir.exists():
            return []
        users = []
        for f in sorted(self.data_dir.glob("*.npz")):
            users.append(f.stem)
        return users

    def is_enrolled(self, username: str) -> bool:
        return self._user_path(username).exists()

    def get_embedding_count(self, username: str) -> int:
        """Get the number of stored embeddings for a user."""
        embeddings = self.load(username)
        return len(embeddings)
