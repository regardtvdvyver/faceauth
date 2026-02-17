"""Wire protocol for daemon <-> client communication.

Simple JSON-over-Unix-socket protocol. Each message is a JSON object
terminated by a newline. Responses are also newline-terminated JSON.

Request format:
    {"action": "verify", "username": "regardtv"}
    {"action": "enroll", "username": "regardtv", "samples": 5}
    {"action": "delete", "username": "regardtv"}
    {"action": "status"}
    {"action": "list"}

Response format:
    {"ok": true, "score": 0.88, ...}
    {"ok": false, "error": "no face detected"}
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Request:
    action: str
    username: str = ""
    samples: int = 5
    threshold: float | None = None
    camera_device: str | None = None

    def to_json(self) -> bytes:
        d = {k: v for k, v in asdict(self).items() if v is not None and v != ""}
        return json.dumps(d).encode() + b"\n"

    @classmethod
    def from_json(cls, data: bytes) -> "Request":
        d = json.loads(data.strip())
        return cls(
            action=d["action"],
            username=d.get("username", ""),
            samples=d.get("samples", 5),
            threshold=d.get("threshold"),
            camera_device=d.get("camera_device"),
        )


@dataclass
class Response:
    ok: bool
    error: str = ""
    score: float = 0.0
    data: dict = field(default_factory=dict)

    def to_json(self) -> bytes:
        d = {"ok": self.ok}
        if self.error:
            d["error"] = self.error
        if self.score != 0.0:
            d["score"] = round(self.score, 4)
        if self.data:
            d.update(self.data)
        return json.dumps(d).encode() + b"\n"

    @classmethod
    def from_json(cls, data: bytes) -> "Response":
        d = json.loads(data.strip())
        return cls(
            ok=d.get("ok", False),
            error=d.get("error", ""),
            score=d.get("score", 0.0),
            data={k: v for k, v in d.items() if k not in ("ok", "error", "score")},
        )
