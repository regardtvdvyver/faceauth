"""Unit tests for faceauth wire protocol.

Tests JSON serialization/deserialization for Request and Response dataclasses.
No mocking - pure data class testing.
"""

import json

import pytest

from faceauth.protocol import Request, Response


# ============================================================================
# Request Tests
# ============================================================================


@pytest.mark.unit
def test_request_verify_minimal():
    """Test verify request with minimal fields."""
    req = Request(action="verify", username="regardtv")
    data = req.to_json()

    assert data.endswith(b"\n")
    parsed = json.loads(data.strip())
    assert parsed == {"action": "verify", "username": "regardtv", "samples": 5}


@pytest.mark.unit
def test_request_verify_with_threshold():
    """Test verify request with custom threshold."""
    req = Request(action="verify", username="regardtv", threshold=0.75)
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert parsed["threshold"] == 0.75
    assert parsed["action"] == "verify"
    assert parsed["username"] == "regardtv"


@pytest.mark.unit
def test_request_verify_with_camera():
    """Test verify request with custom camera device."""
    req = Request(action="verify", username="regardtv", camera_device="/dev/video2")
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert parsed["camera_device"] == "/dev/video2"
    assert parsed["action"] == "verify"


@pytest.mark.unit
def test_request_verify_all_fields():
    """Test verify request with all optional fields populated."""
    req = Request(
        action="verify",
        username="regardtv",
        samples=3,
        threshold=0.8,
        camera_device="/dev/video0",
    )
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert parsed == {
        "action": "verify",
        "username": "regardtv",
        "samples": 3,
        "threshold": 0.8,
        "camera_device": "/dev/video0",
    }


@pytest.mark.unit
def test_request_enroll():
    """Test enroll request with custom sample count."""
    req = Request(action="enroll", username="testuser", samples=10)
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert parsed == {"action": "enroll", "username": "testuser", "samples": 10}


@pytest.mark.unit
def test_request_enroll_default_samples():
    """Test enroll request uses default samples=5."""
    req = Request(action="enroll", username="testuser")
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert parsed["samples"] == 5


@pytest.mark.unit
def test_request_delete():
    """Test delete request."""
    req = Request(action="delete", username="olduser")
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert parsed == {"action": "delete", "username": "olduser", "samples": 5}


@pytest.mark.unit
def test_request_status():
    """Test status request with no username."""
    req = Request(action="status")
    data = req.to_json()

    parsed = json.loads(data.strip())
    # Empty username should not appear in JSON
    assert "username" not in parsed
    assert parsed["action"] == "status"
    assert parsed["samples"] == 5


@pytest.mark.unit
def test_request_list():
    """Test list request."""
    req = Request(action="list")
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert parsed["action"] == "list"
    assert "username" not in parsed


@pytest.mark.unit
def test_request_none_fields_omitted():
    """Test that None fields are not serialized."""
    req = Request(action="verify", username="user", threshold=None, camera_device=None)
    data = req.to_json()

    parsed = json.loads(data.strip())
    assert "threshold" not in parsed
    assert "camera_device" not in parsed


@pytest.mark.unit
def test_request_roundtrip_minimal():
    """Test Request serialization/deserialization roundtrip with minimal data."""
    original = Request(action="verify", username="regardtv")
    serialized = original.to_json()
    restored = Request.from_json(serialized)

    assert restored.action == original.action
    assert restored.username == original.username
    assert restored.samples == 5  # default


@pytest.mark.unit
def test_request_roundtrip_all_fields():
    """Test Request roundtrip with all fields populated."""
    original = Request(
        action="enroll",
        username="testuser",
        samples=7,
        threshold=0.85,
        camera_device="/dev/video2",
    )
    serialized = original.to_json()
    restored = Request.from_json(serialized)

    assert restored.action == original.action
    assert restored.username == original.username
    assert restored.samples == original.samples
    assert restored.threshold == original.threshold
    assert restored.camera_device == original.camera_device


@pytest.mark.unit
def test_request_from_json_with_whitespace():
    """Test Request.from_json handles extra whitespace."""
    data = b'  {"action": "verify", "username": "user"}  \n  '
    req = Request.from_json(data)

    assert req.action == "verify"
    assert req.username == "user"
    assert req.samples == 5


@pytest.mark.unit
def test_request_from_json_missing_optional():
    """Test Request.from_json with missing optional fields uses defaults."""
    data = b'{"action": "status"}\n'
    req = Request.from_json(data)

    assert req.action == "status"
    assert req.username == ""
    assert req.samples == 5
    assert req.threshold is None
    assert req.camera_device is None


@pytest.mark.unit
def test_request_newline_termination():
    """Test all requests end with newline."""
    requests = [
        Request(action="verify", username="user"),
        Request(action="enroll", username="user", samples=3),
        Request(action="delete", username="user"),
        Request(action="status"),
        Request(action="list"),
    ]

    for req in requests:
        data = req.to_json()
        assert data.endswith(b"\n"), f"Request {req.action} missing newline"
        assert data.count(b"\n") == 1, f"Request {req.action} has multiple newlines"


# ============================================================================
# Response Tests
# ============================================================================


@pytest.mark.unit
def test_response_success_minimal():
    """Test successful response with no extra data."""
    resp = Response(ok=True)
    data = resp.to_json()

    assert data.endswith(b"\n")
    parsed = json.loads(data.strip())
    assert parsed == {"ok": True}


@pytest.mark.unit
def test_response_success_with_score():
    """Test successful response with similarity score."""
    resp = Response(ok=True, score=0.8765432)
    data = resp.to_json()

    parsed = json.loads(data.strip())
    assert parsed["ok"] is True
    # Score should be rounded to 4 decimal places
    assert parsed["score"] == 0.8765


@pytest.mark.unit
def test_response_failure_with_error():
    """Test failed response with error message."""
    resp = Response(ok=False, error="no face detected")
    data = resp.to_json()

    parsed = json.loads(data.strip())
    assert parsed == {"ok": False, "error": "no face detected"}


@pytest.mark.unit
def test_response_with_data_dict():
    """Test response with extra data dictionary."""
    resp = Response(ok=True, data={"users": ["alice", "bob"], "count": 2})
    data = resp.to_json()

    parsed = json.loads(data.strip())
    assert parsed["ok"] is True
    assert parsed["users"] == ["alice", "bob"]
    assert parsed["count"] == 2


@pytest.mark.unit
def test_response_all_fields():
    """Test response with all fields populated."""
    resp = Response(
        ok=True,
        error="",  # Should be omitted even if present
        score=0.923456,
        data={"status": "active", "version": "1.0"},
    )
    data = resp.to_json()

    parsed = json.loads(data.strip())
    assert parsed["ok"] is True
    assert "error" not in parsed  # Empty error should be omitted
    assert parsed["score"] == 0.9235
    assert parsed["status"] == "active"
    assert parsed["version"] == "1.0"


@pytest.mark.unit
def test_response_zero_score_omitted():
    """Test that score=0.0 is not included in JSON."""
    resp = Response(ok=True, score=0.0)
    data = resp.to_json()

    parsed = json.loads(data.strip())
    assert "score" not in parsed


@pytest.mark.unit
def test_response_empty_error_omitted():
    """Test that empty error string is not included."""
    resp = Response(ok=True, error="")
    data = resp.to_json()

    parsed = json.loads(data.strip())
    assert "error" not in parsed


@pytest.mark.unit
def test_response_empty_data_omitted():
    """Test that empty data dict is not included."""
    resp = Response(ok=True, data={})
    data = resp.to_json()

    parsed = json.loads(data.strip())
    assert parsed == {"ok": True}


@pytest.mark.unit
def test_response_roundtrip_success():
    """Test Response serialization/deserialization roundtrip for success."""
    original = Response(ok=True, score=0.89)
    serialized = original.to_json()
    restored = Response.from_json(serialized)

    assert restored.ok is True
    assert restored.score == 0.89
    assert restored.error == ""


@pytest.mark.unit
def test_response_roundtrip_failure():
    """Test Response roundtrip for failure with error."""
    original = Response(ok=False, error="camera unavailable")
    serialized = original.to_json()
    restored = Response.from_json(serialized)

    assert restored.ok is False
    assert restored.error == "camera unavailable"
    assert restored.score == 0.0


@pytest.mark.unit
def test_response_roundtrip_with_data():
    """Test Response roundtrip with data dictionary."""
    original = Response(
        ok=True,
        score=0.95,
        data={"username": "regardtv", "enrolled": True, "samples": 5},
    )
    serialized = original.to_json()
    restored = Response.from_json(serialized)

    assert restored.ok is True
    assert restored.score == 0.95
    assert restored.data["username"] == "regardtv"
    assert restored.data["enrolled"] is True
    assert restored.data["samples"] == 5


@pytest.mark.unit
def test_response_from_json_with_whitespace():
    """Test Response.from_json handles extra whitespace."""
    data = b'  {"ok": true, "score": 0.88}  \n  '
    resp = Response.from_json(data)

    assert resp.ok is True
    assert resp.score == 0.88


@pytest.mark.unit
def test_response_from_json_missing_fields():
    """Test Response.from_json with missing optional fields uses defaults."""
    data = b'{"ok": false}\n'
    resp = Response.from_json(data)

    assert resp.ok is False
    assert resp.error == ""
    assert resp.score == 0.0
    assert resp.data == {}


@pytest.mark.unit
def test_response_newline_termination():
    """Test all responses end with newline."""
    responses = [
        Response(ok=True),
        Response(ok=False, error="test error"),
        Response(ok=True, score=0.95),
        Response(ok=True, data={"key": "value"}),
    ]

    for resp in responses:
        data = resp.to_json()
        assert data.endswith(b"\n"), f"Response missing newline"
        assert data.count(b"\n") == 1, f"Response has multiple newlines"


@pytest.mark.unit
def test_response_score_rounding():
    """Test that scores are rounded to 4 decimal places."""
    test_cases = [
        (0.123456789, 0.1235),
        (0.999999, 1.0),
        (0.12345, 0.1235),
        (0.1234, 0.1234),
    ]

    for input_score, expected_score in test_cases:
        resp = Response(ok=True, score=input_score)
        data = resp.to_json()
        parsed = json.loads(data.strip())
        assert parsed["score"] == expected_score, f"Failed for input {input_score}"


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.unit
def test_request_response_workflow():
    """Test realistic request/response workflow."""
    # Client sends verify request
    req = Request(action="verify", username="regardtv", threshold=0.8)
    req_data = req.to_json()

    # Server receives and parses
    received_req = Request.from_json(req_data)
    assert received_req.action == "verify"
    assert received_req.username == "regardtv"
    assert received_req.threshold == 0.8

    # Server sends success response
    resp = Response(ok=True, score=0.8567)
    resp_data = resp.to_json()

    # Client receives and parses
    received_resp = Response.from_json(resp_data)
    assert received_resp.ok is True
    assert received_resp.score == 0.8567


@pytest.mark.unit
def test_request_response_failure_workflow():
    """Test failure workflow with error message."""
    # Client sends enroll request
    req = Request(action="enroll", username="newuser", samples=3)
    req_data = req.to_json()

    # Server receives
    received_req = Request.from_json(req_data)
    assert received_req.samples == 3

    # Server sends failure response
    resp = Response(ok=False, error="camera initialization failed")
    resp_data = resp.to_json()

    # Client receives error
    received_resp = Response.from_json(resp_data)
    assert received_resp.ok is False
    assert received_resp.error == "camera initialization failed"
    assert received_resp.score == 0.0
