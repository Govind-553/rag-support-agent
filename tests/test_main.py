"""
Integration-level tests for the FastAPI app endpoints.
These tests stub out the LLM to avoid loading the 3B model during testing.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.models import SourceCitation


# Patch the LLM pipeline before importing app
with patch("app.orchestrator._get_pipeline") as mock_pipe:
    from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_valid_schema():
    """Chat endpoint returns required fields regardless of LLM output."""
    mock_result = {
        "answer": "Our standard return window is 30 calendar days from delivery.",
        "sources": [SourceCitation(filename="01-returns-policy-current.md", heading="Return Window")],
        "handoff": False,
        "tool_used": False,
        "trace_id": "test-trace-id",
    }
    with patch("app.main.run_turn", return_value=mock_result):
        payload = {"message": "What is the return window?", "session_id": "test-session"}
        response = client.post("/api/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "handoff" in data
    assert "trace_id" in data
    assert "tool_used" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["handoff"], bool)
    assert isinstance(data["tool_used"], bool)


def test_chat_tool_used_field():
    """When order lookup runs, tool_used should be True."""
    mock_result = {
        "answer": "Your order ORD-1007 is shipped and expected August 22, 2026.",
        "sources": [],
        "handoff": False,
        "tool_used": True,
        "trace_id": "test-trace-id-2",
    }
    with patch("app.main.run_turn", return_value=mock_result):
        payload = {"message": "Where is ORD-1007?", "session_id": "test-session-2"}
        response = client.post("/api/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["tool_used"] is True


def test_chat_handoff_field():
    """Handoff flag passes through correctly."""
    mock_result = {
        "answer": "This requires human support agent review.",
        "sources": [],
        "handoff": True,
        "tool_used": False,
        "trace_id": "test-trace-id-3",
    }
    with patch("app.main.run_turn", return_value=mock_result):
        payload = {"message": "My item arrived damaged.", "session_id": "test-session-3"}
        response = client.post("/api/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["handoff"] is True
