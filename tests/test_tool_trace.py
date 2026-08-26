import pytest
from unittest.mock import patch
from app.orchestrator import run_turn
from app.session import Session

@pytest.fixture(autouse=True)
def mock_llm_call():
    with patch("app.orchestrator._call_llm", return_value="Mock response from LLM") as mock:
        yield mock

def test_tool_trace_exact_id():
    # 1. ORD-1007 -> actual tool argument ORD-1007
    session = Session("sess-trace-1")
    result = run_turn("Where is ORD-1007?", session, "trace-1")
    assert result["tool_used"] is True
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "order_lookup"
    assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"
    assert result["tool_calls"][0]["found"] is True

def test_tool_trace_lowercase_id():
    # 2. lowercase order ID -> normalized correct argument
    session = Session("sess-trace-2")
    result = run_turn("track ord-1007 please", session, "trace-2")
    assert result["tool_used"] is True
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"

def test_tool_trace_whitespace_id():
    # 3. whitespace -> normalized correctly
    session = Session("sess-trace-3")
    result = run_turn("status of   ORD-1007   now", session, "trace-3")
    assert result["tool_used"] is True
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"

def test_tool_trace_missing_id():
    # 4. missing ID -> no tool call
    session = Session("sess-trace-4")
    result = run_turn("Where is my order?", session, "trace-4")
    assert result["tool_used"] is False
    assert len(result["tool_calls"]) == 0

def test_tool_trace_unknown_id():
    # 5. unknown ID -> actual requested ID passed to tool
    session = Session("sess-trace-5")
    result = run_turn("Check ORD-9999 please.", session, "trace-5")
    assert result["tool_used"] is True
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-9999"
    assert result["tool_calls"][0]["found"] is False

def test_tool_trace_unrelated_question():
    # 6. unrelated KB question -> no order tool call
    session = Session("sess-trace-6")
    result = run_turn("What is your standard return window?", session, "trace-6")
    assert result["tool_used"] is False
    assert len(result["tool_calls"]) == 0
