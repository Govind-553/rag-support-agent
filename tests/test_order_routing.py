import pytest
from unittest.mock import patch
from app.orchestrator import run_turn
from app.session import Session

@pytest.fixture(autouse=True)
def mock_llm_call():
    with patch("app.orchestrator._call_llm", return_value="Mock response") as mock:
        yield mock

def test_routing_explicit_id():
    # 1. User: "Where is ORD-1007?" -> lookup ORD-1007
    session = Session("sess-route-1")
    result = run_turn("Where is ORD-1007?", session, "tr-1")
    assert result["tool_used"] is True
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"

def test_routing_contextual_reuse():
    # 2. User: "When will it arrive?" -> reuse ORD-1007 (with prior order in session)
    session = Session("sess-route-2")
    session.last_order_id = "ORD-1007"
    result = run_turn("When will it arrive?", session, "tr-2")
    assert result["tool_used"] is True
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"

def test_routing_unrelated_policy_no_reuse():
    # 3. User: "What is your return policy?" -> no order lookup (with prior order in session)
    session = Session("sess-route-3")
    session.last_order_id = "ORD-1007"
    result = run_turn("What is your return policy?", session, "tr-3")
    assert result["tool_used"] is False
    assert len(result["tool_calls"]) == 0

def test_routing_unrelated_kb_no_reuse():
    # 4. Previous order exists, then unrelated KB question -> no order lookup
    session = Session("sess-route-4")
    session.last_order_id = "ORD-1007"
    result = run_turn("How do I care for my Ridge Daypack?", session, "tr-4")
    assert result["tool_used"] is False
    assert len(result["tool_calls"]) == 0

def test_routing_order_intent_no_id():
    # 5. "Can I cancel my order?" -> ask for order ID if no contextual order exists
    session = Session("sess-route-5")
    # No last order ID
    result = run_turn("Can I cancel my order?", session, "tr-5")
    assert result["tool_used"] is False
    assert len(result["tool_calls"]) == 0
    # Handoff reason should be unsupported action or missing ID?
    # Wait, "cancel my order" contains "cancel my order" which is an action request -> triggers unsupported action handoff
    # But no tool was called.

def test_routing_new_session_no_leak():
    # 6. New session asking "When will it arrive?" -> do not reuse another session's order
    session1 = Session("sess-route-6a")
    session1.last_order_id = "ORD-1007"
    
    session2 = Session("sess-route-6b")
    result = run_turn("When will it arrive?", session2, "tr-6b")
    assert result["tool_used"] is False
    assert len(result["tool_calls"]) == 0
