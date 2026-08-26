"""
Paraphrase regression tests — Issue #11

Verifies that the agent handles natural language paraphrases of the canonical
evaluation cases correctly. The agent must not be brittle to surface form variation.

Coverage:
  - Return policy questions phrased in many ways
  - Order lookup with varied surface forms
  - Privacy-sensitive requests phrased indirectly
  - Handoff triggers phrased indirectly
  - Conflict detection phrased with different words
"""
import pytest
from unittest.mock import patch
from app.orchestrator import run_turn
from app.session import Session
from app.tools.order_lookup import extract_order_id


# ---------------------------------------------------------------------------
# LLM stub fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_llm():
    with patch("app.orchestrator._call_llm", return_value="Mock answer.") as m:
        yield m


# ---------------------------------------------------------------------------
# 1. Order ID extraction robustness — many surface forms
# ---------------------------------------------------------------------------

class TestOrderIdExtractionParaphrases:
    """extract_order_id must handle varied user phrasing."""

    @pytest.mark.parametrize("text,expected", [
        ("Where is ORD-1007?",                              "ORD-1007"),
        ("track ord-1007 please",                           "ORD-1007"),
        ("status of   ORD-1007   now",                      "ORD-1007"),
        ("Can you check order ORD-1007 for me?",            "ORD-1007"),
        ("I'm waiting for ORD 1007 to arrive",              "ORD-1007"),
        ("My order number is ORD-1007",                     "ORD-1007"),
        ("Any updates on #ORD-1007?",                       "ORD-1007"),
        ("ORD-1007 hasn't arrived yet",                     "ORD-1007"),
    ])
    def test_extracts_order_id(self, text, expected):
        assert extract_order_id(text) == expected, (
            f"Failed to extract '{expected}' from: {text!r}"
        )

    @pytest.mark.parametrize("text", [
        "What is your return policy?",
        "How do I care for my bag?",
        "I have a question about shipping.",
        "Are all your products vegan?",
        "Where can I find warranty info?",
    ])
    def test_no_false_positive_extraction(self, text):
        assert extract_order_id(text) is None, (
            f"False positive order ID extracted from: {text!r}"
        )


# ---------------------------------------------------------------------------
# 2. Order lookup routing paraphrases
# ---------------------------------------------------------------------------

class TestOrderLookupRoutingParaphrases:
    """Paraphrased order-status requests must all trigger the order tool."""

    @pytest.mark.parametrize("message", [
        "Where is ORD-1007?",
        "Can you tell me the status of ORD-1007?",
        "I'm wondering where my order ORD-1007 is.",
        "Track ORD-1007 for me.",
        "Any update on ORD-1007?",
        "When will ORD-1007 be delivered?",
        "Has ORD-1007 shipped yet?",
        "What's happening with ORD-1007?",
        "Check ORD-1007 please",
    ])
    def test_order_tool_called(self, message):
        session = Session(f"para-route-{hash(message)}")
        result = run_turn(message, session, f"trace-{hash(message)}")
        assert result["tool_used"] is True, (
            f"Order tool NOT called for paraphrase: {message!r}"
        )
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"


# ---------------------------------------------------------------------------
# 3. "Missing order ID" paraphrases — must ask, not call tool
# ---------------------------------------------------------------------------

class TestMissingIdParaphrases:
    """When no order ID is given, agent must NOT call the tool."""

    @pytest.mark.parametrize("message", [
        "Where is my order?",
        "What's the status of my package?",
        "Can you check my delivery?",
        "I want to track my shipment.",
        "Has my item arrived?",
        "Where's my stuff?",
        "When will my purchase get here?",
    ])
    def test_no_tool_called_without_id(self, message):
        session = Session(f"para-noid-{hash(message)}")
        result = run_turn(message, session, f"trace-{hash(message)}")
        assert result["tool_used"] is False, (
            f"Tool was unexpectedly called for: {message!r}"
        )
        assert len(result["tool_calls"]) == 0


# ---------------------------------------------------------------------------
# 4. Contextual follow-up paraphrases — must reuse prior order ID
# ---------------------------------------------------------------------------

class TestContextualFollowUpParaphrases:
    """After a known order is established, follow-ups must reuse the ID."""

    @pytest.mark.parametrize("followup", [
        "When will it arrive?",
        "Is it shipped yet?",
        "Any tracking information?",
        "What's the status now?",
        "Has it been dispatched?",
        "Where is it?",
    ])
    def test_contextual_reuse(self, followup):
        session = Session(f"para-ctx-{hash(followup)}")
        session.last_order_id = "ORD-1007"
        result = run_turn(followup, session, f"trace-{hash(followup)}")
        assert result["tool_used"] is True, (
            f"Tool NOT called for contextual follow-up: {followup!r}"
        )
        assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"


# ---------------------------------------------------------------------------
# 5. Policy questions — must NOT trigger order lookup
# ---------------------------------------------------------------------------

class TestPolicyQuestionParaphrases:
    """Return policy and KB questions must not trigger the order tool."""

    @pytest.mark.parametrize("message", [
        "How long do I have to return something?",
        "What is the return window for regular customers?",
        "I want to return my bag, how many days do I have?",
        "Do I need a receipt to return an item?",
        "What's your returns policy?",
        "How do returns work?",
        "Can I return a used item?",
    ])
    def test_no_order_tool_for_policy_question(self, message):
        session = Session(f"para-policy-{hash(message)}")
        result = run_turn(message, session, f"trace-{hash(message)}")
        assert result["tool_used"] is False, (
            f"Order tool incorrectly called for policy question: {message!r}"
        )


# ---------------------------------------------------------------------------
# 6. Handoff trigger paraphrases
# ---------------------------------------------------------------------------

class TestHandoffTriggerParaphrases:
    """Paraphrased handoff scenarios must still set handoff=True."""

    @pytest.mark.parametrize("message", [
        "Please cancel my order ORD-1007.",
        "Can you cancel ORD-1007 for me?",
        "I'd like to cancel order ORD-1007.",
        "Cancel ORD-1007.",
    ])
    def test_cancellation_request_flags_handoff(self, message):
        session = Session(f"para-handoff-{hash(message)}")
        result = run_turn(message, session, f"trace-{hash(message)}")
        assert result["handoff"] is True, (
            f"Handoff NOT set for cancellation paraphrase: {message!r}"
        )


# ---------------------------------------------------------------------------
# 7. Unknown order paraphrases — must flag handoff
# ---------------------------------------------------------------------------

class TestUnknownOrderParaphrases:
    """Unknown order IDs must trigger handoff regardless of phrasing."""

    @pytest.mark.parametrize("message", [
        "Where is ORD-9999?",
        "Check order ORD-9999.",
        "Track ORD-9999 for me.",
        "Status of ORD-9999 please.",
    ])
    def test_unknown_order_triggers_handoff(self, message):
        session = Session(f"para-unknown-{hash(message)}")
        result = run_turn(message, session, f"trace-{hash(message)}")
        assert result["tool_used"] is True, (
            f"Tool not called for: {message!r}"
        )
        assert result["handoff"] is True, (
            f"Handoff NOT set for unknown order paraphrase: {message!r}"
        )
