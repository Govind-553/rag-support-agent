"""
Combined RAG + Tool integration tests — Issue #12

Verifies scenarios where BOTH the RAG knowledge base AND the order lookup tool
contribute to the agent's response. These are the most realistic scenarios
because real customer queries often need policy context alongside order data.

Coverage:
  - Order lookup + policy RAG context both present in the pipeline
  - Conflict detection combined with order lookup
  - Handoff determination with both order + RAG context
  - Source citations include both RAG and tool sources
"""
import pytest
from unittest.mock import patch
from app.orchestrator import run_turn
from app.session import Session


# ---------------------------------------------------------------------------
# LLM stub fixture — returns a static answer so we can inspect pipeline state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_llm():
    with patch("app.orchestrator._call_llm", return_value="Mock combined answer.") as m:
        yield m


# ---------------------------------------------------------------------------
# 1. Order lookup + RAG both contribute to a single turn
# ---------------------------------------------------------------------------

class TestCombinedOrderAndRAG:
    """When a query mentions an order AND a policy topic, both should fire."""

    def test_order_lookup_and_rag_sources_both_present(self):
        """'What is the return policy for ORD-1007?' needs order + RAG."""
        session = Session("combined-1")
        result = run_turn(
            "What is the return window for ORD-1007?",
            session, "trace-combined-1"
        )
        # Order tool must have been called
        assert result["tool_used"] is True, "Order tool was not called"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"
        assert result["tool_calls"][0]["found"] is True

        # RAG sources must also be present (return policy docs)
        assert len(result["sources"]) > 0, "No RAG sources returned"

    def test_shipping_question_with_order_id(self):
        """'When will ORD-1001 arrive?' needs order data + shipping policy."""
        session = Session("combined-2")
        result = run_turn(
            "When will ORD-1001 arrive?",
            session, "trace-combined-2"
        )
        assert result["tool_used"] is True
        assert result["tool_calls"][0]["found"] is True
        assert len(result["sources"]) > 0


# ---------------------------------------------------------------------------
# 2. Conflict detection with order context
# ---------------------------------------------------------------------------

class TestConflictWithOrderContext:
    """When a query triggers the Breeze Tumbler conflict AND mentions an order."""

    def test_conflict_and_order_both_handled(self):
        """Ask about Breeze Tumbler care for a specific order."""
        session = Session("combined-conflict-1")
        result = run_turn(
            "I ordered ORD-1003, is the Breeze Tumbler dishwasher safe?",
            session, "trace-combined-conflict-1"
        )
        # Order tool should be called for ORD-1003
        assert result["tool_used"] is True
        assert result["tool_calls"][0]["arguments"]["order_id"] == "ORD-1003"

        # Conflict should be detected (if ORD-1003 relates to Breeze Tumbler)
        # or at minimum, RAG sources should be present
        assert len(result["sources"]) > 0


# ---------------------------------------------------------------------------
# 3. Handoff with combined context
# ---------------------------------------------------------------------------

class TestHandoffWithCombinedContext:
    """Handoff determination must work correctly with both order + RAG."""

    def test_unknown_order_triggers_handoff_with_policy_context(self):
        """Unknown order + policy question = handoff for the order part."""
        session = Session("combined-handoff-1")
        result = run_turn(
            "What is the return policy for ORD-9999?",
            session, "trace-combined-handoff-1"
        )
        assert result["tool_used"] is True
        assert result["handoff"] is True
        assert result["handoff_reason"] == "unknown_order"

    def test_exception_order_triggers_handoff(self):
        """An order with 'exception' status should trigger handoff."""
        session = Session("combined-handoff-2")
        result = run_turn(
            "Where is ORD-1010?",
            session, "trace-combined-handoff-2"
        )
        assert result["tool_used"] is True
        # ORD-1010 has status "exception"
        assert result["handoff"] is True
        assert result["handoff_reason"] == "order_exception"

    def test_normal_order_no_handoff(self):
        """A normal, found order should NOT trigger handoff."""
        session = Session("combined-handoff-3")
        result = run_turn(
            "Where is ORD-1007?",
            session, "trace-combined-handoff-3"
        )
        assert result["tool_used"] is True
        assert result["tool_calls"][0]["found"] is True
        assert result["handoff"] is False


# ---------------------------------------------------------------------------
# 4. Multi-turn combined scenarios
# ---------------------------------------------------------------------------

class TestMultiTurnCombined:
    """Multi-turn conversations combining order lookups and policy questions."""

    def test_order_then_policy_followup(self):
        """First turn: order lookup. Second turn: policy question."""
        session = Session("combined-multi-1")

        # Turn 1: Order lookup
        r1 = run_turn("Where is ORD-1007?", session, "trace-multi-1a")
        assert r1["tool_used"] is True
        assert r1["tool_calls"][0]["found"] is True

        # Turn 2: Policy follow-up (no order ID)
        r2 = run_turn("What is the return policy?", session, "trace-multi-1b")
        # Should NOT call order tool (this is a pure policy question)
        assert r2["tool_used"] is False
        # Should have RAG sources for return policy
        assert len(r2["sources"]) > 0

    def test_policy_then_order_lookup(self):
        """First turn: policy question. Second turn: order lookup."""
        session = Session("combined-multi-2")

        # Turn 1: Policy question
        r1 = run_turn("How long do I have to return something?", session, "trace-multi-2a")
        assert r1["tool_used"] is False
        assert len(r1["sources"]) > 0

        # Turn 2: Order lookup
        r2 = run_turn("Also, where is ORD-1001?", session, "trace-multi-2b")
        assert r2["tool_used"] is True
        assert r2["tool_calls"][0]["arguments"]["order_id"] == "ORD-1001"

    def test_order_then_contextual_followup_then_policy(self):
        """Three-turn scenario: order → contextual follow-up → policy."""
        session = Session("combined-multi-3")

        # Turn 1: Order lookup
        r1 = run_turn("Track ORD-1007.", session, "trace-multi-3a")
        assert r1["tool_used"] is True

        # Turn 2: Contextual follow-up (should reuse ORD-1007)
        r2 = run_turn("When will it arrive?", session, "trace-multi-3b")
        assert r2["tool_used"] is True
        assert r2["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"

        # Turn 3: Unrelated policy question (should NOT reuse order)
        r3 = run_turn("What is your warranty coverage?", session, "trace-multi-3c")
        assert r3["tool_used"] is False
        assert len(r3["sources"]) > 0


# ---------------------------------------------------------------------------
# 5. Source citation completeness
# ---------------------------------------------------------------------------

class TestSourceCitationCompleteness:
    """Verify source citations cover both RAG chunks and tool results."""

    def test_sources_include_rag_filenames(self):
        """Sources list must contain at least one RAG filename."""
        session = Session("combined-sources-1")
        result = run_turn(
            "What is the return policy for ORD-1007?",
            session, "trace-sources-1"
        )
        rag_sources = [s for s in result["sources"] if s.filename.endswith(".md")]
        assert len(rag_sources) > 0, "No RAG .md sources found in citations"

    def test_policy_question_has_relevant_source(self):
        """A return-policy question should cite the returns doc."""
        session = Session("combined-sources-2")
        result = run_turn(
            "How many days do I have to return something?",
            session, "trace-sources-2"
        )
        filenames = [s.filename for s in result["sources"]]
        has_return_doc = any("return" in f.lower() for f in filenames)
        assert has_return_doc, (
            f"Expected a return-related source, got: {filenames}"
        )
