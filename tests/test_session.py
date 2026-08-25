"""
Tests for the session store — Phase 6
"""
import pytest
from app.session import SessionStore, Session


def test_create_new_session():
    store = SessionStore()
    session = store.get_or_create("sess-001")
    assert session.session_id == "sess-001"
    assert session.turns == []
    assert session.last_order_id is None


def test_session_reuse():
    store = SessionStore()
    s1 = store.get_or_create("sess-002")
    s1.add_turn("user", "Hello")
    s2 = store.get_or_create("sess-002")
    assert s2 is s1  # Same object
    assert len(s2.turns) == 1


def test_session_isolation():
    """Sessions must not share state."""
    store = SessionStore()
    s1 = store.get_or_create("sess-A")
    s2 = store.get_or_create("sess-B")
    s1.add_turn("user", "Order ORD-1007")
    s1.last_order_id = "ORD-1007"
    assert s2.last_order_id is None
    assert len(s2.turns) == 0


def test_session_history_bounded():
    """History should be bounded to avoid context bloat."""
    store = SessionStore()
    s = store.get_or_create("sess-bound")
    for i in range(30):
        s.add_turn("user", f"Message {i}")
    # MAX_TURNS * 2 = 20
    assert len(s.turns) <= 20


def test_get_history_format():
    store = SessionStore()
    s = store.get_or_create("sess-hist")
    s.add_turn("user", "Hi")
    s.add_turn("assistant", "Hello!")
    history = s.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hi"}
    assert history[1] == {"role": "assistant", "content": "Hello!"}


def test_last_order_id_persists():
    store = SessionStore()
    s = store.get_or_create("sess-order")
    s.last_order_id = "ORD-1007"
    s2 = store.get_or_create("sess-order")
    assert s2.last_order_id == "ORD-1007"
