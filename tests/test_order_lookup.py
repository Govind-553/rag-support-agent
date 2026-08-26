"""
Tests for the order lookup tool — Phase 5
"""
import pytest
from app.tools.order_lookup import (
    normalize_order_id,
    extract_order_id,
    lookup_order,
    sanitize_order,
)


def test_normalize_order_id_uppercase():
    assert normalize_order_id("ord-1007") == "ORD-1007"


def test_normalize_order_id_whitespace():
    assert normalize_order_id("  ORD-1007  ") == "ORD-1007"


def test_normalize_order_id_digits_only():
    # 5. Silently converting arbitrary digits is disabled
    assert normalize_order_id("1007") == "1007"


def test_normalize_order_id_separator():
    # Allowed separator normalization (space, underscore, hyphen)
    assert normalize_order_id("ORD 1007") == "ORD-1007"
    assert normalize_order_id("ORD_1007") == "ORD-1007"
    assert normalize_order_id("ord-1007") == "ORD-1007"
    assert normalize_order_id("ORD1007") == "ORD-1007"


def test_normalize_order_id_malformed():
    # Malformed ID should not be incorrectly guessed/fixed
    assert normalize_order_id("ORD-XYZ") == "ORD-XYZ"
    assert normalize_order_id("ORD-100") == "ORD-100"
    assert normalize_order_id("ABC-1234") == "ABC-1234"


def test_extract_order_id_plain():
    assert extract_order_id("Where is ORD-1007?") == "ORD-1007"


def test_extract_order_id_lowercase():
    assert extract_order_id("track ord-1003") == "ORD-1003"


def test_extract_order_id_none():
    assert extract_order_id("What is your return policy?") is None
    # Standing arbitrary number should not extract as order ID
    assert extract_order_id("Where is 1007?") is None


def test_lookup_order_valid():
    result = lookup_order("ORD-1007")
    assert result["found"] is True
    order = result["order"]
    assert order["order_id"] == "ORD-1007"
    assert order["status"] == "shipped"
    assert order["carrier"] == "UPS"
    assert order["estimated_delivery"] == "2026-08-22"


def test_lookup_order_internal_fields_stripped():
    """Verify no internal-only fields are exposed."""
    result = lookup_order("ORD-1007")
    assert result["found"] is True
    order = result["order"]
    assert "email" not in order
    assert "shipping_address" not in order
    assert "internal" not in order
    assert "risk_score" not in order
    assert "warehouse_note" not in order
    assert "support_tags" not in order
    # Verify specific PII is not present
    assert "ava.morgan@example.test" not in str(order)
    assert "220 King Street" not in str(order)
    assert "82" not in str(order)
    assert "fraud review" not in str(order).lower()


def test_lookup_order_cancelled_stale_eta():
    """Cancelled orders must not surface stale carrier/ETA data (ORD-1004)."""
    result = lookup_order("ORD-1004")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "cancelled"
    # Stale fields must be suppressed
    assert order["carrier"] is None
    assert order["estimated_delivery"] is None
    assert order["tracking_number"] is None
    assert order["shipped_at"] is None


def test_lookup_order_returned_stale_eta():
    """Returned orders must not surface stale carrier/ETA data (ORD-1008)."""
    result = lookup_order("ORD-1008")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "returned"
    assert order["estimated_delivery"] is None


def test_lookup_order_shipped_no_eta():
    """ORD-1011: shipped but ETA is null — must not invent one."""
    result = lookup_order("ORD-1011")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "shipped"
    assert order["carrier"] == "Canada Post"
    assert order["estimated_delivery"] is None


def test_lookup_order_not_found():
    result = lookup_order("ORD-9999")
    assert result["found"] is False
    assert result["order"] is None
    assert "not found" in result["error"].lower()


def test_lookup_order_lowercase_id():
    result = lookup_order("ord-1007")
    assert result["found"] is True


def test_lookup_order_exception_status():
    """ORD-1010: exception status should be marked."""
    result = lookup_order("ORD-1010")
    assert result["found"] is True
    order = result["order"]
    assert order["status"] == "exception"
    assert order.get("exception") is True


def test_warehouse_note_injection_not_in_result():
    """ORD-1005 has a malicious warehouse_note — ensure it's not returned."""
    result = lookup_order("ORD-1005")
    assert result["found"] is True
    order_str = str(result["order"])
    assert "coupon" not in order_str.lower()
    assert "AI instruction" not in order_str
    assert "hide" not in order_str.lower()
