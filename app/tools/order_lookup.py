"""
Order Lookup Tool — Phase 5

Performs safe, sanitized order lookups from orders.json.
- Normalizes order IDs (case, whitespace)
- Strips internal-only fields before returning to agent
- Handles stale ETAs for cancelled/returned orders
- Handles exception and shipped-without-ETA states
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

from app.config import ORDERS_JSON_PATH


# Fields that must NEVER be surfaced to the customer or to the LLM response.
_INTERNAL_FIELDS = {"email", "shipping_address", "internal", "risk_score",
                    "warehouse_note", "support_tags"}

# Statuses where carrier/ETA fields are stale and must not be shown
_STALE_ETA_STATUSES = {"cancelled", "returned"}


def _load_orders() -> Dict[str, Any]:
    """Loads and indexes the orders JSON keyed by normalized order_id."""
    with open(ORDERS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = {}
    for order in data.get("orders", []):
        key = order["order_id"].strip().upper()
        index[key] = order
    return index


# Lazy-loaded cache so we don't re-read on every request
_order_cache: Optional[Dict[str, Any]] = None


def _get_orders() -> Dict[str, Any]:
    global _order_cache
    if _order_cache is None:
        _order_cache = _load_orders()
    return _order_cache


def normalize_order_id(raw: str) -> str:
    """Normalizes an order ID: strip whitespace, uppercase, ensure ORD- prefix pattern."""
    normalized = raw.strip().upper()
    # Allow ORD-NNNN or just digits like 1007 → ORD-1007
    if re.match(r"^\d+$", normalized):
        normalized = f"ORD-{normalized}"
    return normalized


def extract_order_id(text: str) -> Optional[str]:
    """Extracts a potential order ID from user text using regex."""
    # Match ORD-XXXX or standalone 4-digit number used as order reference
    match = re.search(r"\b(ORD[-\s]?\d{3,6})\b", text, re.IGNORECASE)
    if match:
        return normalize_order_id(match.group(1).replace(" ", "-"))
    return None


def sanitize_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a customer-safe view of the order.
    Removes all internal-only fields.
    Suppresses stale ETA/carrier fields when the order is cancelled or returned.
    """
    status = order.get("status", "").lower()
    
    safe = {
        "order_id": order["order_id"],
        "status": order["status"],
        "items": [
            {
                "name": item["name"],
                "quantity": item["quantity"],
                "final_sale": item.get("final_sale", False)
            }
            for item in order.get("items", [])
        ],
        "placed_at": order.get("placed_at"),
        "membership_tier": order.get("membership_tier"),
    }

    if status in _STALE_ETA_STATUSES:
        # Explicitly mark that carrier and ETA are not applicable/stale
        safe["carrier"] = None
        safe["tracking_number"] = None
        safe["estimated_delivery"] = None
        safe["shipped_at"] = None
    else:
        safe["carrier"] = order.get("carrier")
        safe["tracking_number"] = order.get("tracking_number")
        safe["shipped_at"] = order.get("shipped_at")
        # Only include estimated_delivery if it's actually set
        eta = order.get("estimated_delivery")
        safe["estimated_delivery"] = eta  # can be None — agent must handle that

    safe["delivered_at"] = order.get("delivered_at") if status not in _STALE_ETA_STATUSES else None
    safe["customer_safe_message"] = order.get("customer_safe_message", "")
    
    # Handle exception status
    if status == "exception":
        safe["exception"] = True

    return safe


def lookup_order(order_id: str) -> Dict[str, Any]:
    """
    Public API: look up an order by ID and return a sanitized result dict.
    
    Returns:
        {
          "found": bool,
          "order": <sanitized order dict> | None,
          "error": str | None
        }
    """
    try:
        normalized = normalize_order_id(order_id)
    except Exception:
        return {"found": False, "order": None, "error": "Invalid order ID format."}

    orders = _get_orders()
    raw = orders.get(normalized)
    
    if raw is None:
        return {
            "found": False,
            "order": None,
            "error": f"Order {normalized} was not found in our system."
        }

    return {
        "found": True,
        "order": sanitize_order(raw),
        "error": None
    }
