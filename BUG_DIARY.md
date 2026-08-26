# Bug Diary

## Bug 1 - Superseded Policy Was Retrieved

**Problem:**  
The agent could retrieve an older policy instead of the current policy.

**Cause:**  
Semantic similarity alone did not guarantee document authority.

**Fix:**  
Added metadata-based document precedence before retrieved chunks are sent to the LLM.

**Regression Test:**  
Return-policy questions must use the current authoritative policy.

---

## Bug 2 - Cancelled Order Showed Old Delivery Information

**Problem:**  
A cancelled or returned order could contain stale delivery information.

**Cause:**  
Order records still contained shipping and ETA fields from earlier processing states.

**Fix:**  
Added sanitization and status-aware filtering before order data reaches the LLM.

**Regression Test:**  
Cancelled and returned orders must not expose stale ETA or tracking details.

---

## Bug 3 - Previous Order Context Was Reused Too Broadly

**Problem:**  
An earlier order could influence an unrelated follow-up question.

**Cause:**  
Order context reuse logic was too permissive for multi-turn conversations.

**Fix:**  
Order context is now reused only when the current message clearly refers to the previous order.

**Regression Test:**  
Unrelated policy questions must not trigger lookup of the previous order.
