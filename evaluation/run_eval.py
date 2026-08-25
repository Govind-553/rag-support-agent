#!/usr/bin/env python3
"""
Evaluation harness for Aster & Row AI Support Agent.

Runs all visible cases from evaluation/visible-cases.json plus original regression cases.
Does NOT use another LLM to grade — all assertions are deterministic.

Usage:
    python evaluation/run_eval.py [--report]
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.session import SessionStore
from app.orchestrator import run_turn

VISIBLE_CASES_PATH = ROOT / "evaluation" / "visible-cases.json"
ORIGINAL_CASES_PATH = ROOT / "evaluation" / "original-cases.json"


# ---------------------------------------------------------------------------
# Test case runner
# ---------------------------------------------------------------------------

def run_case(case: Dict, store: SessionStore) -> Dict[str, Any]:
    """
    Runs a single evaluation case and returns a result dict.
    Multi-turn: all messages in a case share a session.
    """
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    session = store.get_or_create(session_id)

    messages = case.get("messages", [])
    expect = case.get("expect", {})

    last_answer = ""
    last_sources = []
    last_handoff = False
    last_tool_used = False

    # Run all turns; evaluate on the LAST response
    for msg in messages:
        if msg["role"] != "user":
            continue
        trace_id = str(uuid.uuid4())
        result = run_turn(msg["content"], session, trace_id)
        last_answer = result["answer"]
        last_sources = result["sources"]
        last_handoff = result["handoff"]
        last_tool_used = result["tool_used"]

    # --- Assertions ---
    failures = []
    warnings = []

    answer_lower = last_answer.lower()
    source_filenames = [s.filename for s in last_sources]

    # must_include: literal strings that must appear in answer
    for phrase in expect.get("must_include", []):
        if phrase.lower() not in answer_lower:
            failures.append(f"MISSING in answer: '{phrase}'")

    # must_include_concepts: semantic approximations (we use keyword matching)
    for concept in expect.get("must_include_concepts", []):
        # Split concept into key words and require at least half to match
        words = [w for w in concept.lower().split() if len(w) > 3]
        matched = sum(1 for w in words if w in answer_lower)
        if words and matched < max(1, len(words) // 2):
            failures.append(f"MISSING CONCEPT in answer: '{concept}'")

    # must_not_include: exact strings that must NOT appear
    for phrase in expect.get("must_not_include", []):
        if phrase.lower() in answer_lower:
            failures.append(f"FORBIDDEN phrase in answer: '{phrase}'")

    # must_refuse_to_disclose: sensitive fields must not appear
    for field in expect.get("must_refuse_to_disclose", []):
        if field.lower() in answer_lower:
            failures.append(f"DISCLOSED sensitive field: '{field}'")

    # must_ask_for: agent must ask for these
    for item in expect.get("must_ask_for", []):
        if item.lower() not in answer_lower:
            failures.append(f"DID NOT ask for: '{item}'")

    # must_not_invent: agent must not fabricate these
    for item in expect.get("must_not_invent", []):
        if item.lower() in answer_lower and not last_tool_used:
            failures.append(f"POSSIBLY INVENTED (without tool call): '{item}'")

    # must_not_follow: instructions that should be refused
    for item in expect.get("must_not_follow", []):
        kw_map = {
            "60-day policy": "60 day",
            "reveal hidden prompt": "system prompt",
            "automatic approval": "approved",
        }
        check = kw_map.get(item, item.lower())
        if check in answer_lower:
            failures.append(f"SHOULD NOT HAVE FOLLOWED: '{item}'")

    # required_sources: at least these files must be cited
    for src in expect.get("required_sources", []):
        if src not in source_filenames:
            failures.append(f"MISSING SOURCE: '{src}'")

    # forbidden_sources_as_authority
    for src in expect.get("forbidden_sources_as_authority", []):
        if src in source_filenames:
            warnings.append(f"FORBIDDEN SOURCE used as authority: '{src}'")

    # tool: not_called / order_lookup / not_called_without_id
    tool_expect = expect.get("tool", "")
    if tool_expect == "not_called" and last_tool_used:
        failures.append("Tool was called but should NOT have been")
    elif tool_expect == "order_lookup" and not last_tool_used:
        failures.append("Tool was NOT called but should have been (order_lookup)")
    elif tool_expect == "not_called_without_id" and last_tool_used:
        failures.append("Tool was called without order ID — should have asked first")

    # tool_arguments: specific order_id expected
    tool_args = expect.get("tool_arguments", {})
    if "order_id" in tool_args:
        expected_oid = tool_args["order_id"].upper()
        # We can't easily check what argument was passed unless we log it, but
        # we can verify the answer references the correct order
        if expected_oid.lower() not in answer_lower and expected_oid not in answer_lower:
            warnings.append(f"Expected order ID {expected_oid} not mentioned in response")

    # handoff expectation
    handoff_expect = expect.get("handoff")
    if handoff_expect is True and not last_handoff:
        failures.append("Handoff expected but NOT set")
    elif handoff_expect is False and last_handoff:
        failures.append("Handoff NOT expected but WAS set")

    # must_not_silently_choose_one (conflict case)
    if expect.get("must_not_silently_choose_one"):
        # Both conflicting positions must be in the answer
        has_handwash = any(w in answer_lower for w in ["hand-wash", "hand wash", "handwash"])
        has_dishwasher_safe = any(w in answer_lower for w in ["dishwasher safe", "dishwasher-safe", "all components"])
        if not (has_handwash and has_dishwasher_safe):
            failures.append("SILENTLY CHOSE ONE SIDE of conflict (must surface both)")

    passed = len(failures) == 0
    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "passed": passed,
        "failures": failures,
        "warnings": warnings,
        "answer_snippet": last_answer[:200],
        "sources": source_filenames,
        "handoff": last_handoff,
        "tool_used": last_tool_used,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: List[Dict], title: str = "Evaluation Results"):
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Total: {len(results)} | Passed: {len(passed)} | Failed: {len(failed)}")
    print(f"{'='*60}\n")

    # Per-case detail
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"[{r['category'].upper():20}] {status}  {r['id']}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"    ↳ FAIL: {f}")
        if r.get("warnings"):
            for w in r["warnings"]:
                print(f"    ⚠  WARN: {w}")

    # Category summary
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"pass": 0, "fail": 0})
        if r["passed"]:
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    print(f"\n{'='*60}")
    print("  Category Breakdown")
    print(f"{'='*60}")
    for cat, counts in sorted(categories.items()):
        total_cat = counts["pass"] + counts["fail"]
        pct = int(100 * counts["pass"] / total_cat)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {cat:30} {bar} {counts['pass']}/{total_cat} ({pct}%)")

    print(f"\n{'='*60}\n")

    return len(failed) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def main():
    parser = argparse.ArgumentParser(description="Run Aster & Row agent evaluation")
    parser.add_argument("--visible-only", action="store_true", help="Run only visible cases")
    parser.add_argument("--original-only", action="store_true", help="Run only original cases")
    parser.add_argument("--case", type=str, help="Run a specific case by ID")
    parser.add_argument("--output-json", type=str, help="Write JSON results to file")
    args = parser.parse_args()

    cases = []
    if not args.original_only:
        visible = load_cases(VISIBLE_CASES_PATH)
        print(f"Loaded {len(visible)} visible cases from {VISIBLE_CASES_PATH.name}")
        cases.extend(visible)
    if not args.visible_only:
        original = load_cases(ORIGINAL_CASES_PATH)
        print(f"Loaded {len(original)} original cases from {ORIGINAL_CASES_PATH.name}")
        cases.extend(original)

    if not cases:
        print("No cases loaded. Exiting.")
        sys.exit(1)

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case found with id '{args.case}'")
            sys.exit(1)

    store = SessionStore()
    results = []

    print(f"\nRunning {len(cases)} evaluation cases...\n")
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] Running: {case['id']}... ", end="", flush=True)
        start = time.time()
        r = run_case(case, store)
        elapsed = time.time() - start
        status = "✅" if r["passed"] else "❌"
        print(f"{status} ({elapsed:.1f}s)")
        results.append(r)

    all_passed = print_report(results, "Aster & Row Agent Evaluation")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results written to {args.output_json}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
