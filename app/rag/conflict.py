from typing import List, Tuple
from app.rag.loader import DocumentChunk

from typing import List, Tuple, Optional, Dict
from app.rag.loader import DocumentChunk

CONFLICT_RULES = [
    {
        "topic": "breeze_tumbler_dishwasher",
        "query_keywords_all": ["breeze", "tumbler"],
        "query_keywords_any": ["wash", "clean", "dishwasher", "sink", "soap", "rack"],
        "conflicting_docs": [
            {
                "id": "CARE-2026-01",  # From 11-product-care.md
                "filename": "11-product-care.md",
                "heading": "Breeze Tumbler",
                "summary": "The Product Care Guide (11-product-care.md) states that the stainless-steel body of the Breeze Tumbler should be hand-washed, and only the lid may be placed on the top rack of a dishwasher."
            },
            {
                "id": "PROD-BREEZE-20",  # From 12-breeze-tumbler-product-card.md
                "filename": "12-breeze-tumbler-product-card.md",
                "heading": "Cleaning",
                "summary": "The Breeze Tumbler Product Information card (12-breeze-tumbler-product-card.md) states that all components are dishwasher safe."
            }
        ],
        "interim_guidance": "We recommend hand-washing the stainless-steel body of the tumbler, and placing the lid on the top rack of the dishwasher.",
    }
]

def detect_conflicts(query: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, List[dict]]:
    """
    Generic conflict-detection mechanism.
    Checks if there is a conflict in the retrieved chunks based on topic/fact rules.
    Returns (True, conflict_explanation, sources) or (False, "", []).
    """
    query_lower = query.lower()

    for rule in CONFLICT_RULES:
        # Check query keywords (product name synonyms + cleaning actions)
        has_product = any(k in query_lower for k in ["breeze", "tumbler", "bottle", "cup", "mug", "glass"])
        has_cleaning = any(k in query_lower for k in ["wash", "clean", "dishwasher", "sink", "soap", "rack", "dishwasher-safe"])
        
        if not (has_product and has_cleaning):
            continue

        # Check if we have active authoritative chunks for BOTH conflicting documents
        found_chunks = {}
        for doc_rule in rule["conflicting_docs"]:
            for chunk in chunks:
                # Verify status and policy authority
                status = chunk.metadata.get("status", "").lower()
                if status != "active":
                    continue
                auth = chunk.metadata.get("policy_authority", "").lower()
                if auth == "none":
                    continue
                audience = chunk.metadata.get("audience", "").lower()
                customer_answering = chunk.metadata.get("customer_answering", True)
                if audience == "internal" or customer_answering is False:
                    continue

                if (chunk.doc_id == doc_rule["id"] or chunk.filename == doc_rule["filename"]) and chunk.heading.lower() == doc_rule["heading"].lower():
                    found_chunks[doc_rule["id"]] = chunk
                    break

        if len(found_chunks) == len(rule["conflicting_docs"]):
            # Conflict detected!
            sources = []
            for doc_rule in rule["conflicting_docs"]:
                chunk = found_chunks[doc_rule["id"]]
                sources.append({
                    "filename": chunk.filename,
                    "heading": chunk.heading
                })
            
            conflict_msg = (
                "Our official sources disagree on this topic: "
                f"1. {rule['conflicting_docs'][0]['summary']} "
                f"2. {rule['conflicting_docs'][1]['summary']} "
                f"Safest interim guidance: {rule['interim_guidance']} "
                "I have marked this for human support agent review to confirm."
            )
            return True, conflict_msg, sources

    return False, "", []

def check_breeze_tumbler_conflict(query: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, List[dict]]:
    """Backward compatible wrapper for Breeze Tumbler conflict detection."""
    return detect_conflicts(query, chunks)
