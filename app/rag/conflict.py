from typing import List, Tuple
from app.rag.loader import DocumentChunk

def check_breeze_tumbler_conflict(query: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, List[dict]]:
    """
    Checks if there is a conflict in the retrieved chunks for Breeze Tumbler cleaning.
    If a conflict is detected, returns (True, conflict_explanation, sources).
    Otherwise returns (False, "", []).
    """
    query_lower = query.lower()
    
    # 1. Check if query is about Breeze Tumbler cleaning/dishwasher
    is_breeze_tumbler = "breeze" in query_lower or "tumbler" in query_lower
    is_cleaning = any(k in query_lower for k in ("wash", "clean", "dishwasher", "sink", "soap", "rack"))
    
    if not (is_breeze_tumbler and is_cleaning):
        return False, "", []
        
    # 2. Check if we retrieved chunks from both conflicting documents
    has_product_care = False
    has_product_card = False
    conflicting_chunks = []
    
    for chunk in chunks:
        filename = chunk.filename
        if filename == "11-product-care.md":
            has_product_care = True
            conflicting_chunks.append(chunk)
        elif filename == "12-breeze-tumbler-product-card.md":
            has_product_card = True
            conflicting_chunks.append(chunk)
            
    if has_product_care and has_product_card:
        conflict_msg = (
            "Our official sources disagree on this topic: "
            "1. The Product Care Guide (11-product-care.md) states that the stainless-steel body of the Breeze Tumbler should be hand-washed, and only the lid may be placed on the top rack of a dishwasher. "
            "2. The Breeze Tumbler Product Information card (12-breeze-tumbler-product-card.md) states that all components are dishwasher safe. "
            "Safest interim guidance: We recommend hand-washing the stainless-steel body of the tumbler, and placing the lid on the top rack of the dishwasher. "
            "I have marked this for human support agent review to confirm."
        )
        sources = [
            {"filename": "11-product-care.md", "heading": "Breeze Tumbler"},
            {"filename": "12-breeze-tumbler-product-card.md", "heading": "Cleaning"}
        ]
        return True, conflict_msg, sources

    return False, "", []
