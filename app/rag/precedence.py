from typing import List
from app.rag.loader import DocumentChunk

def filter_authoritative_chunks(chunks_with_scores: List[tuple]) -> List[tuple]:
    """
    Filters retrieved chunks based on version precedence.
    Only allows 'active' documents with 'official' policy authority.
    Excludes superseded, draft, and non-authoritative documents.
    """
    filtered = []
    for chunk, score in chunks_with_scores:
        meta = chunk.metadata
        
        # 1. Check status (must be active, not draft or superseded)
        status = meta.get("status", "").lower()
        if status != "active":
            continue
            
        # 2. Check policy authority
        authority = meta.get("policy_authority", "").lower()
        if authority == "none":
            continue
            
        # 3. Check audience / customer answering
        audience = meta.get("audience", "").lower()
        customer_answering = meta.get("customer_answering", True)
        
        # If it is explicitly marked internal or not for customer answering, skip for customer response grounding
        if audience == "internal" or customer_answering is False:
            continue
            
        filtered.append((chunk, score))
        
    return filtered
