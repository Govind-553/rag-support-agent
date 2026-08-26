from typing import List
from app.rag.loader import DocumentChunk

def filter_authoritative_chunks(chunks_with_scores: List[tuple]) -> List[tuple]:
    """
    Filters retrieved chunks based on version precedence.
    Only allows 'active' documents with 'official' policy authority.
    Excludes superseded, draft, and non-authoritative documents.
    Resolves explicit supersession relationships (supersedes / superseded_by).
    """
    # 1. First pass: basic eligibility filtering
    eligible = []
    for chunk, score in chunks_with_scores:
        meta = chunk.metadata
        
        # Status must be active (not draft, superseded or missing status)
        status = meta.get("status", "").lower()
        if status != "active":
            continue
            
        # Policy authority must not be none
        authority = meta.get("policy_authority", "").lower()
        if authority == "none":
            continue
            
        # Audience must not be internal
        audience = meta.get("audience", "").lower()
        if audience == "internal":
            continue
            
        # Customer answering must not be False
        customer_answering = meta.get("customer_answering", True)
        if customer_answering is False:
            continue
            
        eligible.append((chunk, score))

    # 2. Second pass: resolve explicit supersession relationships
    superseded_ids = set()
    for chunk, _ in chunks_with_scores:
        meta = chunk.metadata
        doc_id = meta.get("document_id") or chunk.doc_id
        
        if "superseded_by" in meta and meta["superseded_by"]:
            superseded_ids.add(doc_id)
            
        if "supersedes" in meta and meta["supersedes"]:
            superseded_ids.add(meta["supersedes"])

    # 3. Third pass: filter out chunks from superseded documents
    filtered = []
    for chunk, score in eligible:
        doc_id = chunk.metadata.get("document_id") or chunk.doc_id
        if doc_id in superseded_ids:
            continue
        filtered.append((chunk, score))
        
    return filtered
