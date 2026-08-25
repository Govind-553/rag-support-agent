import pytest
from app.rag.loader import DocumentChunk
from app.rag.precedence import filter_authoritative_chunks
from app.rag.conflict import check_breeze_tumbler_conflict

def test_filter_authoritative_chunks():
    # Active, Official, Customer
    chunk1 = DocumentChunk(
        doc_id="C1", filename="doc1.md", heading="H1", content="C1 text",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    # Draft
    chunk2 = DocumentChunk(
        doc_id="C2", filename="doc2.md", heading="H2", content="C2 text",
        metadata={"status": "draft", "policy_authority": "official", "audience": "customer"}
    )
    # Superseded
    chunk3 = DocumentChunk(
        doc_id="C3", filename="doc3.md", heading="H3", content="C3 text",
        metadata={"status": "superseded", "policy_authority": "official", "audience": "customer"}
    )
    # Internal
    chunk4 = DocumentChunk(
        doc_id="C4", filename="doc4.md", heading="H4", content="C4 text",
        metadata={"status": "active", "policy_authority": "official", "audience": "internal"}
    )
    # No policy authority
    chunk5 = DocumentChunk(
        doc_id="C5", filename="doc5.md", heading="H5", content="C5 text",
        metadata={"status": "active", "policy_authority": "none", "audience": "customer"}
    )

    candidates = [
        (chunk1, 0.9),
        (chunk2, 0.8),
        (chunk3, 0.7),
        (chunk4, 0.6),
        (chunk5, 0.5),
    ]

    filtered = filter_authoritative_chunks(candidates)
    assert len(filtered) == 1
    assert filtered[0][0].doc_id == "C1"

def test_check_breeze_tumbler_conflict_detected():
    chunk_care = DocumentChunk(
        doc_id="11", filename="11-product-care.md", heading="Breeze Tumbler", content="body should be hand-washed",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_card = DocumentChunk(
        doc_id="12", filename="12-breeze-tumbler-product-card.md", heading="Cleaning", content="all components are dishwasher safe",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )

    # Valid conflict query
    query = "Can I wash my Breeze Tumbler in the dishwasher?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query, [chunk_care, chunk_card])
    assert has_conflict is True
    assert "disagree" in msg
    assert "hand-wash" in msg
    assert len(sources) == 2
    assert sources[0]["filename"] == "11-product-care.md"
    assert sources[1]["filename"] == "12-breeze-tumbler-product-card.md"

def test_check_breeze_tumbler_conflict_not_detected():
    chunk_care = DocumentChunk(
        doc_id="11", filename="11-product-care.md", heading="Breeze Tumbler", content="body should be hand-washed",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_other = DocumentChunk(
        doc_id="other", filename="other.md", heading="Other", content="other info",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )

    # Missing one of the conflicting documents
    query = "Can I wash my Breeze Tumbler in the dishwasher?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query, [chunk_care, chunk_other])
    assert has_conflict is False

    # Query is unrelated to washing
    query_unrelated = "Where can I buy the Breeze Tumbler?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query_unrelated, [chunk_care, chunk_other])
    assert has_conflict is False
