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


def test_check_breeze_tumbler_conflict_superseded_no_conflict():
    # 3. Superseded document does not create conflict
    chunk_care = DocumentChunk(
        doc_id="CARE-2026-01", filename="11-product-care.md", heading="Breeze Tumbler", content="body should be hand-washed",
        metadata={"status": "superseded", "policy_authority": "official", "audience": "customer"}
    )
    chunk_card = DocumentChunk(
        doc_id="PROD-BREEZE-20", filename="12-breeze-tumbler-product-card.md", heading="Cleaning", content="all components are dishwasher safe",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    query = "Can I wash my Breeze Tumbler in the dishwasher?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query, [chunk_care, chunk_card])
    assert has_conflict is False


def test_check_breeze_tumbler_conflict_draft_no_conflict():
    # 4. Draft document does not create conflict
    chunk_care = DocumentChunk(
        doc_id="CARE-2026-01", filename="11-product-care.md", heading="Breeze Tumbler", content="body should be hand-washed",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_card = DocumentChunk(
        doc_id="PROD-BREEZE-20", filename="12-breeze-tumbler-product-card.md", heading="Cleaning", content="all components are dishwasher safe",
        metadata={"status": "draft", "policy_authority": "official", "audience": "customer"}
    )
    query = "Can I wash my Breeze Tumbler in the dishwasher?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query, [chunk_care, chunk_card])
    assert has_conflict is False


def test_check_breeze_tumbler_non_conflicting_no_conflict():
    # 5. Two non-conflicting documents do not create conflict
    chunk_card = DocumentChunk(
        doc_id="PROD-BREEZE-20", filename="12-breeze-tumbler-product-card.md", heading="Cleaning", content="all components are dishwasher safe",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_other = DocumentChunk(
        doc_id="WAR-2026-01", filename="07-warranty.md", heading="Warranty", content="other info",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    query = "Can I wash my Breeze Tumbler in the dishwasher?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query, [chunk_card, chunk_other])
    assert has_conflict is False


def test_check_breeze_tumbler_paraphrased_detected():
    # 6. At least one paraphrased Breeze question still detects conflict
    chunk_care = DocumentChunk(
        doc_id="CARE-2026-01", filename="11-product-care.md", heading="Breeze Tumbler", content="body should be hand-washed",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_card = DocumentChunk(
        doc_id="PROD-BREEZE-20", filename="12-breeze-tumbler-product-card.md", heading="Cleaning", content="all components are dishwasher safe",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    query = "Is the whole metal bottle body safe to clean in the rack of a dishwasher?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query, [chunk_care, chunk_card])
    assert has_conflict is True


def test_conflict_produces_handoff_true():
    # 7. Conflict produces handoff = true
    from app.orchestrator import determine_handoff_reason
    reason = determine_handoff_reason(
        user_message="dishwasher safe body?",
        has_conflict=True,
        order_result=None
    )
    assert reason == "conflict"


def test_normal_policy_question_no_conflict():
    # 8. Normal policy question does not produce conflict
    chunk_care = DocumentChunk(
        doc_id="CARE-2026-01", filename="11-product-care.md", heading="Breeze Tumbler", content="body should be hand-washed",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_card = DocumentChunk(
        doc_id="PROD-BREEZE-20", filename="12-breeze-tumbler-product-card.md", heading="Cleaning", content="all components are dishwasher safe",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    query = "What is the return window?"
    has_conflict, msg, sources = check_breeze_tumbler_conflict(query, [chunk_care, chunk_card])
    assert has_conflict is False


def test_precedence_active_beats_superseded():
    chunk_active = DocumentChunk(
        doc_id="RET-2026-01", filename="01-returns-policy-current.md", heading="Return Window", content="30 days return window",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_superseded = DocumentChunk(
        doc_id="RET-2024-01", filename="02-returns-policy-legacy.md", heading="Return Window", content="60 days return window",
        metadata={"status": "superseded", "policy_authority": "official", "audience": "customer"}
    )
    filtered = filter_authoritative_chunks([(chunk_superseded, 0.9), (chunk_active, 0.7)])
    assert len(filtered) == 1
    assert filtered[0][0].doc_id == "RET-2026-01"


def test_precedence_active_beats_draft():
    chunk_active = DocumentChunk(
        doc_id="RET-2026-01", filename="01-returns-policy-current.md", heading="Return Window", content="30 days return window",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_draft = DocumentChunk(
        doc_id="MIG-TEST-04", filename="14-internal-content-migration-notes.md", heading="Notes", content="60 days draft",
        metadata={"status": "draft", "policy_authority": "official", "audience": "customer"}
    )
    filtered = filter_authoritative_chunks([(chunk_draft, 0.9), (chunk_active, 0.7)])
    assert len(filtered) == 1
    assert filtered[0][0].doc_id == "RET-2026-01"


def test_precedence_official_beats_non_authoritative():
    chunk_official = DocumentChunk(
        doc_id="RET-2026-01", filename="01-returns-policy-current.md", heading="Return Window", content="30 days return window",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_none = DocumentChunk(
        doc_id="MIG-TEST-04", filename="14-internal-content-migration-notes.md", heading="Notes", content="60 days draft",
        metadata={"status": "active", "policy_authority": "none", "audience": "customer"}
    )
    filtered = filter_authoritative_chunks([(chunk_none, 0.9), (chunk_official, 0.7)])
    assert len(filtered) == 1
    assert filtered[0][0].doc_id == "RET-2026-01"


def test_precedence_internal_excluded():
    chunk_internal = DocumentChunk(
        doc_id="SUP-2026-01", filename="13-support-escalation.md", heading="Escalation", content="internal rules",
        metadata={"status": "active", "policy_authority": "official", "audience": "internal"}
    )
    filtered = filter_authoritative_chunks([(chunk_internal, 0.9)])
    assert len(filtered) == 0


def test_precedence_explicit_supersedes_respected():
    chunk_new = DocumentChunk(
        doc_id="RET-2026-01", filename="01-returns-policy-current.md", heading="Return Window", content="30 days return window",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer", "supersedes": "RET-2024-01"}
    )
    chunk_old = DocumentChunk(
        doc_id="RET-2024-01", filename="02-returns-policy-legacy.md", heading="Return Window", content="60 days return window",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    filtered = filter_authoritative_chunks([(chunk_old, 0.9), (chunk_new, 0.7)])
    assert len(filtered) == 1
    assert filtered[0][0].doc_id == "RET-2026-01"


def test_precedence_superseded_document_not_cited():
    chunk_new = DocumentChunk(
        doc_id="RET-2026-01", filename="01-returns-policy-current.md", heading="Return Window", content="30 days return window",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer", "supersedes": "RET-2024-01"}
    )
    chunk_old = DocumentChunk(
        doc_id="RET-2024-01", filename="02-returns-policy-legacy.md", heading="Return Window", content="60 days return window",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    filtered = filter_authoritative_chunks([(chunk_old, 0.9), (chunk_new, 0.7)])
    doc_ids = [c.doc_id for c, _ in filtered]
    assert "RET-2024-01" not in doc_ids
    assert "RET-2026-01" in doc_ids


def test_precedence_unresolved_conflict_preserved():
    chunk_care = DocumentChunk(
        doc_id="CARE-2026-01", filename="11-product-care.md", heading="Breeze Tumbler", content="body should be hand-washed",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    chunk_card = DocumentChunk(
        doc_id="PROD-BREEZE-20", filename="12-breeze-tumbler-product-card.md", heading="Cleaning", content="all components are dishwasher safe",
        metadata={"status": "active", "policy_authority": "official", "audience": "customer"}
    )
    filtered = filter_authoritative_chunks([(chunk_care, 0.9), (chunk_card, 0.8)])
    assert len(filtered) == 2
