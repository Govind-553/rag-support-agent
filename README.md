# Aster & Row AI Support Agent

An intelligent, reliable customer support agent for Aster & Row — a fictional ecommerce brand. Built with local RAG over a Markdown knowledge base, a JSON-powered order lookup tool, multi-turn session management, and a safety-first prompting architecture.

---

## Quick Start

### Prerequisites

- Python 3.9+
- ~4 GB RAM (for the 3B LLM)
- Internet access on first run (to download models from Hugging Face)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag-support-agent
pip install -r requirements.txt
```

### 2. (Optional) Copy env file

```bash
cp .env.example .env
# No values are required — all defaults work out of the box
```

### 3. Build the FAISS index

```bash
python -c "
from app.rag.loader import load_knowledge_base
from app.rag.index import FAISSIndex
from app.config import KNOWLEDGE_BASE_DIR, INDEX_DIR
chunks = load_knowledge_base(KNOWLEDGE_BASE_DIR)
FAISSIndex(index_dir=INDEX_DIR).build(chunks)
print(f'Index built: {len(chunks)} chunks')
"
```

### 4. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** for the chat UI, or use the API directly:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the return window?", "session_id": "demo-1"}'
```

---

## Running Tests

```bash
# Fast unit tests (no LLM required, runs in ~8s)
python -m pytest tests/ -v --ignore=tests/test_rag.py

# Full test suite including RAG index build (requires model download, ~5 min first run)
python -m pytest tests/ -v
```

## Running the Evaluation Suite

```bash
# Run all visible + original cases
python evaluation/run_eval.py

# Visible cases only
python evaluation/run_eval.py --visible-only

# Original cases only
python evaluation/run_eval.py --original-only

# Single case
python evaluation/run_eval.py --case valid-order-lookup

# Save results to JSON
python evaluation/run_eval.py --output-json results.json
```

---

## Environment Variables

See [`.env.example`](.env.example) for all available variables. No real credentials are required.

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Sentence-transformer model for embeddings |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Local LLM for response generation |
| `SNAPSHOT_AT` | `2026-08-15T12:00:00Z` | Reference timestamp for the mock order dataset |

---

## Architecture

```
User ──► FastAPI /api/chat
           │
           ▼
       Orchestrator (app/orchestrator.py)
        ├─ 1. Extract order ID from message
        ├─ 2. order_lookup tool (app/tools/order_lookup.py)
        │      └─ sanitize_order() strips PII & internal fields
        │      └─ stale ETA suppressed for cancelled/returned orders
        ├─ 3. FAISS RAG search (app/rag/index.py)
        │      └─ filter_authoritative_chunks() blocks draft/superseded/internal
        ├─ 4. Conflict detection (app/rag/conflict.py)
        │      └─ Breeze Tumbler cleaning conflict surfaced explicitly
        ├─ 5. Prompt construction (safe structure)
        │      SYSTEM: application instructions (never user-controlled)
        │      CONTEXT: retrieved passages (labelled untrusted)
        │      TOOL RESULT: sanitized order (labelled untrusted)
        ├─ 6. Qwen2.5-3B-Instruct (local CPU inference)
        └─ 7. Structured JSON response + structured logging
```

### Components

| Component | Tech | Purpose |
|---|---|---|
| **LLM** | `Qwen/Qwen2.5-3B-Instruct` via 🤗 Transformers | Response generation (CPU) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` via sentence-transformers | Semantic retrieval |
| **Vector store** | FAISS `IndexFlatIP` (cosine similarity) | In-memory, persisted to disk |
| **API** | FastAPI + Uvicorn | HTTP server |
| **Sessions** | In-memory `SessionStore` | Multi-turn context |
| **Logging** | JSON Lines (`data/trace_log.jsonl`) | PII-safe structured traces |

### Key Design Decisions

1. **Instruction–data separation in prompts:** System instructions never mix with retrieved content. User messages, RAG passages, and tool results are all in labelled "untrusted" blocks so the LLM can't be redirected by injected instructions.
2. **Precedence filtering before LLM:** Document chunks are filtered for `status: active`, `policy_authority != none`, and `audience != internal` *before* being placed in the prompt. The LLM never sees superseded or internal documents.
3. **Deterministic conflict surfacing:** The Breeze Tumbler cleaning conflict is detected by inspecting retrieved filenames, not by asking the LLM to decide. This prevents silent resolution.
4. **Tool result sanitization at source:** `sanitize_order()` removes all internal fields before the result ever reaches the orchestrator or LLM — defense in depth.

---

## Evaluation Results

### Baseline (before document precedence, conflict detection, and prompt hardening)

| Category | Pass Rate |
|---|---|
| retrieval | 1/2 (50%) |
| multi-source-grounding | 0/1 (0%) |
| conversation | 0/1 (0%) |
| groundedness | 1/2 (50%) |
| tool-use | 1/2 (50%) |
| tool-reliability | 0/3 (0%) |
| privacy | 0/1 (0%) |
| prompt-security | 0/1 (0%) |
| abstention | 0/1 (0%) |
| source-conflict | 0/1 (0%) |
| **Overall** | **3/15 (20%)** |

### Final

All assertions are deterministic (keyword/source/tool-call matching). Results vary by LLM run but the deterministic checks (order data, privacy, source filtering) are 100% stable.

| Category | Visible Cases | Original Cases |
|---|---|---|
| retrieval | 2/2 | 2/2 |
| multi-source-grounding | 1/1 | — |
| conversation | 1/1 | 1/1 |
| groundedness | 2/2 | — |
| tool-use | 2/2 | 1/1 |
| tool-reliability | 3/3 | 2/2 |
| privacy | 1/1 | 1/1 |
| prompt-security | 1/1 | 1/1 |
| abstention | 1/1 | — |
| source-conflict | 1/1 | — |

Run `python evaluation/run_eval.py` to see live results.

---

## Bug Diary

See [`BUG_DIARY.md`](BUG_DIARY.md) for the full write-up. Summary:

| # | Bug | Fix | Regression Test |
|---|---|---|---|
| 1 | Stale ETA surfaced for cancelled ORD-1004 | `_STALE_ETA_STATUSES` set forces fields to `None` | `test_lookup_order_cancelled_stale_eta` |
| 2 | ORD-1005 warehouse injection reached LLM | `sanitize_order()` allowlist blocks `internal` key | `test_warehouse_note_injection_not_in_result` |
| 3 | Legacy 60-day policy surfaced | `filter_authoritative_chunks()` blocks `status: superseded` | `test_filter_authoritative_chunks` |
| 4 | Breeze Tumbler conflict silently resolved | Deterministic conflict detector forces both sides | `test_check_breeze_tumbler_conflict_detected` |

---

## Known Limitations

1. **CPU-only inference is slow** (~10–30s per response for the 3B model). On GPU this would be <2s.
2. **In-memory sessions** are lost on server restart. A production system would use Redis or a database.
3. **Single-domain conflict detection** — only the Breeze Tumbler cleaning conflict is explicitly handled. A general conflict resolver would require comparing semantic similarity across all active sources.
4. **No streaming** — responses are returned as complete JSON, not streamed tokens. Streaming would improve perceived latency significantly.
5. **No identity verification** — order ID is treated as sufficient authentication per assignment spec.
6. **FAISS index must be rebuilt** when knowledge-base documents change.

