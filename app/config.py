import os
from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path("c:/Users/choud/rag-support-agent")
KNOWLEDGE_BASE_DIR = WORKSPACE_ROOT / "knowledge-base"
DATA_DIR = WORKSPACE_ROOT / "data"
ORDERS_JSON_PATH = DATA_DIR / "orders.json"
INDEX_DIR = DATA_DIR / "faiss_index"

# Model configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Snapshot timestamp for cancellation/time window logic
SNAPSHOT_AT = "2026-08-15T12:00:00Z"

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
