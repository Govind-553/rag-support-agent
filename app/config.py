import os
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader so local model switches work without extra setup."""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(WORKSPACE_ROOT / ".env")

# Base Paths
KNOWLEDGE_BASE_DIR = WORKSPACE_ROOT / "knowledge-base"
DATA_DIR = WORKSPACE_ROOT / "data"
ORDERS_JSON_PATH = DATA_DIR / "orders.json"
INDEX_DIR = DATA_DIR / "faiss_index"
STATIC_DIR = WORKSPACE_ROOT / "static"

# Model configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
DEVELOPMENT_FALLBACK_LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
INTENDED_SUBMISSION_LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", DEVELOPMENT_FALLBACK_LLM_MODEL_NAME)
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "192"))

# Snapshot timestamp for cancellation/time window logic
SNAPSHOT_AT = os.getenv("SNAPSHOT_AT", "2026-08-15T12:00:00Z")

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
