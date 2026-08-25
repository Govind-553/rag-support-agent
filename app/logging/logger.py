import json
import logging
import os
from pathlib import Path
from datetime import datetime
from app.config import DATA_DIR

# Custom structured logger
TRACE_LOG_PATH = DATA_DIR / "trace_log.jsonl"

def sanitize_for_logging(data: dict) -> dict:
    """Strips sensitive info from logs to prevent PII/secret leaks."""
    if not isinstance(data, dict):
        return data
    
    sanitized = {}
    for k, v in data.items():
        if k in ("email", "shipping_address", "customer", "customer.name", "customer.email", "customer.shipping_address", "name", "address"):
            sanitized[k] = "[REDACTED]"
        elif k in ("internal", "risk_score", "warehouse_note", "support_tags"):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_for_logging(v)
        elif isinstance(v, list):
            sanitized[k] = [sanitize_for_logging(item) if isinstance(item, dict) else item for item in v]
        else:
            sanitized[k] = v
    return sanitized

class StructuredLogger:
    def __init__(self, log_path: Path = TRACE_LOG_PATH):
        self.log_path = log_path
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_turn(self, trace_id: str, session_id: str, turn_data: dict):
        """Logs a single chat turn in JSON lines format."""
        sanitized_data = sanitize_for_logging(turn_data)
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "session_id": session_id,
            "data": sanitized_data
        }
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

logger = StructuredLogger()
