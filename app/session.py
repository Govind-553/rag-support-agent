"""
Session Store — Phase 6

In-memory session management for multi-turn conversation context.
Each session stores a bounded history of turns and a summary
of the last resolved order ID (to handle implicit follow-ups).
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time

MAX_TURNS = 10  # Keep last N turns to avoid context bloat


@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str


@dataclass
class Session:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    last_order_id: Optional[str] = None   # last successfully looked-up order
    last_updated: float = field(default_factory=time.time)

    def add_turn(self, role: str, content: str):
        self.turns.append(Turn(role=role, content=content))
        # Keep only the last MAX_TURNS turns
        if len(self.turns) > MAX_TURNS * 2:
            self.turns = self.turns[-(MAX_TURNS * 2):]
        self.last_updated = time.time()

    def get_history(self) -> List[Dict[str, str]]:
        """Returns turn history as a list of dicts for prompt building."""
        return [{"role": t.role, "content": t.content} for t in self.turns]


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)


# Singleton session store
session_store = SessionStore()
