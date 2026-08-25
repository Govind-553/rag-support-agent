from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str

class SourceCitation(BaseModel):
    filename: str
    heading: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    handoff: bool
    trace_id: str
    tool_used: bool
