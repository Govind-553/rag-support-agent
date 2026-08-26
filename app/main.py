"""
FastAPI entrypoint — Aster & Row AI Support Agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from pathlib import Path

from app.models import ChatRequest, ChatResponse, SourceCitation
from app.logging.logger import logger
from app.session import session_store
from app.orchestrator import run_turn

app = FastAPI(title="Aster & Row AI Support Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    trace_id = str(uuid.uuid4())
    session = session_store.get_or_create(request.session_id)

    result = run_turn(
        user_message=request.message,
        session=session,
        trace_id=trace_id,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        handoff=result["handoff"],
        trace_id=result["trace_id"],
        tool_used=result["tool_used"],
        handoff_reason=result.get("handoff_reason"),
        tool_calls=result.get("tool_calls"),
    )


# Serve static files if index.html exists
static_path = Path("c:/Users/choud/rag-support-agent/static")
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
