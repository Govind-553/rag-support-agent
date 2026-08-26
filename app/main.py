"""
FastAPI entrypoint - Aster & Row AI Support Agent
"""
import sys
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.models import ChatRequest, ChatResponse
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


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
