"""FastAPI chatbot server.  Run:  uvicorn app:app --reload"""
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import database as db
from nlp_engine import ChatEngine

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
engine: ChatEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    db.init_db()
    engine = ChatEngine(os.getenv("FAQ_FILE", "faqs.json"))
    yield


app = FastAPI(title="AI Support Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    session_id: str | None = Field(None, max_length=64)
    message: str = Field(..., min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    message_id: int
    helpful: bool


def require_admin(x_admin_token: str = Header(default="")):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    db.ensure_session(session_id)
    context = db.recent_messages(session_id, limit=6)

    start = time.perf_counter()
    result = engine.respond(req.message, context)
    latency = int((time.perf_counter() - start) * 1000)

    db.log_message(session_id, "user", req.message, sentiment=result.get("sentiment"))
    bot_id = db.log_message(
        session_id, "bot", result["reply"], intent=result["intent"],
        matched_faq=result["matched_faq"], confidence=result["confidence"],
        sentiment=result.get("sentiment"), latency_ms=latency,
    )
    return {
        "session_id": session_id,
        "message_id": bot_id,
        "reply": result["reply"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "suggestions": result["suggestions"],
        "escalate": result["escalate"],
    }


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    return db.history(session_id)


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if not db.set_feedback(req.message_id, req.helpful):
        raise HTTPException(status_code=404, detail="Bot message not found")
    return {"ok": True}


@app.get("/api/logs", dependencies=[Depends(require_admin)])
def get_logs(limit: int = Query(100, le=1000), offset: int = 0):
    return db.logs(limit, offset)


@app.get("/api/stats", dependencies=[Depends(require_admin)])
def get_stats():
    return db.stats()


@app.get("/health")
def health():
    return {"status": "ok", "faqs": len(engine.faqs) if engine else 0}
