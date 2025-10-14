from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.openai_chat import get_advice

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Chatfunktion er ikke konfigureret")
    try:
        reply = get_advice([msg.dict() for msg in req.messages])
    except Exception as exc:  # pragma: no cover - netværk/SDK
        raise HTTPException(status_code=500, detail=f"Chat-fejl: {exc}") from exc
    return ChatResponse(reply=reply)
