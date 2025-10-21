from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.schemas import AgentChatEvent, AgentChatRequest
from app.agents.service import get_orchestrator
from app.agents.tools import execute_tool

router = APIRouter()
LOGGER = logging.getLogger(__name__)
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

_CONVERSATIONS: Dict[str, List[str]] = {}
_MAX_CHUNK = 220


def _resolve_thread_id(request: AgentChatRequest) -> str:
    thread_id = request.thread_id or "default"
    _CONVERSATIONS.setdefault(thread_id, [])
    return thread_id


async def _fetch_similar_context(message: str) -> Optional[dict]:
    match = DATE_PATTERN.search(message)
    target_date = match.group(0) if match else date.today().isoformat()
    try:
        return await execute_tool("find_similar_days", {"date": target_date, "k": 5})
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("similar_days_tool_failed: %s", exc)
        return None


def _format_number(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _context_to_prompt(context: dict) -> str:
    if not context:
        return ""
    summary = context.get("summary") or ""
    similar = context.get("similar_days") or []
    lines = ["Analoge dage til reference:"]
    if summary:
        lines.append(summary)
    for row in similar:
        lines.append(
            f"- {row.get('date')} ({row.get('weekday')}): oms={_format_number(row.get('revenue_pred'))} kr, "
            f"planday={_format_number(row.get('planday_hours'))} t, temp={_format_number(row.get('temp_max'))} °C, "
            f"nedbør={_format_number(row.get('precip_sum'))} mm"
        )
    return "\n".join(lines[:12])


def _context_events(thread_id: str, context: Optional[dict]) -> List[AgentChatEvent]:
    if not context:
        return []
    events: List[AgentChatEvent] = []
    summary = context.get("summary")
    if summary:
        events.append(AgentChatEvent(thread_id=thread_id, content=summary))
    similar = context.get("similar_days") or []
    if similar:
        header = "Dato | Ugedag | Omsætning | Planday timer | Temp | Nedbør"
        events.append(AgentChatEvent(thread_id=thread_id, content=header))
        for row in similar:
            line = (
                f"{row.get('date')} | {row.get('weekday')} | {_format_number(row.get('revenue_pred'))} kr | "
                f"{_format_number(row.get('planday_hours'))} t | {_format_number(row.get('temp_max'))} °C | "
                f"{_format_number(row.get('precip_sum'))} mm"
            )
            events.append(AgentChatEvent(thread_id=thread_id, content=line))
    return events


def _chunk_text(text: str) -> List[str]:
    chunks: List[str] = []
    buffer = ""
    for segment in text.split():
        if len(buffer) + len(segment) + 1 > _MAX_CHUNK:
            if buffer:
                chunks.append(buffer)
            buffer = segment
        else:
            buffer = f"{buffer} {segment}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks or [text]


@router.post("/chat")
async def chat(request: AgentChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="message is required")

    thread_id = _resolve_thread_id(request)
    history = _CONVERSATIONS[thread_id]
    history.append(f"user: {request.message}")

    similar_context = await _fetch_similar_context(request.message)
    context_prompt = _context_to_prompt(similar_context)
    context_events = _context_events(thread_id, similar_context)

    try:
        orchestrator = get_orchestrator()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    async def event_stream():
        try:
            for event in context_events:
                yield f"data: {event.model_dump_json()}\n\n"
                await asyncio.sleep(0)

            reply = await orchestrator.generate_reply(thread_id, request, context_prompt or None)
            history.append(f"assistant: {reply}")
            for chunk in _chunk_text(reply):
                payload = AgentChatEvent(thread_id=thread_id, content=chunk)
                yield f"data: {payload.model_dump_json()}\n\n"
                await asyncio.sleep(0)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("agent_stream_failed: %s", exc)
            error = AgentChatEvent(thread_id=thread_id, content="Der skete en fejl i agenten.")
            yield f"data: {error.model_dump_json()}\n\n"
        finally:
            yield f"data: {AgentChatEvent(thread_id=thread_id, content='', event='done').model_dump_json()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
