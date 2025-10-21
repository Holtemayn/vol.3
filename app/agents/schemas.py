from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Input payload from the frontend to the agent chat endpoint."""

    message: str = Field(..., description="User message that should be routed to the agent")
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation identifier. If omitted, a new thread is created.",
    )


class AgentChatEvent(BaseModel):
    """Single chunk that is streamed back to the client."""

    thread_id: str
    content: str
    event: Literal["message", "done"] = "message"
    meta: Dict[str, Any] = Field(default_factory=dict)


class PlandayParams(BaseModel):
    date: str = Field(..., description="Dato i format YYYY-MM-DD")


class ReconcileParams(BaseModel):
    limit: int = Field(
        10,
        ge=1,
        le=50,
        description="Antal logposter der skal analyseres for afstemning",
    )


class SimilarDaysParams(BaseModel):
    date: str = Field(..., description="Dato i format YYYY-MM-DD, bruges som reference")
    k: int = Field(
        5,
        ge=1,
        le=15,
        description="Antal analoge dage der ønskes (1-15)",
    )


class HistoryQueryParams(BaseModel):
    start_date: str = Field(
        ...,
        description="Startdato (inklusive) i format YYYY-MM-DD",
    )
    end_date: str = Field(
        ...,
        description="Slutdato (inklusive) i format YYYY-MM-DD",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Maksimalt antal dokumenter der returneres",
    )
