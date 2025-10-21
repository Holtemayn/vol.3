from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

from app.services.history import fetch_daily_aggregates, find_similar_days

router = APIRouter()


class HistoryRequest(BaseModel):
    start_date: str = Field(..., description="Startdato i ISO-format (YYYY-MM-DD)")
    end_date: str = Field(..., description="Slutdato i ISO-format (YYYY-MM-DD)")
    limit: int | None = Field(default=None, ge=1, le=365)

    @validator("end_date")
    def _validate_order(cls, value: str, values):
        start = values.get("start_date")
        if start and value < start:
            raise ValueError("end_date skal være efter start_date")
        return value


@router.post("/aggregates")
def read_history(payload: HistoryRequest):
    try:
        documents = fetch_daily_aggregates(
            start_date=payload.start_date,
            end_date=payload.end_date,
            limit=payload.limit,
        )
    except Exception as exc:  # pragma: no cover - eksterne forbindelser
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"items": documents, "count": len(documents)}


@router.get("/similar")
def get_similar_days(
    date_param: str = Query(..., alias="date", description="Dato i ISO-format (YYYY-MM-DD)"),
):
    try:
        target_date = date.fromisoformat(date_param)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Ugyldig dato") from exc

    try:
        payload = find_similar_days(target_date=target_date)
    except Exception as exc:  # pragma: no cover - eksterne forbindelser
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return payload
