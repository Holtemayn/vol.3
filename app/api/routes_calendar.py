from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.services.calendar_forecast import (
    CalendarForecastRequest,
    CalendarForecastResponse,
    calculate_calendar_forecast,
)

router = APIRouter()


@router.post("", response_model=CalendarForecastResponse)
async def calendar_forecast(
    request: CalendarForecastRequest,
) -> CalendarForecastResponse:
    try:
        return await run_in_threadpool(calculate_calendar_forecast, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
