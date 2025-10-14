from __future__ import annotations

from fastapi import APIRouter, Query

from app.services.forecast_log import read_recent_forecasts

router = APIRouter()


@router.get("", summary="Hent seneste forecast-logposter")
def get_forecast_logs(limit: int = Query(5, ge=1, le=50)):
    entries = read_recent_forecasts(limit=limit)
    return {"items": entries}
