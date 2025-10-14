from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.services.forecast import ForecastResult, generate_forecast

router = APIRouter()


class ForecastRequest(BaseModel):
    start_date: date = Field(..., description="Første dato i forecastet")
    horizon_days: int = Field(default=9, ge=1, le=14, description="Antal dage (1-14)")
    wage_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Andel af omsætning til løn")
    avg_hourly_wage: Optional[float] = Field(default=None, gt=0.0, description="Gennemsnitlig timeløn i kr.")

    @validator("start_date")
    def _validate_start(cls, value: date) -> date:
        if value < date(2020, 1, 1):
            raise ValueError("start_date virker forkert – forventer dato efter 2020")
        return value


class DayForecast(BaseModel):
    date: date
    revenue_pred: float
    hours_recommended: float
    planday_hours: Optional[float] = None
    hours_diff: Optional[float] = None
    temp_max: Optional[float] = None
    precip_sum: Optional[float] = None
    sunshine_hours: Optional[float] = None
    wind_max: Optional[float] = None


class ForecastResponse(BaseModel):
    model_version: str
    model_backend: str
    wage_pct: float
    avg_hourly_wage: float
    warnings: List[str]
    items: List[DayForecast]


@router.post("", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    try:
        result: ForecastResult = generate_forecast(
            start_date=req.start_date,
            horizon_days=req.horizon_days,
            wage_pct=req.wage_pct,
            avg_hourly_wage=req.avg_hourly_wage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - afhænger af eksterne services
        raise HTTPException(status_code=500, detail=f"Forecast-fejl: {exc}") from exc

    wage_pct = req.wage_pct if req.wage_pct is not None else settings.WAGE_PCT
    hourly = req.avg_hourly_wage if req.avg_hourly_wage is not None else settings.AVG_HOURLY_WAGE

    items = [
        DayForecast(
            date=row.date,
            revenue_pred=row.revenue_pred,
            hours_recommended=row.hours_recommended,
            planday_hours=row.planday_hours,
            hours_diff=row.hours_diff,
            temp_max=row.temp_max,
            precip_sum=row.precip_sum,
            sunshine_hours=row.sunshine_hours,
            wind_max=row.wind_max,
        )
        for row in result.rows
    ]

    return ForecastResponse(
        model_version=result.model_version,
        model_backend=result.model_backend,
        wage_pct=wage_pct,
        avg_hourly_wage=hourly,
        warnings=result.warnings,
        items=items,
    )
