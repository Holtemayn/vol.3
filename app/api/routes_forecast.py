from __future__ import annotations

from datetime import date
from typing import Any, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.ml.hgb import (
    get_hgb_feature_columns,
    get_hgb_history_dates,
    get_hgb_metrics,
    get_last_history_source,
)
from app.services.forecast import ForecastResult, generate_forecast, generate_forecast_bundle

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


class HistoryCoverage(BaseModel):
    start: date
    end: date


class HGBForecastResponse(BaseModel):
    model_version: str
    model_backend: str
    wage_pct: float
    avg_hourly_wage: float
    warnings: List[str]
    items: List[DayForecast]
    metrics: Optional[dict[str, Any]] = None
    feature_columns: Optional[List[str]] = None
    history_coverage: Optional[HistoryCoverage] = None
    history_source: Optional[str] = None


def _to_day_forecasts(result: ForecastResult) -> List[DayForecast]:
    return [
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


def _gather_hgb_metadata() -> Tuple[Optional[dict[str, Any]], Optional[List[str]], Optional[HistoryCoverage]]:
    metrics: Optional[dict[str, Any]] = None
    feature_columns: Optional[List[str]] = None
    coverage: Optional[HistoryCoverage] = None

    def _to_date(value: Any) -> date:
        converter = getattr(value, "to_pydatetime", None)
        if callable(converter):
            return converter().date()
        date_attr = getattr(value, "date", None)
        if callable(date_attr):
            return date_attr()
        return date.fromisoformat(str(value)[:10])
    try:
        raw_metrics = get_hgb_metrics()
        if raw_metrics:
            metrics = raw_metrics
    except Exception:
        metrics = None

    try:
        columns = get_hgb_feature_columns()
        if columns:
            feature_columns = list(columns)
    except Exception:
        feature_columns = None

    try:
        history_dates = list(get_hgb_history_dates())
    except Exception:
        history_dates = []
    if history_dates:
        history_dates.sort()
        start = _to_date(history_dates[0])
        end = _to_date(history_dates[-1])
        coverage = HistoryCoverage(start=start, end=end)

    return metrics, feature_columns, coverage


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

    items = _to_day_forecasts(result)

    return ForecastResponse(
        model_version=result.model_version,
        model_backend=result.model_backend,
        wage_pct=wage_pct,
        avg_hourly_wage=hourly,
        warnings=result.warnings,
        items=items,
    )


@router.post("/hgb", response_model=HGBForecastResponse)
def forecast_hgb(req: ForecastRequest) -> HGBForecastResponse:
    try:
        primary, variants = generate_forecast_bundle(
            start_date=req.start_date,
            horizon_days=req.horizon_days,
            wage_pct=req.wage_pct,
            avg_hourly_wage=req.avg_hourly_wage,
            additional_backends=["hgb"],
            log_result=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - afhænger af eksterne services
        raise HTTPException(status_code=500, detail=f"Forecast-fejl: {exc}") from exc

    hgb_result = variants.get("hgb")
    if hgb_result is None and primary.model_backend.lower() == "hgb":
        hgb_result = primary
    if hgb_result is None:
        warnings = list(primary.warnings)
        warnings.append("HGB-resultat ikke tilgængeligt.")
        hgb_result = ForecastResult(
            model_version=primary.model_version,
            model_backend="hgb",
            rows=[],
            warnings=warnings,
        )

    wage_pct = req.wage_pct if req.wage_pct is not None else settings.WAGE_PCT
    hourly = req.avg_hourly_wage if req.avg_hourly_wage is not None else settings.AVG_HOURLY_WAGE
    metrics, feature_columns, coverage = _gather_hgb_metadata()
    history_source = get_last_history_source()

    return HGBForecastResponse(
        model_version=hgb_result.model_version,
        model_backend=hgb_result.model_backend,
        wage_pct=wage_pct,
        avg_hourly_wage=hourly,
        warnings=hgb_result.warnings,
        items=_to_day_forecasts(hgb_result),
        metrics=metrics,
        feature_columns=feature_columns,
        history_coverage=coverage,
        history_source=history_source,
    )
