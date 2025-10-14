from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

import pandas as pd

from app.core.config import settings
from app.ml.features import build_feature_frame
from app.ml.model_loader import get_model
from app.ml.regression import predict_regression
from app.services.forecast_log import log_forecast_event
from app.services.planday import get_last_planday_warning, get_planday_hours_for_dates
from app.services.weather import get_last_weather_warning, get_weather_df


@dataclass
class ForecastRow:
    date: date
    revenue_pred: float
    hours_recommended: float
    planday_hours: float | None
    hours_diff: float | None
    temp_max: float | None = None
    precip_sum: float | None = None
    sunshine_hours: float | None = None
    wind_max: float | None = None


@dataclass
class ForecastResult:
    model_version: str
    model_backend: str
    rows: List[ForecastRow]
    warnings: List[str]


def generate_dates(start_date: date, horizon_days: int) -> List[date]:
    return [start_date + timedelta(days=i) for i in range(horizon_days)]


def generate_forecast(
    start_date: date,
    horizon_days: int,
    wage_pct: float | None = None,
    avg_hourly_wage: float | None = None,
) -> ForecastResult:
    dates = generate_dates(start_date, horizon_days)
    weather_payload = get_weather_df(dates)
    feature_frame = build_feature_frame(
        dates=[pd.Timestamp(d) for d in dates],
        weather=weather_payload,
    )

    forecast_df, backend_used, warning_messages = _predict(feature_frame)
    result_df = forecast_df.merge(feature_frame, on="date", how="left")

    wage_pct = wage_pct if wage_pct is not None else settings.WAGE_PCT
    hourly = avg_hourly_wage if avg_hourly_wage is not None else settings.AVG_HOURLY_WAGE
    result_df["hours_recommended"] = (result_df["revenue_pred"] * wage_pct) / hourly

    planday_map = get_planday_hours_for_dates(dates)
    result_df["planday_hours"] = result_df["date"].dt.date.map(planday_map.get)
    result_df["hours_diff"] = result_df.apply(
        lambda row: None
        if row["planday_hours"] is None
        else float(row["hours_recommended"] - row["planday_hours"]),
        axis=1,
    )

    rows = [
        ForecastRow(
            date=row.date.date(),
            revenue_pred=float(row.revenue_pred),
            hours_recommended=float(row.hours_recommended),
            planday_hours=float(row.planday_hours) if row.planday_hours is not None else None,
            hours_diff=float(row.hours_diff) if row.hours_diff is not None else None,
            temp_max=float(row.temp_max) if pd.notna(row.temp_max) else None,
            precip_sum=float(row.precip_sum) if pd.notna(row.precip_sum) else None,
            sunshine_hours=float(row.sunshine_hours) if pd.notna(row.sunshine_hours) else None,
            wind_max=float(row.wind_max) if pd.notna(row.wind_max) else None,
        )
        for row in result_df.itertuples(index=False)
    ]

    warnings = warning_messages.copy()
    weather_warning = get_last_weather_warning()
    if weather_warning:
        warnings.append(weather_warning)
    planday_warning = get_last_planday_warning()
    if planday_warning:
        warnings.append(planday_warning)

    result = ForecastResult(
        model_version=settings.MODEL_VERSION,
        model_backend=backend_used,
        rows=rows,
        warnings=warnings,
    )

    # Log forecast + tilhørende vejrdata. Fejler logningen, fortsætter vi alligevel.
    try:
        log_forecast_event(
            start_date=start_date,
            horizon_days=horizon_days,
            result=result,
            weather_frame=weather_payload.frame,
        )
    except Exception:
        pass

    return result


def _predict(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, str, List[str]]:
    warnings: List[str] = []
    backend = settings.MODEL_BACKEND.lower()
    if backend == "xgboost":
        try:
            model = get_model()
            feature_cols = [col for col in feature_frame.columns if col not in ("date", "rain_group")]
            preds = model.predict(feature_frame[feature_cols])
            df = pd.DataFrame({"date": feature_frame["date"], "revenue_pred": preds})
            return df, "xgboost", warnings
        except Exception as exc:  # pragma: no cover - afhænger af model/fil
            warnings.append(f"XGBoost fallback: {exc}")
            backend = "regression"

    if backend == "regression":
        df = predict_regression(feature_frame)
        return df, "regression", warnings

    raise ValueError(f"Ukendt MODEL_BACKEND: {settings.MODEL_BACKEND}")
