from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, List, Tuple

import pandas as pd

from app.core.config import settings
from app.ml.features import build_feature_frame
from app.ml.hgb import predict_hgb
from app.ml.model_loader import get_model
from app.ml.regression import predict_regression
from app.services.forecast_log import log_forecast_event
from app.services.planday import get_last_planday_warning, get_planday_hours_for_dates
from app.services.weather import WeatherPayload, get_last_weather_warning, get_weather_df


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


@dataclass
class ForecastContext:
    dates: List[date]
    feature_frame: pd.DataFrame
    weather_payload: WeatherPayload
    wage_pct: float
    hourly_wage: float
    planday_map: dict[date, float | None]
    weather_warning: str | None
    planday_warning: str | None


def generate_dates(start_date: date, horizon_days: int) -> List[date]:
    return [start_date + timedelta(days=i) for i in range(horizon_days)]


def generate_forecast(
    start_date: date,
    horizon_days: int,
    wage_pct: float | None = None,
    avg_hourly_wage: float | None = None,
) -> ForecastResult:
    context = _build_context(
        start_date=start_date,
        horizon_days=horizon_days,
        wage_pct=wage_pct,
        avg_hourly_wage=avg_hourly_wage,
    )
    result, variants = _generate_results(context, additional_backends=("hgb",))
    preferred = variants.get("hgb")
    if preferred and preferred.rows:
        result = preferred
    _log_forecast_event(
        start_date=start_date,
        horizon_days=horizon_days,
        result=result,
        weather_frame=context.weather_payload.frame,
    )
    return result


def generate_forecast_bundle(
    start_date: date,
    horizon_days: int,
    wage_pct: float | None = None,
    avg_hourly_wage: float | None = None,
    additional_backends: Iterable[str] | None = None,
    log_result: bool = False,
) -> Tuple[ForecastResult, dict[str, ForecastResult]]:
    context = _build_context(
        start_date=start_date,
        horizon_days=horizon_days,
        wage_pct=wage_pct,
        avg_hourly_wage=avg_hourly_wage,
    )
    result, variants = _generate_results(context, additional_backends or ())
    if log_result:
        _log_forecast_event(
            start_date=start_date,
            horizon_days=horizon_days,
            result=result,
            weather_frame=context.weather_payload.frame,
        )
    return result, variants


def _predict(
    feature_frame: pd.DataFrame,
    backend_override: str | None = None,
    allow_fallback: bool = True,
) -> tuple[pd.DataFrame, str, List[str]]:
    warnings: List[str] = []
    backend = (backend_override or settings.MODEL_BACKEND).lower()
    if backend == "xgboost":
        try:
            model = get_model()
            feature_cols = [col for col in feature_frame.columns if col not in ("date", "rain_group")]
            preds = model.predict(feature_frame[feature_cols])
            df = pd.DataFrame({"date": feature_frame["date"], "revenue_pred": preds})
            return df, "xgboost", warnings
        except Exception as exc:  # pragma: no cover - afhænger af model/fil
            warnings.append(f"XGBoost fallback: {exc}")
            if backend_override and not allow_fallback:
                raise
            backend = "regression"

    if backend == "hgb":
        try:
            df = predict_hgb(feature_frame)
            return df, "hgb", warnings
        except Exception as exc:  # pragma: no cover - afhænger af model/fil
            warnings.append(f"HGB fallback: {exc}")
            if backend_override and not allow_fallback:
                raise
            backend = "regression"

    if backend == "regression":
        df = predict_regression(feature_frame)
        return df, "regression", warnings

    raise ValueError(f"Ukendt MODEL_BACKEND: {settings.MODEL_BACKEND}")


def _build_context(
    start_date: date,
    horizon_days: int,
    wage_pct: float | None,
    avg_hourly_wage: float | None,
) -> ForecastContext:
    dates = generate_dates(start_date, horizon_days)
    weather_payload = get_weather_df(dates)
    feature_frame = build_feature_frame(
        dates=[pd.Timestamp(d) for d in dates],
        weather=weather_payload,
    )
    wage_pct_value = wage_pct if wage_pct is not None else settings.WAGE_PCT
    hourly_value = avg_hourly_wage if avg_hourly_wage is not None else settings.AVG_HOURLY_WAGE
    planday_map = get_planday_hours_for_dates(dates)
    weather_warning = get_last_weather_warning()
    planday_warning = get_last_planday_warning()
    return ForecastContext(
        dates=dates,
        feature_frame=feature_frame,
        weather_payload=weather_payload,
        wage_pct=wage_pct_value,
        hourly_wage=hourly_value,
        planday_map=planday_map,
        weather_warning=weather_warning,
        planday_warning=planday_warning,
    )


def _compose_warnings(
    base_warnings: List[str],
    weather_warning: str | None,
    planday_warning: str | None,
) -> List[str]:
    warnings = list(base_warnings)
    if weather_warning:
        warnings.append(weather_warning)
    if planday_warning:
        warnings.append(planday_warning)
    return warnings


def _prepare_result_rows(
    forecast_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    wage_pct: float,
    hourly_wage: float,
    planday_map: dict[date, float | None],
) -> List[ForecastRow]:
    result_df = forecast_df.merge(feature_frame, on="date", how="left")
    result_df = result_df.sort_values("date").reset_index(drop=True)
    result_df["hours_recommended"] = (result_df["revenue_pred"] * wage_pct) / hourly_wage
    result_df["planday_hours"] = result_df["date"].dt.date.map(planday_map.get)
    result_df["hours_diff"] = result_df.apply(
        lambda row: None
        if row["planday_hours"] is None
        else float(row["hours_recommended"] - row["planday_hours"]),
        axis=1,
    )

    rows: List[ForecastRow] = []
    for row in result_df.itertuples(index=False):
        rows.append(
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
        )
    return rows


def _build_result(
    forecast_df: pd.DataFrame,
    backend_used: str,
    context: ForecastContext,
    base_warnings: List[str],
) -> ForecastResult:
    rows = _prepare_result_rows(
        forecast_df=forecast_df,
        feature_frame=context.feature_frame,
        wage_pct=context.wage_pct,
        hourly_wage=context.hourly_wage,
        planday_map=context.planday_map,
    )
    warnings = _compose_warnings(
        base_warnings=base_warnings,
        weather_warning=context.weather_warning,
        planday_warning=context.planday_warning,
    )
    return ForecastResult(
        model_version=settings.MODEL_VERSION,
        model_backend=backend_used.lower(),
        rows=rows,
        warnings=warnings,
    )


def _generate_results(
    context: ForecastContext,
    additional_backends: Iterable[str],
) -> tuple[ForecastResult, dict[str, ForecastResult]]:
    primary_df, backend_used, base_warnings = _predict(context.feature_frame)
    backend_used = backend_used.lower()
    primary_result = _build_result(primary_df, backend_used, context, base_warnings)

    variants: dict[str, ForecastResult] = {}
    for backend in {b.lower() for b in additional_backends}:
        if backend == backend_used:
            variants[backend] = primary_result
            continue
        try:
            alt_df, alt_backend, alt_warnings = _predict(
                context.feature_frame,
                backend_override=backend,
                allow_fallback=False,
            )
        except Exception as exc:
            alt_backend = backend
            variant = ForecastResult(
                model_version=settings.MODEL_VERSION,
                model_backend=alt_backend,
                rows=[],
                warnings=_compose_warnings(
                    base_warnings=[f"{alt_backend} fejlede: {exc}"],
                    weather_warning=context.weather_warning,
                    planday_warning=context.planday_warning,
                ),
            )
        else:
            variant = _build_result(alt_df, alt_backend.lower(), context, alt_warnings)
        variants[variant.model_backend.lower()] = variant
    return primary_result, variants


def _log_forecast_event(
    start_date: date,
    horizon_days: int,
    result: ForecastResult,
    weather_frame: pd.DataFrame,
) -> None:
    if weather_frame is None:
        weather_frame = pd.DataFrame()
    try:
        log_forecast_event(
            start_date=start_date,
            horizon_days=horizon_days,
            result=result,
            weather_frame=weather_frame,
        )
    except Exception:
        pass
