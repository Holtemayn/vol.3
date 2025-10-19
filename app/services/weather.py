from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable

import pandas as pd

from app.core.config import settings
from app.services.open_meteo import (
    fetch_historic_weather as fetch_open_meteo_historic,
    fetch_weather as fetch_open_meteo,
)

_LAST_WARNING: str | None = None
_CACHE: Dict[tuple[date, date], tuple[datetime, pd.DataFrame]] = {}
_HISTORIC_CACHE: Dict[tuple[date, date], tuple[datetime, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 3600


@dataclass
class WeatherPayload:
    frame: pd.DataFrame
    warning: str | None = None


def get_weather_df(dates: Iterable[date]) -> WeatherPayload:
    """
    Hent daglige vejrdata for de ønskede datoer.

    Returnerer WeatherPayload med en DataFrame, hvor kolonnen ``date`` er datetime64[ns].
    Ved fejl returneres en syntetisk stub og warning sættes.
    """
    dates = list(dates)
    if not dates:
        return WeatherPayload(_empty_frame())

    start = min(dates)
    end = max(dates)
    cache_key = (start, end)
    now = datetime.utcnow()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < _CACHE_TTL_SECONDS:
        frame = cached[1]
        frame = frame[frame["date"].dt.date.isin({d for d in dates})].reset_index(drop=True)
        return WeatherPayload(frame)
    try:
        frame = _fetch_daily_range(start, end)
        frame = frame[frame["date"].dt.date.isin({d for d in dates})].reset_index(drop=True)
        if frame.empty:
            raise ValueError("Open-Meteo returnerede ingen rækker")
        _set_warning(None)
        _CACHE[cache_key] = (now, frame)
        return WeatherPayload(frame)
    except Exception as exc:  # pragma: no cover - netværks-/API-fejl
        _set_warning(f"Open-Meteo fallback: {exc}")
        fallback = _fallback_frame(dates)
        return WeatherPayload(fallback, warning=_LAST_WARNING)


def _fetch_daily_range(start: date, end: date) -> pd.DataFrame:
    frame = fetch_open_meteo(
        latitude=settings.WEATHER_LATITUDE,
        longitude=settings.WEATHER_LONGITUDE,
        start_date=start,
        end_date=end,
    )
    if frame.empty:
        raise ValueError("Open-Meteo returnerede tom 'daily'-blok")
    cols = [
        "date",
        "temp_max",
        "temp_min",
        "precip_sum",
        "wind_max",
        "sunshine_hours",
    ]
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Mangler kolonner fra Open-Meteo: {', '.join(missing)}")
    subset = frame[cols].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    return subset


def _fallback_frame(dates: Iterable[date]) -> pd.DataFrame:
    rows = []
    for dt in sorted(dates):
        ts = pd.Timestamp(dt)
        rows.append(
            {
                "date": ts,
                "temp_max": 15.0,
                "temp_min": 8.0,
                "precip_sum": 0.0,
                "wind_max": 5.0,
                "sunshine_hours": 6.0,
            }
        )
    return pd.DataFrame(rows)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "temp_max", "temp_min", "precip_sum", "wind_max", "sunshine_hours"])


def _set_warning(message: str | None) -> None:
    global _LAST_WARNING
    _LAST_WARNING = message


def get_last_weather_warning() -> str | None:
    return _LAST_WARNING


def get_historic_weather_map(dates: Iterable[date]) -> dict[date, dict[str, float]]:
    dates = list(dates)
    if not dates:
        return {}

    start = min(dates)
    end = max(dates)
    cache_key = (start, end)
    now = datetime.utcnow()
    cached = _HISTORIC_CACHE.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < _CACHE_TTL_SECONDS:
        frame = cached[1]
    else:
        try:
            frame = fetch_open_meteo_historic(start, end)
        except Exception:
            return {}
        frame = frame[frame["date"].dt.date.isin({d for d in dates})].reset_index(drop=True)
        _HISTORIC_CACHE[cache_key] = (now, frame)

    result: dict[date, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        dt = row.date.date()
        result[dt] = {
            "temp_max": float(row.temp_max) if row.temp_max is not None else None,
            "temp_min": float(row.temp_min) if row.temp_min is not None else None,
            "precip_sum": float(row.precip_sum) if row.precip_sum is not None else None,
            "wind_max": float(row.wind_max) if row.wind_max is not None else None,
            "sunshine_hours": float(row.sunshine_hours) if row.sunshine_hours is not None else None,
        }
    return result
