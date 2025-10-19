from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from app.core.config import settings

DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,sunshine_duration"
HOURLY_FIELDS = "precipitation"
ARCHIVE_DAILY_FIELDS = DAILY_FIELDS
RAIN_BINS = [-0.1, 0.1, 1, 4, 10, 20, 1_000]
RAIN_LABELS = ["0", "0.1-1", "1.1-4", "4.1-10", "10.1-20", "20+"]

OPENING_HOUR_START = 8
OPENING_HOUR_END = 22


def fetch_weather(
    latitude: float | None = None,
    longitude: float | None = None,
    days: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Hent daglige Open-Meteo værdier.

    Hvis ``start_date``/``end_date`` angives, bruges de som eksakt interval.
    Ellers anvendes ``forecast_days`` med udgangspunkt i Open-Meteos nuværende dato.
    """
    lat = latitude or settings.WEATHER_LATITUDE
    lon = longitude or settings.WEATHER_LONGITUDE
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_FIELDS,
        "hourly": HOURLY_FIELDS,
        "timezone": settings.WEATHER_TIMEZONE,
    }
    if start_date and end_date:
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()
    else:
        params["forecast_days"] = days or settings.DEFAULT_FORECAST_DAYS
    base_url = settings.WEATHER_API_BASE_URL or "https://api.open-meteo.com/v1/forecast"
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily")
    if not daily:
        raise ValueError("Open-Meteo svarede uden 'daily'")

    hourly = payload.get("hourly") or {}

    df = pd.DataFrame(daily).copy()
    if df.empty:
        raise ValueError("Ingen 'daily'-rækker fra Open-Meteo")

    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.normalize()
    df = df.rename(
        columns={
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "precip_sum",
            "wind_speed_10m_max": "wind_max",
            "sunshine_duration": "sunshine_sec",
        }
    )
    df["sunshine_hours"] = df["sunshine_sec"].fillna(0.0) / 3600.0
    df["month"] = df["time"].dt.month
    df["weekday"] = df["time"].dt.weekday + 1
    df["year"] = df["time"].dt.year
    df["rain_group"] = pd.cut(df["precip_sum"], bins=RAIN_BINS, labels=RAIN_LABELS)

    if hourly:
        hourly_df = pd.DataFrame(hourly).copy()
        if not hourly_df.empty and "precipitation" in hourly_df:
            times = pd.to_datetime(hourly_df["time"])
            if getattr(times.dt, "tz", None) is not None:
                times = times.dt.tz_convert(settings.WEATHER_TIMEZONE).dt.tz_localize(None)
            hourly_df["time"] = times
            hourly_df["date"] = hourly_df["time"].dt.normalize()
            hourly_df["hour"] = hourly_df["time"].dt.hour
            mask = hourly_df["hour"].between(OPENING_HOUR_START, OPENING_HOUR_END)
            filtered = hourly_df.loc[mask]
            if not filtered.empty:
                daily_precip = filtered.groupby("date", as_index=True)["precipitation"].sum()
                df["precip_sum"] = df["date"].map(daily_precip).fillna(df["precip_sum"])

    return df


def fetch_historic_weather(
    start_date: date,
    end_date: date,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    lat = latitude or settings.WEATHER_LATITUDE
    lon = longitude or settings.WEATHER_LONGITUDE
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ARCHIVE_DAILY_FIELDS,
        "timezone": settings.WEATHER_TIMEZONE,
    }
    base_url = (
        settings.WEATHER_ARCHIVE_API_BASE_URL
        or "https://archive-api.open-meteo.com/v1/archive"
    )
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily")
    if not daily:
        raise ValueError("Open-Meteo archive svarede uden 'daily'")
    df = pd.DataFrame(daily).copy()
    if df.empty:
        raise ValueError("Open-Meteo archive returnerede tom 'daily'")
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.normalize()
    df = df.rename(
        columns={
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "precip_sum",
            "wind_speed_10m_max": "wind_max",
            "sunshine_duration": "sunshine_sec",
        }
    )
    df["sunshine_hours"] = df["sunshine_sec"].fillna(0.0) / 3600.0
    return df[
        [
            "date",
            "temp_max",
            "temp_min",
            "precip_sum",
            "wind_max",
            "sunshine_hours",
        ]
    ].copy()
