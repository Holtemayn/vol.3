from __future__ import annotations

from calendar import day_name
from datetime import date as date_cls, datetime
from math import sqrt
from typing import Any, Dict, List, Optional

from bson import json_util
from pymongo import MongoClient
from pymongo.collection import Collection

from app.core.config import settings
from app.services.weather import get_weather_df

_client: Optional[MongoClient] = None
_collection: Optional[Collection] = None


def _get_collection() -> Collection:
    global _client, _collection
    if _collection is not None:
        return _collection
    if not settings.MONGO_URI:
        raise RuntimeError("MONGO_URI is not configured")
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    collection = db[settings.MONGO_DAILY_COLLECTION]
    _client = client
    _collection = collection
    return collection


def _to_datetime(value: str, end_of_day: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if end_of_day:
        return dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def fetch_daily_aggregates(start_date: str, end_date: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    collection = _get_collection()
    start_dt = _to_datetime(start_date)
    end_dt = _to_datetime(end_date, end_of_day=True)

    cursor = collection.find(
        {"dato": {"$gte": start_dt, "$lte": end_dt}}
    ).sort("dato", 1)
    if limit:
        cursor = cursor.limit(limit)

    documents: List[Dict[str, Any]] = []
    for doc in cursor:
        clean = json_util.loads(json_util.dumps(doc))
        dato = clean.pop("dato", None)
        clean["date"] = _to_iso_value(dato)
        clean.pop("_id", None)
        documents.append(clean)
    return documents


def _fetch_target_document(collection: Collection, target: date_cls) -> Dict[str, Any]:
    start_dt = datetime.combine(target, datetime.min.time())
    end_dt = datetime.combine(target, datetime.max.time())
    doc = collection.find_one({"dato": {"$gte": start_dt, "$lte": end_dt}})
    if not doc:
        return _build_target_from_forecast(target)
    clean = json_util.loads(json_util.dumps(doc))
    dato = clean.pop("dato", None)
    clean["date"] = _to_iso_value(dato)
    clean["source"] = "history"
    clean.pop("_id", None)
    clean["rain"] = float(clean.get("rain") or 0.0)
    clean["sunshine_duration"] = float(clean.get("sunshine_duration") or 0.0)
    clean["temperature_2m"] = float(clean.get("temperature_2m") or 0.0)
    clean["wind_gusts_10m"] = float(clean.get("wind_gusts_10m") or 0.0)
    return clean


def _build_target_from_forecast(target: date_cls) -> Dict[str, Any]:
    payload = get_weather_df([target])
    frame = payload.frame
    if frame is None or frame.empty:
        raise ValueError(f"Ingen historik eller forecast fundet for {target.isoformat()}")
    row = frame.iloc[0]
    rain = float(row.get("precip_sum") or 0.0)
    sunshine_hours = float(row.get("sunshine_hours") or 0.0)
    temp = float(row.get("temp_max") or 0.0)
    wind = float(row.get("wind_max") or 0.0)
    return {
        "date": target.isoformat(),
        "ugedag": target.weekday(),
        "ugenummer": target.isocalendar().week,
        "år": target.year,
        "rain": rain,
        "sunshine_duration": sunshine_hours * 3600.0,
        "temperature_2m": temp,
        "wind_gusts_10m": wind,
        "source": "forecast",
    }


def _day_label(weekday: int) -> str:
    try:
        return day_name[weekday].capitalize()
    except IndexError:
        return str(weekday)


def find_similar_days(target_date: date_cls, max_candidates: int = 500) -> Dict[str, Any]:
    collection = _get_collection()
    target_doc = _fetch_target_document(collection, target_date)

    weekday = target_doc.get("ugedag")
    week_number = target_doc.get("ugenummer")
    target_year = target_doc.get("år")
    rain_target = float(target_doc.get("rain") or 0.0)
    sun_target = float(target_doc.get("sunshine_duration") or 0.0)
    temp_target = float(target_doc.get("temperature_2m") or 0.0)

    query: Dict[str, Any] = {
        "ugedag": weekday,
        "ugenummer": {"$gte": max(1, week_number - 2), "$lte": week_number + 2},
    }
    if target_year is not None:
        query["år"] = {"$lt": target_year}

    cursor = (
        collection.find(query)
        .sort("dato", -1)
        .limit(max_candidates)
    )

    candidates: List[Dict[str, Any]] = []
    for doc in cursor:
        clean = json_util.loads(json_util.dumps(doc))
        date_value = clean.pop("dato", None)
        clean["date"] = _to_iso_value(date_value)
        clean.pop("_id", None)
        if not date_value:
            continue
        clean["rain"] = float(clean.get("rain") or 0.0)
        clean["sunshine_duration"] = float(clean.get("sunshine_duration") or 0.0)
        clean["temperature_2m"] = float(clean.get("temperature_2m") or 0.0)
        clean["wind_gusts_10m"] = float(clean.get("wind_gusts_10m") or 0.0)
        candidates.append(clean)

    if not candidates:
        return {
            "target": target_doc,
            "matches": [],
        }

    def rain_diff(item: Dict[str, Any]) -> float:
        return abs(float(item.get("rain") or 0.0) - rain_target)

    def sun_diff(item: Dict[str, Any]) -> float:
        return abs(float(item.get("sunshine_duration") or 0.0) - sun_target)

    def temp_diff(item: Dict[str, Any]) -> float:
        return abs(float(item.get("temperature_2m") or 0.0) - temp_target)

    def rain_sun_diff(item: Dict[str, Any]) -> float:
        return rain_diff(item) + sun_diff(item) / 3600.0

    def overall_diff(item: Dict[str, Any]) -> float:
        rain_d = rain_diff(item)
        sun_d = sun_diff(item) / 3600.0
        temp_d = temp_diff(item)
        wind_d = abs(float(item.get("wind_gusts_10m") or 0.0) - float(target_doc.get("wind_gusts_10m") or 0.0))
        return sqrt(rain_d**2 + sun_d**2 + temp_d**2 + (wind_d * 0.1) ** 2)

    metrics = [
        ("rain", "Regn", rain_diff),
        ("sunshine", "Solskin", sun_diff),
        ("rain_and_sun", "Regn + sol", rain_sun_diff),
        ("temperature", "Temperatur", temp_diff),
        ("overall", "Samlet", overall_diff),
    ]

    used_dates: set[str] = set()
    matches: List[Dict[str, Any]] = []

    for key, label, scorer in metrics:
        sorted_candidates = sorted(candidates, key=scorer)
        selected = None
        for item in sorted_candidates:
            iso = item.get("date")
            if not iso or iso in used_dates:
                continue
            selected = item
            break
        if not selected:
            continue
        used_dates.add(selected["date"])

        matches.append(
            {
                "criterion": key,
                "label": label,
                "date": selected["date"],
                "weekday": _day_label(selected.get("ugedag")),
                "week": selected.get("ugenummer"),
                "year": selected.get("år"),
                "rain": selected.get("rain"),
                "sunshine_duration": selected.get("sunshine_duration"),
                "temperature_2m": selected.get("temperature_2m"),
                "wind_gusts_10m": selected.get("wind_gusts_10m"),
                "difference": scorer(selected),
            }
        )

    return {
        "target": target_doc,
        "matches": matches,
        "candidates_considered": len(candidates),
        "metrics": [metric for metric, _, _ in metrics],
    }


def _to_iso_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        inner = value.get("$date")
        if isinstance(inner, dict):
            inner_value = inner.get("$numberLong") or inner.get("$numberDouble")
            if inner_value:
                try:
                    return datetime.fromtimestamp(int(inner_value) / 1000).isoformat()
                except Exception:
                    return str(inner)
        return inner if isinstance(inner, str) else str(inner)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)
