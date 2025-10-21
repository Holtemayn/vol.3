from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime
from statistics import median
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

import httpx

from app.agents.schemas import (
    PlandayParams,
    ReconcileParams,
    SimilarDaysParams,
    HistoryQueryParams,
)
from app.core.config import settings

ToolCallable = Callable[[dict], Awaitable[dict]]
TOOL_REGISTRY: Dict[str, ToolCallable] = {}

_default_port = os.getenv("PORT", "8000")
BASE_URL = getattr(settings, "INTERNAL_BASE_URL", None) or f"http://127.0.0.1:{_default_port}"
LOGGER = logging.getLogger(__name__)


async def execute_tool(name: str, args: dict) -> dict:
    handler = TOOL_REGISTRY.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool '{name}'")
    LOGGER.info("tool_call_start name=%s args=%s", name, args)
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            result = await handler(args)
            LOGGER.info("tool_call_success name=%s status=ok", name)
            return result
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            LOGGER.warning(
                "tool_call_http_error name=%s status_code=%s detail=%s attempt=%s",
                name,
                exc.response.status_code,
                exc.response.text[:200],
                attempt + 1,
            )
            if exc.response.status_code in {502, 503} and attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            last_exc = exc
            LOGGER.exception("tool_call_exception name=%s attempt=%s", name, attempt + 1)
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Tool '{name}' failed without raising an exception")


async def _call_get_planday(args: dict) -> dict:
    params = PlandayParams(**args)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/planday/{params.date}")
        response.raise_for_status()
        return response.json()


TOOL_REGISTRY["get_planday"] = _call_get_planday


async def _call_reconcile(args: dict) -> dict:
    params = ReconcileParams(**args)
    payload = {"limit": params.limit}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{BASE_URL}/reconcile", json=payload)
        response.raise_for_status()
        return response.json()


TOOL_REGISTRY["reconcile_accounts"] = _call_reconcile


async def _call_history(args: dict) -> dict:
    params = HistoryQueryParams(**args)
    payload = {
        "start_date": params.start_date,
        "end_date": params.end_date,
        "limit": params.limit,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{BASE_URL}/history/aggregates", json=payload)
        response.raise_for_status()
        return response.json()


TOOL_REGISTRY["lookup_history"] = _call_history


RAIN_BINS: Tuple[float, ...] = (-0.1, 0.1, 1, 4, 10, 20, float("inf"))
TEMP_BINS: Tuple[float, ...] = (-50, 0, 5, 10, 15, 20, 25, 35, float("inf"))


def _bin_value(value: Optional[float], bins: Tuple[float, ...]) -> int:
    if value is None:
        return -1
    for idx, limit in enumerate(bins):
        if value <= limit:
            return idx
    return len(bins) - 1


def _weekday(dt: date) -> int:
    return dt.weekday()


def _safe_date(value: str) -> Optional[date]:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


async def _call_similar_days(args: dict) -> dict:
    params = SimilarDaysParams(**args)
    target_date = _safe_date(params.date)
    if target_date is None:
        raise ValueError("Invalid date format – expected YYYY-MM-DD")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{BASE_URL}/logs", params={"limit": 50})
        response.raise_for_status()
        data = response.json()

    items: Iterable[dict] = data.get("items", [])
    records: List[dict] = []
    for entry in items:
        rows = entry.get("rows", [])[:50]
        for row in rows:
            row_date = _safe_date(str(row.get("date", "")))
            if row_date is None:
                continue
            record = {
                "date": row_date,
                "week": row_date.isocalendar().week,
                "weekday": _weekday(row_date),
                "month": row_date.month,
                "temp_max": row.get("temp_max"),
                "precip_sum": row.get("precip_sum"),
                "sunshine_hours": row.get("sunshine_hours"),
                "revenue_pred": row.get("revenue_pred"),
                "revenue_actual": row.get("revenue_actual"),
                "planday_hours": row.get("planday_hours"),
            }
            record["precip_bin"] = _bin_value(record["precip_sum"], RAIN_BINS)
            record["temp_bin"] = _bin_value(record["temp_max"], TEMP_BINS)
            records.append(record)

    if not records:
        return {"similar_days": [], "summary": "Ingen logposter tilgængelige"}

    target_week = target_date.isocalendar().week
    target_weekday = _weekday(target_date)
    target_month = target_date.month
    target_temp = None
    target_precip_bin = None

    for rec in records:
        if rec["date"] == target_date:
            target_temp = rec["temp_max"]
            target_precip_bin = rec["precip_bin"]
            break

    def score(rec: dict) -> float:
        value = abs(rec["week"] - target_week) * 0.2
        if rec["weekday"] != target_weekday:
            value += 1.0
        value += abs(rec["month"] - target_month) * 0.1
        if target_temp is not None and rec["temp_max"] is not None:
            value += abs(rec["temp_max"] - target_temp) * 0.1
        if target_precip_bin is not None and rec["precip_bin"] != target_precip_bin:
            value += 0.5
        return value

    ranked = sorted(records, key=score)
    top = [
        {
            "date": rec["date"].isoformat(),
            "weekday": rec["date"].strftime("%A"),
            "revenue_pred": rec.get("revenue_pred"),
            "revenue_actual": rec.get("revenue_actual"),
            "planday_hours": rec.get("planday_hours"),
            "temp_max": rec.get("temp_max"),
            "precip_sum": rec.get("precip_sum"),
            "sunshine_hours": rec.get("sunshine_hours"),
        }
        for rec in ranked[: params.k]
    ]

    revenues = [rec["revenue_pred"] for rec in ranked if rec.get("revenue_pred") is not None]
    median_revenue = median(revenues[: params.k]) if revenues else None

    summary = "Ingen analoge dage fundet"
    if top:
        parts = [f"{row['date']} ({row['weekday']})" for row in top]
        summary = f"Analoge dage: {', '.join(parts)}"
        if median_revenue is not None:
            formatted = f"{int(round(median_revenue)):,.0f}".replace(",", ".")
            summary += f" · median omsætning ~ {formatted} kr"

    return {
        "similar_days": top,
        "summary": summary,
        "median_revenue_pred": median_revenue,
    }


TOOL_REGISTRY["find_similar_days"] = _call_similar_days


def _schema(model: type) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_planday",
        "description": "Hent planlagte timer for en bestemt dato fra Planday.",
        "function": {
            "name": "get_planday",
            "description": "Returnerer et JSON-objekt med planlagte timer for datoen.",
            "parameters": _schema(PlandayParams),
        },
    },
    {
        "type": "function",
        "name": "reconcile_accounts",
        "description": "Afstem forecast mod Planday omsætning baseret på loghistorik.",
        "function": {
            "name": "reconcile_accounts",
            "description": "Returnerer forskellen mellem forecast og faktisk omsætning for valgte logposter.",
            "parameters": _schema(ReconcileParams),
        },
    },
    {
        "type": "function",
        "name": "find_similar_days",
        "description": "Finder analoge dage via forecast-logs til brug i anbefalinger.",
        "function": {
            "name": "find_similar_days",
            "description": "Returnerer de mest lignende dage og deres nøgletal.",
            "parameters": _schema(SimilarDaysParams),
        },
    },
    {
        "type": "function",
        "name": "lookup_history",
        "description": "Slår historiske dagsaggregater op fra MongoDB baseret på dato-interval.",
        "function": {
            "name": "lookup_history",
            "description": "Returnerer liste over daglige nøgletal (omsætning, vejr mv.).",
            "parameters": _schema(HistoryQueryParams),
        },
    },
]
