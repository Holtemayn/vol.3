from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from collections import deque
from typing import Iterable, TYPE_CHECKING, List, Optional

import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    select,
    func,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings

if TYPE_CHECKING:  # pragma: no cover - kun til type hints
    from app.services.forecast import ForecastResult, ForecastRow

LOGGER = logging.getLogger(__name__)

_TABLE_READY = False
_metadata = MetaData()
forecast_logs = Table(
    "forecast_logs",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("forecast_date", Date, nullable=False),
    Column("horizon_days", Integer, nullable=False),
    Column("model_version", String(length=64), nullable=False),
    Column("model_backend", String(length=64), nullable=False),
    Column("warnings", JSON, nullable=False),
    Column("rows", JSON, nullable=False),
    Column("weather", JSON, nullable=False),
)

def _rows_to_serializable(rows: Iterable["ForecastRow"]) -> list[dict]:
    serializable = []
    for row in rows:
        serializable.append(
            {
                "date": row.date.isoformat(),
                "revenue_pred": row.revenue_pred,
                "hours_recommended": row.hours_recommended,
                "planday_hours": row.planday_hours,
                "hours_diff": row.hours_diff,
                "temp_max": row.temp_max,
                "precip_sum": row.precip_sum,
                "sunshine_hours": row.sunshine_hours,
                "wind_max": row.wind_max,
            }
        )
    return serializable


def _weather_to_serializable(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    working = frame.copy()
    if "date" in working:
        working["date"] = pd.to_datetime(working["date"]).dt.date.astype(str)
    return working.to_dict(orient="records")


def log_forecast_event(
    start_date: date,
    horizon_days: int,
    result: "ForecastResult",
    weather_frame: pd.DataFrame,
) -> None:
    """
    Append forecast + vejrdata til en logfil (newline-delimited JSON).
    Svigter logningen, nøjes vi med en warning – forecast skal ikke fejle.
    """
    rows_payload = _rows_to_serializable(result.rows[:5])
    weather_subset = None
    if weather_frame is not None:
        try:
            weather_subset = weather_frame.head(5)
        except AttributeError:
            weather_subset = weather_frame
    weather_payload = _weather_to_serializable(weather_subset)
    if settings.DATABASE_URL:
        try:
            _log_to_database(start_date, horizon_days, result, rows_payload, weather_payload)
            return
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Kunne ikke logge til database: %s", exc)
    _log_to_file(start_date, horizon_days, result, rows_payload, weather_payload)


def read_recent_forecasts(limit: int = 10) -> List[dict]:
    if settings.DATABASE_URL:
        try:
            return _read_from_database(limit)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Kunne ikke læse logposter fra database: %s", exc)
    return _read_from_file(limit)


# ---------------------------------------------------------------------------
# Database helpers

def _get_session() -> Optional[Session]:
    from app.services.db import get_session  # lokal import for at undgå cirkler

    if not settings.DATABASE_URL:
        return None
    try:
        return get_session()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Database session fejlede: %s", exc)
        return None


def _ensure_table(session: Session) -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    engine = session.get_bind()
    _metadata.create_all(engine, tables=[forecast_logs])
    _TABLE_READY = True


def _log_to_database(
    start_date: date,
    horizon_days: int,
    result: "ForecastResult",
    rows_payload: list[dict],
    weather_payload: list[dict],
) -> None:
    session = _get_session()
    if session is None:
        raise RuntimeError("Database session ikke tilgængelig")
    with session:
        _ensure_table(session)
        exists = session.execute(
            select(forecast_logs.c.id).where(forecast_logs.c.forecast_date == start_date)
        ).first()
        if exists:
            return  # Kun log én gang pr. dag
        session.execute(
            forecast_logs.insert().values(
                forecast_date=start_date,
                horizon_days=min(horizon_days, 5),
                model_version=result.model_version,
                model_backend=result.model_backend,
                warnings=result.warnings,
                rows=rows_payload,
                weather=weather_payload,
            )
        )
        session.commit()


def _read_from_database(limit: int) -> List[dict]:
    session = _get_session()
    if session is None:
        raise RuntimeError("Database session ikke tilgængelig")
    with session:
        _ensure_table(session)
        rows = session.execute(
            select(
                forecast_logs.c.forecast_date,
                forecast_logs.c.horizon_days,
                forecast_logs.c.model_version,
                forecast_logs.c.model_backend,
                forecast_logs.c.warnings,
                forecast_logs.c.rows,
                forecast_logs.c.weather,
            )
            .order_by(forecast_logs.c.forecast_date.desc())
            .limit(limit)
        ).all()
    return [
        {
            "forecast_date": record.forecast_date.isoformat(),
            "horizon_days": record.horizon_days,
            "model_version": record.model_version,
            "model_backend": record.model_backend,
            "warnings": record.warnings,
            "rows": record.rows,
            "weather": record.weather,
        }
        for record in rows
    ]


# ---------------------------------------------------------------------------
# File fallback (lokal udvikling)

def _log_to_file(
    start_date: date,
    horizon_days: int,
    result: "ForecastResult",
    rows_payload: list[dict],
    weather_payload: list[dict],
) -> None:
    try:
        path = Path(settings.FORECAST_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = _read_from_file(limit=1000)
        if any(entry.get("forecast_date") == start_date.isoformat() for entry in entries):
            return
        payload = {
            "forecast_date": start_date.isoformat(),
            "horizon_days": min(horizon_days, 5),
            "model_version": result.model_version,
            "model_backend": result.model_backend,
            "warnings": result.warnings,
            "rows": rows_payload,
            "weather": weather_payload,
        }
        with path.open("a", encoding="utf-8") as handler:
            handler.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Kunne ikke logge forecast-event (fil): %s", exc)


def _read_from_file(limit: int) -> List[dict]:
    path = Path(settings.FORECAST_LOG_PATH)
    if not path.exists():
        return []
    entries: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handler:
        for line in handler:
            line = line.strip()
            if line:
                entries.append(line)
    results: List[dict] = []
    for line in reversed(entries):
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results
