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

def _rows_to_serializable(
    rows: Iterable["ForecastRow"],
    actuals: Optional[dict[date, float]] = None,
    actual_weather: Optional[dict[date, dict[str, float]]] = None,
) -> list[dict]:
    serializable = []
    for row in rows:
        actual_value: float | None = None
        if actuals:
            actual_value = actuals.get(row.date)
        weather_actual: dict[str, float] | None = None
        if actual_weather:
            weather_actual = actual_weather.get(row.date)
        serializable.append(
            {
                "date": row.date.isoformat(),
                "revenue_pred": row.revenue_pred,
                "hours_recommended": row.hours_recommended,
                "planday_hours": row.planday_hours,
                "hours_diff": row.hours_diff,
                "revenue_actual": actual_value,
                "temp_max": row.temp_max,
                "precip_sum": row.precip_sum,
                "sunshine_hours": row.sunshine_hours,
                "wind_max": row.wind_max,
                "temp_max_actual": weather_actual.get("temp_max") if weather_actual else None,
                "temp_min_actual": weather_actual.get("temp_min") if weather_actual else None,
                "precip_sum_actual": weather_actual.get("precip_sum") if weather_actual else None,
                "sunshine_hours_actual": weather_actual.get("sunshine_hours") if weather_actual else None,
                "wind_max_actual": weather_actual.get("wind_max") if weather_actual else None,
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
    sample_rows = list(result.rows)
    actual_map: dict[date, float] | None = None
    weather_actual_map: dict[date, dict[str, float]] | None = None
    try:
        from app.services.planday import get_planday_revenue_for_dates
        from app.services.weather import get_historic_weather_map

        if sample_rows:
            actual_map = get_planday_revenue_for_dates([row.date for row in sample_rows])
            weather_actual_map = get_historic_weather_map([row.date for row in sample_rows])
    except Exception as exc:  # pragma: no cover - afhænger af API
        LOGGER.warning("Kunne ikke hente omsætning fra Planday: %s", exc)

    rows_payload = _rows_to_serializable(sample_rows, actuals=actual_map, actual_weather=weather_actual_map)
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
    entries: List[dict]
    if settings.DATABASE_URL:
        try:
            entries = _read_from_database(limit)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Kunne ikke læse logposter fra database: %s", exc)
            entries = _read_from_file(limit)
    else:
        entries = _read_from_file(limit)
    _hydrate_planday_actuals(entries)
    _hydrate_weather_actuals(entries)
    return entries


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
        values = {
            "horizon_days": min(horizon_days, 5),
            "model_version": result.model_version,
            "model_backend": result.model_backend,
            "warnings": result.warnings,
            "rows": rows_payload,
            "weather": weather_payload,
        }
        if exists:
            session.execute(
                forecast_logs.update()
                .where(forecast_logs.c.forecast_date == start_date)
                .values(**values)
            )
        else:
            session.execute(
                forecast_logs.insert().values(
                    forecast_date=start_date,
                    **values,
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
        payload = {
            "forecast_date": start_date.isoformat(),
            "horizon_days": min(horizon_days, 5),
            "model_version": result.model_version,
            "model_backend": result.model_backend,
            "warnings": result.warnings,
            "rows": rows_payload,
            "weather": weather_payload,
        }
        lines: list[str] = []
        replaced = False
        if path.exists():
            with path.open("r", encoding="utf-8") as handler:
                for line in handler:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("forecast_date") == start_date.isoformat():
                        lines.append(json.dumps(payload, ensure_ascii=False))
                        replaced = True
                    else:
                        lines.append(json.dumps(entry, ensure_ascii=False))
        if not replaced:
            lines.append(json.dumps(payload, ensure_ascii=False))
        with path.open("w", encoding="utf-8") as handler:
            handler.write("\n".join(lines) + "\n")
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


def _hydrate_planday_actuals(entries: List[dict]) -> None:
    missing_dates: set[date] = set()
    today = date.today()

    for entry in entries:
        for row in entry.get("rows") or []:
            row_date = _parse_row_date(row.get("date"))
            if row_date and row_date <= today and row.get("revenue_actual") is None:
                missing_dates.add(row_date)

    if not missing_dates:
        return

    try:
        from app.services.planday import get_planday_revenue_for_dates

        actual_map = get_planday_revenue_for_dates(sorted(missing_dates))
    except Exception as exc:  # pragma: no cover - afhænger af eksterne services
        LOGGER.warning("Kunne ikke opdatere Planday omsætning: %s", exc)
        return

    for entry in entries:
        for row in entry.get("rows") or []:
            row_date = _parse_row_date(row.get("date"))
            if not row_date:
                continue
            actual_value = actual_map.get(row_date)
            if actual_value is not None:
                row["revenue_actual"] = actual_value


def _parse_row_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _hydrate_weather_actuals(entries: List[dict]) -> None:
    missing_dates: set[date] = set()
    today = date.today()

    for entry in entries:
        for row in entry.get("rows") or []:
            row_date = _parse_row_date(row.get("date"))
            if not row_date or row_date > today:
                continue
            if any(
                row.get(key) is not None
                for key in (
                    "temp_max_actual",
                    "temp_min_actual",
                    "precip_sum_actual",
                    "sunshine_hours_actual",
                    "wind_max_actual",
                )
            ):
                continue
            missing_dates.add(row_date)

    if not missing_dates:
        return

    try:
        from app.services.weather import get_historic_weather_map

        weather_map = get_historic_weather_map(sorted(missing_dates))
    except Exception as exc:  # pragma: no cover - ekstern afhængighed
        LOGGER.warning("Kunne ikke hente historisk vejr: %s", exc)
        return

    for entry in entries:
        for row in entry.get("rows") or []:
            row_date = _parse_row_date(row.get("date"))
            if not row_date:
                continue
            weather_actual = weather_map.get(row_date)
            if not weather_actual:
                continue
            row["temp_max_actual"] = weather_actual.get("temp_max")
            row["temp_min_actual"] = weather_actual.get("temp_min")
            row["precip_sum_actual"] = weather_actual.get("precip_sum")
            row["sunshine_hours_actual"] = weather_actual.get("sunshine_hours")
            row["wind_max_actual"] = weather_actual.get("wind_max")
