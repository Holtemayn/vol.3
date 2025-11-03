import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

from fastapi import HTTPException
from pydantic import BaseModel, Field, validator
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.core.config import settings

logger = logging.getLogger("cafecaster.calendar")


@dataclass(frozen=True)
class MongoConfig:
    uri: str
    db: str
    coll: str
    date_field: str
    revenue_field: str


class CalendarForecastRequest(BaseModel):
    start_date: Optional[date] = Field(default=None)
    horizon_days: Optional[int] = Field(default=30)
    wage_pct: Optional[float] = Field(default=0.25)

    @validator("start_date", pre=True, always=True)
    def parse_start_date(cls, value: Optional[str]) -> date:
        if value in (None, ""):
            return date.today()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ValueError("start_date must be formatted as YYYY-MM-DD") from exc
        raise ValueError("start_date must be a date or YYYY-MM-DD string")

    @validator("horizon_days", pre=True, always=True)
    def validate_horizon(cls, value: Optional[int]) -> int:
        if value is None:
            value = 30
        try:
            horizon = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("horizon_days must be an integer") from exc
        if horizon < 7 or horizon > 60:
            raise ValueError("horizon_days must be between 7 og 60")
        return horizon

    @validator("wage_pct", pre=True, always=True)
    def validate_wage_pct(cls, value: Optional[float]) -> float:
        if value is None:
            value = 0.25
        try:
            wage = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("wage_pct must be a float") from exc
        if wage < 0.10 or wage > 0.50:
            raise ValueError("wage_pct must be between 0.10 og 0.50")
        return wage


class DailyItem(BaseModel):
    date: date
    iso_year: int
    iso_week: int
    iso_weekday: int
    forecast: Optional[float]
    wage: Optional[float]
    n_candidates: int
    n_hits_prev1: int
    n_hits_prev2: int


class WeeklyItem(BaseModel):
    iso_year: int
    iso_week: int
    days: int
    sum_forecast: Optional[float]
    sum_wage: Optional[float]


class ForecastMeta(BaseModel):
    start_date: date
    end_date: date
    horizon_days: int
    wage_pct: float
    n_days: int
    n_days_forecasted: int
    total_forecast: Optional[float]
    total_wage: Optional[float]


class CalendarForecastResponse(BaseModel):
    meta: ForecastMeta
    daily: List[DailyItem]
    weekly: List[WeeklyItem]


_mongo_config: Optional[MongoConfig] = None
_mongo_client: Optional[MongoClient] = None
_mongo_client_lock = threading.Lock()
_growth_factor_log_cache: Set[Tuple[int, int]] = set()


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is not None:
        return value
    return default


def get_mongo_config() -> MongoConfig:
    global _mongo_config
    if _mongo_config is None:
        uri = _get_env("MONGO_URI", settings.MONGO_URI)
        if not uri:
            raise ValueError("Environment variable MONGO_URI is required")
        db = _get_env("MONGO_DB", settings.MONGO_DB_NAME or "cafecaster")
        coll = _get_env("MONGO_COLL", settings.MONGO_DAILY_COLLECTION or "daily_sales")
        date_field = _get_env("DATE_FIELD", "date")
        revenue_field = _get_env("REVENUE_FIELD", "revenue")
        _mongo_config = MongoConfig(
            uri=uri,
            db=db,
            coll=coll,
            date_field=date_field,
            revenue_field=revenue_field,
        )
    return _mongo_config


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        config = get_mongo_config()
        with _mongo_client_lock:
            if _mongo_client is None:
                try:
                    _mongo_client = MongoClient(config.uri, tz_aware=True)
                except PyMongoError as exc:
                    logger.exception("Failed to initialize MongoDB client")
                    raise RuntimeError("Could not connect to MongoDB") from exc
    return _mongo_client


def iso_parts(target_day: date) -> Tuple[int, int, int]:
    iso_year, iso_week, iso_weekday = target_day.isocalendar()
    return iso_year, iso_week, iso_weekday


def build_candidate_dates(target_day: date) -> Dict[int, List[date]]:
    _, iso_week, iso_weekday = iso_parts(target_day)
    candidates: Dict[int, List[date]] = {}
    for source_year in (target_day.year - 1, target_day.year - 2):
        dates: List[date] = []
        for delta_week in range(-3, 4):
            week = iso_week + delta_week
            try:
                candidate = date.fromisocalendar(source_year, week, iso_weekday)
            except ValueError:
                continue
            dates.append(candidate)
        candidates[source_year] = dates
    return candidates


def _normalize_revenue(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        revenue = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(revenue):
        return None
    return revenue


def fetch_revenues_mongo(
    client: MongoClient,
    db_name: str,
    coll_name: str,
    date_field: str,
    revenue_field: str,
    dates_by_year: Dict[int, List[date]],
) -> Dict[date, float]:
    collection = client[db_name][coll_name]
    unique_dates: Set[date] = set()
    for date_list in dates_by_year.values():
        unique_dates.update(date_list)
    if not unique_dates:
        return {}

    dt_values = [
        datetime(d.year, d.month, d.day, tzinfo=timezone.utc) for d in unique_dates
    ]
    str_values = [d.strftime("%Y-%m-%d") for d in unique_dates]
    revenues: Dict[date, float] = {}
    date_fields = []
    for candidate in (date_field, "date", "dato"):
        if candidate not in date_fields:
            date_fields.append(candidate)
    revenue_fields = []
    for candidate in (revenue_field, "revenue", "oms"):
        if candidate not in revenue_fields:
            revenue_fields.append(candidate)

    try:
        for field in date_fields:
            if dt_values:
                for doc in collection.find({field: {"$in": dt_values}}):
                    raw_date = doc.get(field)
                    if not isinstance(raw_date, datetime):
                        continue
                    revenue_value = None
                    for rev_field in revenue_fields:
                        revenue_value = _normalize_revenue(doc.get(rev_field))
                        if revenue_value is not None:
                            break
                    if revenue_value is None:
                        continue
                    normalized = (
                        raw_date.astimezone(timezone.utc).date()
                        if raw_date.tzinfo
                        else raw_date.date()
                    )
                    if normalized not in revenues:
                        revenues[normalized] = revenue_value
            if str_values:
                for doc in collection.find({field: {"$in": str_values}}):
                    raw_date = doc.get(field)
                    if not isinstance(raw_date, str):
                        continue
                    revenue_value = None
                    for rev_field in revenue_fields:
                        revenue_value = _normalize_revenue(doc.get(rev_field))
                        if revenue_value is not None:
                            break
                    if revenue_value is None:
                        continue
                    try:
                        normalized = datetime.strptime(raw_date, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    if normalized in revenues:
                        continue
                    revenues[normalized] = revenue_value
    except PyMongoError as exc:
        logger.exception("MongoDB query failed")
        raise RuntimeError("Database query failed") from exc

    return revenues


def growth_factor_for(source_year: int, target_year: int) -> float:
    growth_map: Dict[Tuple[int, int], float] = {
        (2024, 2025): 1.25,
        (2023, 2025): 1.56,
    }
    if (source_year, target_year) in growth_map:
        return growth_map[(source_year, target_year)]
    combination = (source_year, target_year)
    if combination not in _growth_factor_log_cache:
        logger.info(
            "Using default growth factor 1.0 for source_year=%s -> target_year=%s",
            source_year,
            target_year,
        )
        _growth_factor_log_cache.add(combination)
    return 1.0


def forecast_for_day(
    target_day: date,
    revenues_map: Dict[date, float],
    dates_by_year: Dict[int, List[date]],
) -> Tuple[Optional[float], int, int, int]:
    total = 0.0
    count = 0
    n_candidates = sum(len(values) for values in dates_by_year.values())
    hits_prev1 = 0
    hits_prev2 = 0

    for source_year, candidate_dates in dates_by_year.items():
        factor = growth_factor_for(source_year, target_day.year)
        source_hits = 0
        for candidate in candidate_dates:
            revenue = revenues_map.get(candidate)
            if revenue is None:
                continue
            scaled = revenue * factor
            if not math.isfinite(scaled):
                continue
            total += scaled
            count += 1
            source_hits += 1
        if source_year == target_day.year - 1:
            hits_prev1 = source_hits
        elif source_year == target_day.year - 2:
            hits_prev2 = source_hits

    forecast_value = total / count if count else None
    return forecast_value, n_candidates, hits_prev1, hits_prev2


def make_daily_items(
    start_date: date,
    horizon_days: int,
    wage_pct: float,
    candidate_map: Dict[date, Dict[int, List[date]]],
    revenues_map: Dict[date, float],
) -> Tuple[List[DailyItem], Dict[str, float]]:
    items: List[DailyItem] = []
    totals = {"forecast": 0.0, "wage": 0.0, "count_forecast": 0}

    for offset in range(horizon_days):
        target_day = start_date + timedelta(days=offset)
        dates_by_year = candidate_map[target_day]
        forecast_value, n_candidates, hits_prev1, hits_prev2 = forecast_for_day(
            target_day, revenues_map, dates_by_year
        )
        wage_value: Optional[float] = None
        if forecast_value is not None:
            wage_value = forecast_value * wage_pct
            totals["forecast"] += forecast_value
            totals["wage"] += wage_value
            totals["count_forecast"] += 1
        iso_year, iso_week, iso_weekday = iso_parts(target_day)
        items.append(
            DailyItem(
                date=target_day,
                iso_year=iso_year,
                iso_week=iso_week,
                iso_weekday=iso_weekday,
                forecast=round(forecast_value, 2) if forecast_value is not None else None,
                wage=round(wage_value, 2) if wage_value is not None else None,
                n_candidates=n_candidates,
                n_hits_prev1=hits_prev1,
                n_hits_prev2=hits_prev2,
            )
        )

    return items, totals


def weekly_aggregate(daily_items: List[DailyItem]) -> List[WeeklyItem]:
    weekly_map: Dict[Tuple[int, int], Dict[str, float]] = {}
    for item in daily_items:
        key = (item.iso_year, item.iso_week)
        entry = weekly_map.setdefault(
            key,
            {
                "days": 0,
                "sum_forecast": 0.0,
                "sum_wage": 0.0,
                "count_forecast": 0,
                "count_wage": 0,
            },
        )
        entry["days"] += 1
        if item.forecast is not None:
            entry["sum_forecast"] += item.forecast
            entry["count_forecast"] += 1
        if item.wage is not None:
            entry["sum_wage"] += item.wage
            entry["count_wage"] += 1

    weekly_items: List[WeeklyItem] = []
    for (iso_year, iso_week), entry in sorted(weekly_map.items()):
        sum_forecast = entry["sum_forecast"] if entry["count_forecast"] else None
        sum_wage = entry["sum_wage"] if entry["count_wage"] else None
        weekly_items.append(
            WeeklyItem(
                iso_year=iso_year,
                iso_week=iso_week,
                days=int(entry["days"]),
                sum_forecast=round(sum_forecast, 2) if sum_forecast is not None else None,
                sum_wage=round(sum_wage, 2) if sum_wage is not None else None,
            )
        )
    return weekly_items


def calculate_calendar_forecast(
    request: CalendarForecastRequest,
) -> CalendarForecastResponse:
    start_date = request.start_date
    horizon_days = request.horizon_days
    wage_pct = request.wage_pct

    config = get_mongo_config()
    client = get_mongo_client()

    candidate_map: Dict[date, Dict[int, List[date]]] = {}
    aggregated_dates: Dict[int, Set[date]] = {}

    for offset in range(horizon_days):
        target_day = start_date + timedelta(days=offset)
        candidates = build_candidate_dates(target_day)
        candidate_map[target_day] = candidates
        for source_year, dates in candidates.items():
            aggregated_dates.setdefault(source_year, set()).update(dates)

    dates_by_year = {
        year: sorted(list(dates)) for year, dates in aggregated_dates.items()
    }

    revenues_map = fetch_revenues_mongo(
        client,
        config.db,
        config.coll,
        config.date_field,
        config.revenue_field,
        dates_by_year,
    )

    daily_items, totals = make_daily_items(
        start_date, horizon_days, wage_pct, candidate_map, revenues_map
    )
    weekly_items = weekly_aggregate(daily_items)

    total_forecast = (
        round(totals["forecast"], 2) if totals["count_forecast"] else None
    )
    total_wage = round(totals["wage"], 2) if totals["count_forecast"] else None

    meta = ForecastMeta(
        start_date=start_date,
        end_date=start_date + timedelta(days=horizon_days - 1),
        horizon_days=horizon_days,
        wage_pct=round(wage_pct, 2),
        n_days=horizon_days,
        n_days_forecasted=int(totals["count_forecast"]),
        total_forecast=total_forecast,
        total_wage=total_wage,
    )

    return CalendarForecastResponse(meta=meta, daily=daily_items, weekly=weekly_items)


def ensure_calendar_forecast(
    request: CalendarForecastRequest,
) -> CalendarForecastResponse:
    try:
        return calculate_calendar_forecast(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - uforudsete fejl logges
        logger.exception("Unhandled calendar forecast error")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
