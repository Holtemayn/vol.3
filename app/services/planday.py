from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Union

import pandas as pd
import requests

from app.core.config import settings

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://openapi.planday.com"
TOKEN_URLS = (
    "https://auth.planday.com/connect/token",
    "https://id.planday.com/connect/token",
)
DEFAULT_TZ = settings.WEATHER_TIMEZONE or "Europe/Copenhagen"
_TOKEN_CACHE: Dict[str, Union[str, datetime]] = {"token": None, "expires_at": None}
_LAST_WARNING: Optional[str] = None


def get_planday_hours_for_dates(dates: List[date]) -> Dict[date, float | None]:
    """
    Returnér {date: planlagte timer}. Mangler integration → None.
    """
    global _LAST_WARNING
    _LAST_WARNING = None
    if not dates:
        return {}
    if not _planday_configured():
        _LAST_WARNING = "Planday credentials mangler (CLIENT_ID/REFRESH_TOKEN)."
        return {d: None for d in dates}

    unique_dates = sorted({d for d in dates})
    start, end = unique_dates[0], unique_dates[-1]
    try:
        df = _fetch_hours_dataframe(start, end, tz=DEFAULT_TZ)
    except Exception as exc:  # pragma: no cover - netværksfejl mv.
        LOGGER.warning("Planday fetch failed: %s", exc)
        _LAST_WARNING = str(exc)
        return {d: None for d in dates}

    if df is None or df.empty:
        _LAST_WARNING = f"Ingen planlagte timer i Planday for intervallet {start}..{end}."
        return {d: None for d in dates}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.groupby("date", as_index=False)["planday_hours"].sum()
    df_map = {row.date: float(row.planday_hours) for row in df.itertuples(index=False)}

    result: Dict[date, float | None] = {}
    for d in dates:
        value = df_map.get(d)
        result[d] = round(value, 2) if value and value > 0 else None
    return result


# ------------------------------------------------------------------------------
# Access tokens & headers

def _planday_configured() -> bool:
    return bool(settings.PLANDAY_CLIENT_ID and settings.PLANDAY_REFRESH_TOKEN)


def _get_access_token() -> str:
    now = datetime.utcnow()
    cached_token = _TOKEN_CACHE.get("token")
    expires_at = _TOKEN_CACHE.get("expires_at")
    if cached_token and isinstance(expires_at, datetime) and expires_at > now + timedelta(seconds=30):
        return cached_token  # type: ignore[return-value]

    payload = {
        "client_id": settings.PLANDAY_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": settings.PLANDAY_REFRESH_TOKEN,
    }
    if settings.PLANDAY_CLIENT_SECRET:
        payload["client_secret"] = settings.PLANDAY_CLIENT_SECRET
    last_error: Optional[str] = None
    for token_url in TOKEN_URLS:
        try:
            response = requests.post(
                token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            if response.status_code == 404:
                last_error = f"{token_url} returned 404"
                continue
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:  # pragma: no cover - afhænger af netværk
            last_error = str(exc)
            continue
        token = data.get("access_token")
        if not token:
            last_error = f"No access_token in response from {token_url}"
            continue
        expires_in = data.get("expires_in", 1800)
        _TOKEN_CACHE["token"] = token
        _TOKEN_CACHE["expires_at"] = now + timedelta(seconds=int(expires_in))
        return token
    raise RuntimeError(f"Unable to fetch Planday access token: {last_error}")


def _headers() -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {_get_access_token()}",
        "X-ClientId": settings.PLANDAY_CLIENT_ID,
        "Accept": "application/json",
    }
    if settings.PLANDAY_TENANT_ID:
        headers["X-Tenant"] = str(settings.PLANDAY_TENANT_ID)
    return headers


# ------------------------------------------------------------------------------
# HTTP helpers

def _ensure_list(payload, list_keys=("items", "data", "results", "shifts", "scheduleDays", "days")) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in list_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _fetch_schedule_day(
    from_date: date,
    to_date: date,
    department_id: Optional[int],
    employee_group_id: Optional[int],
) -> List[dict]:
    params = {
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
    }
    if department_id:
        params["departmentId"] = department_id
    if employee_group_id:
        params["employeeGroupId"] = employee_group_id
    response = requests.get(
        f"{BASE_URL}/scheduling/v1.0/scheduleDay",
        headers=_headers(),
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return _ensure_list(response.json(), list_keys=("items", "data", "results", "scheduleDays", "days"))


def _fetch_shifts(
    from_date: date,
    to_date_exclusive: date,
    department_id: Optional[int],
    employee_group_id: Optional[int],
) -> List[dict]:
    params = {
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date_exclusive.strftime("%Y-%m-%d"),
    }
    if department_id:
        params["departmentId"] = department_id
    if employee_group_id:
        params["employeeGroupId"] = employee_group_id
    response = requests.get(
        f"{BASE_URL}/scheduling/v1.0/shifts",
        headers=_headers(),
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return _ensure_list(response.json(), list_keys=("items", "data", "results", "shifts"))


# ------------------------------------------------------------------------------
# Aggregation helpers

def _split_shift_over_days(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[tuple[pd.Timestamp, float]]:
    out: List[tuple[pd.Timestamp, float]] = []
    cursor = start_ts
    while cursor.date() < end_ts.date():
        midnight = cursor.normalize() + pd.Timedelta(days=1)
        if cursor.tz is not None and midnight.tz is None:
            midnight = midnight.tz_localize(cursor.tz)
        out.append((pd.Timestamp(cursor.date(), tz=cursor.tz), (midnight - cursor).total_seconds() / 3600.0))
        cursor = midnight
    out.append((pd.Timestamp(cursor.date(), tz=cursor.tz), (end_ts - cursor).total_seconds() / 3600.0))
    return out


def _to_local(series: pd.Series, tz: str) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize(tz)
    return dt.dt.tz_convert(tz)


def _aggregate_schedule_days(payload: Iterable[dict], employee_group_ids: Optional[Set[int]]) -> pd.DataFrame:
    items = _ensure_list(payload, list_keys=("items", "data", "results", "scheduleDays", "days"))
    if not items:
        return pd.DataFrame(columns=["date", "planday_hours"])
    rows = []
    for it in items:
        if employee_group_ids and not _item_matches_group(it, employee_group_ids):
            continue
        date_str = (it.get("date") or it.get("day") or it.get("scheduleDate"))
        if not date_str:
            continue
        date_val = pd.to_datetime(str(date_str)[:10], errors="coerce")
        if pd.isna(date_val):
            continue
        seconds = (
            it.get("totalTimeInSeconds")
            or it.get("timeInSeconds")
            or it.get("scheduledTimeInSeconds")
            or 0
        )
        if not seconds:
            shifts = it.get("shifts") or []
            total = 0
            for shift in shifts:
                st = pd.to_datetime(shift.get("startDateTime"), errors="coerce")
                en = pd.to_datetime(shift.get("endDateTime"), errors="coerce")
                if pd.isna(st) or pd.isna(en) or en <= st:
                    continue
                total += (en - st).total_seconds()
            seconds = total
        rows.append({"date": date_val.date(), "planday_hours": seconds / 3600.0})
    if not rows:
        return pd.DataFrame(columns=["date", "planday_hours"])
    return pd.DataFrame(rows).groupby("date", as_index=False)["planday_hours"].sum()


def _aggregate_shifts(payload: Iterable[dict], tz: str, employee_group_ids: Optional[Set[int]]) -> pd.DataFrame:
    items = _ensure_list(payload, list_keys=("items", "data", "results", "shifts"))
    if not items:
        return pd.DataFrame(columns=["date", "planday_hours"])
    df = pd.DataFrame(items)
    if df.empty or "startDateTime" not in df or "endDateTime" not in df:
        return pd.DataFrame(columns=["date", "planday_hours"])
    if employee_group_ids:
        df = _filter_shifts_by_group(df, employee_group_ids)
        if df.empty:
            return pd.DataFrame(columns=["date", "planday_hours"])
    df["start"] = _to_local(df["startDateTime"], tz)
    df["end"] = _to_local(df["endDateTime"], tz)
    rows: List[dict] = []
    for start_ts, end_ts in zip(df["start"], df["end"]):
        if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
            continue
        for day_ts, hours in _split_shift_over_days(start_ts, end_ts):
            rows.append({"date": day_ts.date(), "hours": hours})
    if not rows:
        return pd.DataFrame(columns=["date", "planday_hours"])
    agg = pd.DataFrame(rows).groupby("date", as_index=False)["hours"].sum()
    return agg.rename(columns={"hours": "planday_hours"})


# ------------------------------------------------------------------------------
# Fetch orchestration
        
def _fetch_hours_dataframe(start: date, end: date, tz: str) -> pd.DataFrame:
    department_id = _coerce_optional_int(settings.PLANDAY_DEPARTMENT_ID, "PLANDAY_DEPARTMENT_ID")
    employee_group_ids = _coerce_optional_int_set(settings.PLANDAY_EMPLOYEE_GROUP_ID, "PLANDAY_EMPLOYEE_GROUP_ID")
    api_employee_group_id: Optional[int] = None
    if employee_group_ids and len(employee_group_ids) == 1:
        api_employee_group_id = next(iter(employee_group_ids))

    days = (end - start).days + 1
    if days > 7:
        frames: List[pd.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=6), end)
            frame = _fetch_hours_dataframe(cursor, chunk_end, tz)
            if frame is not None and not frame.empty:
                frames.append(frame)
            cursor = chunk_end + timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=["date", "planday_hours"])
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.groupby("date", as_index=False)["planday_hours"].sum()
        return _restrict_to_range(combined, start, end)

    # ScheduleDay first
    try:
        schedule_items = _fetch_schedule_day(start, end, department_id, api_employee_group_id)
        schedule_df = _aggregate_schedule_days(schedule_items, employee_group_ids)
        if not schedule_df.empty and schedule_df["planday_hours"].sum() > 0:
            return _restrict_to_range(schedule_df, start, end)
    except Exception as exc:  # pragma: no cover - afhænger af API
        LOGGER.info("ScheduleDay fallback: %s", exc)
        global _LAST_WARNING
        _LAST_WARNING = f"ScheduleDay fallback: {exc}"

    shifts_items = _fetch_shifts(start, end + timedelta(days=1), department_id, api_employee_group_id)
    shifts_df = _aggregate_shifts(shifts_items, tz=tz, employee_group_ids=employee_group_ids)
    return _restrict_to_range(shifts_df, start, end)


# ------------------------------------------------------------------------------
# Optional helper

def list_planday_departments() -> pd.DataFrame:
    response = requests.get(f"{BASE_URL}/hr/v1.0/departments", headers=_headers(), timeout=30)
    response.raise_for_status()
    items = _ensure_list(response.json(), list_keys=("items", "data", "results", "departments"))
    if not items:
        return pd.DataFrame(columns=["id", "name"])
    df = pd.DataFrame(items)
    if {"id", "name"}.issubset(df.columns):
        return df[["id", "name"]]
    return df


def _coerce_optional_int(value: Optional[str], label: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid %s: %s", label, value)
        return None


def _coerce_optional_int_set(value: Optional[str], label: str) -> Optional[Set[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        raw_parts = re.split(r"[,\s;]+", text)
    results: Set[int] = set()
    for part in raw_parts:
        part = str(part).strip()
        if not part:
            continue
        try:
            results.add(int(part))
        except (TypeError, ValueError):
            LOGGER.warning("Invalid %s entry: %s", label, part)
    return results or None


def _restrict_to_range(df: Optional[pd.DataFrame], start: date, end: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "planday_hours"])
    working = df.copy()
    parsed_dates = pd.to_datetime(working["date"], errors="coerce").dt.date
    mask = parsed_dates.notna() & (parsed_dates >= start) & (parsed_dates <= end)
    if not mask.any():
        return pd.DataFrame(columns=["date", "planday_hours"])
    working = working.loc[mask].reset_index(drop=True)
    working["date"] = parsed_dates.loc[mask]
    return working


def get_last_planday_warning() -> Optional[str]:
    return _LAST_WARNING


def _filter_shifts_by_group(df: pd.DataFrame, employee_group_ids: Set[int]) -> pd.DataFrame:
    if not employee_group_ids:
        return df
    targets = {str(employee_group_id) for employee_group_id in employee_group_ids}
    mask = pd.Series(False, index=df.index)
    if "employeeGroupId" in df:
        normalized = df["employeeGroupId"].apply(
            lambda val: str(int(val)) if val is not None and not pd.isna(val) else None
        )
        mask |= normalized.isin(targets)
    if "employeeGroup" in df:
        mask |= df["employeeGroup"].apply(lambda val: bool(_extract_group_ids(val) & targets))
    if "employeeGroups" in df:
        mask |= df["employeeGroups"].apply(lambda val: bool(_extract_group_ids(val) & targets))
    if "groups" in df:
        mask |= df["groups"].apply(lambda val: bool(_extract_group_ids(val) & targets))
    return df[mask] if mask.any() else df.iloc[0:0]


def _item_matches_group(item: dict, employee_group_ids: Set[int]) -> bool:
    if not employee_group_ids:
        return True
    item_ids = _extract_group_ids(item)
    return any(str(employee_group_id) in item_ids for employee_group_id in employee_group_ids)


def _extract_group_ids(value) -> set[str]:
    result: set[str] = set()
    if value is None:
        return result
    if isinstance(value, dict):
        for key in ("employeeGroupId", "id"):
            if key in value and value[key] is not None:
                result.add(str(value[key]))
        for key in ("employeeGroupIds", "groups", "employeeGroups"):
            nested = value.get(key)
            result.update(_extract_group_ids(nested))
    elif isinstance(value, list):
        for item in value:
            result.update(_extract_group_ids(item))
    else:
        try:
            result.add(str(int(value)))
        except (TypeError, ValueError):
            pass
    return result
