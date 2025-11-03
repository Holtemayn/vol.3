from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent
_HGB_DIR = _BASE_DIR / "HGB"
_MODEL_PATH = _HGB_DIR / "model_hgb_daily.pkl"
_FEATURE_COLUMNS_PATH = _HGB_DIR / "feature_columns.json"
_METRICS_PATH = _HGB_DIR / "metrics.json"
_HISTORY_PATH = Path("data/daily_aggregates.mongo.jsonl")


@lru_cache
def _load_feature_columns() -> List[str]:
    with _FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("feature_columns.json skal indeholde en liste")
    return [str(column) for column in data]


@lru_cache
def _load_metrics() -> Dict[str, Any]:
    if not _METRICS_PATH.exists():
        return {}
    with _METRICS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("metrics.json skal være et JSON-objekt")
    return payload


@lru_cache
def _load_model():
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"HGB-model ikke fundet: {_MODEL_PATH}")
    return joblib.load(_MODEL_PATH)


@lru_cache
def _load_history_frame() -> pd.DataFrame:
    if not _HISTORY_PATH.exists():
        raise FileNotFoundError(f"Historiske data ikke fundet: {_HISTORY_PATH}")
    rows: List[Dict[str, Any]] = []
    with _HISTORY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            raw_date = payload.get("dato")
            if isinstance(raw_date, dict):
                raw_date = raw_date.get("$date")
            if not raw_date:
                continue
            dt = pd.to_datetime(raw_date).tz_localize(None)
            oms_value = payload.get("oms")
            try:
                oms = float(oms_value) if oms_value is not None else None
            except (TypeError, ValueError):
                oms = None
            rows.append({"date": dt.floor("D"), "oms": oms})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Ingen rækker i {_HISTORY_PATH}")
    frame = frame.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _lookup(history: pd.DataFrame, target_date: pd.Timestamp) -> float:
    match = history[history["date"] == target_date]
    if match.empty:
        return math.nan
    value = match.iloc[-1]["oms"]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    return float(value)


def _rolling_mean(history: pd.DataFrame, end_date: pd.Timestamp, window: int) -> float:
    subset = history[history["date"] < end_date]["oms"].dropna()
    if subset.empty:
        return math.nan
    tail = subset.tail(window)
    if tail.empty:
        return math.nan
    return float(tail.mean())


def predict_hgb(features: pd.DataFrame) -> pd.DataFrame:
    """
    Returnér DataFrame med kolonnerne ``date`` og ``revenue_pred`` baseret på HGB-modellen.
    """
    if features is None or features.empty:
        return pd.DataFrame(columns=["date", "revenue_pred"])

    model = _load_model()
    feature_columns = _load_feature_columns()

    history = _load_history_frame().copy()
    history["oms"] = history["oms"].astype(float)
    history = history.sort_values("date").reset_index(drop=True)

    working_history = history.copy()
    forecast_rows: List[Dict[str, Any]] = []

    sorted_features = features.copy()
    sorted_features["date"] = pd.to_datetime(sorted_features["date"]).dt.floor("D")
    sorted_features = sorted_features.sort_values("date").reset_index(drop=True)

    for row in sorted_features.itertuples(index=False):
        dt = row.date
        lag1 = _lookup(working_history, dt - pd.Timedelta(days=1))
        lag7 = _lookup(working_history, dt - pd.Timedelta(days=7))
        ma7 = _rolling_mean(working_history, dt, window=7)

        sunshine_hours = _safe_float(getattr(row, "sunshine_hours", None), default=0.0)
        feature_row = {
            "ugedag": dt.weekday(),
            "måned": dt.month,
            "år": dt.year,
            "temperature_2m": _safe_float(getattr(row, "temp_max", None), default=0.0),
            "wind_gusts_10m": _safe_float(getattr(row, "wind_max", None), default=0.0),
            "rain": _safe_float(getattr(row, "precip_sum", None), default=0.0),
            "sunshine_duration": sunshine_hours * 3600.0,
            "lag1": lag1,
            "lag7": lag7,
            "ma7": ma7,
        }

        feature_df = pd.DataFrame([feature_row])
        for column in feature_columns:
            if column not in feature_df.columns:
                feature_df[column] = 0.0
        feature_df = feature_df[feature_columns]

        prediction = float(model.predict(feature_df)[0])
        forecast_rows.append({"date": dt, "revenue_pred": prediction})

        working_history = (
            pd.concat([working_history, pd.DataFrame({"date": [dt], "oms": [prediction]})], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

    return pd.DataFrame(forecast_rows)


def get_hgb_feature_columns() -> List[str]:
    return list(_load_feature_columns())


def get_hgb_metrics() -> Dict[str, Any]:
    metrics = _load_metrics()
    return json.loads(json.dumps(metrics)) if metrics else {}


def get_hgb_history_dates() -> Iterable[pd.Timestamp]:
    frame = _load_history_frame()
    return tuple(frame["date"].tolist())
