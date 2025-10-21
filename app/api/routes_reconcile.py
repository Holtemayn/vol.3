from __future__ import annotations

from statistics import mean, median
from typing import Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.forecast_log import read_recent_forecasts

router = APIRouter()

_TARGET_DAYS = (1, 3, 5, 10)


class ReconcileRequest(BaseModel):
    limit: int = Field(
        10,
        ge=1,
        le=50,
        description="Antal logposter der analyseres for afstemning",
    )


class DayDelta(BaseModel):
    day: int
    count: int
    average_difference: float | None
    median_difference: float | None
    sum_difference: float
    average_predicted: float | None
    average_actual: float | None


class ReconcileResponse(BaseModel):
    logs_analyzed: int
    targets: List[int]
    totals: Dict[str, float]
    days: List[DayDelta]


def _collect_differences(entries: List[dict]) -> Dict[int, List[dict]]:
    bucket: Dict[int, List[dict]] = {day: [] for day in _TARGET_DAYS}
    for entry in entries:
        rows = entry.get("rows") or []
        for index, row in enumerate(rows, start=1):
            if index not in bucket:
                continue
            actual = row.get("revenue_actual")
            predicted = row.get("revenue_pred")
            if actual is None or predicted is None:
                continue
            bucket[index].append(
                {
                    "forecast_date": entry.get("forecast_date"),
                    "date": row.get("date"),
                    "predicted": float(predicted),
                    "actual": float(actual),
                    "difference": float(actual) - float(predicted),
                }
            )
    return bucket


def _aggregate(bucket: Dict[int, List[dict]]) -> List[DayDelta]:
    days: List[DayDelta] = []
    for day in _TARGET_DAYS:
        records = bucket.get(day) or []
        if not records:
            days.append(
                DayDelta(
                    day=day,
                    count=0,
                    average_difference=None,
                    median_difference=None,
                    sum_difference=0.0,
                    average_predicted=None,
                    average_actual=None,
                )
            )
            continue
        diffs = [rec["difference"] for rec in records]
        preds = [rec["predicted"] for rec in records]
        actuals = [rec["actual"] for rec in records]
        days.append(
            DayDelta(
                day=day,
                count=len(records),
                average_difference=mean(diffs),
                median_difference=median(diffs),
                sum_difference=sum(diffs),
                average_predicted=mean(preds),
                average_actual=mean(actuals),
            )
        )
    return days


@router.post("", response_model=ReconcileResponse)
def reconcile(req: ReconcileRequest) -> ReconcileResponse:
    entries = read_recent_forecasts(limit=req.limit)
    bucket = _collect_differences(entries)
    day_metrics = _aggregate(bucket)

    totals = {
        "predicted": 0.0,
        "actual": 0.0,
        "difference": 0.0,
    }
    for records in bucket.values():
        for rec in records:
            totals["predicted"] += rec["predicted"]
            totals["actual"] += rec["actual"]
            totals["difference"] += rec["difference"]

    return ReconcileResponse(
        logs_analyzed=len(entries),
        targets=list(_TARGET_DAYS),
        totals=totals,
        days=day_metrics,
    )
