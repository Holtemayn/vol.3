"""
Sync logged Planday actuals from forecast.log into MongoDB daily aggregates.

Usage:
    python3 scripts/sync_forecast_log_to_mongo.py [--dry-run] [--log-path PATH]

Assumptions:
    - forecast.log er newline-delimited JSON (standardformatet i appen).
    - hvert entry indeholder feltet ``rows`` med ``date`` (ISO) og
      ``revenue_actual`` samt evt. vejr-/planday felter.
    - relevante Mongo-indstillinger hentes fra app.core.config.settings.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from pymongo import MongoClient, UpdateOne

from app.core.config import settings


def _load_entries(log_path: Path) -> Iterable[dict]:
    if not log_path.exists():
        raise FileNotFoundError(f"Logfil ikke fundet: {log_path}")
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _build_operations(entries: Iterable[dict]) -> List[UpdateOne]:
    operations: List[UpdateOne] = []
    for entry in entries:
        rows = entry.get("rows") or []
        for row in rows:
            revenue_actual = row.get("revenue_actual")
            if revenue_actual is None:
                continue
            date_str: Optional[str] = row.get("date")
            if not date_str:
                continue
            try:
                row_date = datetime.fromisoformat(date_str).date()
            except ValueError:
                continue
            dt_value = datetime.combine(row_date, time.min, tzinfo=timezone.utc)
            sunshine_hours = row.get("sunshine_hours")
            sunshine_duration = float(sunshine_hours) * 3600.0 if sunshine_hours is not None else None
            update_values = {
                "oms": float(revenue_actual),
                "planday_hours": float(row["planday_hours"]) if row.get("planday_hours") is not None else None,
                "temperature_2m": float(row["temp_max"]) if row.get("temp_max") is not None else None,
                "wind_gusts_10m": float(row["wind_max"]) if row.get("wind_max") is not None else None,
                "rain": float(row["precip_sum"]) if row.get("precip_sum") is not None else None,
                "sunshine_duration": sunshine_duration,
                "ugedag": row_date.weekday(),
                "ugenummer": row_date.isocalendar().week,
                "måned": row_date.month,
                "år": row_date.year,
            }
            operations.append(
                UpdateOne(
                    {"dato": dt_value},
                    {
                        "$set": {k: v for k, v in update_values.items() if v is not None},
                        "$setOnInsert": {
                            "dato": dt_value,
                            "lag1": None,
                            "lag7": None,
                            "ma7": None,
                        },
                    },
                    upsert=True,
                )
            )
    return operations


def _write_to_mongo(operations: List[UpdateOne], dry_run: bool) -> int:
    if dry_run or not operations:
        return len(operations)
    if not settings.MONGO_URI:
        raise RuntimeError("MONGO_URI er ikke konfigureret")
    client = MongoClient(settings.MONGO_URI)
    collection = client[settings.MONGO_DB_NAME][settings.MONGO_DAILY_COLLECTION]
    result = collection.bulk_write(operations, ordered=False)
    return (result.upserted_count or 0) + (result.modified_count or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync forecast log actuals into MongoDB.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(settings.FORECAST_LOG_PATH),
        help="Path til forecast.log (default: settings.FORECAST_LOG_PATH).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Vis antal operationer uden at skrive til Mongo.")
    args = parser.parse_args()

    entries = _load_entries(args.log_path)
    operations = _build_operations(entries)
    count = _write_to_mongo(operations, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry-run: ville opdatere {count} dokumenter.")
    else:
        print(f"Opdaterede {count} dokumenter i MongoDB.")


if __name__ == "__main__":
    main()
