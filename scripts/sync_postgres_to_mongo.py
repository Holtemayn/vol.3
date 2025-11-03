"""
Sync daily aggregates (including lag features) from Postgres into MongoDB.

Usage:
    python3 scripts/sync_postgres_to_mongo.py [--dry-run]

Requires that the following environment variables are configured:
    - DATABASE_URL
    - MONGO_URI
    - (optional) MONGO_DB_NAME, MONGO_DAILY_COLLECTION
"""

from __future__ import annotations

import argparse
from datetime import timezone
from typing import Iterable, List

import pandas as pd
from pymongo import MongoClient, ReplaceOne
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import settings


def _get_engine() -> Engine:
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return create_engine(settings.DATABASE_URL)


def _fetch_postgres_frame(engine: Engine) -> pd.DataFrame:
    query = text(
        """
        SELECT
            dato::date AS date,
            oms,
            temperature_2m,
            wind_gusts_10m,
            rain,
            sunshine_duration,
            EXTRACT(DOW FROM dato)::int + 1 AS ugedag,
            EXTRACT(WEEK FROM dato)::int AS ugenummer,
            EXTRACT(MONTH FROM dato)::int AS måned,
            EXTRACT(YEAR FROM dato)::int AS år,
            LAG(oms, 1) OVER w AS lag1,
            LAG(oms, 7) OVER w AS lag7,
            AVG(oms) OVER (w ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS ma7
        FROM {table}
        WINDOW w AS (ORDER BY dato)
        ORDER BY date;
        """.format(table=settings.POSTGRES_DAILY_TABLE)
    )
    frame = pd.read_sql_query(query, engine)
    if frame.empty:
        raise ValueError("Postgres returned no rows for daily aggregates")
    return frame


def _prepare_documents(frame: pd.DataFrame) -> List[dict]:
    docs: List[dict] = []
    frame = frame.sort_values("date").reset_index(drop=True)
    for row in frame.itertuples(index=False):
        data = row._asdict()
        dt = pd.Timestamp(data["date"]).to_pydatetime().replace(tzinfo=timezone.utc)
        doc = {
            "dato": dt,
            "oms": float(data.get("oms")) if data.get("oms") is not None else None,
            "temperature_2m": float(data.get("temperature_2m")) if data.get("temperature_2m") is not None else None,
            "wind_gusts_10m": float(data.get("wind_gusts_10m")) if data.get("wind_gusts_10m") is not None else None,
            "rain": float(data.get("rain")) if data.get("rain") is not None else None,
            "sunshine_duration": float(data.get("sunshine_duration")) if data.get("sunshine_duration") is not None else None,
            "ugedag": int(data.get("ugedag")) if data.get("ugedag") is not None else None,
            "ugenummer": int(data.get("ugenummer")) if data.get("ugenummer") is not None else None,
            "måned": int(data.get("måned")) if data.get("måned") is not None else None,
            "år": int(data.get("år")) if data.get("år") is not None else None,
            "lag1": float(data.get("lag1")) if data.get("lag1") is not None else None,
            "lag7": float(data.get("lag7")) if data.get("lag7") is not None else None,
            "ma7": float(data.get("ma7")) if data.get("ma7") is not None else None,
        }
        docs.append(doc)
    return docs


def _write_to_mongo(docs: Iterable[dict], dry_run: bool = False) -> int:
    if not settings.MONGO_URI:
        raise RuntimeError("MONGO_URI is not configured")

    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    collection = db[settings.MONGO_DAILY_COLLECTION]

    operations = [
        ReplaceOne({"dato": doc["dato"]}, doc, upsert=True)
        for doc in docs
    ]

    if dry_run:
        return len(operations)

    if operations:
        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync lagged daily aggregates from Postgres to MongoDB.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but do not write to MongoDB.")
    args = parser.parse_args()

    engine = _get_engine()
    frame = _fetch_postgres_frame(engine)
    docs = _prepare_documents(frame)
    count = _write_to_mongo(docs, dry_run=args.dry_run)

    if args.dry_run:
        print(f"Fetched {len(docs)} documents (dry-run, nothing written).")
    else:
        print(f"Written/updated {count} documents in MongoDB.")


if __name__ == "__main__":
    main()
