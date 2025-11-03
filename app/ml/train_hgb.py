from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("data/daily_aggregates.mongo.jsonl")
MODEL_PATH = BASE_DIR / "HGB" / "model_hgb_daily.pkl"
METRICS_PATH = BASE_DIR / "HGB" / "metrics.json"
FEATURE_COLUMNS_PATH = BASE_DIR / "HGB" / "feature_columns.json"


def _load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset mangler: {DATA_PATH}")
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record = {}
            for key, value in payload.items():
                if key == "dato":
                    if isinstance(value, dict):
                        value = value.get("$date")
                    record["date"] = pd.to_datetime(value).tz_localize(None)
                else:
                    record[key] = value
            rows.append(record)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["oms"] = frame["oms"].astype(float)
    return frame


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data = data.dropna(subset=feature_cols + ["oms"])
    X = data[feature_cols].astype(float)
    y = data["oms"].astype(float)
    return X, y


def _export_metrics(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    cutoff_date: pd.Timestamp | None,
) -> None:
    metrics = {
        "train": {
            "MAE": float(mean_absolute_error(y_true_train, y_pred_train)),
            "RMSE": float(mean_squared_error(y_true_train, y_pred_train, squared=False)),
            "R2": float(r2_score(y_true_train, y_pred_train)),
        },
        "test": {
            "MAE": float(mean_absolute_error(y_true_test, y_pred_test)),
            "RMSE": float(mean_squared_error(y_true_test, y_pred_test, squared=False)),
            "R2": float(r2_score(y_true_test, y_pred_test)),
        },
    }
    if cutoff_date is not None:
        metrics["cutoff_date"] = cutoff_date.date().isoformat()
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    frame = _load_dataset()
    X, y = _prepare_features(frame, feature_columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False,
    )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=600,
        max_leaf_nodes=25,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_features=1.0,
        max_bins=255,
        early_stopping=True,
        scoring="neg_mean_absolute_error",
        validation_fraction=0.2,
        n_iter_no_change=30,
        tol=1e-7,
        random_state=30,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    cutoff = frame["date"].max() if not frame.empty else None

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    _export_metrics(y_train.to_numpy(), y_pred_train, y_test.to_numpy(), y_pred_test, cutoff)


if __name__ == "__main__":
    main()
