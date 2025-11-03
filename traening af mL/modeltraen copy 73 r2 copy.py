import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Load & clean ----------
base_dir = Path(__file__).resolve().parent
repo_root = base_dir.parent
excel_path = base_dir / "OMSAETNING.xlsx"
raw = pd.read_excel(excel_path, sheet_name="Sheet 1 - OMSAETNING")
raw.columns = raw.iloc[0].tolist()
df = raw.iloc[1:].copy()

# Type-cast
df["dato"] = pd.to_datetime(df["dato"], errors="coerce")
for c in ["oms", "ugedag", "ugenummer", "måned", "år",
          "temperature_2m", "rain", "sunshine_duration", "wind_gusts_10m"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop tomme rækker 
df = df.dropna(subset=["dato", "oms"])

# ---------- Aggregér til dagsniveau ----------
daily = df.groupby("dato").agg({
    "oms": "sum",
    "temperature_2m": "mean",
    "wind_gusts_10m": "mean",
    "rain": "sum",
    "sunshine_duration": "sum",
    "ugedag": "first",
    "ugenummer": "first",
    "måned": "first",
    "år": "first"
}).reset_index()

# Sikr samme 1-baserede ugedage som FastAPI-pipelinen
daily["ugedag"] = daily["ugedag"].fillna(daily["dato"].dt.weekday + 1).astype(int)

# Tilføj features
daily["is_weekend"] = daily["dato"].dt.weekday.isin([5, 6]).astype(int)
daily["day_of_year"] = daily["dato"].dt.dayofyear.astype(int)
daily = daily.sort_values("dato").reset_index(drop=True)

# ---------- Features & target ----------
target_col = "oms"
lag_feature_cols = ["lag1", "lag7", "ma7"]
base_feature_cols = [
    "ugedag", "måned", "år",
    "temperature_2m", "wind_gusts_10m",
    "rain", "sunshine_duration",
]

daily["lag1"] = daily[target_col].shift(1)
daily["lag7"] = daily[target_col].shift(7)
daily["ma7"] = daily[target_col].rolling(window=7, min_periods=1).mean().shift(1)

feature_cols = base_feature_cols + lag_feature_cols

X = daily[feature_cols].astype(float)
y = daily[target_col].astype(float)

# ---------- Utilities ----------
def prepare_train_test(cutoff_date: pd.Timestamp):
    train_idx = daily["dato"] <= cutoff_date
    test_idx = ~train_idx

    X_train_split, X_test_split = X[train_idx].copy(), X[test_idx].copy()
    y_train_split, y_test_split = y[train_idx].copy(), y[test_idx].copy()

    valid_train_mask = ~X_train_split[lag_feature_cols].isna().any(axis=1)
    removed_rows_split = len(X_train_split) - valid_train_mask.sum()
    if removed_rows_split > 0:
        X_train_split = X_train_split[valid_train_mask]
        y_train_split = y_train_split[valid_train_mask]

    X_test_split.loc[:, lag_feature_cols] = X_test_split[lag_feature_cols].ffill().bfill()

    return X_train_split, X_test_split, y_train_split, y_test_split, removed_rows_split


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }


base_model_params = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    max_iter=600,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.2,
    random_state=42,
)


def run_validation_windows(offset_days_list, params):
    max_date = daily["dato"].max()
    min_date = daily["dato"].min()
    for offset_days in offset_days_list:
        cutoff_date = max_date - pd.Timedelta(days=offset_days)
        if cutoff_date <= min_date:
            print(f"[Validation] Skipping offset={offset_days}: cutoff {cutoff_date.date()} before data start.")
            continue

        X_train_v, X_test_v, y_train_v, y_test_v, removed_rows_v = prepare_train_test(cutoff_date)
        if len(X_test_v) == 0:
            print(f"[Validation] Skipping offset={offset_days}: no test rows left.")
            continue

        model_v = HistGradientBoostingRegressor(**params)
        model_v.fit(X_train_v, y_train_v)

        train_metrics_v = metrics(y_train_v, model_v.predict(X_train_v))
        test_metrics_v = metrics(y_test_v, model_v.predict(X_test_v))

        print(f"[Validation] offset={offset_days} days (cutoff {cutoff_date.date()}): "
              f"train_rows={len(X_train_v)} (removed {removed_rows_v}), test_rows={len(X_test_v)}")
        print(f"  Train: {train_metrics_v}")
        print(f"  Test: {test_metrics_v}")

    print("")


def evaluate_param_set(params, offsets):
    maes = []
    print(f"=== Evaluating params: {params} ===")
    max_date = daily["dato"].max()
    min_date = daily["dato"].min()
    for offset_days in offsets:
        cutoff_date = max_date - pd.Timedelta(days=offset_days)
        if cutoff_date <= min_date:
            print(f"[Grid] Skipping offset={offset_days}: cutoff {cutoff_date.date()} before data start.")
            continue

        X_train_v, X_test_v, y_train_v, y_test_v, removed_rows_v = prepare_train_test(cutoff_date)
        if len(X_test_v) == 0:
            print(f"[Grid] Skipping offset={offset_days}: no test rows left.")
            continue

        model_v = HistGradientBoostingRegressor(**params)
        model_v.fit(X_train_v, y_train_v)

        test_metrics_v = metrics(y_test_v, model_v.predict(X_test_v))
        maes.append(test_metrics_v["MAE"])
        print(f"[Grid] offset={offset_days} -> Test MAE: {test_metrics_v['MAE']:.2f}, "
              f"R2: {test_metrics_v['R2']:.3f}")

    avg_mae = float(np.mean(maes)) if maes else np.inf
    print(f"=== Avg Test MAE: {avg_mae:.2f} ===\n")
    return avg_mae


param_grid = [
    {**base_model_params},
    {**base_model_params, "learning_rate": 0.03, "max_leaf_nodes": 25},
    {**base_model_params, "learning_rate": 0.08, "max_leaf_nodes": 20},
    {**base_model_params, "learning_rate": 0.05, "max_leaf_nodes": 20, "min_samples_leaf": 20},
    {**base_model_params, "learning_rate": 0.05, "max_leaf_nodes": 25, "min_samples_leaf": 25},
    {**base_model_params, "loss": "absolute_error", "learning_rate": 0.04, "max_leaf_nodes": 25},
]

grid_offsets = [45, 30]
best_params = None
best_score = np.inf
for params in param_grid:
    avg_mae = evaluate_param_set(params, grid_offsets)
    if avg_mae < best_score:
        best_score = avg_mae
        best_params = params

if best_params is None:
    best_params = base_model_params
    print("No valid parameter set found; falling back to base params.")
else:
    print(f"Selected params with lowest avg MAE ({best_score:.2f}): {best_params}\n")

model_params = best_params

validation_offsets = [80, 60, 45, 30]
run_validation_windows(validation_offsets, model_params)

# ---------- Train/test split ----------
main_offset_days = 30
cutoff = daily["dato"].max() - pd.Timedelta(days=main_offset_days)
X_train, X_test, y_train, y_test, removed_rows = prepare_train_test(cutoff)

print(f"Training rows after lag drop: {len(X_train)} (removed {removed_rows})")
print(f"X_train columns: {X_train.columns.tolist()}")

# ---------- Model ----------
model_params = {**model_params, "random_state": 42}

model = HistGradientBoostingRegressor(**model_params)
model.fit(X_train, y_train)

train_metrics = metrics(y_train, model.predict(X_train))
test_metrics = metrics(y_test, model.predict(X_test))

# ---------- Persist ----------
artifacts_dir = repo_root / "app/ml/HGB"
artifacts_dir.mkdir(parents=True, exist_ok=True)
model_path = artifacts_dir / "model_hgb_daily.pkl"
joblib.dump(model, model_path)

feature_list_path = artifacts_dir / "feature_columns.json"
with open(feature_list_path, "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)

metrics_path = artifacts_dir / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({"train": train_metrics, "test": test_metrics, "cutoff_date": str(cutoff.date())},
              f, ensure_ascii=False, indent=2)

print("✅ Model trained and saved")
print("Train:", train_metrics)
print("Test:", test_metrics)
