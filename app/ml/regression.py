from __future__ import annotations

import pandas as pd

from app.models.regression import predict_revenue as vol2_predict


def predict_regression(features: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper omkring vol.2 regressionsmodel så vi genbruger samme kodebase.
    Konverterer ``date`` -> ``time`` og returnerer dataframe med ``date`` og ``revenue_pred``.
    """
    vol2_features = features.copy().rename(columns={"date": "time"})
    vol2_features["time"] = pd.to_datetime(vol2_features["time"])
    result = vol2_predict(vol2_features)
    if "time" not in result or "predicted_revenue" not in result:
        raise ValueError("Regression-resultatet mangler forventede kolonner")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(result["time"]),
            "revenue_pred": result["predicted_revenue"].astype(float),
        }
    )
    return out
