from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from app.services.weather import WeatherPayload


RAIN_BINS = [-0.1, 0.1, 1, 4, 10, 20, np.inf]
RAIN_LABELS = ["0", "0.1-1", "1.1-4", "4.1-10", "10.1-20", "20+"]


def build_feature_frame(dates: Iterable[pd.Timestamp], weather: WeatherPayload) -> pd.DataFrame:
    """
    Kombinér datoer og vejrdata til en feature-ramme.

    Bruges både af regressions-fallback og en evt. ML-model.
    """
    dates_df = pd.DataFrame({"date": list(dates)})
    frame = dates_df.merge(weather.frame, on="date", how="left")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month
    frame["weekday"] = frame["date"].dt.weekday + 1  # match vol.2
    frame["year"] = frame["date"].dt.year

    frame["precip_sum"] = frame["precip_sum"].fillna(0.0)
    frame["sunshine_hours"] = frame["sunshine_hours"].fillna(0.0)
    frame["wind_max"] = frame["wind_max"].fillna(0.0)
    frame["temp_max"] = frame["temp_max"].ffill().bfill().fillna(15.0)

    frame["rain_group"] = pd.cut(frame["precip_sum"], bins=RAIN_BINS, labels=RAIN_LABELS)
    frame["rain_group"] = frame["rain_group"].astype(str)

    return frame
