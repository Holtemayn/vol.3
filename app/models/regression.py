import pandas as pd

def predict_revenue(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    intercept = -6091.86
    revenue = (
        intercept
        + 1091.45 * df["temp_max"]
        - 161.57 * df["wind_max"]
        + 1050.98 * df["sunshine_hours"]
    )

    month_coef = {2:0,3:1921.83,4:3218.33,5:4395.99,6:3272.20,7:-635.25,8:1713.98,9:-524.20,10:-218.02,11:5332.89,12:7911.60}
    revenue += df["month"].map(month_coef).fillna(0)

    weekday_coef = {1:0,2:-675.05,3:507.61,4:3441.95,5:3488.52,6:3736.35,7:2099.48}
    revenue += df["weekday"].map(weekday_coef).fillna(0)

    rain_coef = {"0":0,"0.1-1":-2542.21,"1.1-4":-4584.56,"4.1-10":-5169.57,"10.1-20":-7182.38,"20+":-7340.26}
    revenue += df["rain_group"].astype(str).map(rain_coef).fillna(0)

    year_coef = {2023:6727.87, 2024:10734.00, 2025:16610.22}
    revenue += df["year"].map(year_coef).fillna(0)

    df["predicted_revenue"] = revenue
    df["predicted_medarbejder_timer"] = (df["predicted_revenue"] * 0.2) / 155
    return df[["time","temp_max","wind_max","precip_sum","sunshine_hours","predicted_revenue","predicted_medarbejder_timer"]]
