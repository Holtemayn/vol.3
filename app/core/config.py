from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app configuration loaded fra miljø-variabler eller .env.

    Felt-navne holdes tæt på tidligere versioner for at gøre migration enklere.
    """

    # ------------------------------------------------------------------
    # Application / model
    MODEL_VERSION: str = "v3.0"
    MODEL_BACKEND: Literal["xgboost", "regression"] = "regression"
    MODEL_PATH: str = "./models/xgb_v3.pkl"

    DEFAULT_FORECAST_DAYS: int = 10
    WAGE_PCT: float = 0.20
    AVG_HOURLY_WAGE: float = 155.0

    # ------------------------------------------------------------------
    # Weather / Open-Meteo
    WEATHER_LATITUDE: float = 55.6761
    WEATHER_LONGITUDE: float = 12.5683
    WEATHER_TIMEZONE: str = "Europe/Copenhagen"
    WEATHER_API_BASE_URL: str | None = None

    # ------------------------------------------------------------------
    # Planday OAuth
    PLANDAY_CLIENT_ID: str | None = None
    PLANDAY_CLIENT_SECRET: str | None = None
    PLANDAY_REFRESH_TOKEN: str | None = None
    PLANDAY_REDIRECT_URI: str | None = None
    PLANDAY_TENANT_ID: str | None = None
    PLANDAY_DEPARTMENT_ID: str | None = None
    PLANDAY_EMPLOYEE_GROUP_ID: str | None = None
    PLANDAY_REVENUE_UNIT_ID: str | None = None

    # ------------------------------------------------------------------
    # Google Sheets (service account JSON forventes base64-enkodet)
    GOOGLE_SHEETS_CREDENTIALS_JSON: str | None = None
    GOOGLE_SHEETS_SPREADSHEET_ID: str | None = None
    GOOGLE_SHEETS_WORKSHEET_NAME: str | None = None

    # ------------------------------------------------------------------
    # Misc / future extensions
    DATABASE_URL: str | None = Field(default=None)
    OPENAI_API_KEY: str | None = None
    FORECAST_LOG_PATH: str = "./logs/forecast.log"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
