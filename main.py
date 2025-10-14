
from datetime import date
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.api.routes_forecast import router as forecast_router
from app.api.routes_planday import router as planday_router
from app.api.routes_reconcile import router as reconcile_router
from app.core.config import settings
from app.services.forecast import ForecastResult, generate_forecast

app = FastAPI(title="CaféCaster v3")
templates = Jinja2Templates(directory="app/ui/templates")

app.include_router(forecast_router, prefix="/forecast", tags=["forecast"])
app.include_router(planday_router, prefix="/planday", tags=["planday"])
app.include_router(reconcile_router, prefix="/reconcile", tags=["reconcile"])


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    today = date.today()
    try:
        forecast: ForecastResult = generate_forecast(today, settings.DEFAULT_FORECAST_DAYS)
        items = [
            {
                "date": row.date.isoformat(),
                "revenue_pred": row.revenue_pred,
                "hours_recommended": row.hours_recommended,
                "planday_hours": row.planday_hours,
                "hours_diff": row.hours_diff,
                "temp_max": row.temp_max,
                "precip_sum": row.precip_sum,
                "sunshine_hours": row.sunshine_hours,
                "wind_max": row.wind_max,
            }
            for row in forecast.rows
        ]
        payload: dict[str, Any] = {
            "model_version": forecast.model_version,
            "model_backend": forecast.model_backend,
            "wage_pct": settings.WAGE_PCT,
            "avg_hourly_wage": settings.AVG_HOURLY_WAGE,
            "warnings": forecast.warnings,
            "items": items,
        }
    except Exception as exc:  # pragma: no cover - vi viser fejl i UI
        payload = {
            "model_version": settings.MODEL_VERSION,
            "model_backend": settings.MODEL_BACKEND,
            "wage_pct": settings.WAGE_PCT,
            "avg_hourly_wage": settings.AVG_HOURLY_WAGE,
            "warnings": [f"Dashboard kunne ikke hente forecast: {exc}"],
            "items": [],
        }
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "initial_forecast": payload,
            "default_start": today.isoformat(),
            "default_horizon": settings.DEFAULT_FORECAST_DAYS,
            "openai_ready": bool(settings.OPENAI_API_KEY),
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": settings.MODEL_VERSION}


@app.get("/chat/config")
def chat_config():
    """
    Klar til OpenAI ChatKit – returnerer kun flag nu.
    Frontenden kan bruge endpointet til at aktivere widget når nøglen findes.
    """
    return JSONResponse(
        {
            "ready": bool(settings.OPENAI_API_KEY),
            "default_model": "gpt-4o-mini",
            "instructions": "Attach OpenAI ChatKit when ready. Endpoint prepared.",
        }
    )
