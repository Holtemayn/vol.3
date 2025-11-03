import logging
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.services.calendar_forecast import (
    CalendarForecastRequest,
    CalendarForecastResponse,
    calculate_calendar_forecast,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cafecaster.app")

app = FastAPI(title="Cafecaster Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    logger.debug("Validation error for request %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=CalendarForecastResponse)
async def forecast_endpoint(request: CalendarForecastRequest) -> CalendarForecastResponse:
    try:
        return await run_in_threadpool(calculate_calendar_forecast, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Unhandled calendar forecast error")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
