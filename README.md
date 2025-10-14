
# CaféCaster v3 (FastAPI + Railway + Planday + XGBoost)

Vol.2 logikken er migreret til FastAPI v3 med et Jinja-dashboard (Skeleton CSS). Kør lokalt med uvicorn, deploy på Railway (Docker).
**Formål:** Forecast af omsætning (regression fallback + XGBoost når klar), beregn anbefalede timer, hent Planday-timer, vis afvigelser, skriv til Google Sheets – og gør klar til OpenAI ChatKit.

## Hurtig start (lokalt)
```bash
uv venv && source .venv/bin/activate  # hvis du bruger uv/venv, ellers brug pip/venv
pip install -r requirements.txt
uvicorn main:app --reload
# Åbn http://127.0.0.1:8000/ for dashboard + /health til ping
```

## Endpoints & UI
- `GET  /` → dashboard med Skeleton CSS + fetch mod /forecast (vol.2 funktionalitet).
- `POST /forecast` → beregner omsætning + anbefalet bemanding (regression fallback).
- `GET  /planday/{date}` → henter planlagte timer (kræver Planday config).
- `POST /reconcile` → skriver forecasts/afstemning til Google Sheet.
- `GET  /logs` → JSON-overblik over de seneste forecast-logposter (Railway Postgres, falder tilbage til lokal fil under udvikling).
- `GET  /chat/config` → flag til ChatKit (viser `ready=true` når `OPENAI_API_KEY` er sat).
- `GET  /health` → simpel status.

## Krav
- Python 3.11+
- En XGBoost model-fil i `models/xgb_v3.pkl` med samme feature-order som i `build_feature_frame`.

## Konfiguration
Udfyld `.env` (se `.env.example`). På Railway sættes env vars i dashboardet.

## TODO før produktion
1. **Model**: Læg din trænede `.pkl` i `models/` og skift `MODEL_BACKEND=xgboost` når klar.
2. **Features**: Finpuds `app/ml/features.py` + `app/services/weather.py` så de matcher din træning.
3. **Planday OAuth2**: Bekræft credentials og scopes; udfyld `PLANDAY_*` i `.env`.
4. **Google Sheets**: Indsæt service account JSON (base64) i `GOOGLE_SHEETS_CREDENTIALS_JSON` og ark-id i `GOOGLE_SHEETS_SPREADSHEET_ID`.
5. **ChatKit**: Når `OPENAI_API_KEY` er sat, kan du montere OpenAI ChatKit-klienten og bruge `/chat/config` til bootstrap.
6. **Cron/Deployment**: Opret Railway Cron-job eller intern scheduler.
7. **Logging DB**: Peg `DATABASE_URL` på en Railway Postgres instans, så forecast-loggen overlever deploys (ellers uses lokal fil).
8. **Tests**: Læg `pytest` og smoke-tests ind (feature-pipeline + endpoints).

## License
MIT
