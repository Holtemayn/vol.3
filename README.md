# CaféCaster v3 Skeleton

Denne kodebase er et FastAPI-baseret skelet til CaféCaster-tjenesten. Den indeholder:

1. **Forecast & staffing**: Endpoint `/forecast` beregner daglige omsætning/ timer, med støtte for både regression og XGBoost modeller.
2. **Logs & dashboard**: `/logs` eksponerer logposter, og dashboardet visualiserer forecast, loghistorik og agent.
3. **Planday integration**: `/planday/{date}` henter planlagte timer, og loggen afstemmes automatisk mod Planday-omsætning.
4. **Mongo historik**: `/history/similar` og `/history/aggregates` henter daglige aggregater fra MongoDB.
5. **OpenAI agent**: `/agent/chat` streamer svar fra en agent med værktøjerne `get_planday`, `reconcile_accounts`, `find_similar_days` og `lookup_history`.

## Kom i gang

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Konfigurér miljøvariabler via `.env` eller Railway variables. Minimum:

```env
OPENAI_API_KEY=...
PLANDAY_CLIENT_ID=...
PLANDAY_REFRESH_TOKEN=...
MONGO_URI=mongodb://...
```

## Struktur

```
app/
  api/                # FastAPI routers
  agents/             # Agent schemata, tool registry og orchestration
  services/           # Planday, weather, history, forecast logging m.m.
  ui/                 # Jinja2-dashboard med forecasting, log og chat
```

## Kørselsnoter

- Loggerne gemmes i Postgres når `DATABASE_URL` er sat, ellers i `logs/forecast.log`.
- MongoDB bruges til at vise lignende historiske dage og til agentens historikværktøj.
- Planday omsætning omskrives når forecastet gøres om, så loggen altid har nyeste faktiske tal.
