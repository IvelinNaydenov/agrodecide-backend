# AgroDecide Backend Proxy

FastAPI backend proxy за AgroDecide платформата.

## Endpoints

| Endpoint | Описание | Cache |
|----------|----------|-------|
| `GET /` | Health check | — |
| `GET /api/token` | Copernicus auth token | 10 мин |
| `GET /api/stac?lat=&lon=&days=` | Sentinel-2 сцени | 6 ч |
| `GET /api/meteo?lat=&lon=` | OpenMeteo 14-дневна прогноза | 15 мин |
| `GET /api/meteo/history?lat=&lon=&days=` | OpenMeteo история | 6 ч |
| `POST /api/ai` | Gemini Flash AI | — |

## Deploy на Render.com

1. Fork/push тoва repo в GitHub
2. render.com → New → Web Service → свържи repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Добави Environment Variables (от .env.example)

## Environment Variables

Задължителни в Render dashboard:
- `COP_CLIENT_ID`
- `COP_CLIENT_SECRET`
- `PROXY_URL`
- `PROXY_USER`
- `PROXY_PASS`
- `GEMINI_KEY`
