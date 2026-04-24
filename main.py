from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import time
from datetime import datetime, timedelta

app = FastAPI(title="AgroDecide Backend Proxy", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ─────────────────────────────────────────────────
COP_CLIENT_ID     = os.getenv("COP_CLIENT_ID")
COP_CLIENT_SECRET = os.getenv("COP_CLIENT_SECRET")
PROXY_URL         = os.getenv("PROXY_URL")
PROXY_USER        = os.getenv("PROXY_USER")
PROXY_PASS        = os.getenv("PROXY_PASS")
GROQ_KEY          = os.getenv("GROQ_KEY")

# ── Cache ──────────────────────────────────────────────────
_cache = {}

def cache_get(key):
    e = _cache.get(key)
    if e and time.time() < e["exp"]:
        return e["data"]
    return None

def cache_set(key, data, ttl):
    _cache[key] = {"data": data, "exp": time.time() + ttl}

# ── HTTP clients ───────────────────────────────────────────
def make_client(use_proxy=True, timeout=30):
    if use_proxy and PROXY_URL and PROXY_USER:
        proxy = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_URL.replace('http://', '')}"
        return httpx.AsyncClient(proxy=proxy, timeout=timeout, verify=False)
    return httpx.AsyncClient(timeout=timeout)

# ── Copernicus token ───────────────────────────────────────
async def get_token():
    cached = cache_get("cop_token")
    if cached:
        return cached

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": COP_CLIENT_ID,
        "client_secret": COP_CLIENT_SECRET,
    }
    last_err = None
    for use_proxy in [True, False]:
        try:
            async with make_client(use_proxy) as client:
                r = await client.post(url, data=data)
                r.raise_for_status()
                td = r.json()
                token = td["access_token"]
                cache_set("cop_token", token, td.get("expires_in", 600) - 30)
                return token
        except Exception as e:
            last_err = e
    raise HTTPException(502, f"Copernicus auth failed: {last_err}")

# ── Routes ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "AgroDecide Proxy", "version": "1.1.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "cache_keys": len(_cache)}

@app.get("/api/token")
async def copernicus_token():
    token = await get_token()
    return {"access_token": token}

@app.get("/api/stac")
async def stac_search(lat: float, lon: float, days: int = 60, limit: int = 10):
    key = f"stac:{lat:.3f}:{lon:.3f}:{days}"
    cached = cache_get(key)
    if cached:
        return {**cached, "_cache": "hit"}

    end   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bbox  = f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}"

    # Without sortby — more compatible across STAC versions
    url = (
        f"https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items"
        f"?bbox={bbox}&datetime={start}/{end}&limit={limit}"
    )

    token = await get_token()
    headers = {"Authorization": f"Bearer {token}"}

    last_err = None
    for use_proxy in [True, False]:
        try:
            async with make_client(use_proxy) as client:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
                data = r.json()
                cache_set(key, data, 6 * 3600)
                return {**data, "_cache": "miss"}
        except Exception as e:
            last_err = e
    raise HTTPException(502, f"STAC error: {last_err}")

@app.get("/api/meteo")
async def meteo(lat: float, lon: float):
    key = f"meteo:{lat:.3f}:{lon:.3f}"
    cached = cache_get(key)
    if cached:
        return {**cached, "_cache": "hit"}

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,precipitation"
        f"&hourly=et0_fao_evapotranspiration"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration"
        f"&forecast_days=14&timezone=Europe%2FSofia"
    )
    async with make_client(False) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    cache_set(key, data, 15 * 60)
    return {**data, "_cache": "miss"}

@app.get("/api/meteo/history")
async def meteo_history(lat: float, lon: float, days: int = 90):
    key = f"meteo_hist:{lat:.3f}:{lon:.3f}:{days}"
    cached = cache_get(key)
    if cached:
        return {**cached, "_cache": "hit"}

    end   = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max,et0_fao_evapotranspiration"
        f"&start_date={start}&end_date={end}&timezone=Europe%2FSofia"
    )
    async with make_client(False) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    cache_set(key, data, 6 * 3600)
    return {**data, "_cache": "miss"}

@app.post("/api/ai")
async def ai_chat(body: dict):
    """Groq LLaMA — fast, free tier"""
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_KEY not configured")

    # Convert Gemini-style contents to OpenAI-style messages
    messages = []
    system_text = body.get("system", "You are a helpful assistant.")
    messages.append({"role": "system", "content": system_text})

    for item in body.get("contents", []):
        role = item.get("role", "user")
        # Gemini uses "model", OpenAI uses "assistant"
        if role == "model":
            role = "assistant"
        parts = item.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if "text" in p)
        if text:
            messages.append({"role": role, "content": text})

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7,
    }

    async with make_client(False) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if r.status_code != 200:
            detail = r.text[:300]
            raise HTTPException(r.status_code, f"Groq error: {detail}")
        data = r.json()

    text = data["choices"][0]["message"]["content"]
    return {"text": text, "model": data.get("model"), "provider": "groq"}
