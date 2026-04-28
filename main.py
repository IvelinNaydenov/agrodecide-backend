from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import time
from datetime import datetime, timedelta

app = FastAPI(title="AgroDecide Backend Proxy", version="1.2.0")

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
    import asyncio as _aio
    async with make_client(False) as client:
        for _attempt in range(3):
            r = await client.get(url)
            if r.status_code == 429:
                await _aio.sleep(2 ** _attempt)
                continue
            r.raise_for_status()
            break
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
    # Use archive API for historical data (forecast only works for future/recent)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
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

@app.get("/api/wms")
async def wms_proxy(request: Request):
    """
    Proxy Sentinel Hub WMS requests server-side with auth token.
    Frontend calls /api/wms?LAYERS=TRUE-COLOR&BBOX=...&WIDTH=...&HEIGHT=...
    Backend adds Bearer token and forwards to Sentinel Hub.
    Tiles cached 6h (Sentinel updates max twice daily).
    """
    WMS_INSTANCE = os.getenv("WMS_INSTANCE", "69051eb2-80ae-466a-9501-850209a883db")
    WMS_URL = f"https://sh.dataspace.copernicus.eu/ogc/wms/{WMS_INSTANCE}"

    # Forward all query params from frontend
    params = dict(request.query_params)
    # Normalize to uppercase — Sentinel Hub requires it
    params = {k.upper(): v for k, v in params.items()}
    params.setdefault("SERVICE", "WMS")
    params.setdefault("REQUEST", "GetMap")
    params.setdefault("VERSION", "1.3.0")
    params.setdefault("FORMAT", "image/jpeg")
    params.setdefault("WIDTH", "512")
    params.setdefault("HEIGHT", "512")
    params.setdefault("STYLES", "")
    # Sentinel Hub uses CRS (keep as-is)
    params.setdefault("CRS", "EPSG:3857")
    # Remove SRSNAME if present — Sentinel Hub doesn't use it
    params.pop("SRSNAME", None)

    # Cache key from the full param set
    cache_key = "wms:" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    cached = cache_get(cache_key)
    if cached:
        fmt = params.get("FORMAT", "image/jpeg")
        return Response(content=cached, media_type=fmt,
                        headers={"X-Cache": "HIT"})

    token = await get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/jpeg,image/png,*/*",
    }

    last_err = None
    for use_proxy in [True, False]:
        try:
            async with make_client(use_proxy, timeout=20) as client:
                r = await client.get(WMS_URL, params=params, headers=headers)
                r.raise_for_status()
                img_bytes = r.content
                fmt = r.headers.get("content-type", "image/jpeg")
                cache_set(cache_key, img_bytes, 6 * 3600)
                return Response(content=img_bytes, media_type=fmt,
                                headers={"X-Cache": "MISS"})
        except Exception as e:
            last_err = e

    raise HTTPException(502, f"WMS proxy error: {last_err}")

@app.get("/api/ndvi/classify")
async def classify_crop(lat: float, lon: float, days: int = 180):
    """
    Classify crop type from NDVI time series.
    Fetches OpenMeteo ET0+precipitation history and derives NDVI proxy series,
    then uses Groq AI to classify the most likely crop based on the curve shape.
    """
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_KEY not configured")

    cache_key = f"classify:{lat:.3f}:{lon:.3f}:{days}"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    # Fetch meteo history for NDVI proxy
    end   = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    # Use archive API for historical data (forecast only works for future/recent)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max,et0_fao_evapotranspiration"
        f"&start_date={start}&end_date={end}&timezone=Europe%2FSofia"
    )

    async with make_client(False) as client:
        r = await client.get(url)
        r.raise_for_status()
        hist = r.json()

    dates = hist["daily"]["time"]
    et0   = hist["daily"]["et0_fao_evapotranspiration"]
    rain  = hist["daily"]["precipitation_sum"]
    tmax  = hist["daily"]["temperature_2m_max"]

    # Derive NDVI proxy series (weekly averages for cleaner signal)
    ndvi_series = []
    for i in range(0, len(dates), 7):
        chunk_et0  = et0[i:i+7]
        chunk_rain = rain[i:i+7]
        chunk_tmax = tmax[i:i+7]
        avg_et0  = sum(v or 0 for v in chunk_et0) / max(len(chunk_et0), 1)
        sum_rain = sum(v or 0 for v in chunk_rain)
        avg_tmax = sum(v or 0 for v in chunk_tmax) / max(len(chunk_tmax), 1)
        wb = (sum_rain - avg_et0 * 7) / 30
        hs = -((avg_tmax - 33) * 0.01) if avg_tmax > 33 else 0
        ndvi = min(0.95, max(0.05, 0.55 + min(0.18, max(-0.22, wb)) + hs))
        ndvi_series.append({
            "date": dates[i],
            "ndvi": round(ndvi, 3),
            "rain_mm": round(sum_rain, 1),
            "tmax": round(avg_tmax, 1)
        })

    # Build prompt for Groq
    series_str = ", ".join(f"{s['date'][:7]}:{s['ndvi']}" for s in ndvi_series)
    months = [s['date'][5:7] for s in ndvi_series]

    system = """You are an expert agronomist AI specializing in crop classification from NDVI time series data.

NDVI crop signatures (Northern Bulgaria / SE Europe):
- Wheat/Barley: high NDVI (0.6-0.8) March-May, sharp drop June (harvest), low summer
- Sunflower: low NDVI until May, peak July-August (0.5-0.75), drops September  
- Rapeseed: early peak February-March (flowering), drops April-May, very low summer
- Maize/Corn: starts May, peak August-September (0.7-0.85), drops October
- Sugar beet: steady growth April-September, long season
- Lucerne: multiple peaks (cut 3-4 times), never very low in summer
- Fallow/bare: consistently low NDVI (0.1-0.3) throughout

Respond ONLY in JSON format: {"crop": "name_in_Bulgarian", "confidence": 0-100, "reasoning": "brief explanation in Bulgarian", "rotation_hint": "what was likely here last year"}"""

    prompt = f"""Класифицирай културата по NDVI времева серия за координати {lat:.3f}°N {lon:.3f}°E:

Седмични NDVI стойности (дата:ndvi): {series_str}

Текущ месец: {datetime.utcnow().strftime('%B %Y')}
Регион: Добрич, България (умерено-континентален климат)

Определи: каква култура е най-вероятно засята тази година?"""

    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": prompt}]

    async with make_client(False) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "max_tokens": 300, "temperature": 0.3, "response_format": {"type": "json_object"}},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()

    ai_text = data["choices"][0]["message"]["content"]
    try:
        import json
        ai_result = json.loads(ai_text)
    except Exception:
        ai_result = {"crop": "Неизвестна", "confidence": 0, "reasoning": ai_text, "rotation_hint": "—"}

    result = {
        "lat": lat, "lon": lon,
        "ndvi_series": ndvi_series,
        "classification": ai_result,
        "series_summary": f"{len(ndvi_series)} седмици · {dates[0][:7]} → {dates[-1][:7]}",
        "_cache": "miss"
    }

    cache_set(cache_key, result, 6 * 3600)
    return result

# ── Supabase integration ───────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jmqwasmthyxdppbcpwis.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImptcXdhc210aHl4ZHBwYmNwd2lzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzcxMTkyNTAsImV4cCI6MjA5MjY5NTI1MH0.1NTo3BfnlflQ6f2uo1InNxug9hl1MoPcMsQ_5QLEFDk")

def supa_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

@app.post("/api/parcels")
async def save_parcel(body: dict):
    """Save parcel to Supabase"""
    import json as _json
    coords = body.get("coords", [])
    payload = {
        "name": body.get("name"),
        "crop": body.get("crop"),
        "area_ha": body.get("area"),
        "source": body.get("source", "drawn"),
        "lpis_id": body.get("lpis_id"),
        "coords_json": _json.dumps(coords) if coords else None,
    }
    # Convert coords to WKT polygon for PostGIS
    if coords:
        pts = ", ".join(f"{c[1]} {c[0]}" for c in coords)
        first = f"{coords[0][1]} {coords[0][0]}"
        payload["geom"] = f"SRID=4326;POLYGON(({pts}, {first}))"

    async with make_client(False) as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/parcels",
            headers=supa_headers(),
            json=payload
        )
        if r.status_code not in (200, 201):
            raise HTTPException(r.status_code, f"Supabase error: {r.text}")
        return r.json()

@app.get("/api/parcels")
async def get_parcels():
    """Get all parcels from Supabase"""
    async with make_client(False) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/parcels?select=*&order=created_at.desc",
            headers=supa_headers()
        )
        r.raise_for_status()
        return r.json()

@app.delete("/api/parcels/{parcel_id}")
async def delete_parcel(parcel_id: str):
    """Delete parcel from Supabase"""
    async with make_client(False) as client:
        r = await client.delete(
            f"{SUPABASE_URL}/rest/v1/parcels?id=eq.{parcel_id}",
            headers=supa_headers()
        )
        r.raise_for_status()
        return {"deleted": parcel_id}

@app.post("/api/parcels/{parcel_id}/eucrops")
async def save_eucrops(parcel_id: str, body: dict):
    """Save EuroCrops history for a parcel"""
    rows = [
        {"parcel_id": parcel_id, "year": h["y"], "crop": h["c"],
         "source": "eucrops_v11", "country_code": "BG"}
        for h in body.get("history", [])
    ]
    if not rows:
        return {"saved": 0}
    async with make_client(False) as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/eucrops_history",
            headers={**supa_headers(), "Prefer": "return=minimal"},
            json=rows
        )
        r.raise_for_status()
        return {"saved": len(rows)}

@app.get("/api/parcels/{parcel_id}/eucrops")
async def get_eucrops(parcel_id: str):
    """Get EuroCrops history for a parcel"""
    async with make_client(False) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/eucrops_history?parcel_id=eq.{parcel_id}&order=year.asc",
            headers=supa_headers()
        )
        r.raise_for_status()
        return r.json()

@app.get("/api/crop-history")
async def crop_history(lat: float, lon: float, radius_km: float = 10):
    """
    Get crop history for a location from eucrops_reference table.
    Uses spatial proximity — finds points within radius_km kilometers.
    Sources: LUCAS (Bulgaria), EuroCrops (other EU countries).
    """
    cache_key = f"crophistory:{lat:.3f}:{lon:.3f}:{radius_km}"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    # Query Supabase — filter by lat/lon bounding box (fast, no PostGIS needed)
    # 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
    import math
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

    url = (
        f"{SUPABASE_URL}/rest/v1/eucrops_reference"
        f"?lat=gte.{lat - lat_delta}&lat=lte.{lat + lat_delta}"
        f"&lon=gte.{lon - lon_delta}&lon=lte.{lon + lon_delta}"
        f"&select=crop_name,crop_code,year,country,nuts3,data_source"
        f"&order=year.asc"
    )

    async with make_client(False) as client:
        r = await client.get(url, headers=supa_headers())
        r.raise_for_status()
        points = r.json()

    if not points:
        return {"found": False, "history": [], "source": "none",
                "message": "Няма данни за този район. Използва се AI предикция."}

    # Aggregate by year — most common crop per year
    from collections import Counter
    by_year = {}
    for p in points:
        y = str(p.get("year", ""))
        c = p.get("crop_name", "Unknown")
        if y not in by_year:
            by_year[y] = []
        by_year[y].append(c)

    history = []
    for year in sorted(by_year.keys()):
        most_common = Counter(by_year[year]).most_common(1)[0][0]
        history.append({"year": year, "crop": most_common})

    # Determine source
    sources = list(set(p.get("data_source", "") for p in points))
    source_label = "LUCAS 2022 (Eurostat)" if "lucas_2022" in sources else "EuroCrops v11"

    result = {
        "found": True,
        "history": history,
        "points_found": len(points),
        "source": source_label,
        "sources": sources,
        "_cache": "miss"
    }
    cache_set(cache_key, result, 24 * 3600)  # Cache 24h — historical data doesn't change
    return result

@app.get("/api/eurostat/agriculture")
async def eurostat_agriculture(country: str = "BG"):
    """
    Fetch agricultural land use data from Eurostat API.
    Dataset: ef_lus_main — Main farm land use by NUTS2 region
    """
    cache_key = f"eurostat:agri:{country}"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    import asyncio as _aio

    # Multiple Eurostat datasets
    datasets = {
        "land_use": f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ef_lus_main?geo={country}&lang=en&format=JSON",
        "crop_production": f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/apro_cpshr?geo={country}&lang=en&format=JSON&unit=THA",
        "land_overview": f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lan_use_ovw?geo={country}&lang=en&format=JSON",
    }

    results = {}
    async with make_client(False, timeout=30) as client:
        for key, url in datasets.items():
            for attempt in range(2):
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        results[key] = r.json()
                        break
                    elif r.status_code == 429:
                        await _aio.sleep(2)
                except Exception as e:
                    results[key] = {"error": str(e)}

    result = {"country": country, "datasets": results, "_cache": "miss"}
    cache_set(cache_key, result, 24 * 3600)
    return result

@app.get("/api/ndvi/real")
async def real_ndvi(lat: float, lon: float, days: int = 30):
    """
    Fetch real Sentinel-2 NDVI values via Copernicus Process API (evalscript).
    Returns time series of actual NDVI pixel values for the given coordinates.
    """
    cache_key = f"real_ndvi:{lat:.4f}:{lon:.4f}:{days}"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    # Date range
    end_date   = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Bounding box ~500m around point
    delta = 0.005  # ~500m
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

    # Evalscript — returns NDVI, EVI, cloud coverage
    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B04", "B08", "B8A", "CLM"], units: "REFLECTANCE"}],
    output: [
      {id: "ndvi", bands: 1, sampleType: "FLOAT32"},
      {id: "evi",  bands: 1, sampleType: "FLOAT32"},
      {id: "cloud",bands: 1, sampleType: "FLOAT32"}
    ],
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  let ndvis = [], evis = [], clouds = [];
  for (let s of samples) {
    if (s.CLM > 0.5) { clouds.push(1); continue; }
    clouds.push(0);
    let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 0.0001);
    ndvis.push(ndvi);
    let evi = 2.5 * (s.B08 - s.B04) / (s.B08 + 6*s.B04 - 7.5*0.0002 + 1);
    evis.push(evi);
  }
  let avgNDVI = ndvis.length > 0 ? ndvis.reduce((a,b)=>a+b,0)/ndvis.length : -9999;
  let avgEVI  = evis.length > 0  ? evis.reduce((a,b)=>a+b,0)/evis.length   : -9999;
  let cloudPct= clouds.length > 0 ? clouds.reduce((a,b)=>a+b,0)/clouds.length : 1;
  return {ndvi: [avgNDVI], evi: [avgEVI], cloud: [cloudPct]};
}
"""

    request_body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": 80
                }
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": 1,
            "height": 1,
            "responses": [
                {"identifier": "ndvi",  "format": {"type": "image/tiff"}},
                {"identifier": "evi",   "format": {"type": "image/tiff"}},
                {"identifier": "cloud", "format": {"type": "image/tiff"}}
            ]
        }
    }

    # Use Statistical API instead — returns JSON time series directly
    stat_body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
                    "maxCloudCoverage": 80
                }
            }]
        },
        "aggregation": {
            "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
            "aggregationInterval": {"of": "P7D"},
            "evalscript": """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B04","B08","CLM"], units: "REFLECTANCE"}],
    output: [
      {id:"ndvi", bands:1, sampleType:"FLOAT32"},
      {id:"dataMask", bands:1, sampleType:"UINT8"}
    ]
  };
}
function evaluatePixel(s) {
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 0.0001);
  let valid = s.CLM < 0.5 ? 1 : 0;
  return {ndvi: [ndvi * valid + (-9999) * (1-valid)], dataMask: [valid]};
}
""",
            "resampling": {"downsampling": "BILINEAR", "upsampling": "BILINEAR"},
            "width": 10,
            "height": 10
        },
        "calculations": {
            "ndvi": {
                "histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}},
                "statistics": {"default": {"percentiles": {"k": [25, 50, 75]}, "noDataValues": [-9999]}}
            }
        }
    }

    try:
        token = await get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        last_err = None
        for use_proxy in [True, False]:
            try:
                async with make_client(use_proxy, timeout=30) as client:
                    r = await client.post(
                        "https://services.sentinel-hub.com/api/v1/statistics",
                        headers=headers,
                        json=stat_body
                    )
                    if r.status_code == 200:
                        data = r.json()
                        # Parse the statistical response
                        intervals = data.get("data", [])
                        series = []
                        for interval in intervals:
                            date = interval.get("interval", {}).get("from", "")[:10]
                            outputs = interval.get("outputs", {})
                            ndvi_stats = outputs.get("ndvi", {}).get("statistics", {}).get("default", {})
                            mean = ndvi_stats.get("mean")
                            median = ndvi_stats.get("percentiles", {}).get("50.0")
                            sample_count = ndvi_stats.get("sampleCount", 0)
                            no_data = ndvi_stats.get("noDataCount", 0)
                            # Skip intervals with too many no-data pixels (clouds)
                            if sample_count > 0 and no_data / max(sample_count, 1) < 0.7:
                                ndvi_val = median or mean
                                if ndvi_val is not None and ndvi_val > -1:
                                    series.append({
                                        "date": date,
                                        "ndvi": round(ndvi_val, 4),
                                        "mean": round(mean, 4) if mean else None,
                                        "source": "sentinel-2-l2a"
                                    })

                        if not series:
                            # Fallback to WMS-derived if no clear scenes
                            raise ValueError("No clear Sentinel-2 scenes in period")

                        # Current NDVI = latest valid value
                        current_ndvi = series[-1]["ndvi"] if series else None

                        result = {
                            "lat": lat, "lon": lon,
                            "current_ndvi": current_ndvi,
                            "series": series,
                            "series_count": len(series),
                            "period": f"{start_date} → {end_date}",
                            "source": "Copernicus Process API · Sentinel-2 L2A · Statistical API",
                            "_cache": "miss"
                        }
                        cache_set(cache_key, result, 6 * 3600)
                        return result
                    else:
                        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = str(e)

        raise HTTPException(502, f"Process API error: {last_err}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Real NDVI error: {e}")

@app.get("/api/ndvi/real")
async def real_ndvi(lat: float, lon: float, days: int = 30):
    """Real Sentinel-2 NDVI via Copernicus Statistical API (evalscript)."""
    cache_key = f"real_ndvi:{lat:.4f}:{lon:.4f}:{days}"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    end_date   = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    delta = 0.005
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

    evalscript = """
//VERSION=3
function setup(){return{input:[{bands:["B04","B08","CLM"],units:"REFLECTANCE"}],output:[{id:"ndvi",bands:1,sampleType:"FLOAT32"},{id:"dataMask",bands:1,sampleType:"UINT8"}]};}
function evaluatePixel(s){
  let ndvi=(s.B08-s.B04)/(s.B08+s.B04+0.0001);
  let valid=s.CLM<0.5?1:0;
  return{ndvi:[ndvi*valid+(-9999)*(1-valid)],dataMask:[valid]};
}
"""

    stat_body = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"}, "maxCloudCoverage": 80}}]
        },
        "aggregation": {
            "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
            "aggregationInterval": {"of": "P7D"},
            "evalscript": evalscript,
            "resampling": {"downsampling": "BILINEAR", "upsampling": "BILINEAR"},
            "width": 10, "height": 10
        },
        "calculations": {
            "ndvi": {
                "histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}},
                "statistics": {"default": {"percentiles": {"k": [25, 50, 75]}, "noDataValues": [-9999]}}
            }
        }
    }

    try:
        token = await get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
        last_err = None

        for use_proxy in [True, False]:
            try:
                async with make_client(use_proxy, timeout=30) as client:
                    r = await client.post(
                        "https://services.sentinel-hub.com/api/v1/statistics",
                        headers=headers, json=stat_body
                    )
                    if r.status_code == 200:
                        data = r.json()
                        intervals = data.get("data", [])
                        series = []
                        for interval in intervals:
                            date = interval.get("interval", {}).get("from", "")[:10]
                            outputs = interval.get("outputs", {})
                            ndvi_stats = outputs.get("ndvi", {}).get("statistics", {}).get("default", {})
                            mean = ndvi_stats.get("mean")
                            median_val = ndvi_stats.get("percentiles", {}).get("50.0")
                            sample_count = ndvi_stats.get("sampleCount", 0)
                            no_data = ndvi_stats.get("noDataCount", 0)
                            if sample_count > 0 and (no_data / max(sample_count, 1)) < 0.7:
                                ndvi_val = median_val or mean
                                if ndvi_val is not None and ndvi_val > -1:
                                    series.append({"date": date, "ndvi": round(ndvi_val, 4), "mean": round(mean, 4) if mean else None, "source": "sentinel-2-l2a"})

                        current_ndvi = series[-1]["ndvi"] if series else None
                        result = {
                            "lat": lat, "lon": lon,
                            "current_ndvi": current_ndvi,
                            "series": series,
                            "series_count": len(series),
                            "period": f"{start_date} → {end_date}",
                            "source": "Copernicus Statistical API · Sentinel-2 L2A",
                            "_cache": "miss"
                        }
                        cache_set(cache_key, result, 6 * 3600)
                        return result
                    else:
                        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = str(e)

        raise HTTPException(502, f"Statistical API error: {last_err}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Real NDVI error: {e}")

@app.get("/api/market/prices")
async def market_prices():
    """
    Fetch real MATIF/Euronext commodity prices via Yahoo Finance.
    Returns €/ton prices for major Bulgarian crops.
    Cached 15 minutes.
    """
    cache_key = "market:prices:matif"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    # MATIF/CBOT tickers on Yahoo Finance
    # EBM.PA, ECO.PA, EMA.PA = MATIF Paris (€/t)
    # ZW=F, ZC=F, ZS=F = CBOT Chicago (USD/bushel → convert to EUR/t)
    tickers = {
        "wheat":     {"symbol": "EBM.PA", "name": "Пшеница мелница", "bg_discount": -15, "unit": "€/т", "convert": None},
        "rapeseed":  {"symbol": "ECO.PA", "name": "Рапица",          "bg_discount": -18, "unit": "€/т", "convert": None},
        "corn":      {"symbol": "EMA.PA", "name": "Царевица",        "bg_discount": -12, "unit": "€/т", "convert": None},
        "sunflower": {"symbol": "ZS=F",   "name": "Слънчоглед (соя proxy)", "bg_discount": -25, "unit": "$/bu→€/т", "convert": "soy_to_eur_t"},
        "barley":    {"symbol": "ZW=F",   "name": "Ечемик (пшеница CBOT)", "bg_discount": -30, "unit": "$/bu→€/т", "convert": "wheat_to_eur_t"},
    }

    import asyncio as _aio
    results = {}

    async with make_client(False, timeout=15) as client:
        for key, info in tickers.items():
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{info['symbol']}?interval=1d&range=30d"
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    data = r.json()
                    meta = data["chart"]["result"][0]["meta"]
                    raw_price = meta.get("regularMarketPrice") or meta.get("previousClose")
                    raw_prev  = meta.get("chartPreviousClose") or meta.get("previousClose")
                    currency = meta.get("currency", "EUR")

                    # Convert CBOT USD/bushel → EUR/tonne if needed
                    def convert_price(p, conv_type):
                        if not p: return p
                        if conv_type == "soy_to_eur_t":
                            # Soybeans: 1 bushel = 27.2155 kg → 1 USD/bu * (1/27.2155*1000) * 0.92 EUR/USD
                            return round(p * (1000/27.2155) * 0.92, 2)
                        elif conv_type == "wheat_to_eur_t":
                            # Wheat: 1 bushel = 27.2155 kg
                            return round(p * (1000/27.2155) * 0.92, 2)
                        return p

                    conv = info.get("convert")
                    price = convert_price(raw_price, conv) if conv else raw_price
                    prev  = convert_price(raw_prev,  conv) if conv else raw_prev

                    # Historical closes for trend
                    timestamps = data["chart"]["result"][0].get("timestamp", [])
                    closes     = data["chart"]["result"][0]["indicators"]["quote"][0].get("close", [])
                    history = []
                    for t, cl in zip(timestamps[-20:], closes[-20:]):
                        if cl is not None:
                            converted_cl = convert_price(cl, conv) if conv else cl
                            history.append({"date": datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"), "price": round(converted_cl, 2)})

                    change_pct = ((price - prev) / prev * 100) if prev and prev != 0 else 0

                    results[key] = {
                        "name":        info["name"],
                        "symbol":      info["symbol"],
                        "price":       round(price, 2),
                        "prev_close":  round(prev, 2),
                        "change_pct":  round(change_pct, 2),
                        "currency":    "EUR",
                        "unit":        "€/т",
                        "bg_price":    round(price + info["bg_discount"], 2),
                        "bg_discount": info["bg_discount"],
                        "history":     history,
                        "source":      "MATIF Euronext" if not conv else "CBOT converted to €/т"
                    }
                else:
                    results[key] = {"name": info["name"], "error": f"HTTP {r.status_code}"}
            except Exception as e:
                results[key] = {"name": info["name"], "error": str(e)[:100]}

    result = {
        "prices": results,
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "note": "МАТИФ Euronext фючърси · 15мин забавяне · Цени в €/т",
        "_cache": "miss"
    }
    cache_set(cache_key, result, 15 * 60)
    return result
