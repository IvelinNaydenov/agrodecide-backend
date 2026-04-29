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
        from urllib.parse import quote
        safe_pass = quote(PROXY_PASS or "", safe="")
        proxy = f"http://{PROXY_USER}:{safe_pass}@{PROXY_URL.replace('http://', '')}"
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
async def real_ndvi(lat: float, lon: float, days: int = 90):
    """Real Sentinel-2 NDVI via Copernicus Statistical API."""
    cache_key = f"ndvi_real:{lat:.4f}:{lon:.4f}:{days}"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    delta = 0.003  # ~300m bbox
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

    # Fixed evalscript: NO CLM (not available in CDSE), use dataMask instead
    evalscript = """//VERSION=3
function setup(){return{input:["B04","B08","dataMask"],output:[{id:"ndvi",bands:1,sampleType:"FLOAT32"},{id:"dataMask",bands:1}]};}
function evaluatePixel(s){
  var ndvi=(s.B08-s.B04)/(s.B08+s.B04+0.001);
  return{ndvi:[ndvi],dataMask:[s.dataMask]};
}"""

    body = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
                "maxCloudCoverage": 70
            }}]
        },
        "aggregation": {
            "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
            "aggregationInterval": {"of": "P7D"},
            "evalscript": evalscript,
            "width": 5, "height": 5
        },
        "calculations": {"ndvi": {"statistics": {"default": {"percentiles": {"k": [50]}}}}}
    }

    try:
        token = await get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        last_err = None

        # Use CDSE endpoint, not old services.sentinel-hub.com
        endpoints = [
            "https://sh.dataspace.copernicus.eu/api/v1/statistics",
            "https://services.sentinel-hub.com/api/v1/statistics",
        ]

        for stat_url in endpoints:
            for use_proxy in [True, False]:
                try:
                    async with make_client(use_proxy, timeout=30) as client:
                        r = await client.post(
                            stat_url,
                            headers=headers, json=body
                        )
                        if r.status_code == 200:
                            data = r.json()
                            series = []
                            for interval in data.get("data", []):
                                dt = interval.get("interval", {}).get("from", "")[:10]
                                stats = interval.get("outputs", {}).get("ndvi", {}).get("statistics", {}).get("default", {})
                                mean_val = stats.get("mean")
                                median_val = stats.get("percentiles", {}).get("50.0")
                                sample_count = stats.get("sampleCount", 0)
                                no_data = stats.get("noDataCount", 0)
                                valid_ratio = (sample_count - no_data) / max(sample_count, 1)
                                ndvi_val = median_val if median_val is not None else mean_val
                                if ndvi_val is not None and valid_ratio > 0.3 and 0 < ndvi_val < 1:
                                    series.append({"date": dt, "ndvi": round(ndvi_val, 4), "source": "sentinel-2"})

                            result = {
                                "current_ndvi": series[-1]["ndvi"] if series else None,
                                "series": series,
                                "series_count": len(series),
                                "period": f"{start_date} to {end_date}",
                                "source": "Copernicus Sentinel-2 L2A Statistical API",
                                "real": True,
                                "_cache": "miss"
                            }
                            cache_set(cache_key, result, 6 * 3600)
                            return result
                        else:
                            last_err = f"HTTP {r.status_code}: {r.text[:100]}"
                except Exception as e:
                    last_err = str(e)[:100]

        raise HTTPException(502, f"Sentinel Hub error: {last_err}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))




# ── Price History (Supabase) ───────────────────────────
@app.post("/api/prices/save")
async def save_prices(body: dict):
    """Save daily prices to Supabase price_history table."""
    payload = {
        "date": body.get("date"),
        "wheat_matif": body.get("wheat_matif"),
        "corn_matif": body.get("corn_matif"),
        "rapeseed_matif": body.get("rapeseed_matif"),
        "sunflower_euronext": body.get("sunflower_euronext"),
        "barley_bg": body.get("barley_bg"),
        "wheat_bg": body.get("wheat_bg"),
        "corn_bg": body.get("corn_bg"),
        "sunflower_bg": body.get("sunflower_bg"),
        "source": body.get("source", "agroportal"),
    }
    async with make_client(False) as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/price_history",
            headers={**supa_headers(), "Prefer": "return=minimal,resolution=merge-duplicates"},
            json=payload
        )
        return {"saved": r.status_code in (200, 201, 409)}


@app.get("/api/prices/history")
async def get_price_history(days: int = 30):
    """Get price history from Supabase."""
    cache_key = f"price_hist:{days}"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    async with make_client(False) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/price_history?date=gte.{since}&order=date.asc",
            headers=supa_headers()
        )
        r.raise_for_status()
        data = r.json()

    result = {"history": data, "days": days, "_cache": "miss"}
    cache_set(cache_key, result, 10 * 60)
    return result


# ── Market Prices (scrape agroportal.bg + borsaagro.com) ──
@app.get("/api/market/prices")
async def market_prices():
    """
    Scrape real commodity prices from agroportal.bg and borsaagro.com.
    Returns MATIF futures + BG physical prices.
    Also saves to Supabase price_history for charting.
    Cached 15 minutes.
    """
    cache_key = "market:prices:v3"
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "_cache": "hit"}

    import re as _re

    matif_prices = {}
    bg_prices = {}
    source_used = "none"

    # --- Try agroportal.bg first ---
    try:
        async with make_client(True, timeout=15) as client:
            r = await client.get(
                "https://agroportal.bg/",
                headers={"User-Agent": "Mozilla/5.0 (compatible; AgroDecide/1.0)",
                         "Accept-Language": "bg-BG,bg;q=0.9"}
            )
            if r.status_code == 200:
                html = r.text
                source_used = "agroportal.bg"

                # Parse the scrolling ticker bar at top
                # Format: "Царевица MATIF ... 222.75 €/мт"
                ticker_crops = {
                    "wheat_matif": ["Пшеница MATIF", "Пшеница МАТИФ"],
                    "corn_matif": ["Царевица MATIF", "Царевица МАТИФ"],
                    "rapeseed_matif": ["Рапица MATIF", "Рапица МАТИФ"],
                    "sunflower_euronext": ["Слънчоглед - 44-9-2", "Слънчоглед 44-9-2"],
                }
                for key, names in ticker_crops.items():
                    for name in names:
                        idx = html.find(name)
                        if idx != -1:
                            snippet = html[idx:idx+200]
                            m = _re.search(r'([\d,.]+)\s*[€$]', snippet)
                            if m:
                                val = m.group(1).replace(",", "").replace(" ", "")
                                try:
                                    matif_prices[key] = float(val)
                                except ValueError:
                                    pass
                            break

                # Parse BG physical prices
                bg_crops = {
                    "wheat_bg": ["Хлебна пшеница", "ХЛЕБНА ПШЕНИЦА", "Пшеница хлебна"],
                    "corn_bg": ["Царевица", "ЦАРЕВИЦА"],
                    "sunflower_bg": ["Слънчоглед маслодаен", "СЛЪНЧОГЛЕД МАСЛОДАЕН", "Слънчоглед - маслодаен"],
                    "barley_bg": ["Ечемик фуражен", "ЕЧЕМИК ФУРАЖЕН", "Ечемик"],
                    "rapeseed_bg": ["Рапица", "РАПИЦА"],
                }
                for key, names in bg_crops.items():
                    for name in names:
                        # Look in the physical market section
                        idx = html.find(name)
                        if idx != -1:
                            snippet = html[idx:idx+200]
                            m = _re.search(r'([\d,.]+)\s*[€лв]', snippet)
                            if m:
                                val = m.group(1).replace(",", "").replace(" ", "")
                                try:
                                    bg_prices[key] = float(val)
                                except ValueError:
                                    pass
                            break
    except Exception as e:
        pass

    # --- Fallback: borsaagro.com ---
    if not matif_prices:
        try:
            async with make_client(True, timeout=15) as client:
                r = await client.get(
                    "https://borsaagro.com/",
                    headers={"User-Agent": "Mozilla/5.0 (compatible; AgroDecide/1.0)"}
                )
                if r.status_code == 200:
                    html = r.text
                    source_used = "borsaagro.com"

                    for key, names in {
                        "wheat_matif": ["Пшеница MATIF"],
                        "corn_matif": ["Царевица MATIF"],
                        "rapeseed_matif": ["Рапица MATIF"],
                        "sunflower_euronext": ["Слънчоглед - 44-9-2"],
                    }.items():
                        for name in names:
                            idx = html.find(name)
                            if idx != -1:
                                snippet = html[idx:idx+300]
                                m = _re.search(r'([\d,.]+)\s*€', snippet)
                                if m:
                                    val = m.group(1).replace(",", "")
                                    try:
                                        matif_prices[key] = float(val)
                                    except ValueError:
                                        pass

                    # BG physical from borsaagro
                    for key, names in {
                        "wheat_bg": ["ПШЕНИЦА ФУРАЖНА", "ПШЕНИЦА ХЛЕБНА"],
                        "corn_bg": ["ЦАРЕВИЦА"],
                        "sunflower_bg": ["СЛЪНЧОГЛЕД МАСЛОДАЕН"],
                        "barley_bg": ["ЕЧЕМИК ФУРАЖЕН"],
                    }.items():
                        for name in names:
                            idx = html.find(name)
                            if idx != -1:
                                snippet = html[idx:idx+200]
                                m = _re.search(r'([\d,.]+)\s*€', snippet)
                                if m:
                                    val = m.group(1).replace(",", "")
                                    try:
                                        bg_prices[key] = float(val)
                                    except ValueError:
                                        pass
        except Exception:
            pass

    # --- Build response ---
    names_map = {
        "wheat":     "Пшеница мелница (МАТИФ)",
        "corn":      "Царевица (МАТИФ)",
        "rapeseed":  "Рапица (МАТИФ)",
        "sunflower": "Слънчоглед (Euronext 44-9-2)",
        "barley":    "Ечемик (БГ физически)",
    }

    # Get history from Supabase for change calculation
    history_data = []
    try:
        since = (datetime.utcnow() - timedelta(days=35)).strftime("%Y-%m-%d")
        async with make_client(False) as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/price_history?date=gte.{since}&order=date.asc",
                headers=supa_headers()
            )
            if r.status_code == 200:
                history_data = r.json()
    except Exception:
        pass

    def get_history_for(field):
        return [{"date": row["date"], "price": row.get(field)} for row in history_data if row.get(field)]

    def make_entry(key, name, matif_key, bg_key):
        price = matif_prices.get(matif_key)
        bg = bg_prices.get(bg_key)
        if price is None and bg is not None:
            price = bg
        if price is None:
            return {"name": name, "error": "Няма данни"}
        hist = get_history_for(matif_key)
        prev = hist[-2]["price"] if len(hist) >= 2 else price
        chg = round((price - prev) / prev * 100, 2) if prev else 0
        return {
            "name": name, "price": round(price, 2),
            "prev_close": round(prev, 2), "change_pct": chg,
            "currency": "EUR", "unit": "EUR/t",
            "bg_price": round(bg, 2) if bg else None,
            "history": hist,
            "source": source_used,
        }

    prices = {
        "wheat":     make_entry("wheat", names_map["wheat"], "wheat_matif", "wheat_bg"),
        "corn":      make_entry("corn", names_map["corn"], "corn_matif", "corn_bg"),
        "rapeseed":  make_entry("rapeseed", names_map["rapeseed"], "rapeseed_matif", "rapeseed_bg"),
        "sunflower": make_entry("sunflower", names_map["sunflower"], "sunflower_euronext", "sunflower_bg"),
        "barley":    make_entry("barley", names_map["barley"], "barley_bg", "barley_bg"),
    }

    # Save today prices to Supabase
    today = datetime.utcnow().strftime("%Y-%m-%d")
    save_payload = {
        "date": today,
        "wheat_matif": matif_prices.get("wheat_matif"),
        "corn_matif": matif_prices.get("corn_matif"),
        "rapeseed_matif": matif_prices.get("rapeseed_matif"),
        "sunflower_euronext": matif_prices.get("sunflower_euronext"),
        "barley_bg": bg_prices.get("barley_bg"),
        "wheat_bg": bg_prices.get("wheat_bg"),
        "corn_bg": bg_prices.get("corn_bg"),
        "sunflower_bg": bg_prices.get("sunflower_bg"),
        "source": source_used,
    }
    try:
        async with make_client(False) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/price_history",
                headers={**supa_headers(), "Prefer": "return=minimal,resolution=merge-duplicates"},
                json=save_payload
            )
    except Exception:
        pass

    result = {
        "prices": prices,
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "source": source_used,
        "note": "Live prices from " + source_used,
        "_cache": "miss",
    }
    cache_set(cache_key, result, 15 * 60)
    return result

GEMINI_KEY = os.getenv("GEMINI_KEY", "")

@app.post("/api/gemini")
async def gemini_proxy(body: dict):
    """Proxy for Gemini API — keeps API key server-side."""
    if not GEMINI_KEY:
        raise HTTPException(503, "Gemini API key not configured")
    
    try:
        async with make_client(False, timeout=25) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}",
                headers={"Content-Type": "application/json"},
                json=body
            )
            if r.status_code == 200:
                data = r.json()
                text = ""
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = parts[0].get("text", "") if parts else ""
                return {"text": text, "model": "gemini-2.5-flash"}
            else:
                raise HTTPException(r.status_code, f"Gemini error: {r.text[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Gemini proxy error: {e}")
