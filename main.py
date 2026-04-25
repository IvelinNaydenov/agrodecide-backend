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

# ── NDVI Crop Classification ───────────────────────────────

NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(samples) {
  let ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04);
  // Exclude nodata, clouds (SCL 8,9,10), cloud shadows (3), water (6)
  let validMask = samples.dataMask;
  if (samples.SCL == 3 || samples.SCL == 6 || samples.SCL == 8 || samples.SCL == 9 || samples.SCL == 10) {
    validMask = 0;
  }
  if (samples.B08 + samples.B04 == 0) { validMask = 0; }
  return { ndvi: [ndvi], dataMask: [validMask] };
}
"""

# Phenological NDVI profiles for common Bulgarian/EU crops (monthly averages Jan-Dec)
# Based on literature: wheat peaks March-May, sunflower peaks July-Aug, rapeseed peaks April-May, maize peaks July-Aug
CROP_PROFILES = {
    "Пшеница":    [0.15, 0.20, 0.45, 0.70, 0.80, 0.65, 0.25, 0.15, 0.18, 0.20, 0.18, 0.15],
    "Ечемик":     [0.15, 0.22, 0.50, 0.72, 0.75, 0.55, 0.20, 0.15, 0.17, 0.20, 0.18, 0.15],
    "Рапица":     [0.25, 0.30, 0.55, 0.75, 0.70, 0.35, 0.15, 0.12, 0.15, 0.25, 0.30, 0.28],
    "Слънчоглед": [0.10, 0.10, 0.12, 0.18, 0.35, 0.60, 0.78, 0.75, 0.45, 0.15, 0.10, 0.10],
    "Царевица":   [0.10, 0.10, 0.12, 0.15, 0.30, 0.55, 0.80, 0.82, 0.55, 0.20, 0.12, 0.10],
    "Люцерна":    [0.15, 0.20, 0.40, 0.60, 0.65, 0.55, 0.60, 0.55, 0.50, 0.35, 0.20, 0.15],
    "Угар":       [0.08, 0.08, 0.10, 0.12, 0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.08, 0.08],
}


def classify_crop(ndvi_series: list) -> dict:
    """
    Classify crop from NDVI time series by comparing to known phenological profiles.
    ndvi_series: list of {date, mean_ndvi} dicts
    Returns: {crop, confidence, scores}
    """
    if not ndvi_series:
        return {"crop": "Неизвестно", "confidence": 0, "scores": {}}

    # Build monthly averages from the time series
    monthly = {}
    for pt in ndvi_series:
        if pt.get("mean") is not None and pt["mean"] > -0.5:
            month = int(pt["date"].split("-")[1])
            monthly.setdefault(month, []).append(pt["mean"])
    monthly_avg = {m: sum(v) / len(v) for m, v in monthly.items()}

    if len(monthly_avg) < 3:
        return {"crop": "Недостатъчно данни", "confidence": 0, "scores": {}}

    # Compare with each crop profile using correlation-like score
    scores = {}
    for crop, profile in CROP_PROFILES.items():
        diffs = []
        for m, avg in monthly_avg.items():
            ref = profile[m - 1]
            diffs.append((avg - ref) ** 2)
        rmse = (sum(diffs) / len(diffs)) ** 0.5
        # Convert RMSE to similarity score (0-1, higher = better match)
        scores[crop] = max(0, 1 - rmse * 4)

    best = max(scores, key=scores.get)
    conf = scores[best]
    # Normalize confidence to percentage
    total = sum(scores.values()) or 1
    conf_pct = round(conf / total * 100)

    return {
        "crop": best,
        "confidence": conf_pct,
        "scores": {k: round(v / total * 100) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
    }


@app.post("/api/ndvi/timeseries")
async def ndvi_timeseries(body: dict):
    """
    Get NDVI time series from Sentinel Hub Statistical API.
    Body: {lat, lon, bbox_size?: 0.005, days?: 180}
    Returns: {timeseries: [...], classification: {...}}
    """
    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        raise HTTPException(400, "lat and lon required")

    days = body.get("days", 180)
    bbox_half = body.get("bbox_size", 0.005) / 2

    key = f"ndvi_ts:{lat:.4f}:{lon:.4f}:{days}"
    cached = cache_get(key)
    if cached:
        return {**cached, "_cache": "hit"}

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    bbox = [lon - bbox_half, lat - bbox_half, lon + bbox_half, lat + bbox_half]

    stats_request = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": 40,
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": start.strftime("%Y-%m-%dT00:00:00Z"),
                "to": end.strftime("%Y-%m-%dT23:59:59Z"),
            },
            "aggregationInterval": {"of": "P10D"},
            "evalscript": NDVI_EVALSCRIPT,
            "resx": 20,
            "resy": 20,
        },
    }

    token = await get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    stat_url = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

    last_err = None
    for use_proxy in [True, False]:
        try:
            async with make_client(use_proxy, timeout=45) as client:
                r = await client.post(stat_url, headers=headers, json=stats_request)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                    continue
                raw = r.json()
                break
        except Exception as e:
            last_err = str(e)
    else:
        raise HTTPException(502, f"Statistical API error: {last_err}")

    # Parse response into clean time series
    timeseries = []
    for interval in raw.get("data", []):
        ts_from = interval.get("interval", {}).get("from", "")[:10]
        outputs = interval.get("outputs", {})
        ndvi_out = outputs.get("ndvi", {})
        bands = ndvi_out.get("bands", {})
        b0 = bands.get("B0", {})
        stats = b0.get("stats", {})
        sample_count = stats.get("sampleCount", 0)
        no_data = stats.get("noDataCount", 0)

        if sample_count > 0 and sample_count > no_data:
            timeseries.append({
                "date": ts_from,
                "mean": round(stats.get("mean", 0), 4),
                "min": round(stats.get("min", 0), 4),
                "max": round(stats.get("max", 0), 4),
                "stdev": round(stats.get("stDev", 0), 4),
                "samples": sample_count,
            })

    # Classify crop from NDVI profile
    classification = classify_crop(timeseries)

    result = {
        "timeseries": timeseries,
        "classification": classification,
        "bbox": bbox,
        "period": {"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")},
        "profiles": {k: v for k, v in CROP_PROFILES.items()},
    }

    cache_set(key, result, 6 * 3600)
    return {**result, "_cache": "miss"}


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
