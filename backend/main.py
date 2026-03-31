"""
TerraAlert — FastAPI ML Backend
Serves disaster risk predictions for Wildfires, Floods, Earthquakes

Run with: uvicorn main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import requests
from datetime import datetime, timedelta
import os

app = FastAPI(
    title="TerraAlert ML API",
    description="Multi-hazard disaster risk prediction — Wildfires, Floods, Earthquakes",
    version="0.1.0"
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class WildfireRequest(BaseModel):
    latitude: float
    longitude: float
    month: int                        # 1-12
    temperature_celsius: float
    humidity_percent: float
    wind_speed_kmh: float
    drought_index: Optional[float] = 0.5  # 0-1

class FloodRequest(BaseModel):
    latitude: float
    longitude: float
    rainfall_mm_24h: float
    soil_moisture: Optional[float] = 0.5   # 0-1
    elevation_m: Optional[float] = 100.0
    river_level_m: Optional[float] = 1.0

class EarthquakeRequest(BaseModel):
    latitude: float
    longitude: float
    depth_km: float
    time_since_last_event_hours: Optional[float] = 24.0
    recent_mag_mean: Optional[float] = 3.0

class RiskResponse(BaseModel):
    disaster_type: str
    risk_level: str           # LOW / MEDIUM / HIGH / CRITICAL
    risk_score: float         # 0.0 - 1.0
    confidence: float         # 0.0 - 1.0
    model_used: str
    top_factors: List[str]
    timestamp: str
    location: dict

# ─────────────────────────────────────────────
# PLACEHOLDER PREDICTION LOGIC
# Real models (RF, XGBoost, TFT) will replace
# these rule-based stubs in Phase 2
# ─────────────────────────────────────────────

def classify_risk(score: float) -> str:
    if score < 0.25: return "LOW"
    elif score < 0.5: return "MEDIUM"
    elif score < 0.75: return "HIGH"
    else: return "CRITICAL"

def predict_wildfire_risk(req: WildfireRequest) -> dict:
    """
    STUB — to be replaced with trained XGBoost/LightGBM model
    Current logic: rule-based score from temperature, humidity, wind
    """
    score = 0.0
    factors = []

    if req.temperature_celsius > 35:
        score += 0.3
        factors.append(f"High temperature ({req.temperature_celsius}°C)")
    if req.humidity_percent < 20:
        score += 0.25
        factors.append(f"Low humidity ({req.humidity_percent}%)")
    if req.wind_speed_kmh > 50:
        score += 0.2
        factors.append(f"High wind speed ({req.wind_speed_kmh} km/h)")
    if req.drought_index > 0.7:
        score += 0.15
        factors.append(f"Drought index elevated ({req.drought_index:.2f})")
    if req.month in [6, 7, 8, 9]:  # peak fire season
        score += 0.1
        factors.append("Peak wildfire season (Jun-Sep)")

    score = min(score, 1.0)
    return {
        "score": round(score, 3),
        "factors": factors or ["Conditions within normal range"],
        "model": "Rule-based stub (XGBoost model pending — Phase 2)"
    }

def predict_flood_risk(req: FloodRequest) -> dict:
    """
    STUB — to be replaced with trained Random Forest / ResNet-50 model
    """
    score = 0.0
    factors = []

    if req.rainfall_mm_24h > 100:
        score += 0.4
        factors.append(f"Extreme rainfall ({req.rainfall_mm_24h}mm in 24h)")
    elif req.rainfall_mm_24h > 50:
        score += 0.2
        factors.append(f"Heavy rainfall ({req.rainfall_mm_24h}mm in 24h)")
    if req.soil_moisture and req.soil_moisture > 0.8:
        score += 0.2
        factors.append("Saturated soil moisture")
    if req.elevation_m and req.elevation_m < 20:
        score += 0.2
        factors.append(f"Low elevation flood zone ({req.elevation_m}m)")
    if req.river_level_m and req.river_level_m > 3.0:
        score += 0.2
        factors.append(f"Elevated river level ({req.river_level_m}m)")

    score = min(score, 1.0)
    return {
        "score": round(score, 3),
        "factors": factors or ["Conditions within normal range"],
        "model": "Rule-based stub (RF + ResNet-50 model pending — Phase 2)"
    }

def predict_earthquake_risk(req: EarthquakeRequest) -> dict:
    """
    STUB — to be replaced with trained LSTM / TFT model
    """
    score = 0.0
    factors = []

    if req.depth_km < 10:
        score += 0.3
        factors.append(f"Shallow depth ({req.depth_km}km — higher surface impact)")
    if req.recent_mag_mean and req.recent_mag_mean > 4.0:
        score += 0.3
        factors.append(f"Elevated recent seismic activity (mean mag {req.recent_mag_mean:.1f})")
    if req.time_since_last_event_hours and req.time_since_last_event_hours < 6:
        score += 0.2
        factors.append(f"Recent event {req.time_since_last_event_hours:.1f}h ago (aftershock zone)")

    # High-risk geographic zones (simplified)
    high_risk_zones = [
        (35, -120, 5),   # California
        (37, 144, 5),    # Japan
        (-33, -70, 5),   # Chile
    ]
    for zlat, zlon, radius in high_risk_zones:
        if abs(req.latitude - zlat) < radius and abs(req.longitude - zlon) < radius:
            score += 0.2
            factors.append("Known high-seismicity zone")
            break

    score = min(score, 1.0)
    return {
        "score": round(score, 3),
        "factors": factors or ["Low seismic activity in this region"],
        "model": "Rule-based stub (LSTM vs TFT model pending — Phase 2)"
    }

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "TerraAlert ML API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": ["/predict/wildfire", "/predict/flood", "/predict/earthquake", "/live/earthquakes"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict/wildfire", response_model=RiskResponse)
def predict_wildfire(req: WildfireRequest):
    """Predict wildfire risk given environmental conditions"""
    result = predict_wildfire_risk(req)
    return RiskResponse(
        disaster_type="wildfire",
        risk_level=classify_risk(result["score"]),
        risk_score=result["score"],
        confidence=0.72,  # will be real model confidence in Phase 2
        model_used=result["model"],
        top_factors=result["factors"],
        timestamp=datetime.utcnow().isoformat(),
        location={"lat": req.latitude, "lon": req.longitude}
    )

@app.post("/predict/flood", response_model=RiskResponse)
def predict_flood(req: FloodRequest):
    """Predict flood risk given hydrological conditions"""
    result = predict_flood_risk(req)
    return RiskResponse(
        disaster_type="flood",
        risk_level=classify_risk(result["score"]),
        risk_score=result["score"],
        confidence=0.68,
        model_used=result["model"],
        top_factors=result["factors"],
        timestamp=datetime.utcnow().isoformat(),
        location={"lat": req.latitude, "lon": req.longitude}
    )

@app.post("/predict/earthquake", response_model=RiskResponse)
def predict_earthquake(req: EarthquakeRequest):
    """Predict earthquake aftershock risk"""
    result = predict_earthquake_risk(req)
    return RiskResponse(
        disaster_type="earthquake",
        risk_level=classify_risk(result["score"]),
        risk_score=result["score"],
        confidence=0.61,
        model_used=result["model"],
        top_factors=result["factors"],
        timestamp=datetime.utcnow().isoformat(),
        location={"lat": req.latitude, "lon": req.longitude}
    )

@app.get("/live/earthquakes")
def live_earthquakes(min_magnitude: float = 4.0, hours: int = 24):
    """
    Fetch live earthquake data from USGS API
    Real data — updates automatically
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.isoformat(),
        "endtime": end_time.isoformat(),
        "minmagnitude": min_magnitude,
        "orderby": "magnitude"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        features = data.get("features", [])

        events = []
        for f in features[:50]:  # cap at 50
            props = f["properties"]
            coords = f["geometry"]["coordinates"]
            events.append({
                "magnitude": props.get("mag"),
                "place": props.get("place"),
                "time": datetime.utcfromtimestamp(props["time"] / 1000).isoformat(),
                "latitude": coords[1],
                "longitude": coords[0],
                "depth_km": coords[2],
                "usgs_url": props.get("url")
            })

        return {
            "source": "USGS Earthquake Catalog",
            "last_updated": datetime.utcnow().isoformat(),
            "count": len(events),
            "min_magnitude": min_magnitude,
            "hours_back": hours,
            "events": events
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"USGS API unavailable: {str(e)}")

@app.get("/live/wildfires")
def live_wildfires():
    """
    Placeholder — will connect to NASA FIRMS API
    Requires NASA API key (free at firms.modaps.eosdis.nasa.gov)
    """
    return {
        "source": "NASA FIRMS",
        "status": "pending_api_key",
        "message": "Add NASA_FIRMS_KEY to .env to enable live wildfire data",
        "get_key_at": "https://firms.modaps.eosdis.nasa.gov/api/"
    }
