"""
STEP 12 — MOBILE API FOR IOS APP
================================
Personal-use backend that wraps the final ordinal departure-delay model in a
mobile-friendly JSON API.

The API is designed for a one-flight-at-a-time workflow:
  - 1 AviationStack call for the requested flight
  - 2 NWS calls for origin/destination weather
  - 3 horizon predictions (6h / 3h / 1h)

Run:
  uvicorn 12_mobile_api:app --reload --host 0.0.0.0 --port 8000

Then point the iOS app at:
  http://<your-machine-ip>:8000
"""

from __future__ import annotations

import gc
import os
import shutil
import threading
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch_geometric.data import HeteroData

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
GRAPH_DATA_DIR = BASE_DIR / "graph_data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_KEY", "")
MODEL_URL = os.getenv("MODEL_URL", "").strip()

CHECKPOINT_NAME = os.getenv("CHECKPOINT_NAME", "best_model_dep_12k_ordinal_ep25.pt")
SEVERE_ALERT_THRESHOLD = 0.60
AIRPORT_FEATURE_DIM = 30
FLIGHT_FEATURE_DIM = 19


def load_module(module_name: str, file_path: Path):
    spec = spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_checkpoint_present(ckpt_path: Path) -> None:
    if ckpt_path.exists():
        return
    if not MODEL_URL:
        raise RuntimeError(
            f"Checkpoint not found at {ckpt_path} and MODEL_URL is not set. "
            "Set MODEL_URL to a public download URL for the checkpoint."
        )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".part")
    print(f"[startup] downloading checkpoint from MODEL_URL -> {ckpt_path.name}")

    try:
        with requests.get(MODEL_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        shutil.move(str(tmp_path), str(ckpt_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


dash_mod = load_module("dashboard_step7", BASE_DIR / "07_dashboard.py")

AIRPORTS = [
    "ANC","ATL","BNA","BOS","BWI","CLE","CLT","CMH","DEN","DFW",
    "DTW","EWR","FLL","HOU","IAD","IAH","IND","JFK","LAS","LAX",
    "LGA","MCO","MCI","MIA","MSP","OAK","ORD","PDX","PHL","PHX",
    "PIT","SAN","SEA","SFO","SJC","SLC",
]

AIRPORT_NAMES = {
    "ANC":"Anchorage","ATL":"Atlanta","BNA":"Nashville","BOS":"Boston",
    "BWI":"Baltimore","CLE":"Cleveland","CLT":"Charlotte","CMH":"Columbus",
    "DEN":"Denver","DFW":"Dallas/FW","DTW":"Detroit","EWR":"Newark",
    "FLL":"Ft. Lauderdale","HOU":"Houston Hobby","IAD":"Washington Dulles",
    "IAH":"Houston Intercontinental","IND":"Indianapolis","JFK":"New York JFK",
    "LAS":"Las Vegas","LAX":"Los Angeles","LGA":"New York LaGuardia",
    "MCO":"Orlando","MCI":"Kansas City","MIA":"Miami","MSP":"Minneapolis",
    "OAK":"Oakland","ORD":"Chicago O'Hare","PDX":"Portland","PHL":"Philadelphia",
    "PHX":"Phoenix","PIT":"Pittsburgh","SAN":"San Diego","SEA":"Seattle",
    "SFO":"San Francisco","SJC":"San Jose","SLC":"Salt Lake City",
}

NWS_STATION_MAP = {
    "ANC":"PANC","ATL":"KATL","BNA":"KBNA","BOS":"KBOS","BWI":"KBWI",
    "CLE":"KCLE","CLT":"KCLT","CMH":"KCMH","DEN":"KDEN","DFW":"KDFW",
    "DTW":"KDTW","EWR":"KEWR","FLL":"KFLL","HOU":"KHOU","IAD":"KIAD",
    "IAH":"KIAH","IND":"KIND","JFK":"KJFK","LAS":"KLAS","LAX":"KLAX",
    "LGA":"KLGA","MCO":"KMCO","MCI":"KMCI","MIA":"KMIA","MSP":"KMSP",
    "OAK":"KOAK","ORD":"KORD","PDX":"KPDX","PHL":"KPHL","PHX":"KPHX",
    "PIT":"KPIT","SAN":"KSAN","SEA":"KSEA","SFO":"KSFO","SJC":"KSJC",
    "SLC":"KSLC",
}


def fetch_nws_weather(iata_code: str) -> dict[str, float]:
    icao = NWS_STATION_MAP.get(iata_code, f"K{iata_code}")
    try:
        url = f"https://api.weather.gov/stations/{icao}/observations/latest"
        resp = requests.get(url, headers={"User-Agent": "FlightDelayGNN/1.0"}, timeout=8)
        if resp.status_code == 200:
            props = resp.json()["properties"]
            return {
                "wind_speed_ms": props.get("windSpeed", {}).get("value") or 0.0,
                "visibility_m": props.get("visibility", {}).get("value") or 10000.0,
                "precip_depth_mm": props.get("precipitationLastHour", {}).get("value") or 0.0,
            }
    except Exception:
        pass
    return {"wind_speed_ms": 0.0, "visibility_m": 10000.0, "precip_depth_mm": 0.0}


def build_airport_features_simple(ap_idx_map, weather_map, snap_time):
    n = len(ap_idx_map)
    X = np.zeros((n, 30), dtype=np.float32)
    h = snap_time.hour
    mo = snap_time.month

    for ap, idx in ap_idx_map.items():
        wx = weather_map.get(ap, {})
        X[idx, 14] = float(wx.get("wind_speed_ms", 0.0) or 0.0) / 30.0
        X[idx, 15] = float(wx.get("visibility_m", 10000.0) or 10000.0) / 10000.0
        X[idx, 16] = float(wx.get("precip_depth_mm", 0.0) or 0.0) / 50.0
        X[idx, 26] = np.sin(2 * np.pi * h / 24)
        X[idx, 27] = np.cos(2 * np.pi * h / 24)
        X[idx, 28] = np.sin(2 * np.pi * mo / 12)
        X[idx, 29] = np.cos(2 * np.pi * mo / 12)
    return X


def build_flight_features_single(origin, dest, dep_hour, dow, h2dep, route_stats, dep_delay=0.0):
    X = np.zeros(19, dtype=np.float32)
    max_delay = 300.0
    mask_part = 1.0 / 24
    time_to_dep = min(h2dep / 24.0, 1.0)

    X[1] = np.sin(2 * np.pi * dep_hour / 24)
    X[2] = np.cos(2 * np.pi * dep_hour / 24)
    X[6] = 1.0
    X[8] = np.sin(2 * np.pi * dow / 7)
    X[9] = np.cos(2 * np.pi * dow / 7)
    X[10] = 1.0
    X[14] = time_to_dep

    if time_to_dep < mask_part and dep_delay != 0:
        X[0] = np.clip(dep_delay / max_delay, -1, 1)

    if route_stats is not None:
        lf, lh, lr, gm, gs = route_stats
        key_f = (origin, dest, dep_hour, dow)
        key_h = (origin, dest, dep_hour)
        key_r = (origin, dest)
        if key_f in lf:
            h_avg, h_std = lf[key_f]
        elif key_h in lh:
            h_avg, h_std = lh[key_h]
        elif key_r in lr:
            h_avg, h_std = lr[key_r]
        else:
            h_avg, h_std = gm, gs
        X[17] = np.clip(h_avg / max_delay, -1, 1)
        X[18] = np.clip(h_std / max_delay, 0, 1)

    return X


def tier_label(delay_min: float) -> str:
    if delay_min < 0:
        return "On Time / Early (<0)"
    if delay_min < 15:
        return "Minor (0-15)"
    if delay_min < 60:
        return "Moderate (15-60)"
    if delay_min < 120:
        return "Heavy (60-120)"
    if delay_min < 240:
        return "Severe (120-240)"
    if delay_min < 720:
        return "Extreme (240-720)"
    return "Ultra (720+)"


def tier_code(delay_min: float) -> str:
    if delay_min < 0:
        return "ontime"
    if delay_min < 15:
        return "minor"
    if delay_min < 60:
        return "moderate"
    if delay_min < 120:
        return "heavy"
    if delay_min < 240:
        return "severe"
    if delay_min < 720:
        return "extreme"
    return "ultra"


def color_for_tier(code: str) -> str:
    return {
        "ontime": "#2E8B57",
        "minor": "#5DADE2",
        "moderate": "#F5B041",
        "heavy": "#EB984E",
        "severe": "#E74C3C",
        "extreme": "#C0392B",
        "ultra": "#7D3C98",
    }.get(code, "#5DADE2")


class FlightSearchRequest(BaseModel):
    flight_number: str | None = Field(default=None, description="IATA flight number, e.g. UA328")
    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    departure_date: str = Field(..., description="YYYY-MM-DD")
    departure_time: str = Field(default="14:00", description="HH:MM, local scheduled departure")


class HealthResponse(BaseModel):
    status: str
    checkpoint: str
    aviationstack_enabled: bool
    model_loaded: bool
    model_loading: bool
    model_error: str | None = None


@dataclass
class LoadedAssets:
    model: Any
    device: torch.device
    airports: list[str]
    airport_index: dict[str, int]
    static: dict[str, Any]
    route_stats: Any
    ordinal_thresholds: list[float]
    severe_idx: int


def load_assets() -> LoadedAssets:
    device = torch.device("cpu")
    ckpt_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    ensure_checkpoint_present(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    if ck.get("ordinal_thresholds") is not None:
        dash_mod.ORDINAL_THRESHOLDS = [float(x) for x in ck["ordinal_thresholds"]]
    dash_mod.REGRESSION_TARGET_TRANSFORM = ck.get("regression_target_transform")
    use_tail_uplift = ck.get("use_tail_uplift")
    if use_tail_uplift is None:
        use_tail_uplift = any(k.startswith("fl_tail_uplift.") for k in ck["model_state"].keys())
    dash_mod.TAIL_UPLIFT_THRESHOLDS = [
        float(x) for x in ck.get("tail_uplift_thresholds", dash_mod.TAIL_UPLIFT_THRESHOLDS)
    ]
    tail_uplift_detach_gates = ck.get("tail_uplift_detach_gates", True)
    cls_out_dim = ck.get("cls_out_dim")
    if cls_out_dim is None:
        w = ck["model_state"].get("fl_cls.3.weight")
        cls_out_dim = int(w.shape[0]) if w is not None else 1

    airport_df = pd.read_parquet(GRAPH_DATA_DIR / "airport_index.parquet")
    airports = airport_df["airport"].tolist()
    airport_index = {ap: i for i, ap in enumerate(airports)}

    static = torch.load(GRAPH_DATA_DIR / "static_edges.pt", map_location=device, weights_only=False)

    route_stats = None
    rs_path = GRAPH_DATA_DIR / "route_stats.parquet"
    rg_path = GRAPH_DATA_DIR / "route_stats_global.parquet"
    if rs_path.exists():
        rs = pd.read_parquet(rs_path)
        rg = pd.read_parquet(rg_path)
        lf = {
            (r.ORIGIN, r.DEST, int(r.dep_hour), int(r.DayOfWeek)): (float(r.hist_avg), float(r.hist_std))
            for r in rs.itertuples(index=False)
        }
        lh, lr = {}, {}
        for r in rs.itertuples(index=False):
            lh.setdefault((r.ORIGIN, r.DEST, int(r.dep_hour)), (float(r.hist_avg), float(r.hist_std)))
            lr.setdefault((r.ORIGIN, r.DEST), (float(r.hist_avg), float(r.hist_std)))
        gm = float(rg.iloc[0]["global_mean"])
        gs = float(rg.iloc[0]["global_std"])
        route_stats = (lf, lh, lr, gm, gs)

    ap_in = AIRPORT_FEATURE_DIM
    fl_in = FLIGHT_FEATURE_DIM
    n_ap = ck.get("num_airports", len(airports))
    n_tails = ck.get("num_tails", 12000)

    model = dash_mod.FlightDelayGNN(
        ap_in=ap_in,
        fl_in=fl_in,
        hidden=ck.get("hidden_dim", dash_mod.HIDDEN_DIM),
        heads=dash_mod.NUM_HEADS,
        layers=dash_mod.NUM_GNN_LAYERS,
        gru_h=ck.get("gru_hidden", dash_mod.GRU_HIDDEN_DIM),
        gru_layers=ck.get("gru_num_layers", dash_mod.GRU_NUM_LAYERS),
        mlp_h=dash_mod.MLP_HIDDEN_DIM,
        tail_h=ck.get("tail_hidden", dash_mod.TAIL_HIDDEN_DIM),
        n_airports=n_ap,
        n_tails=n_tails,
        dropout=0.0,
        cls_out_dim=cls_out_dim,
        use_tail_uplift=use_tail_uplift,
        tail_uplift_thresholds=dash_mod.TAIL_UPLIFT_THRESHOLDS,
        tail_uplift_detach_gates=tail_uplift_detach_gates,
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    del ck
    gc.collect()

    ordinal_thresholds = [float(x) for x in dash_mod.ORDINAL_THRESHOLDS]
    severe_idx = ordinal_thresholds.index(120.0) if 120.0 in ordinal_thresholds else max(len(ordinal_thresholds) - 1, 0)
    return LoadedAssets(
        model=model,
        device=device,
        airports=airports,
        airport_index=airport_index,
        static=static,
        route_stats=route_stats,
        ordinal_thresholds=ordinal_thresholds,
        severe_idx=severe_idx,
    )


_ASSETS: LoadedAssets | None = None
_ASSETS_LOCK = threading.Lock()
_MODEL_LOADING = False
_MODEL_ERROR: str | None = None

app = FastAPI(title="Flight Delay Mobile API", version="1.0.0")


def get_assets() -> LoadedAssets:
    global _ASSETS, _MODEL_LOADING, _MODEL_ERROR
    if _ASSETS is not None:
        return _ASSETS
    with _ASSETS_LOCK:
        if _ASSETS is None:
            _MODEL_LOADING = True
            _MODEL_ERROR = None
            try:
                _ASSETS = load_assets()
            except Exception as exc:
                _MODEL_ERROR = str(exc)
                raise
            finally:
                _MODEL_LOADING = False
    return _ASSETS


def fetch_live_flight_details(flight_number: str, flight_date: str, origin: str, destination: str) -> dict[str, Any] | None:
    if not AVIATIONSTACK_KEY or not flight_number:
        return None
    try:
        resp = requests.get(
            "http://api.aviationstack.com/v1/flights",
            params={
                "access_key": AVIATIONSTACK_KEY,
                "flight_iata": flight_number.upper().strip(),
                "flight_date": flight_date,
            },
            timeout=12,
        )
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            return None

        def score(row: dict[str, Any]) -> tuple[int, int]:
            dep = row.get("departure", {}) or {}
            arr = row.get("arrival", {}) or {}
            same_route = int(dep.get("iata", "").upper() == origin and arr.get("iata", "").upper() == destination)
            scheduled = int(bool(dep.get("scheduled")))
            return (same_route, scheduled)

        row = sorted(rows, key=score, reverse=True)[0]
        dep = row.get("departure", {}) or {}
        arr = row.get("arrival", {}) or {}
        airline = row.get("airline", {}) or {}
        aircraft = row.get("aircraft", {}) or {}
        flight = row.get("flight", {}) or {}
        return {
            "flight_number": flight.get("iata") or flight_number.upper().strip(),
            "airline_name": airline.get("name"),
            "airline_iata": airline.get("iata"),
            "flight_status": row.get("flight_status"),
            "tail_number": aircraft.get("registration"),
            "origin": dep.get("iata"),
            "destination": arr.get("iata"),
            "scheduled_departure": dep.get("scheduled"),
            "scheduled_arrival": arr.get("scheduled"),
            "departure_terminal": dep.get("terminal"),
            "departure_gate": dep.get("gate"),
            "arrival_terminal": arr.get("terminal"),
            "arrival_gate": arr.get("gate"),
            "live_departure_delay_min": float(dep.get("delay") or 0.0),
            "live_arrival_delay_min": float(arr.get("delay") or 0.0),
        }
    except Exception:
        return None


def validate_airport(code: str) -> str:
    code = (code or "").upper().strip()
    if code not in AIRPORTS:
        raise HTTPException(status_code=400, detail=f"Unsupported airport: {code}")
    return code


def parse_departure_datetime(date_str: str, time_str: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(f"{date_str} {time_str}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid departure date/time: {exc}") from exc


def build_single_snapshot(
    origin: str,
    destination: str,
    dep_dt: pd.Timestamp,
    hours_before: int,
    dep_delay: float,
    weather_map: dict[str, dict[str, float]],
) -> HeteroData:
    assets = get_assets()
    snap_time = dep_dt - pd.Timedelta(hours=hours_before)
    h2dep = float(hours_before)
    dow = dep_dt.dayofweek
    dep_hour = dep_dt.hour

    X_ap = build_airport_features_simple(assets.airport_index, weather_map, snap_time)
    X_ap_t = torch.tensor(X_ap, dtype=torch.float16)

    X_fl = build_flight_features_single(
        origin,
        destination,
        dep_hour,
        dow,
        h2dep,
        assets.route_stats,
        dep_delay if hours_before == 1 else 0.0,
    )
    X_fl_t = torch.tensor(X_fl, dtype=torch.float16).unsqueeze(0)
    X_fl_t = dash_mod.apply_masking(X_fl_t)

    cg_ei = assets.static["congestion_ei"].to(assets.device)
    cg_ea = assets.static.get("congestion_ea")
    cg_ea = cg_ea.to(assets.device) if cg_ea is not None else torch.zeros((0, 1), dtype=torch.float, device=assets.device)
    nw_ei = assets.static["network_ei"].to(assets.device)
    nw_ea = assets.static["network_ea"].to(assets.device)

    origin_idx = assets.airport_index.get(origin, 0)
    dest_idx = assets.airport_index.get(destination, 0)

    snap = HeteroData()
    snap["airport"].x = X_ap_t
    snap["airport"].num_nodes = len(assets.airports)
    snap["airport"].y = torch.zeros(len(assets.airports))
    snap["airport"].y_mask = torch.zeros(len(assets.airports), dtype=torch.bool)

    snap["flight"].x = X_fl_t
    snap["flight"].num_nodes = 1
    snap["flight"].y = torch.zeros(1)
    snap["flight"].tail_id = torch.tensor([0], dtype=torch.long)
    for attr in ["y_mask_0h", "y_mask_1h", "y_mask_3h", "y_mask_6h"]:
        setattr(snap["flight"], attr, torch.zeros(1, dtype=torch.bool))

    snap["airport", "rotation", "airport"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    snap["airport", "rotation", "airport"].edge_attr = torch.zeros((0, 3))
    snap["airport", "congestion", "airport"].edge_index = cg_ei
    snap["airport", "congestion", "airport"].edge_attr = cg_ea
    snap["airport", "network", "airport"].edge_index = nw_ei
    snap["airport", "network", "airport"].edge_attr = nw_ea
    snap["airport", "departs_to", "flight"].edge_index = torch.tensor([[origin_idx], [0]], dtype=torch.long)
    snap["airport", "departs_to", "flight"].edge_attr = torch.zeros((1, 1))
    snap["airport", "arrives_from", "flight"].edge_index = torch.tensor([[dest_idx], [0]], dtype=torch.long)
    snap["airport", "arrives_from", "flight"].edge_attr = torch.zeros((1, 1))
    snap["flight", "rotation", "flight"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    snap["flight", "rotation", "flight"].edge_attr = torch.zeros((0, 4))
    snap["flight", "departs_from", "airport"].edge_index = torch.tensor([[0], [origin_idx]], dtype=torch.long)
    snap["flight", "departs_from", "airport"].edge_attr = torch.zeros((1, 1))
    snap["flight", "arrives_at", "airport"].edge_index = torch.tensor([[0], [dest_idx]], dtype=torch.long)
    snap["flight", "arrives_at", "airport"].edge_attr = torch.zeros((1, 1))

    return snap.to(assets.device)


def run_horizon_prediction(
    origin: str,
    destination: str,
    dep_dt: pd.Timestamp,
    hours_before: int,
    dep_delay: float,
    weather_map: dict[str, dict[str, float]],
) -> dict[str, Any]:
    assets = get_assets()
    snap = build_single_snapshot(origin, destination, dep_dt, hours_before, dep_delay, weather_map)
    ap_h, tail_h = assets.model.init_hidden(assets.device)

    with torch.no_grad():
        fl_pred, fl_logits, _, _ = assets.model(snap, ap_h, tail_h)

    pred_delay = float(fl_pred[0].item()) if len(fl_pred) else 0.0
    if fl_logits.ndim == 1:
        severe_prob = float(torch.sigmoid(fl_logits[0]).item()) if len(fl_logits) else 0.0
        bucket_probs = None
    else:
        raw_probs = torch.sigmoid(fl_logits[0]).cpu().numpy()
        severe_prob = float(raw_probs[assets.severe_idx])
        bucket_probs = dash_mod.ordinal_bucket_probs(raw_probs[: len(assets.ordinal_thresholds)])

    tier = tier_label(pred_delay)
    code = tier_code(pred_delay)
    return {
        "hours_before_departure": hours_before,
        "predicted_delay_min": round(pred_delay, 1),
        "predicted_tier": tier,
        "predicted_tier_code": code,
        "tier_color": color_for_tier(code),
        "severe_prob": round(severe_prob, 3),
        "high_conf_severe_alert": bool(severe_prob >= SEVERE_ALERT_THRESHOLD),
        "regression_severe": bool(pred_delay >= 120.0),
        "consensus_severe_alert": bool(severe_prob >= SEVERE_ALERT_THRESHOLD and pred_delay >= 120.0),
        "confidence_label": "High" if hours_before == 1 else ("Medium" if hours_before == 3 else "Low"),
        "bucket_probs": bucket_probs,
        "weather": {
            "origin": weather_map[origin],
            "destination": weather_map[destination],
        },
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        checkpoint=CHECKPOINT_NAME,
        aviationstack_enabled=bool(AVIATIONSTACK_KEY),
        model_loaded=_ASSETS is not None,
        model_loading=_MODEL_LOADING,
        model_error=_MODEL_ERROR,
    )


@app.get("/")
def root():
    return {
        "service": "flight-delay-mobile-api",
        "status": "ok",
        "checkpoint": CHECKPOINT_NAME,
        "model_loaded": _ASSETS is not None,
        "model_loading": _MODEL_LOADING,
        "model_error": _MODEL_ERROR,
    }


@app.get("/airports")
def airports():
    return [
        {"code": ap, "name": AIRPORT_NAMES.get(ap, ap)}
        for ap in sorted(AIRPORTS)
    ]


@app.post("/warmup")
def warmup():
    if _ASSETS is not None:
        return {"status": "ready", "checkpoint": CHECKPOINT_NAME}
    try:
        get_assets()
        return {"status": "ready", "checkpoint": CHECKPOINT_NAME}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model warmup failed: {exc}") from exc


@app.post("/predict-flight")
def predict_flight(req: FlightSearchRequest):
    origin = validate_airport(req.origin)
    destination = validate_airport(req.destination)
    dep_dt = parse_departure_datetime(req.departure_date, req.departure_time)

    live_flight = None
    if req.flight_number:
        live_flight = fetch_live_flight_details(
            req.flight_number,
            req.departure_date,
            origin,
            destination,
        )
        if live_flight:
            if live_flight.get("origin") in AIRPORTS:
                origin = live_flight["origin"]
            if live_flight.get("destination") in AIRPORTS:
                destination = live_flight["destination"]
            if live_flight.get("scheduled_departure"):
                dep_dt = pd.Timestamp(live_flight["scheduled_departure"])

    live_dep_delay = float(live_flight.get("live_departure_delay_min", 0.0)) if live_flight else 0.0
    weather_map = {
        origin: fetch_nws_weather(origin),
        destination: fetch_nws_weather(destination),
    }

    try:
        horizons = [
            run_horizon_prediction(
                origin,
                destination,
                dep_dt,
                hours_before=h,
                dep_delay=live_dep_delay,
                weather_map=weather_map,
            )
            for h in [6, 3, 1]
        ]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction unavailable: {exc}") from exc

    summary = {
        "max_predicted_delay_min": max(h["predicted_delay_min"] for h in horizons),
        "max_severe_prob": max(h["severe_prob"] for h in horizons),
        "has_high_conf_severe_alert": any(h["high_conf_severe_alert"] for h in horizons),
        "strict_mode_threshold": SEVERE_ALERT_THRESHOLD,
        "top_tier": max(horizons, key=lambda h: h["predicted_delay_min"])["predicted_tier"],
    }

    return {
        "query": {
            "flight_number": (req.flight_number or "").upper().strip() or None,
            "origin": origin,
            "destination": destination,
            "departure_datetime": dep_dt.isoformat(),
        },
        "live_flight": live_flight,
        "summary": summary,
        "horizons": horizons,
        "model": {
            "checkpoint": CHECKPOINT_NAME,
            "severe_alert_threshold": SEVERE_ALERT_THRESHOLD,
            "ordinal_thresholds": get_assets().ordinal_thresholds,
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("12_mobile_api:app", host="0.0.0.0", port=port, reload=False)
