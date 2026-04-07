"""FastAPI backend for longitudinal ECG trend analysis.

Endpoints:
    GET  /scenarios          — list available preset scenarios
    POST /simulate           — run a scenario (preset or custom), return trends + alerts
    GET  /scenario-defaults  — get default transform parameter ranges for custom builder

Usage:
    uv run uvicorn ecg_monitor.api:app --reload
"""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ecg_monitor.pipeline import DS1_RECORDS, DS2_RECORDS
from ecg_monitor.scenarios import (
    DEFAULT_PARAMS, PRESET_SCENARIOS, build_custom_scenario,
)
from ecg_monitor.simulation import SimulationContext, run_scenario
from ecg_monitor.trends import (
    ALERT_RULES, DISPLAY_METRICS, TRACKED_METRICS,
    compute_trends, detect_alerts, detect_forecast_alerts, summarize_alerts,
)


# ---------------------------------------------------------------------------
# App state — loaded once at startup
# ---------------------------------------------------------------------------

_ctx: SimulationContext | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ctx
    print("Loading models and data...")
    _ctx = SimulationContext()
    print("Ready.")
    yield
    _ctx = None


app = FastAPI(
    title="ECG Trend Analysis API",
    description="Longitudinal ECG monitoring: disease simulation, trend detection, clinical alerting.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    """Request body for /simulate endpoint."""
    # Either specify a preset name...
    preset: str | None = None
    # ...or provide custom scenario parameters
    base_record: str | None = None
    n_sessions: int = 14
    param_ranges: dict[str, list[float]] | None = None  # {param: [start, end]}

    model_type: str = 'gb'  # 'gb' or 'hybrid_cnn'
    n_baseline: int = 3
    ewma_span: int = 3


class AlertResponse(BaseModel):
    session: int
    metric: str
    severity: str
    value: float | None
    baseline_mean: float
    baseline_std: float
    z_score: float
    direction: str
    interpretation: str
    unit: str


class ScenarioInfo(BaseModel):
    key: str
    name: str
    description: str
    n_sessions: int
    base_record: str
    param_schedule: dict[str, list[float]]


class ForecastResponse(BaseModel):
    metric: str
    fit_session: int
    projected_sessions: list[float]
    projected_values: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    warn_crossing: float | None
    danger_crossing: float | None
    prob_warn: float
    prob_danger: float


class SimulateResponse(BaseModel):
    scenario_name: str
    session_metrics: list[dict[str, Any]]
    trends: list[dict[str, Any]]
    alerts: list[AlertResponse]
    forecasts: dict[str, list[ForecastResponse]]
    summary: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/scenarios", response_model=list[ScenarioInfo])
def list_scenarios():
    """List all available preset scenarios."""
    return [
        ScenarioInfo(
            key=key,
            name=s.name,
            description=s.description,
            n_sessions=s.n_sessions,
            base_record=s.base_record,
            param_schedule=s.param_schedule,
        )
        for key, s in PRESET_SCENARIOS.items()
    ]


@app.get("/scenario-defaults")
def scenario_defaults():
    """Return default parameter values and descriptions for custom scenario builder."""
    return {
        'default_params': DEFAULT_PARAMS,
        'available_records': sorted(DS1_RECORDS | DS2_RECORDS),
        'display_metrics': DISPLAY_METRICS,
        'tracked_metrics': TRACKED_METRICS,
        'alert_rules': {k: {kk: vv for kk, vv in v.items()
                            if not kk.startswith('interpretation')}
                        for k, v in ALERT_RULES.items()},
    }


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """Run a simulation and return session metrics, trends, and alerts."""
    if _ctx is None:
        raise HTTPException(503, "Server still loading models")

    # Build scenario from request
    if req.preset:
        if req.preset not in PRESET_SCENARIOS:
            raise HTTPException(
                404, f"Unknown preset '{req.preset}'. "
                     f"Available: {list(PRESET_SCENARIOS.keys())}")
        scenario = PRESET_SCENARIOS[req.preset]
        # Allow overriding the base record
        if req.base_record and req.base_record != scenario.base_record:
            scenario = dataclasses.replace(scenario, base_record=req.base_record)
    elif req.param_ranges and req.base_record:
        # Custom scenario: param_ranges is {param: [start, end]}
        ranges = {k: (v[0], v[1]) for k, v in req.param_ranges.items()
                  if len(v) == 2}
        scenario = build_custom_scenario(
            name="Custom",
            base_record=req.base_record,
            n_sessions=req.n_sessions,
            param_ranges=ranges,
            description="User-defined custom scenario",
        )
    else:
        raise HTTPException(
            400, "Provide either 'preset' name or 'base_record' + 'param_ranges'")

    if req.model_type not in ('gb', 'hybrid_cnn'):
        raise HTTPException(400, "model_type must be 'gb' or 'hybrid_cnn'")

    # Run simulation
    df_sessions = run_scenario(
        _ctx, scenario, model_type=req.model_type)

    if df_sessions.empty:
        raise HTTPException(500, "Simulation produced no results")

    # Compute trends and alerts
    df_trends = compute_trends(
        df_sessions, n_baseline=req.n_baseline, ewma_span=req.ewma_span)
    actual_alerts = detect_alerts(df_trends, n_baseline=req.n_baseline)
    forecast_alerts, forecasts_by_metric = detect_forecast_alerts(
        df_trends, actual_alerts, n_baseline=req.n_baseline)

    # Merge all alerts chronologically
    all_alerts = actual_alerts + forecast_alerts
    severity_order = {'forecast': 0, 'danger': 1, 'warning': 2}
    all_alerts.sort(key=lambda a: (a.session, severity_order.get(a.severity, 3)))

    summary = summarize_alerts(all_alerts, df_sessions)

    # Serialize
    session_metrics = df_sessions.to_dict(orient='records')
    trends = df_trends.to_dict(orient='records')
    alert_responses = [
        AlertResponse(**dataclasses.asdict(a)) for a in all_alerts
    ]
    forecast_responses = {
        metric: [ForecastResponse(**dataclasses.asdict(f)) for f in flist]
        for metric, flist in forecasts_by_metric.items()
    }

    return SimulateResponse(
        scenario_name=scenario.name,
        session_metrics=session_metrics,
        trends=trends,
        alerts=alert_responses,
        forecasts=forecast_responses,
        summary=summary,
    )
