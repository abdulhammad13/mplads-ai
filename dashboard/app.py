from __future__ import annotations

import json
import math
import os
from typing import Any

import polars as pl
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

# ============================================================
# MPLADS AI MONITOR — DASH FRONTEND
# ============================================================
# Architecture:
#   Dash/Plotly UI -> FastAPI -> processed MPLADS analytical data
#
# This file intentionally does NOT recompute risk, ML, feature
# engineering, or business logic. Those remain in backend/.
# ============================================================

API_BASE = os.getenv("MPLADS_API_URL", "http://127.0.0.1:8000").rstrip("/")
DASH_HOST = os.getenv("MPLADS_DASH_HOST", "127.0.0.1")
DASH_PORT = int(os.getenv("MPLADS_DASH_PORT", "8050"))
REQUEST_TIMEOUT = int(os.getenv("MPLADS_DASH_TIMEOUT", "30"))
APP_VERSION = "5.1-Dash"
QUEUE_LIMIT = 500

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_RANGES = {
    "LOW": "0–25",
    "MEDIUM": ">25–50",
    "HIGH": ">50–75",
    "CRITICAL": ">75–100",
}

FONT_HEAD = "Bahnschrift, Segoe UI, Arial, sans-serif"
FONT_BODY = "Segoe UI, Arial, sans-serif"
FONT_MONO = "Cascadia Mono, Consolas, monospace"


# ============================================================
# SAFE / DATA HELPERS
# ============================================================

def api_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        f"{API_BASE}{endpoint}",
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(safe_float(value, float(default))))
    except (TypeError, ValueError):
        return default


def pct(value: Any) -> str:
    return "—" if value is None else f"{safe_float(value):,.1f}%"


def inr(value: Any) -> str:
    amount = safe_float(value, float("nan"))
    if not math.isfinite(amount):
        return "—"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1e7:
        return f"{sign}₹{amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"{sign}₹{amount / 1e5:,.2f} L"
    return f"{sign}₹{amount:,.0f}"


def clean_options(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return sorted(result, key=str.casefold)


def option_values(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def build_filter_params(
    state: str | None,
    district: str | None,
    mp: str | None,
    constituency: str | None,
    category: str | None,
    status: str | None,
    risk: str | None,
    completion: str | None,
    search: str | None,
    min_sanction: float | None,
    max_sanction: float | None,
    risk_range: list[float] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    if risk_range:
        params["min_risk"] = risk_range[0]
        params["max_risk"] = risk_range[1]

    values = {
        "state": state,
        "district": district,
        "mp": mp,
        "constituency": constituency,
        "work_category": category,
        "work_status": status,
        "risk_category": risk,
        "completion_status": completion,
    }

    for key, value in values.items():
        if value and value != "All":
            params[key] = value

    if search and search.strip():
        params["search"] = search.strip()

    if min_sanction and min_sanction > 0:
        params["min_sanction"] = min_sanction

    if max_sanction and max_sanction > 0:
        params["max_sanction"] = max_sanction

    return params


def to_polars(records: Any) -> pl.DataFrame:
    if not isinstance(records, list) or not records:
        return pl.DataFrame()
    try:
        return pl.DataFrame(records)
    except Exception:
        return pl.DataFrame(
            [dict(item) for item in records if isinstance(item, dict)]
        )


def records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    if isinstance(payload, list):
        return payload
    return []


# ============================================================
# PLOTLY VISUAL SYSTEM
# ============================================================

def base_figure(title: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "font": {"family": FONT_HEAD, "size": 17}},
        height=height,
        margin={"l": 14, "r": 18, "t": 58, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_BODY, "color": "#DDE6ED"},
        hoverlabel={"font": {"family": FONT_BODY}},
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#202A34",
        zeroline=False,
        linecolor="#2B3642",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#202A34",
        zeroline=False,
        linecolor="#2B3642",
    )
    return fig


def empty_figure(message: str, height: int = 380) -> go.Figure:
    fig = base_figure("", height)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"family": FONT_BODY, "size": 14, "color": "#81909E"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def risk_figure(data: Any, title: str) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or not {"risk_category", "count"}.issubset(frame.columns):
        return empty_figure("Risk distribution unavailable")

    rows = frame.to_dicts()
    rows.sort(
        key=lambda row: RISK_ORDER.index(str(row.get("risk_category")))
        if str(row.get("risk_category")) in RISK_ORDER
        else 99
    )

    fig = base_figure(title, 390)
    fig.add_trace(
        go.Bar(
            x=[safe_int(r.get("count")) for r in rows],
            y=[str(r.get("risk_category")) for r in rows],
            orientation="h",
            text=[f"{safe_float(r.get('pct')):.1f}%" for r in rows],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:,} works<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": RISK_ORDER},
        xaxis_title="Works",
        yaxis_title="",
    )
    return fig


def reasons_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or not {"reason", "count"}.issubset(frame.columns):
        return empty_figure("Risk-signal frequency unavailable")

    frame = frame.with_columns(
        pl.col("count").cast(pl.Float64, strict=False).fill_null(0)
    ).sort("count")

    rows = frame.to_dicts()
    fig = base_figure("Detected risk-signal frequency", 390)
    fig.add_trace(
        go.Bar(
            x=[safe_float(r.get("count")) for r in rows],
            y=[str(r.get("reason")) for r in rows],
            orientation="h",
            hovertemplate="%{y}<br>%{x:,.0f} works<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Works", yaxis_title="")
    return fig


def time_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or "financial_year" not in frame.columns:
        return empty_figure("Time-series data unavailable")

    rows = frame.to_dicts()
    x = [row.get("financial_year") for row in rows]

    fig = base_figure("Financial flow across financial years", 400)

    if "sanctioned_amount" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[safe_float(r.get("sanctioned_amount")) for r in rows],
                mode="lines+markers",
                name="Sanctioned",
                hovertemplate="FY %{x}<br>Sanctioned: ₹%{y:,.0f}<extra></extra>",
            )
        )

    if "total_expenditure" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[safe_float(r.get("total_expenditure")) for r in rows],
                mode="lines+markers",
                name="Recorded expenditure",
                hovertemplate="FY %{x}<br>Expenditure: ₹%{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        yaxis_title="Amount (₹)",
        hovermode="x unified",
    )
    return fig


def state_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    required = {"state", "high_or_critical_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("State analytics unavailable", 500)

    frame = (
        frame.with_columns(
            pl.col("high_or_critical_rate_pct")
            .cast(pl.Float64, strict=False)
            .fill_null(0)
        )
        .sort("high_or_critical_rate_pct", descending=True)
        .head(20)
        .sort("high_or_critical_rate_pct")
    )

    rows = frame.to_dicts()
    fig = base_figure("High / critical signal rate by state", 510)
    fig.add_trace(
        go.Bar(
            x=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],
            y=[str(r.get("state", "Unknown")) for r in rows],
            orientation="h",
            customdata=[
                [safe_int(r.get("works")), safe_int(r.get("completed_works"))]
                for r in rows
            ],
            hovertemplate=(
                "%{y}<br>High/Critical: %{x:.1f}%"
                "<br>Works: %{customdata[0]:,}"
                "<br>Completed: %{customdata[1]:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(xaxis_title="Rate (%)", yaxis_title="")
    return fig


def sector_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty():
        return empty_figure("Sector analytics unavailable")

    label_col = (
        "sector"
        if "sector" in frame.columns
        else "work_category"
        if "work_category" in frame.columns
        else None
    )
    value_col = (
        "sanctioned_amount"
        if "sanctioned_amount" in frame.columns
        else "works"
        if "works" in frame.columns
        else None
    )

    if not label_col or not value_col:
        return empty_figure("Sector analytics unavailable")

    frame = (
        frame.with_columns(
            pl.col(value_col).cast(pl.Float64, strict=False).fill_null(0)
        )
        .sort(value_col, descending=True)
        .head(12)
    )

    rows = frame.to_dicts()
    fig = base_figure("Portfolio by sector / work category", 400)
    fig.add_trace(
        go.Bar(
            x=[safe_float(r.get(value_col)) for r in rows],
            y=[str(r.get(label_col, "Unknown")) for r in rows],
            orientation="h",
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Amount (₹)" if value_col == "sanctioned_amount" else "Works",
        yaxis_title="",
    )
    return fig


def financial_scatter(data: list[dict[str, Any]]) -> go.Figure:
    frame = to_polars(data)
    required = {"sanction_amount", "total_expenditure"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("Sanction / expenditure fields unavailable", 420)

    rows = frame.to_dicts()
    valid = []

    for row in rows:
        x = safe_float(row.get("sanction_amount"), float("nan"))
        y = safe_float(row.get("total_expenditure"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            valid.append((x, y, row))

    if not valid:
        return empty_figure("No numeric financial observations", 420)

    max_value = max(
        max(x for x, _, _ in valid),
        max(y for _, y, _ in valid),
        1.0,
    )

    fig = base_figure("Sanction vs recorded expenditure · current queue", 430)

    fig.add_trace(
        go.Scatter(
            x=[x for x, _, _ in valid],
            y=[y for _, y, _ in valid],
            mode="markers",
            marker={"size": 9, "opacity": 0.78},
            customdata=[
                [
                    row.get("work_uid"),
                    row.get("risk_category"),
                    safe_float(row.get("final_risk_score")),
                ]
                for _, _, row in valid
            ],
            hovertemplate=(
                "Work %{customdata[0]}"
                "<br>Band: %{customdata[1]}"
                "<br>Risk: %{customdata[2]:.1f}"
                "<br>Sanction: ₹%{x:,.0f}"
                "<br>Expenditure: ₹%{y:,.0f}<extra></extra>"
            ),
            name="Works",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            name="1:1 reference",
            line={"dash": "dash"},
        )
    )

    fig.update_layout(
        xaxis_title="Sanction amount (₹)",
        yaxis_title="Recorded expenditure (₹)",
    )
    return fig


def execution_figure(data: list[dict[str, Any]]) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty():
        return empty_figure("Execution data unavailable")

    fields = {
        "days_open_since_sanction": "Open age",
        "days_since_last_expenditure": "Days since expenditure",
        "days_sanction_to_complete": "Sanction → completion",
    }

    available = [field for field in fields if field in frame.columns]
    if not available:
        return empty_figure("Execution-duration fields unavailable")

    fig = base_figure("Execution-duration distribution · current queue", 410)

    for field in available:
        values = (
            frame.get_column(field)
            .cast(pl.Float64, strict=False)
            .drop_nulls()
            .to_list()
        )
        if values:
            fig.add_trace(
                go.Box(
                    y=values,
                    name=fields[field],
                    boxpoints="outliers",
                    hovertemplate=f"{fields[field]}<br>%{{y:.0f}} days<extra></extra>",
                )
            )

    fig.update_layout(yaxis_title="Days")
    return fig


def component_figure(components: dict[str, Any]) -> go.Figure:
    mapping = [
        ("ml_anomaly_percentile", "ML anomaly"),
        ("financial_risk_score", "Financial"),
        ("execution_risk_score", "Execution"),
        ("duplicate_risk_score", "Similarity"),
        ("data_integrity_risk_score", "Integrity"),
    ]

    pairs = [
        (label, safe_float(components.get(key)))
        for key, label in mapping
        if components.get(key) is not None
    ]

    if not pairs:
        return empty_figure("Component scores unavailable", 300)

    fig = base_figure("Risk decomposition", 310)
    fig.add_trace(
        go.Bar(
            x=[value for _, value in pairs],
            y=[label for label, _ in pairs],
            orientation="h",
            text=[f"{value:.1f}" for _, value in pairs],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"range": [0, 100], "title": "Score (0–100)"},
        yaxis_title="",
    )
    return fig


# ============================================================
# COMPREHENSIVE ANALYTICS CHARTS
# ============================================================

def _numeric_values(frame: pl.DataFrame, column: str) -> list[float]:
    if column not in frame.columns:
        return []
    return [
        safe_float(v, float("nan"))
        for v in frame.get_column(column).cast(pl.Float64, strict=False).to_list()
        if math.isfinite(safe_float(v, float("nan")))
    ]


def risk_basis_figure(analytics: dict[str, Any]) -> go.Figure:
    final = to_polars(analytics.get("risk_final"))
    rule = to_polars(analytics.get("risk_rule"))
    ml = to_polars(analytics.get("risk_ml"))
    if final.is_empty() and rule.is_empty() and ml.is_empty():
        return empty_figure("Risk-basis comparison unavailable")

    def by_band(frame: pl.DataFrame) -> dict[str, float]:
        if frame.is_empty() or "risk_category" not in frame.columns:
            return {}
        return {str(r.get("risk_category")): safe_float(r.get("pct")) for r in frame.to_dicts()}

    datasets = [
        ("Final composite", by_band(final)),
        ("Rule-based", by_band(rule)),
        ("ML anomaly", by_band(ml)),
    ]
    fig = base_figure("Risk-band composition across scoring layers", 420)
    for name, values in datasets:
        if not values:
            continue
        fig.add_trace(go.Bar(
            x=RISK_ORDER,
            y=[values.get(b, 0.0) for b in RISK_ORDER],
            name=name,
            hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Risk band",
        yaxis_title="Share of filtered works (%)",
        yaxis={"range": [0, 100]},
    )
    return fig


def risk_score_histogram(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or "final_risk_score" not in frame.columns:
        return empty_figure("Risk-score distribution unavailable")
    vals = _numeric_values(frame, "final_risk_score")
    if not vals:
        return empty_figure("No numeric risk scores in current queue")
    bins = [i for i in range(0, 101, 10)]
    counts = [0] * 10
    for v in vals:
        idx = min(9, max(0, int(v // 10)))
        counts[idx] += 1
    labels = [f"{i}–{i+10}" for i in range(0, 100, 10)]
    fig = base_figure("Final-risk score distribution · priority queue", 390)
    fig.add_trace(go.Bar(x=labels, y=counts, name="Works", hovertemplate="Score %{x}<br>%{y:,} works<extra></extra>"))
    fig.update_layout(xaxis_title="Final risk score", yaxis_title="Works")
    return fig


def completion_risk_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    required = {"final_risk_score", "utilization_pct"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("Risk vs utilisation relationship unavailable")
    rows=[]
    for r in frame.to_dicts():
        x=safe_float(r.get("utilization_pct"), float("nan")); y=safe_float(r.get("final_risk_score"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            rows.append((x,y,r))
    if not rows:
        return empty_figure("No numeric observations for risk vs utilisation")
    fig=base_figure("Risk score vs utilisation", 430)
    fig.add_trace(go.Scatter(
        x=[r[0] for r in rows], y=[r[1] for r in rows], mode="markers", name="Works",
        customdata=[[r[2].get("work_uid"), r[2].get("risk_category")] for r in rows],
        hovertemplate="Work %{customdata[0]}<br>Band: %{customdata[1]}<br>Utilisation: %{x:.1f}%<br>Risk: %{y:.1f}<extra></extra>",
        marker={"size":8,"opacity":0.65},
    ))
    fig.update_layout(xaxis_title="Recorded expenditure / sanction (%)", yaxis_title="Final risk score (0–100)")
    return fig


def priority_risk_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"final_risk_score","priority_score"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("Priority vs risk relationship unavailable")
    rows=[]
    for r in frame.to_dicts():
        x=safe_float(r.get("final_risk_score"),float("nan")); y=safe_float(r.get("priority_score"),float("nan"))
        if math.isfinite(x) and math.isfinite(y): rows.append((x,y,r))
    if not rows: return empty_figure("No numeric priority observations")
    fig=base_figure("Risk score vs investigation priority",430)
    fig.add_trace(go.Scatter(
        x=[r[0] for r in rows], y=[r[1] for r in rows], mode="markers", name="Works",
        customdata=[[r[2].get("work_uid"),r[2].get("risk_category")] for r in rows],
        hovertemplate="Work %{customdata[0]}<br>Band: %{customdata[1]}<br>Risk: %{x:.1f}<br>Priority: %{y:.1f}<extra></extra>",
        marker={"size":8,"opacity":0.65},
    ))
    fig.update_layout(xaxis_title="Final risk score", yaxis_title="Priority score")
    return fig


def risk_reason_rate_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "reason" not in frame.columns: return empty_figure("Risk-signal rates unavailable")
    value="pct_of_filtered_works" if "pct_of_filtered_works" in frame.columns else "pct_of_all_works" if "pct_of_all_works" in frame.columns else None
    if not value: return empty_figure("Risk-signal rates unavailable")
    frame=frame.with_columns(pl.col(value).cast(pl.Float64,strict=False).fill_null(0)).sort(value).tail(12)
    rows=frame.to_dicts()
    fig=base_figure("Risk-signal prevalence",430)
    fig.add_trace(go.Bar(x=[safe_float(r.get(value)) for r in rows], y=[str(r.get("reason")) for r in rows], orientation="h", hovertemplate="%{y}<br>%{x:.1f}% of filtered works<extra></extra>"))
    fig.update_layout(xaxis_title="Share of filtered works (%)",yaxis_title="")
    return fig


def sector_completion_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"work_category","completion_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("Sector completion data unavailable")
    frame=frame.with_columns(pl.col("completion_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("completion_rate_pct").tail(12)
    rows=frame.to_dicts()
    fig=base_figure("Completion rate by work category",430)
    fig.add_trace(go.Bar(x=[safe_float(r.get("completion_rate_pct")) for r in rows],y=[str(r.get("work_category")) for r in rows],orientation="h",hovertemplate="%{y}<br>Completion: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Completed works (%)",yaxis_title="")
    return fig


def sector_risk_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"work_category","high_or_critical_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("Sector risk data unavailable")
    frame=frame.with_columns(pl.col("high_or_critical_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("high_or_critical_rate_pct").tail(12)
    rows=frame.to_dicts()
    fig=base_figure("High / critical signal rate by work category",430)
    fig.add_trace(go.Bar(x=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],y=[str(r.get("work_category")) for r in rows],orientation="h",hovertemplate="%{y}<br>High/Critical: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Rate (%)",yaxis_title="")
    return fig


def state_exposure_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"state","sanctioned_amount"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("State financial exposure unavailable")
    frame=frame.with_columns(pl.col("sanctioned_amount").cast(pl.Float64,strict=False).fill_null(0)).sort("sanctioned_amount").tail(15)
    rows=frame.to_dicts()
    fig=base_figure("Sanctioned amount by state",480)
    fig.add_trace(go.Bar(x=[safe_float(r.get("sanctioned_amount")) for r in rows],y=[str(r.get("state")) for r in rows],orientation="h",hovertemplate="%{y}<br>Sanctioned: ₹%{x:,.0f}<extra></extra>"))
    fig.update_layout(xaxis_title="Sanctioned amount (₹)",yaxis_title="")
    return fig


def state_completion_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"state","completion_rate_pct"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("State completion data unavailable")
    frame=frame.with_columns(pl.col("completion_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("completion_rate_pct").tail(15)
    rows=frame.to_dicts()
    fig=base_figure("Completion rate by state",480)
    fig.add_trace(go.Bar(x=[safe_float(r.get("completion_rate_pct")) for r in rows],y=[str(r.get("state")) for r in rows],orientation="h",hovertemplate="%{y}<br>Completion: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Completion rate (%)",yaxis_title="")
    return fig


def time_performance_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "financial_year" not in frame.columns: return empty_figure("Financial-year performance unavailable")
    rows=frame.to_dicts(); fig=base_figure("Completion and high/critical rates over time",420)
    if "completion_rate_pct" in frame.columns:
        fig.add_trace(go.Scatter(x=[r.get("financial_year") for r in rows],y=[safe_float(r.get("completion_rate_pct")) for r in rows],mode="lines+markers",name="Completion rate",hovertemplate="FY %{x}<br>Completion: %{y:.1f}%<extra></extra>"))
    if "high_or_critical_rate_pct" in frame.columns:
        fig.add_trace(go.Scatter(x=[r.get("financial_year") for r in rows],y=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],mode="lines+markers",name="High/Critical rate",hovertemplate="FY %{x}<br>High/Critical: %{y:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Financial year",yaxis_title="Rate (%)",yaxis={"range":[0,100]},hovermode="x unified")
    return fig


def time_risk_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or not {"financial_year","median_risk"}.issubset(frame.columns): return empty_figure("Median-risk trend unavailable")
    rows=frame.to_dicts(); fig=base_figure("Median final risk by financial year",400)
    fig.add_trace(go.Scatter(x=[r.get("financial_year") for r in rows],y=[safe_float(r.get("median_risk")) for r in rows],mode="lines+markers",name="Median risk",hovertemplate="FY %{x}<br>Median risk: %{y:.1f}<extra></extra>"))
    fig.update_layout(xaxis_title="Financial year",yaxis_title="Median final risk (0–100)",yaxis={"range":[0,100]})
    return fig


def financial_gap_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"sanction_amount","total_expenditure"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("Financial-gap relationship unavailable")
    rows=[]
    for r in frame.to_dicts():
        s=safe_float(r.get("sanction_amount"),float("nan")); e=safe_float(r.get("total_expenditure"),float("nan"))
        if math.isfinite(s) and math.isfinite(e) and s>0: rows.append((s,e,r))
    if not rows: return empty_figure("No valid financial observations")
    fig=base_figure("Sanction, expenditure and financial position",430)
    fig.add_trace(go.Scatter(x=[r[0] for r in rows],y=[r[1] for r in rows],mode="markers",name="Works",customdata=[[r[2].get("work_uid"),r[2].get("risk_category")] for r in rows],hovertemplate="Work %{customdata[0]}<br>Band: %{customdata[1]}<br>Sanction: ₹%{x:,.0f}<br>Expenditure: ₹%{y:,.0f}<extra></extra>",marker={"size":8,"opacity":0.6}))
    maxv=max(max(r[0] for r in rows),max(r[1] for r in rows))
    fig.add_trace(go.Scatter(x=[0,maxv],y=[0,maxv],mode="lines",name="Sanction = expenditure",line={"dash":"dash"}))
    fig.update_layout(xaxis_title="Sanction amount (₹)",yaxis_title="Recorded expenditure (₹)")
    return fig


def utilization_distribution_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "utilization_pct" not in frame.columns: return empty_figure("Utilisation distribution unavailable")
    vals=_numeric_values(frame,"utilization_pct")
    if not vals: return empty_figure("No valid utilisation values")
    bounds=[0,25,50,75,100,125,150,200,300,float("inf")]
    labels=["0–25%","25–50%","50–75%","75–100%","100–125%","125–150%","150–200%","200–300%","300%+"]
    counts=[0]*len(labels)
    for v in vals:
        for i in range(len(bounds)-1):
            if bounds[i] <= v < bounds[i+1]: counts[i]+=1; break
    fig=base_figure("Utilisation distribution · priority queue",400)
    fig.add_trace(go.Bar(x=labels,y=counts,name="Works",hovertemplate="Utilisation %{x}<br>%{y:,} works<extra></extra>"))
    fig.update_layout(xaxis_title="Recorded expenditure / sanction",yaxis_title="Works")
    return fig


def execution_age_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or "days_open_since_sanction" not in frame.columns: return empty_figure("Execution-age distribution unavailable")
    vals=_numeric_values(frame,"days_open_since_sanction")
    if not vals: return empty_figure("No valid execution-age values")
    bins=[0,90,180,365,730,1095,1825,float("inf")]
    labels=["0–90","91–180","181–365","366–730","731–1095","1096–1825","1826+"]
    counts=[0]*len(labels)
    for v in vals:
        for i in range(len(bounds:=bins)-1):
            if bounds[i] <= v < bounds[i+1]: counts[i]+=1; break
    fig=base_figure("Open-work age distribution",400)
    fig.add_trace(go.Bar(x=labels,y=counts,name="Works",hovertemplate="Open age %{x} days<br>%{y:,} works<extra></extra>"))
    fig.update_layout(xaxis_title="Days since sanction",yaxis_title="Open works")
    return fig



# ============================================================
# OVERVIEW-SPECIFIC FIGURES
# ============================================================

def overview_status_figure(scope: Any) -> go.Figure:
    completed = safe_int(scope.get("completed_works")) if isinstance(scope, dict) else 0
    open_works = safe_int(scope.get("open_works")) if isinstance(scope, dict) else 0
    total = completed + open_works
    if total <= 0:
        return empty_figure("Project-status data unavailable", 260)

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["Completed", "Open"],
        values=[completed, open_works],
        hole=.72,
        sort=False,
        textinfo="none",
        hovertemplate="%{label}<br>%{value:,} works<br>%{percent}<extra></extra>",
        marker={"line": {"color": "#0B1117", "width": 3}},
    ))
    fig.add_annotation(
        x=.5, y=.54,
        text=f"<b>{total:,}</b>",
        showarrow=False,
        font={"family": FONT_HEAD, "size": 25, "color": "#EEF4F8"},
    )
    fig.add_annotation(
        x=.5, y=.39,
        text="WORKS",
        showarrow=False,
        font={"family": FONT_MONO, "size": 9, "color": "#7E93A3"},
    )
    fig.update_layout(
        title={"text":"Project status", "font":{"family":FONT_HEAD,"size":16}},
        height=260,
        margin={"l":8,"r":8,"t":45,"b":8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family":FONT_BODY,"color":"#DDE7EE"},
        showlegend=True,
        legend={"orientation":"h","y":-0.02,"x":.5,"xanchor":"center"},
    )
    return fig


def overview_utilization_figure(scope: Any) -> go.Figure:
    value = safe_float(scope.get("portfolio_utilization_pct"), float("nan")) if isinstance(scope, dict) else float("nan")
    if not math.isfinite(value):
        return empty_figure("Utilisation unavailable", 260)
    value = max(0.0, min(value, 100.0))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix":"%", "font":{"family":FONT_HEAD,"size":31}},
        title={"text":"Portfolio utilisation", "font":{"family":FONT_HEAD,"size":16}},
        gauge={
            "axis":{"range":[0,100],"tickvals":[0,25,50,75,100],"tickfont":{"family":FONT_MONO,"size":8,"color":"#71818D"}},
            "bar":{"color":"#8FD0FF","thickness":.22},
            "bgcolor":"#101820",
            "bordercolor":"#2A3743",
            "borderwidth":1,
            "steps":[
                {"range":[0,25],"color":"#17222B"},
                {"range":[25,50],"color":"#1A2730"},
                {"range":[50,75],"color":"#1D2A32"},
                {"range":[75,100],"color":"#213039"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin={"l":22,"r":22,"t":45,"b":20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family":FONT_BODY,"color":"#DDE7EE"},
    )
    return fig


def overview_risk_spectrum_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    if frame.is_empty() or not {"risk_category","count"}.issubset(frame.columns):
        return empty_figure("Risk spectrum unavailable", 285)
    lookup = {str(r.get("risk_category")): safe_int(r.get("count")) for r in frame.to_dicts()}
    total = sum(lookup.values())
    if total <= 0:
        return empty_figure("Risk spectrum unavailable", 285)
    fig = go.Figure()
    for band in RISK_ORDER:
        count = lookup.get(band,0)
        fig.add_trace(go.Bar(
            x=[count], y=["Current scope"], orientation="h", name=band,
            customdata=[[count, count/total*100]],
            hovertemplate=f"{band}<br>%{{customdata[0]:,}} works<br>%{{customdata[1]:.1f}}% of scope<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title={"text":"Risk spectrum · same filtered population", "font":{"family":FONT_HEAD,"size":16}},
        height=285,
        margin={"l":14,"r":18,"t":52,"b":35},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family":FONT_BODY,"color":"#DDE7EE"},
        legend={"orientation":"h","y":1.08,"x":0},
        xaxis={"showgrid":True,"gridcolor":"#202A34","title":"Works"},
        yaxis={"showgrid":False,"title":""},
    )
    return fig


def overview_state_attention_figure(data: Any) -> go.Figure:
    frame = to_polars(data)
    required={"state","completion_rate_pct","high_or_critical_rate_pct","works"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return empty_figure("State comparison unavailable", 390)
    frame=(frame.with_columns([
        pl.col("completion_rate_pct").cast(pl.Float64,strict=False).fill_null(0),
        pl.col("high_or_critical_rate_pct").cast(pl.Float64,strict=False).fill_null(0),
        pl.col("works").cast(pl.Float64,strict=False).fill_null(0),
    ]).filter(pl.col("works")>0).sort("works",descending=True).head(35))
    if frame.is_empty(): return empty_figure("State comparison unavailable",390)
    rows=frame.to_dicts()
    fig=base_figure("State execution × signal landscape",390)
    fig.add_trace(go.Scatter(
        x=[safe_float(r.get("completion_rate_pct")) for r in rows],
        y=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],
        mode="markers+text",
        text=[str(r.get("state","Unknown"))[:18] for r in rows],
        textposition="top center",
        textfont={"size":8,"color":"#91A5B4"},
        marker={"size":[max(7,min(25,7+math.sqrt(max(0,safe_float(r.get("works")))/10))) for r in rows],"line":{"width":1,"color":"#8FD0FF"}},
        customdata=[[safe_int(r.get("works")),safe_float(r.get("completion_rate_pct")),safe_float(r.get("high_or_critical_rate_pct"))] for r in rows],
        hovertemplate="%{text}<br>Works: %{customdata[0]:,}<br>Completion: %{customdata[1]:.1f}%<br>High/Critical: %{customdata[2]:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=50,line_dash="dot",line_color="#3A4B58")
    fig.add_hline(y=10,line_dash="dot",line_color="#3A4B58")
    fig.update_layout(xaxis_title="Completion rate (%)",yaxis_title="High / Critical signal rate (%)",hovermode="closest")
    return fig


def overview_financial_flow_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    required={"financial_year","sanctioned_amount","total_expenditure"}
    if frame.is_empty() or not required.issubset(frame.columns): return empty_figure("Financial flow unavailable",390)
    frame=frame.with_columns([
        pl.col("sanctioned_amount").cast(pl.Float64,strict=False).fill_null(0),
        pl.col("total_expenditure").cast(pl.Float64,strict=False).fill_null(0),
    ]).sort("financial_year")
    rows=frame.to_dicts()
    x=[str(r.get("financial_year")) for r in rows]
    fig=base_figure("Financial flow by financial year",390)
    fig.add_trace(go.Scatter(x=x,y=[safe_float(r.get("sanctioned_amount")) for r in rows],name="Sanctioned",mode="lines+markers",line={"width":2.5},hovertemplate="FY %{x}<br>Sanctioned: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x,y=[safe_float(r.get("total_expenditure")) for r in rows],name="Expenditure",mode="lines+markers",line={"width":2.5},hovertemplate="FY %{x}<br>Expenditure: ₹%{y:,.0f}<extra></extra>"))
    fig.update_layout(yaxis_title="Amount (₹)",hovermode="x unified")
    return fig


def overview_signal_figure(data: Any) -> go.Figure:
    frame=to_polars(data)
    if frame.is_empty() or not {"reason","count","pct_of_filtered_works"}.issubset(frame.columns): return empty_figure("Signal profile unavailable",390)
    frame=frame.with_columns(pl.col("pct_of_filtered_works").cast(pl.Float64,strict=False).fill_null(0)).sort("pct_of_filtered_works")
    rows=frame.to_dicts()
    fig=base_figure("Signal prevalence · not findings",390)
    fig.add_trace(go.Bar(
        x=[safe_float(r.get("pct_of_filtered_works")) for r in rows],
        y=[str(r.get("reason")) for r in rows],orientation="h",
        customdata=[[safe_int(r.get("count"))] for r in rows],
        hovertemplate="%{y}<br>%{x:.1f}% of filtered works<br>Count: %{customdata[0]:,}<extra></extra>"
    ))
    fig.update_layout(xaxis_title="Share of filtered works (%)",yaxis_title="")
    return fig


# ============================================================
# UI COMPONENTS
# ============================================================

def kpi_card(
    label: str,
    value: str,
    hint: str = "",
    tone: str = "",
) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(hint, className="kpi-hint") if hint else None,
        ],
        className=f"kpi-card {tone}".strip(),
    )


def section_header(kicker: str, title: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(kicker.upper(), className="section-kicker"),
            html.H2(title, className="section-title"),
            html.Div(note, className="section-note") if note else None,
        ],
        className="section-header",
    )


def empty_panel(message: str) -> html.Div:
    return html.Div(message, className="empty-panel")


def queue_table(
    data: list[dict[str, Any]],
    selectable: bool = False,
    table_id: str = "queue-table",
) -> dash_table.DataTable:
    if not data:
        return dash_table.DataTable(
            data=[],
            columns=[{"name": "Status", "id": "status"}],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#151D26", "color": "#A7B5C2"},
            style_cell={
                "backgroundColor": "#0F151C",
                "color": "#DCE5EC",
                "padding": "10px",
            },
        )

    fields = [
        "priority_rank",
        "work_uid",
        "work",
        "state",
        "ida",
        "mp",
        "constituency",
        "work_category",
        "financial_year",
        "sanction_amount",
        "total_expenditure",
        "utilization_pct",
        "days_open_since_sanction",
        "final_risk_score",
        "risk_category",
        "priority_score",
        "confidence_score",
        "primary_risk_reason",
    ]

    frame = to_polars(data)
    available = [field for field in fields if field in frame.columns]
    frame = frame.select(available)

    labels = {
        "priority_rank": "#",
        "work_uid": "Work ID",
        "work": "Work",
        "state": "State",
        "ida": "District / IDA",
        "mp": "MP",
        "constituency": "Constituency",
        "work_category": "Category",
        "financial_year": "FY",
        "sanction_amount": "Sanction",
        "total_expenditure": "Expenditure",
        "utilization_pct": "Utilisation",
        "days_open_since_sanction": "Open age",
        "final_risk_score": "Risk",
        "risk_category": "Band",
        "priority_score": "Priority",
        "confidence_score": "Confidence",
        "primary_risk_reason": "Primary signal",
    }

    return dash_table.DataTable(
        id=table_id,
        data=frame.to_dicts(),
        columns=[
            {"name": labels.get(field, field), "id": field}
            for field in available
        ],
        page_action="none",
        sort_action="native",
        filter_action="native",
        row_selectable="single" if selectable else False,
        style_table={
            "overflowX": "auto",
            "maxHeight": "600px",
            "overflowY": "auto",
        },
        style_header={
            "backgroundColor": "#151D26",
            "color": "#A7B5C2",
            "fontFamily": FONT_BODY,
            "fontWeight": "700",
            "border": "1px solid #27323D",
            "position": "sticky",
            "top": 0,
            "zIndex": 1,
        },
        style_cell={
            "backgroundColor": "#0F151C",
            "color": "#DCE5EC",
            "fontFamily": FONT_BODY,
            "fontSize": "12px",
            "padding": "10px",
            "border": "1px solid #202A34",
            "textAlign": "left",
            "maxWidth": "300px",
            "whiteSpace": "normal",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": '{risk_category} = "CRITICAL"'},
                "backgroundColor": "#24171A",
            },
            {
                "if": {"filter_query": '{risk_category} = "HIGH"'},
                "backgroundColor": "#211D16",
            },
            {
                "if": {"column_id": "final_risk_score"},
                "fontWeight": "700",
            },
            {
                "if": {"column_id": "priority_score"},
                "fontWeight": "700",
            },
        ],
        sort_by=(
            [{"column_id": "priority_score", "direction": "desc"}]
            if "priority_score" in available
            else []
        ),
    )


# ============================================================
# APPLICATION
# ============================================================

app = Dash(
    __name__,
    title="MPLADS AI Monitor",
    update_title="MPLADS AI Monitor · updating",
    suppress_callback_exceptions=False,
    serve_locally=True,
)
server = app.server

app.index_string = r"""
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

app.layout = html.Div(
    [
        dcc.Store(id="health-store"),
        dcc.Store(id="options-store"),
        dcc.Store(id="national-store"),
        dcc.Store(id="filtered-store"),
        dcc.Store(id="works-store"),
        dcc.Store(id="analytics-store"),
        dcc.Download(id="download-queue"),

        dcc.Interval(
            id="startup",
            interval=1000,
            n_intervals=0,
            max_intervals=1,
        ),

        html.Div(
            [
                # ---------------- SIDEBAR ----------------
                html.Aside(
                    [
                        html.Div(
                            [
                                html.Div("MPLADS", className="brand-mark"),
                                html.Div("AI MONITOR", className="brand-sub"),
                                html.Div(
                                    "MONITOR · DETECT · EXPLAIN · REVIEW",
                                    className="brand-micro",
                                ),
                            ],
                            className="brand",
                        ),

                        html.Div("VIEW", className="control-label"),
                        dcc.Dropdown(
                            id="view-mode",
                            options=option_values(
                                [
                                    "Overview",
                                    "Risk Intelligence",
                                    "Financial & Execution",
                                    "Geography",
                                    "Work Explorer",
                                    "Methodology",
                                ]
                            ),
                            value="Overview",
                            clearable=False,
                            className="dark-dropdown",
                        ),

                        html.Div("SCOPE", className="control-label"),
                        dcc.Dropdown(
                            id="state-filter",
                            options=[{"label": "All", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        dcc.Dropdown(
                            id="district-filter",
                            options=[{"label": "All", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        dcc.Dropdown(
                            id="mp-filter",
                            options=[{"label": "All", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        dcc.Dropdown(
                            id="constituency-filter",
                            options=[{"label": "All", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        dcc.Dropdown(
                            id="category-filter",
                            options=[{"label": "All", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        dcc.Dropdown(
                            id="status-filter",
                            options=[{"label": "All", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        dcc.Dropdown(
                            id="risk-filter",
                            options=[{"label": "All", "value": "All"}],
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),
                        dcc.Dropdown(
                            id="completion-filter",
                            options=option_values(["All", "Completed", "Open"]),
                            value="All",
                            clearable=False,
                            className="dark-dropdown",
                        ),

                        html.Div("SEARCH", className="control-label"),
                        dcc.Input(
                            id="search-filter",
                            type="search",
                            placeholder="Work, ID, MP, constituency…",
                            className="dark-input",
                        ),

                        html.Div("RISK / VALUE", className="control-label"),
                        dcc.RangeSlider(
                            id="risk-range",
                            min=0,
                            max=100,
                            step=1,
                            value=[0, 100],
                            marks={
                                0: "0",
                                25: "25",
                                50: "50",
                                75: "75",
                                100: "100",
                            },
                            tooltip={"placement": "bottom"},
                        ),

                        dcc.Input(
                            id="min-sanction",
                            type="number",
                            min=0,
                            step=100000,
                            value=0,
                            placeholder="Minimum sanction ₹",
                            className="dark-input",
                        ),
                        dcc.Input(
                            id="max-sanction",
                            type="number",
                            min=0,
                            step=100000,
                            value=0,
                            placeholder="Maximum sanction ₹",
                            className="dark-input",
                        ),

                        html.Button(
                            "↻  REFRESH DATA",
                            id="refresh-button",
                            className="action-button",
                        ),

                        html.Div("WORK LOOKUP", className="control-label"),
                        dcc.Input(
                            id="work-id-input",
                            type="text",
                            placeholder="Enter Work ID…",
                            debounce=True,
                            className="dark-input",
                        ),
                        html.Button(
                            "OPEN WORK PROFILE",
                            id="work-lookup-button",
                            className="action-button",
                        ),
                        html.Button(
                            "DOWNLOAD CURRENT QUEUE",
                            id="download-button",
                            className="action-button",
                        ),

                        html.Div(
                            id="connection-state",
                            className="connection-state",
                        ),

                        html.Div(
                            [
                                html.Div("FASTAPI", className="meta-label"),
                                html.Div(API_BASE, className="meta-value"),
                                html.Div("DASH", className="meta-label"),
                                html.Div(
                                    f"127.0.0.1:{DASH_PORT}",
                                    className="meta-value",
                                ),
                            ],
                            className="sidebar-meta",
                        ),
                    ],
                    className="sidebar",
                ),

                # ---------------- MAIN ----------------
                html.Main(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "PUBLIC WORKS · ANALYTICS CONTROL CENTER",
                                            className="eyebrow",
                                        ),
                                        html.H1(
                                            "MPLADS AI Monitor",
                                            className="page-title",
                                        ),
                                        html.P(
                                            "Explainable monitoring of financial, execution, consistency and anomaly signals.",
                                            className="page-subtitle",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="runtime-badges",
                                    className="runtime-badges",
                                ),
                            ],
                            className="topbar",
                        ),

                        html.Div(
                            id="scope-banner",
                            className="scope-banner",
                        ),

                        html.Div(
                            id="kpi-grid",
                            className="kpi-grid",
                        ),

                        html.Div(
                            id="secondary-kpi-grid",
                            className="secondary-kpi-grid",
                        ),

                        html.Div(
                            id="active-view",
                            className="view-container",
                        ),

                        html.Div(
                            id="selected-work-detail",
                            className="detail-shell",
                        ),

                        html.Div(
                            "MPLADS AI Monitor · Analytical decision-support interface · Signals are for review and verification, not adjudication.",
                            className="footer-note",
                        ),
                    ],
                    className="main-content",
                ),
            ],
            className="app-shell",
        ),
    ]
)


# ============================================================
# CSS — DARK EDITORIAL / ANALYTICS SYSTEM
# ============================================================

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        app.layout
    ]
) if False else app.layout

app.index_string = app.index_string.replace(
    "</head>",
    """
<style>
:root{
  --bg:#090D12;
  --panel:#10161E;
  --panel2:#141C25;
  --panel3:#0D131A;
  --border:#26313C;
  --border2:#1C2630;
  --text:#EEF4F8;
  --muted:#91A0AD;
  --muted2:#687681;
  --accent:#8FD0FF;
  --green:#79D6A8;
  --amber:#F1C16A;
  --red:#FF8585;
  --cyan:#6FE3E7;
  --shadow:0 18px 50px rgba(0,0,0,.22);
}
*{box-sizing:border-box}
html,body,#_dash-app-content{margin:0;min-height:100%;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}
body{background:
 radial-gradient(circle at 88% 4%,rgba(112,190,255,.08),transparent 24rem),
 radial-gradient(circle at 10% 18%,rgba(96,226,205,.035),transparent 25rem),
 var(--bg)}
.app-shell{display:flex;min-height:100vh}
.sidebar{position:sticky;top:0;width:286px;height:100vh;overflow-y:auto;padding:24px 18px;background:rgba(12,17,23,.96);border-right:1px solid var(--border);backdrop-filter:blur(18px);z-index:10}
.brand{padding:6px 8px 22px;border-bottom:1px solid var(--border2);margin-bottom:18px}
.brand-mark{font:700 30px/1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.06em}
.brand-sub{font:600 12px/1.2 Cascadia Mono,Consolas,monospace;letter-spacing:.18em;color:var(--accent);margin-top:7px}
.brand-micro{font:500 8px/1.5 Cascadia Mono,Consolas,monospace;letter-spacing:.12em;color:var(--muted2);margin-top:10px}
.control-label{font:700 9px/1 Cascadia Mono,Consolas,monospace;letter-spacing:.15em;color:#6F7F8C;margin:19px 3px 8px}
.dark-dropdown{margin-bottom:8px;position:relative}
.dark-dropdown .Select-control{background:#111821!important;border:1px solid #2A3743!important;border-radius:9px!important;min-height:42px!important;box-shadow:none!important}
.dark-dropdown .Select-control:hover{border-color:#4A6376!important}
.dark-dropdown .Select-value-label{color:#EAF2F7!important;font-weight:500!important}
.dark-dropdown .Select-placeholder{color:#83929E!important}
.dark-dropdown .Select-input>input{color:#EAF2F7!important;background:transparent!important}
.dark-dropdown .Select-arrow{border-top-color:#91A7B6!important}
.dark-dropdown .Select-clear{color:#91A7B6!important}
.dark-dropdown .Select-menu-outer{background:#10171F!important;border:1px solid #2A3743!important;border-radius:9px!important;box-shadow:0 18px 40px rgba(0,0,0,.45)!important;overflow:hidden!important;z-index:9999!important}
.dark-dropdown .Select-menu{background:#10171F!important}
.dark-dropdown .Select-option{background:#10171F!important;color:#DCE6ED!important;padding:10px 12px!important}
.dark-dropdown .Select-option.is-focused{background:#1B2935!important;color:#FFFFFF!important}
.dark-dropdown .Select-option.is-selected{background:#183246!important;color:#FFFFFF!important}
.dark-dropdown .VirtualizedSelectFocusedOption{background:#1B2935!important;color:#FFFFFF!important}
.dark-dropdown .VirtualizedSelectOption{background:#10171F!important;color:#DCE6ED!important}
/* React-Select v5 / current Dash selectors */
.dark-dropdown .Select__control{background:#111821!important;border:1px solid #2A3743!important;border-radius:9px!important;min-height:42px!important;box-shadow:none!important}
.dark-dropdown .Select__control:hover{border-color:#4A6376!important}
.dark-dropdown .Select__single-value{color:#EAF2F7!important}
.dark-dropdown .Select__placeholder{color:#83929E!important}
.dark-dropdown .Select__input-container,.dark-dropdown .Select__input-container input{color:#EAF2F7!important}
.dark-dropdown .Select__dropdown-indicator,.dark-dropdown .Select__clear-indicator{color:#91A7B6!important}
.dark-dropdown .Select__menu{background:#10171F!important;border:1px solid #2A3743!important;border-radius:9px!important;box-shadow:0 18px 40px rgba(0,0,0,.45)!important;overflow:hidden!important;z-index:9999!important}
.dark-dropdown .Select__option{background:#10171F!important;color:#DCE6ED!important;padding:10px 12px!important}
.dark-dropdown .Select__option--is-focused{background:#1B2935!important;color:#FFFFFF!important}
.dark-dropdown .Select__option--is-selected{background:#183246!important;color:#FFFFFF!important}
.dark-input{width:100%;background:#10171F;color:var(--text);border:1px solid var(--border);border-radius:9px;padding:10px 11px;margin:5px 0;outline:none}
.dark-input:focus{border-color:#4C7795;box-shadow:0 0 0 2px rgba(143,208,255,.08)}
.rc-slider{margin:8px 5px 26px}
.rc-slider-track{background:var(--accent)}
.rc-slider-handle{border-color:var(--accent);background:#0E151C}
.rc-slider-mark-text{color:#6F7F8C!important;font-size:9px}
.action-button{width:100%;border:1px solid #31546C;background:#132331;color:#DFF2FF;border-radius:10px;padding:11px 13px;font:700 10px Cascadia Mono,Consolas,monospace;letter-spacing:.08em;cursor:pointer;margin-top:16px}
.action-button:hover{background:#183044;border-color:#4E7897}
.action-button.compact{width:auto;margin:12px 0}
.connection-state{font:500 10px Cascadia Mono,Consolas,monospace;color:var(--green);padding:13px 3px 4px}
.sidebar-meta{border-top:1px solid var(--border2);margin-top:14px;padding-top:14px}
.meta-label{font:700 8px Cascadia Mono,Consolas,monospace;color:#667582;letter-spacing:.12em;margin-top:8px}
.meta-value{font:500 9px Cascadia Mono,Consolas,monospace;color:#9DABB6;word-break:break-all;margin-top:3px}
.main-content{width:calc(100% - 286px);max-width:1700px;margin:0 auto;padding:28px 34px 38px}
.topbar{display:flex;justify-content:space-between;gap:30px;align-items:flex-start}
.eyebrow,.section-kicker{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.16em;color:#71818D}
.page-title{font:700 43px/1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.055em;margin:8px 0 8px}
.page-subtitle{color:var(--muted);margin:0;max-width:720px;font-size:14px;line-height:1.6}
.runtime-badges{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}
.runtime-badge{border:1px solid var(--border);background:#10171F;border-radius:999px;padding:7px 10px;font:500 9px Cascadia Mono,Consolas,monospace;color:#AAB8C3}
.scope-banner{margin-top:22px}
.banner{border:1px solid #263F51;background:linear-gradient(135deg,#101C26,#0D151C);border-radius:14px;padding:13px 16px;box-shadow:var(--shadow)}
.banner.error{border-color:#63363A;background:#1C1215;color:#FFB7B7}
.banner-kicker{font:700 9px Cascadia Mono,Consolas,monospace;color:var(--accent);letter-spacing:.15em}
.banner-main{font:600 17px Bahnschrift,Segoe UI,sans-serif;margin-top:3px}
.banner-sub{font-size:11px;color:var(--muted);margin-top:3px}
.kpi-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-top:14px}
.kpi-card{min-height:116px;padding:15px 15px 13px;border:1px solid var(--border);border-radius:14px;background:linear-gradient(145deg,#111821,#0E141B);box-shadow:0 10px 30px rgba(0,0,0,.13)}
.kpi-card.warning{border-color:#4D422C}
.kpi-card.danger{border-color:#553136}
.kpi-label{font:700 8px Cascadia Mono,Consolas,monospace;letter-spacing:.12em;color:#71818D}
.kpi-value{font:700 24px/1.1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.04em;margin-top:12px}
.kpi-hint{font-size:9px;color:var(--muted2);margin-top:7px}

/* ============================================================
   OVERVIEW — CITIZEN-FIRST ANALYTICS SURFACE
   ============================================================ */
.overview-hero{
  display:grid;
  grid-template-columns:minmax(0,1.55fr) minmax(290px,.75fr);
  gap:14px;
  margin-bottom:14px;
}
.overview-intro{
  position:relative;
  overflow:hidden;
  min-height:230px;
  padding:24px 25px;
  border:1px solid #29465A;
  border-radius:18px;
  background:
    radial-gradient(circle at 92% 12%,rgba(125,191,232,.12),transparent 34%),
    linear-gradient(135deg,#111F2A 0%,#0E171F 58%,#0B1117 100%);
  box-shadow:0 18px 48px rgba(0,0,0,.20);
}
.overview-intro:after{
  content:"";
  position:absolute;
  right:-80px;
  bottom:-105px;
  width:260px;
  height:260px;
  border:1px solid rgba(143,208,255,.12);
  border-radius:50%;
  box-shadow:0 0 0 34px rgba(143,208,255,.025),0 0 0 68px rgba(143,208,255,.018);
}
.overview-kicker{font:700 9px Cascadia Mono,Consolas,monospace;letter-spacing:.18em;color:#8FD0FF}
.overview-headline{font:700 clamp(30px,3.5vw,48px)/.98 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.055em;margin:12px 0 10px;max-width:780px}
.overview-copy{font-size:13px;line-height:1.65;color:#AEBCC7;max-width:760px;margin:0}
.overview-meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}
.overview-chip{border:1px solid #2B4455;background:#0C151D;border-radius:999px;padding:7px 10px;color:#9EB1BF;font:600 9px Cascadia Mono,Consolas,monospace}
.overview-hero-panel{border:1px solid var(--border);border-radius:18px;background:linear-gradient(145deg,#111A23,#0C131A);padding:12px;box-shadow:var(--shadow);display:flex;align-items:center;justify-content:center}
.overview-hero-panel .js-plotly-plot{width:100%}
.overview-section-label{display:flex;justify-content:space-between;align-items:end;gap:14px;margin:24px 0 11px}
.overview-section-label h3{font:650 19px Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.025em;margin:3px 0 0}
.overview-section-label p{font-size:10px;color:#748692;margin:0;max-width:700px;line-height:1.5;text-align:right}
.overview-signal-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:14px}
.signal-card{min-height:90px;border:1px solid var(--border);border-radius:13px;background:#0E151C;padding:13px 14px}
.signal-card .signal-label{font:700 8px Cascadia Mono,Consolas,monospace;letter-spacing:.12em;color:#70818E}
.signal-card .signal-value{font:700 23px Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.04em;margin-top:9px}
.signal-card .signal-note{font-size:9px;color:#73838F;margin-top:5px}
.signal-card.warning{border-color:#51452D}.signal-card.danger{border-color:#58343A}.signal-card.accent{border-color:#2B4C62}
.secondary-kpi-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin-top:10px}
.secondary-kpi-grid:empty{display:none}
.secondary-kpi-grid .kpi-card{min-height:88px;padding:12px 13px;background:#0D141B;border-radius:12px}
.secondary-kpi-grid .kpi-value{font-size:20px;margin-top:8px}
.secondary-kpi-grid .kpi-hint{margin-top:5px}
.overview-queue-head{display:flex;justify-content:space-between;align-items:end;gap:15px;margin-bottom:10px}
.overview-queue-title{font:650 20px Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.025em;margin:3px 0 0}
.overview-queue-note{font-size:10px;color:#748692;line-height:1.5;max-width:620px;text-align:right}
@media(max-width:1050px){.overview-hero{grid-template-columns:1fr}.secondary-kpi-grid{grid-template-columns:repeat(3,1fr)}.overview-signal-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:850px){.secondary-kpi-grid{grid-template-columns:repeat(2,1fr)}.overview-signal-grid{grid-template-columns:1fr}.overview-section-label{display:block}.overview-section-label p,.overview-queue-note{text-align:left;margin-top:5px}}
.view-container{margin-top:28px}
.section-header{margin-bottom:15px}
.section-title{font:650 27px/1.1 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.035em;margin:5px 0 7px}
.section-note{font-size:11px;color:var(--muted);line-height:1.55}
.overview-grid,.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:14px}
.panel{border:1px solid var(--border);border-radius:15px;background:rgba(15,21,28,.86);padding:7px 9px;box-shadow:var(--shadow);margin-bottom:14px}
.queue-meta{font:500 10px Cascadia Mono,Consolas,monospace;color:#7E8D99;margin:4px 0 10px}
.detail-shell{margin-top:18px}
.detail-card{border:1px solid #31536A;border-radius:17px;background:linear-gradient(145deg,#101B24,#0E151C);padding:18px;box-shadow:var(--shadow)}
.detail-head{display:flex;justify-content:space-between;gap:20px;align-items:center}
.detail-title{font:650 25px/1.2 Bahnschrift,Segoe UI,sans-serif;letter-spacing:-.03em}
.detail-meta{font-size:11px;color:var(--muted);margin-top:8px}
.mono{font:500 9px Cascadia Mono,Consolas,monospace;color:#7E93A3;margin-top:8px}
.score-orb{width:110px;height:110px;border:1px solid #39566B;border-radius:50%;display:flex;flex-direction:column;justify-content:center;align-items:center;background:radial-gradient(circle,#172631,#0D151C 65%);flex:none}
.score-number{font:700 30px Bahnschrift,Segoe UI,sans-serif}
.score-band{font:700 8px Cascadia Mono,Consolas,monospace;color:var(--accent);letter-spacing:.1em;margin-top:4px}
.detail-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:16px 0}
.explanation{border-left:3px solid var(--accent);background:#111A22;border-radius:0 10px 10px 0;padding:13px;line-height:1.6;font-size:12px;margin:10px 0 15px}
.evidence{white-space:pre-wrap;max-height:350px;overflow:auto;background:#0A1016;border:1px solid var(--border2);border-radius:10px;padding:12px;color:#9FB0BC;font:10px/1.5 Cascadia Mono,Consolas,monospace}
.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.method-flow{font:11px/1.8 Cascadia Mono,Consolas,monospace;color:#AFC1CD;background:#0A1016;border:1px solid var(--border2);padding:15px;border-radius:10px;overflow:auto}
.band-row{display:flex;justify-content:space-between;border-bottom:1px solid var(--border2);padding:10px 4px;font-size:11px}
.band-row span{color:var(--muted)}
.band-low b{color:var(--green)}.band-medium b{color:var(--amber)}.band-high b{color:#F0A36B}.band-critical b{color:var(--red)}
.guardrail{border:1px solid #5B4930;background:#1C1710;color:#D9C39B;border-radius:12px;padding:14px;margin-top:14px;font-size:11px;line-height:1.6}
.empty-panel{border:1px dashed #2B3945;border-radius:13px;padding:30px;color:#74838F;text-align:center;font-size:12px;background:#0D131A}
.footer-note{text-align:center;color:#5E6B76;font:9px Cascadia Mono,Consolas,monospace;margin-top:30px;padding-top:18px;border-top:1px solid var(--border2)}
.js-plotly-plot .plotly .modebar{background:transparent!important}
@media(max-width:1200px){.kpi-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:850px){.sidebar{position:relative;width:100%;height:auto}.app-shell{display:block}.main-content{width:100%;padding:20px}.overview-grid,.chart-grid,.method-grid{grid-template-columns:1fr}.kpi-grid{grid-template-columns:repeat(2,1fr)}.topbar{display:block}.runtime-badges{justify-content:flex-start;margin-top:15px}}

/* HARD OVERRIDE: Dash React-Select / DCC dropdown surfaces */
.sidebar .dark-dropdown,
.sidebar .dark-dropdown > div,
.sidebar .dark-dropdown .Select,
.sidebar .dark-dropdown .Select-control,
.sidebar .dark-dropdown [class*="control"],
.sidebar .dark-dropdown [class*="Control"]{
  background-color:#111821 !important;
  background:#111821 !important;
  color:#EEF4F8 !important;
  border-color:#2A3743 !important;
  box-shadow:none !important;
}
.sidebar .dark-dropdown .Select-value,
.sidebar .dark-dropdown [class*="singleValue"],
.sidebar .dark-dropdown [class*="SingleValue"],
.sidebar .dark-dropdown [class*="placeholder"],
.sidebar .dark-dropdown [class*="Placeholder"]{
  background:transparent !important;
  color:#EAF2F7 !important;
  opacity:1 !important;
}
.sidebar .dark-dropdown [class*="placeholder"]{color:#8796A3 !important}
.sidebar .dark-dropdown .Select-menu-outer,
.sidebar .dark-dropdown [class*="menu"]{
  background:#0F161E !important;
  background-color:#0F161E !important;
  color:#EAF2F7 !important;
  border-color:#2A3743 !important;
  box-shadow:0 18px 45px rgba(0,0,0,.55) !important;
}
.sidebar .dark-dropdown .Select-option,
.sidebar .dark-dropdown [class*="option"]{
  background:#0F161E !important;
  color:#DDE7EE !important;
}
.sidebar .dark-dropdown .Select-option.is-focused,
.sidebar .dark-dropdown [class*="option--is-focused"]{
  background:#1B2D3C !important;
  color:#FFFFFF !important;
}
.sidebar .dark-dropdown .Select-option.is-selected,
.sidebar .dark-dropdown [class*="option--is-selected"]{
  background:#17384D !important;
  color:#FFFFFF !important;
}
.sidebar .dark-dropdown input,
.sidebar .dark-dropdown textarea{
  color:#EEF4F8 !important;
  background:transparent !important;
}
.sidebar .dark-dropdown svg{color:#9FB2C1 !important;fill:#9FB2C1 !important}
</style>
</head>"""
)


# ============================================================
# RUNTIME / FILTER CALLBACKS
# ============================================================

@app.callback(
    Output("health-store", "data"),
    Output("options-store", "data"),
    Output("national-store", "data"),
    Output("connection-state", "children"),
    Input("startup", "n_intervals"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=False,
)
def load_runtime(_startup: int, _refresh: int | None):
    try:
        health = api_get("/health")
        options = api_get("/api/v1/filter-options")
        national = api_get("/api/v1/dashboard-summary")
        source = health.get("data_source", "unknown") if isinstance(health, dict) else "unknown"
        return health, options, national, f"● API ONLINE · {source}"
    except requests.RequestException as exc:
        return (
            {"error": str(exc)},
            {},
            {"error": "FastAPI unavailable"},
            "● API OFFLINE · start FastAPI on :8000",
        )


@app.callback(
    Output("state-filter", "options"),
    Output("district-filter", "options"),
    Output("mp-filter", "options"),
    Output("constituency-filter", "options"),
    Output("category-filter", "options"),
    Output("status-filter", "options"),
    Output("risk-filter", "options"),
    Input("options-store", "data"),
)
def populate_filter_options(options):
    if not isinstance(options, dict):
        fallback = [{"label": "All", "value": "All"}]
        return (fallback,) * 7

    names = [
        "states",
        "districts",
        "mps",
        "constituencies",
        "work_categories",
        "work_statuses",
    ]
    result = [
        ["All"] + clean_options(options.get(name, []))
        for name in names
    ]
    risks = [
        "All"
    ] + [
        value
        for value in RISK_ORDER
        if value in clean_options(options.get("risk_categories", []))
    ]
    result.append(risks)

    return tuple(option_values(values) for values in result)


@app.callback(
    Output("district-filter", "value"),
    Input("state-filter", "value"),
    State("district-filter", "value"),
    prevent_initial_call=True,
)
def reset_district_on_state_change(state, current):
    # A state change invalidates the previously selected district.
    if state is None:
        return no_update
    return "All" if state != "All" else ("All" if current != "All" else no_update)


# ============================================================
# MAIN DATA SYNCHRONISATION
# ============================================================

@app.callback(
    Output("filtered-store", "data"),
    Output("works-store", "data"),
    Output("analytics-store", "data"),
    Output("scope-banner", "children"),
    Input("state-filter", "value"),
    Input("district-filter", "value"),
    Input("mp-filter", "value"),
    Input("constituency-filter", "value"),
    Input("category-filter", "value"),
    Input("status-filter", "value"),
    Input("risk-filter", "value"),
    Input("completion-filter", "value"),
    Input("search-filter", "value"),
    Input("min-sanction", "value"),
    Input("max-sanction", "value"),
    Input("risk-range", "value"),
    Input("refresh-button", "n_clicks"),
)
def load_monitoring_data(
    state,
    district,
    mp,
    constituency,
    category,
    status,
    risk,
    completion,
    search,
    min_sanction,
    max_sanction,
    risk_range,
    _refresh,
):
    if (
        min_sanction
        and max_sanction
        and min_sanction > 0
        and max_sanction > 0
        and min_sanction > max_sanction
    ):
        return (
            {"error": "Minimum sanction cannot exceed maximum sanction."},
            {"items": []},
            {},
            html.Div(
                "Correct the sanction range to continue.",
                className="banner error",
            ),
        )

    params = build_filter_params(
        state,
        district,
        mp,
        constituency,
        category,
        status,
        risk,
        completion,
        search,
        min_sanction,
        max_sanction,
        risk_range,
    )

    try:
        filtered = api_get("/api/v1/filtered-summary", params)

        works = api_get(
            "/api/v1/works",
            {
                **params,
                "page": 1,
                "page_size": QUEUE_LIMIT,
                "sort_by": "priority_score",
                "sort_order": "desc",
            },
        )

        # One authoritative filtered-analytics request keeps every graph
        # synchronized with exactly the same active scope as the KPI cards.
        # This avoids accidentally mixing filtered KPIs with national charts.
        analytics = api_get(
            "/api/v1/filtered-analytics",
            params,
        )

        total = safe_int(
            works.get("total_matching"),
            safe_int(filtered.get("total_works")),
        )

        return (
            filtered,
            works,
            analytics,
            html.Div(
                [
                    html.Div(
                        "FILTERED MONITORING UNIVERSE",
                        className="banner-kicker",
                    ),
                    html.Div(
                        f"{total:,} works in current scope",
                        className="banner-main",
                    ),
                    html.Div(
                        f"High/Critical signal rate: "
                        f"{safe_float(filtered.get('high_or_critical_rate_pct')):.1f}%"
                        f" · Queue capped at {QUEUE_LIMIT:,}",
                        className="banner-sub",
                    ),
                ],
                className="banner",
            ),
        )

    except requests.RequestException as exc:
        return (
            {"error": str(exc)},
            {"items": []},
            {},
            html.Div(
                "FastAPI could not be reached. Start backend on 127.0.0.1:8000.",
                className="banner error",
            ),
        )


@app.callback(
    Output("kpi-grid", "children"),
    Output("secondary-kpi-grid", "children"),
    Output("runtime-badges", "children"),
    Input("filtered-store", "data"),
    Input("health-store", "data"),
    Input("view-mode", "value"),
)
def render_kpis(filtered, health, view):
    if not isinstance(filtered, dict) or filtered.get("error"):
        return (
            [kpi_card("STATUS", "API unavailable", "Start FastAPI on :8000", "danger")],
            [],
            [],
        )

    primary = [
        kpi_card("WORKS", f"{safe_int(filtered.get('total_works')):,}", "current scope"),
        kpi_card("COMPLETED", f"{safe_int(filtered.get('completed_works')):,}", "completed works"),
        kpi_card("OPEN", f"{safe_int(filtered.get('open_works')):,}", "open works"),
        kpi_card("SANCTIONED", inr(filtered.get("sanctioned_amount")), "recorded scope"),
        kpi_card("EXPENDITURE", inr(filtered.get("total_expenditure")), "recorded expenditure"),
        kpi_card("UTILISATION", pct(filtered.get("portfolio_utilization_pct")), "expenditure ÷ sanctioned"),
    ]

    secondary = [
        kpi_card("COMPLETION RATE", pct(filtered.get("completion_rate_pct")), "completed ÷ works"),
        kpi_card("MEAN RISK", f"{safe_float(filtered.get('mean_risk')):.1f}", "composite score / 100"),
        kpi_card("HIGH / CRITICAL", f"{safe_int(filtered.get('high_or_critical')):,}", pct(filtered.get("high_or_critical_rate_pct")), "warning"),
        kpi_card("CRITICAL", f"{safe_int(filtered.get('critical')):,}", "requires verification", "danger"),
        kpi_card("OVERDUE OPEN", f"{safe_int(filtered.get('overdue_open_works')):,}", "timing signal"),
        kpi_card("DUPLICATE CANDIDATES", f"{safe_int(filtered.get('duplicate_candidates')):,}", "similarity signal"),
    ]

    # Keep secondary risk indicators visible on analytical views while the
    # Overview gets a cleaner, citizen-facing hierarchy.
    if view != "Overview":
        secondary.extend([
            kpi_card("MULTI-SIGNAL", f"{safe_int(filtered.get('multi_signal_cases')):,}", "2+ independent signals"),
        ])

    source = health.get("data_source", "unknown") if isinstance(health, dict) else "unknown"
    badges = [
        html.Div("● API ONLINE", className="runtime-badge"),
        html.Div(f"SOURCE · {source}", className="runtime-badge"),
        html.Div(f"DASH · {APP_VERSION}", className="runtime-badge"),
    ]
    return primary, secondary, badges


# ============================================================
# VIEW RENDERING
# ============================================================

def overview_view(works, analytics):
    scope = analytics.get("scope", {}) if isinstance(analytics, dict) else {}
    return html.Div([
        html.Div([
            html.Div([
                html.Div("PORTFOLIO AT A GLANCE", className="overview-kicker"),
                html.Div("Understand the whole picture before opening a work.", className="overview-headline"),
                html.P(
                    "A filtered, evidence-oriented view of MPLADS works: scale first, then project status, financial flow, risk signals and geographic concentration. Every analytical card below uses the same active filter scope.",
                    className="overview-copy",
                ),
                html.Div([
                    html.Div("SCALE", className="overview-chip"),
                    html.Div("MONEY", className="overview-chip"),
                    html.Div("EXECUTION", className="overview-chip"),
                    html.Div("RISK SIGNALS", className="overview-chip"),
                    html.Div("HUMAN REVIEW", className="overview-chip"),
                ], className="overview-meta"),
            ], className="overview-intro"),
            html.Div(
                dcc.Graph(figure=overview_status_figure(scope), config={"displaylogo":False}),
                className="overview-hero-panel",
            ),
        ], className="overview-hero"),

        html.Div([
            html.Div([
                html.Div("PORTFOLIO STATUS", className="overview-section-label"),
                html.Div(
                    dcc.Graph(figure=overview_utilization_figure(scope), config={"displaylogo":False}),
                    className="panel",
                ),
            ]),
            html.Div([
                html.Div("RISK PROFILE", className="overview-section-label"),
                html.Div(
                    dcc.Graph(figure=overview_risk_spectrum_figure(analytics.get("risk_final")), config={"displaylogo":False}),
                    className="panel",
                ),
            ]),
        ], className="chart-grid"),

        html.Div(
            [
                kpi_card("MULTI-SIGNAL", f"{safe_int(scope.get('multi_signal_cases')):,}", "2+ independent signals", "accent"),
                kpi_card("OVERDUE OPEN", f"{safe_int(scope.get('overdue_open_works')):,}", "timing signal"),
                kpi_card("DUPLICATE CANDIDATES", f"{safe_int(scope.get('duplicate_candidates')):,}", "similarity signal"),
            ],
            className="overview-signal-grid",
        ),

        html.Div([
            html.Div(
                [html.Div("FINANCIAL + EXECUTION", className="overview-section-label"),
                 html.Div(dcc.Graph(figure=overview_financial_flow_figure(analytics.get("time")), config={"displaylogo":False}), className="panel")]
            ),
            html.Div(
                [html.Div("SIGNAL PROFILE", className="overview-section-label"),
                 html.Div(dcc.Graph(figure=overview_signal_figure(analytics.get("risk_reasons")), config={"displaylogo":False}), className="panel")]
            ),
        ], className="chart-grid"),

        html.Div([
            html.Div(dcc.Graph(figure=overview_state_attention_figure(analytics.get("state")), config={"displaylogo":False}), className="panel"),
            html.Div([
                html.Div("READ THIS VIEW", className="section-kicker"),
                html.Div("From population to review", className="overview-queue-title"),
                html.P(
                    "Start with the six primary KPIs. Project status shows completed versus open works. Utilisation compares recorded expenditure with sanctioned scope. The risk spectrum shows how the composite score is distributed. Signal prevalence counts analytical flags; it does not establish misconduct.",
                    className="section-note",
                ),
                html.Div([
                    html.Div([html.B("1 · Scope"), html.Span(" — what is in the current filter")], className="band-row"),
                    html.Div([html.B("2 · Flow"), html.Span(" — sanctioned → expenditure")], className="band-row"),
                    html.Div([html.B("3 · Execution"), html.Span(" — completion and open work")], className="band-row"),
                    html.Div([html.B("4 · Signals"), html.Span(" — independent analytical flags")], className="band-row"),
                    html.Div([html.B("5 · Review"), html.Span(" — inspect individual works")], className="band-row"),
                ], style={"marginTop":"14px"}),
            ], className="panel", style={"padding":"18px"}),
        ], className="chart-grid"),

        html.Div([
            html.Div([
                html.Div("PRIORITY INVESTIGATION QUEUE", className="section-kicker"),
                html.Div("Review candidates from the active scope", className="overview-queue-title"),
            ]),
            html.Div("The queue is capped for usability; analytical charts above use the complete filtered population.", className="overview-queue-note"),
        ], className="overview-queue-head"),
        html.Div(queue_table(works, table_id="overview-queue-table"), className="panel"),
    ])

def risk_view(works, analytics):
    return html.Div([
        section_header("DETECT + EXPLAIN", "Risk Intelligence", "Separate the composite score from its underlying rule and machine-learning signals."),
        html.Div([
            html.Div(dcc.Graph(figure=risk_basis_figure(analytics), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=risk_score_histogram(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=risk_reason_rate_figure(analytics.get("risk_reasons")), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=priority_risk_figure(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=completion_risk_figure(works), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=state_figure(analytics.get("state")), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div(dcc.Graph(figure=sector_risk_figure(analytics.get("sector")), config={"displaylogo":False}), className="panel"),
        html.Div([html.Div("CURRENT INVESTIGATION QUEUE", className="section-kicker"), html.Div(f"{len(works):,} displayed records", className="queue-meta"), queue_table(works, table_id="risk-queue-table")], className="panel"),
    ])
def finance_view(works, analytics):
    return html.Div([
        section_header("MONEY + EXECUTION", "Financial & Execution Intelligence", "Track sanctioned scope, recorded expenditure, utilisation, completion and execution age without conflating signals with audit conclusions."),
        html.Div([
            html.Div(dcc.Graph(figure=time_figure(analytics.get("time")), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=time_risk_figure(analytics.get("time")), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=financial_gap_figure(works), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=utilization_distribution_figure(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=execution_age_figure(works), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=execution_figure(works), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=sector_completion_figure(analytics.get("sector")), config={"displaylogo":False}), className="panel"),
            html.Div(dcc.Graph(figure=sector_figure(analytics.get("sector")), config={"displaylogo":False}), className="panel"),
        ], className="chart-grid"),
    ])
def geography_view(works, analytics):
    district_data=analytics.get("district")
    district_frame=to_polars(district_data)
    if district_frame.is_empty() or not {"ida","high_or_critical_rate_pct"}.issubset(district_frame.columns):
        district_panel=empty_panel("District analytics unavailable for the active scope.")
    else:
        district_frame=district_frame.with_columns(pl.col("high_or_critical_rate_pct").cast(pl.Float64,strict=False).fill_null(0)).sort("high_or_critical_rate_pct",descending=True).head(30).sort("high_or_critical_rate_pct")
        rows=district_frame.to_dicts(); fig=base_figure("District / IDA high-critical signal rate",540)
        fig.add_trace(go.Bar(x=[safe_float(r.get("high_or_critical_rate_pct")) for r in rows],y=[str(r.get("ida","Unknown")) for r in rows],orientation="h",customdata=[[safe_int(r.get("works")),safe_int(r.get("completed_works"))] for r in rows],hovertemplate="%{y}<br>High/Critical: %{x:.1f}%<br>Works: %{customdata[0]:,}<br>Completed: %{customdata[1]:,}<extra></extra>"))
        fig.update_layout(xaxis_title="Rate (%)",yaxis_title="")
        district_panel=html.Div(dcc.Graph(figure=fig,config={"displaylogo":False}),className="panel")
    return html.Div([
        section_header("WHERE", "Geographic Intelligence", "Compare descriptive geographic concentration, exposure and completion patterns. Map points are shown only when valid source coordinates exist."),
        html.Div([
            html.Div(dcc.Graph(figure=state_figure(analytics.get("state")),config={"displaylogo":False}),className="panel"),
            html.Div(dcc.Graph(figure=state_exposure_figure(analytics.get("state")),config={"displaylogo":False}),className="panel"),
        ],className="chart-grid"),
        html.Div([
            html.Div(dcc.Graph(figure=state_completion_figure(analytics.get("state")),config={"displaylogo":False}),className="panel"),
            district_panel,
        ],className="chart-grid"),
        html.Div([
            html.Div("WORK-LEVEL RISK MAP",className="section-kicker"),
            html.Div("The map uses the filtered priority queue and valid latitude/longitude fields. It is descriptive, not a geographic proof of anomaly.",className="section-note"),
            dcc.Graph(figure=map_figure_from_works(works),config={"displaylogo":False}),
        ],className="panel"),
    ])
def map_figure_from_works(works):
    frame = to_polars(works)
    if frame.is_empty():
        return empty_figure("No work records available for mapping", 560)

    lat_col = (
        "lat"
        if "lat" in frame.columns
        else "latitude"
        if "latitude" in frame.columns
        else None
    )
    lon_col = (
        "lon"
        if "lon" in frame.columns
        else "longitude"
        if "longitude" in frame.columns
        else None
    )

    if not lat_col or not lon_col:
        return empty_figure(
            "No latitude/longitude fields in the current API response",
            560,
        )

    points = []
    for row in frame.to_dicts():
        lat = safe_float(row.get(lat_col), float("nan"))
        lon = safe_float(row.get(lon_col), float("nan"))
        if (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        ):
            points.append((lat, lon, row))

    if not points:
        return empty_figure("No valid coordinates available", 560)

    fig = go.Figure(
        go.Scattermapbox(
            lat=[p[0] for p in points],
            lon=[p[1] for p in points],
            mode="markers",
            marker={"size": 9},
            customdata=[
                [
                    p[2].get("work_uid"),
                    p[2].get("risk_category"),
                    safe_float(p[2].get("final_risk_score")),
                ]
                for p in points
            ],
            hovertemplate=(
                "Work %{customdata[0]}"
                "<br>Band: %{customdata[1]}"
                "<br>Risk: %{customdata[2]:.1f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        mapbox={"style": "open-street-map", "center": {"lat": 22.5, "lon": 79.0}, "zoom": 3.8},
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def explorer_view(works):
    total = len(works)
    return html.Div(
        [
            section_header(
                "EXPLAIN + REVIEW",
                "Work Explorer",
                "Select a row to open the work-level evidence profile.",
            ),
            html.Div(
                [
                    html.Div(
                        f"{total:,} records returned by the active API query.",
                        className="queue-meta",
                    ),
                    queue_table(works, selectable=True, table_id="explorer-queue-table"),
                ],
                className="panel",
            ),
        ]
    )


def methodology_view(health):
    health = health if isinstance(health, dict) else {}
    status = [
        {"Item": "API version", "Value": health.get("version")},
        {"Item": "Data source", "Value": health.get("data_source")},
        {"Item": "Works loaded", "Value": safe_int(health.get("works_loaded"))},
        {"Item": "Data as-of", "Value": health.get("data_as_of")},
        {"Item": "Dashboard", "Value": APP_VERSION},
    ]

    return html.Div(
        [
            section_header(
                "TRUST",
                "Methodology & Guardrails",
                "The interface exposes analytical signals while keeping human verification in the loop.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "ANALYTICAL HIERARCHY",
                                className="section-kicker",
                            ),
                            html.Pre(
                                "Official source data\n"
                                "      ↓\n"
                                "Validation + feature engineering\n"
                                "      ↓\n"
                                "Rule signals + peer statistics\n"
                                "      ↓\n"
                                "Isolation Forest anomaly signal\n"
                                "      ↓\n"
                                "Composite analytical risk\n"
                                "      ↓\n"
                                "Financial exposure + evidence support\n"
                                "      ↓\n"
                                "Investigation priority\n"
                                "      ↓\n"
                                "Human review",
                                className="method-flow",
                            ),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "SCORE SEMANTICS",
                                className="section-kicker",
                            ),
                            html.P(
                                "ML anomaly percentile is a population-relative "
                                "abnormality measure, not a probability of fraud."
                            ),
                            html.P(
                                "Final risk is a composite analytical score. "
                                "Its weights are project design parameters, not "
                                "official MPLADS weights."
                            ),
                            html.P(
                                "Investigation priority is a triage measure "
                                "combining analytical risk, financial exposure "
                                "and evidence support."
                            ),
                            html.Div(
                                "RISK BANDS",
                                className="section-kicker",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.B(band),
                                            html.Span(RISK_RANGES[band]),
                                        ],
                                        className=f"band-row band-{band.lower()}",
                                    )
                                    for band in RISK_ORDER
                                ]
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="method-grid",
            ),
            html.Div(
                [
                    html.Div(
                        "RUNTIME STATUS",
                        className="section-kicker",
                    ),
                    dash_table.DataTable(
                        data=status,
                        columns=[
                            {"name": "Item", "id": "Item"},
                            {"name": "Value", "id": "Value"},
                        ],
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#151D26",
                            "color": "#A7B5C2",
                            "fontWeight": "700",
                        },
                        style_cell={
                            "backgroundColor": "#0F151C",
                            "color": "#DCE5EC",
                            "padding": "10px",
                            "border": "1px solid #202A34",
                        },
                    ),
                ],
                className="panel",
            ),
            html.Div(
                "Guardrail: an anomaly is a review signal; a similarity candidate "
                "is not a confirmed duplicate; data-integrity issues are data-quality "
                "signals; timing benchmarks require rule-specific verification; "
                "the system does not establish fraud, corruption, legal "
                "non-compliance, misappropriation or causality.",
                className="guardrail",
            ),
        ]
    )


@app.callback(
    Output("active-view", "children"),
    Input("view-mode", "value"),
    Input("filtered-store", "data"),
    Input("works-store", "data"),
    Input("analytics-store", "data"),
    Input("health-store", "data"),
)
def render_view(view, filtered, works_payload, analytics, health):
    if not isinstance(filtered, dict) or filtered.get("error"):
        return empty_panel("Monitoring data is unavailable.")

    works = records(works_payload)
    analytics = analytics if isinstance(analytics, dict) else {}

    if view == "Risk Intelligence":
        return risk_view(works, analytics)

    if view == "Financial & Execution":
        return finance_view(works, analytics)

    if view == "Geography":
        return geography_view(works, analytics)

    if view == "Work Explorer":
        return explorer_view(works)

    if view == "Methodology":
        return methodology_view(health)

    return overview_view(works, analytics)


# ============================================================
# WORK DRILL-DOWN
# ============================================================

# ============================================================
# WORK PROFILE LOOKUP
# ============================================================

@app.callback(
    Output("selected-work-detail", "children"),
    Input("work-lookup-button", "n_clicks"),
    State("work-id-input", "value"),
    prevent_initial_call=True,
)
def lookup_work(_clicks, work_uid):
    uid = str(work_uid or "").strip()

    if not uid:
        return empty_panel("Enter a Work ID to open its evidence profile.")

    try:
        detail = api_get(f"/api/v1/works/{uid}")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            return empty_panel(f"No work was found for Work ID: {uid}")
        return empty_panel(f"Work profile request failed: HTTP {status or 'error'}")
    except requests.RequestException as exc:
        return empty_panel(f"FastAPI request failed: {exc}")

    row = detail.get("work", {}) if isinstance(detail, dict) else {}
    components = detail.get("risk_components", {}) if isinstance(detail, dict) else {}
    evidence = detail.get("evidence", {}) if isinstance(detail, dict) else {}

    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            evidence = {}

    return html.Div(
        [
            html.Div("WORK INTELLIGENCE", className="section-kicker"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(str(row.get("work", uid)), className="detail-title"),
                            html.Div(f"Work ID · {uid}", className="mono"),
                            html.Div(
                                f"{row.get('work_category', 'Unclassified')} · "
                                f"{row.get('state', 'Unknown')} · "
                                f"{row.get('ida', 'Unknown')}",
                                className="detail-meta",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                f"{safe_float(components.get('final_risk_score')):.1f}",
                                className="score-number",
                            ),
                            html.Div(
                                str(row.get("risk_category", "UNSCORED")),
                                className="score-band",
                            ),
                        ],
                        className="score-orb",
                    ),
                ],
                className="detail-head",
            ),
            html.Div(
                [
                    kpi_card("RECOMMENDED", inr(row.get("recommended_amount"))),
                    kpi_card("SANCTIONED", inr(row.get("sanction_amount"))),
                    kpi_card("EXPENDITURE", inr(row.get("total_expenditure"))),
                    kpi_card("UTILISATION", pct(row.get("utilization_pct"))),
                ],
                className="detail-kpis",
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            figure=component_figure(components),
                            config={"displaylogo": False},
                        ),
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.Div("WHY FLAGGED?", className="section-kicker"),
                            html.Div(
                                str(row.get("risk_explanation") or "No strong rule-based signal."),
                                className="explanation",
                            ),
                            html.Div("EVIDENCE", className="section-kicker"),
                            html.Pre(
                                json.dumps(
                                    evidence,
                                    indent=2,
                                    ensure_ascii=False,
                                    default=str,
                                )[:6000],
                                className="evidence",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="chart-grid",
            ),
        ],
        className="detail-card",
    )


# ============================================================
# DOWNLOAD
# ============================================================

@app.callback(
    Output("download-queue", "data"),
    Input("download-button", "n_clicks"),
    State("works-store", "data"),
    prevent_initial_call=True,
)
def download_queue(_clicks, works_payload):
    works = records(works_payload)
    frame = to_polars(works)

    if frame.is_empty():
        return no_update

    return dcc.send_string(
        frame.write_csv(),
        "mplads_current_queue.csv",
    )


# ============================================================
# ENTRY POINT
# ============================================================
# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("MPLADS AI MONITOR — DASH")
    print(f"Dashboard : http://{DASH_HOST}:{DASH_PORT}")
    print(f"FastAPI   : {API_BASE}")
    print(f"Version   : {APP_VERSION}")
    print("=" * 72)

    app.run(
        host=DASH_HOST,
        port=DASH_PORT,
        debug=False,
        dev_tools_hot_reload=False,
    )
