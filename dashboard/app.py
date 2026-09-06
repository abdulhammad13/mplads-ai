from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# PAGE / CONFIGURATION
# ============================================================

API_BASE = os.getenv("MPLADS_API_URL", "http://127.0.0.1:8000").rstrip("/")
APP_VERSION = "4.0"
REQUEST_TIMEOUT = 30
PRIORITY_PAGE_SIZE = 500

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_RANGES = {
    "LOW": "0–<25",
    "MEDIUM": "25–<50",
    "HIGH": "50–<75",
    "CRITICAL": "75–100",
}
RISK_SYMBOL = {
    "LOW": "✓",
    "MEDIUM": "!",
    "HIGH": "▲",
    "CRITICAL": "⚠",
}

st.set_page_config(
    page_title="MPLADS AI Monitor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #0b0f14;
        --panel: #11161d;
        --panel-2: #151b23;
        --border: #27303a;
        --muted: #9aa6b2;
        --text: #eef3f7;
        --accent: #7cc4ff;
        --positive: #63d39c;
        --warning: #f2bd64;
        --danger: #ff8a8a;
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.0rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #0d1218;
        border-right: 1px solid var(--border);
    }

    .hero {
        border: 1px solid var(--border);
        background: linear-gradient(135deg, #121922 0%, #0e1319 100%);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin: 0;
    }

    .hero-sub {
        color: var(--muted);
        margin-top: 0.45rem;
        font-size: 0.95rem;
    }

    .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: .9rem;
    }

    .status-pill {
        border: 1px solid var(--border);
        background: #0f151c;
        border-radius: 999px;
        padding: .32rem .65rem;
        color: #d6dee6;
        font-size: .78rem;
    }

    .section-note {
        color: var(--muted);
        font-size: .82rem;
        margin-top: -.3rem;
        margin-bottom: .75rem;
    }

    .insight {
        border-left: 3px solid var(--accent);
        background: #101720;
        padding: .75rem .9rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: .55rem;
    }

    .risk-card {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: .85rem 1rem;
        background: var(--panel);
        min-height: 120px;
    }

    .risk-value {
        font-size: 1.65rem;
        font-weight: 740;
        margin-top: .1rem;
    }

    .small-muted {
        color: var(--muted);
        font-size: .78rem;
    }

    .selected-work {
        border: 1px solid #31536f;
        background: #0e1822;
        border-radius: 16px;
        padding: 1rem;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        padding: .75rem .85rem;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: .78rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    .footer-note {
        color: #74808c;
        text-align: center;
        font-size: .75rem;
        padding-top: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

class APIError(RuntimeError):
    pass


@st.cache_data(ttl=30, show_spinner=False)
def api_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    try:
        response = requests.get(
            f"{API_BASE}{endpoint}",
            params=params or {},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            try:
                detail = f" | {exc.response.text[:250]}"
            except Exception:
                pass
        raise APIError(f"API request failed: {endpoint}{detail}") from exc


@st.cache_data(ttl=30, show_spinner=False)
def api_get_uncached_label(endpoint: str, params: tuple[tuple[str, Any], ...] = ()) -> Any:
    return api_get(endpoint, dict(params))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{safe_float(value):,.1f}%"


def inr(value: Any) -> str:
    amount = safe_float(value, float("nan"))
    if pd.isna(amount):
        return "—"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1e7:
        return f"{sign}₹{amount/1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"{sign}₹{amount/1e5:,.2f} L"
    return f"{sign}₹{amount:,.0f}"


def risk_badge(category: Any) -> str:
    value = str(category).upper() if category is not None else "UNSCORED"
    symbol = RISK_SYMBOL.get(value, "•")
    return f"{symbol} {value}"


def clean_options(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if text and text.casefold() not in seen:
            cleaned.append(text)
            seen.add(text.casefold())
    return sorted(cleaned, key=str.casefold)


def build_filter_params(
    state: str,
    district: str,
    mp: str,
    constituency: str,
    category: str,
    work_status: str,
    risk_category: str,
    completion: str,
    search: str,
    min_sanction: float | None,
    max_sanction: float | None,
    risk_range: tuple[float, float],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "min_risk": risk_range[0],
        "max_risk": risk_range[1],
    }
    for key, value in {
        "state": state,
        "district": district,
        "mp": mp,
        "constituency": constituency,
        "work_category": category,
        "work_status": work_status,
        "risk_category": risk_category,
        "completion_status": completion,
    }.items():
        if value and value != "All":
            params[key] = value
    if search.strip():
        params["search"] = search.strip()
    if min_sanction is not None:
        params["min_sanction"] = min_sanction
    if max_sanction is not None:
        params["max_sanction"] = max_sanction
    return params


def extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return items
    if isinstance(payload, list):
        return payload
    return []


def parse_evidence(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


# ============================================================
# CONNECTIVITY / GLOBAL DATA
# ============================================================

try:
    health = api_get("/health")
    options = api_get("/api/v1/filter-options")
    national_summary = api_get("/api/v1/dashboard-summary")
except Exception as error:
    st.error("🔴 MPLADS API is unavailable.")
    st.code("python -m uvicorn backend.main:app --reload")
    st.caption(str(error))
    st.stop()


# ============================================================
# SIDEBAR / CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("## 🎛️ Monitoring Controls")
    st.caption("Filters affect the KPI layer and investigation queue.")

    view_mode = st.selectbox(
        "Monitoring view",
        ["Executive", "Financial", "Execution", "Risk", "Geography", "Investigation"],
        index=0,
    )

    st.divider()

    state_options = ["All"] + clean_options(options.get("states", []))
    district_options = ["All"] + clean_options(options.get("districts", []))
    mp_options = ["All"] + clean_options(options.get("mps", []))
    constituency_options = ["All"] + clean_options(options.get("constituencies", []))
    category_options = ["All"] + clean_options(options.get("work_categories", []))
    status_options = ["All"] + clean_options(options.get("work_statuses", []))
    risk_options = ["All"] + [x for x in RISK_ORDER if x in clean_options(options.get("risk_categories", []))]

    state = st.selectbox("State", state_options)
    district = st.selectbox("District / IDA", district_options)
    mp = st.selectbox("Member of Parliament", mp_options)
    constituency = st.selectbox("Constituency", constituency_options)
    category = st.selectbox("Work category", category_options)
    work_status = st.selectbox("Work status", status_options)
    risk_category = st.selectbox("Risk category", ["All"] + risk_options)
    completion = st.selectbox("Completion", ["All", "Completed", "Open"])

    search = st.text_input(
        "🔎 Search",
        placeholder="Work, description, MP, constituency or ID",
    )

    st.subheader("Risk / Value")
    risk_range = st.slider("Final risk score", 0, 100, (0, 100))

    min_sanction = st.number_input(
        "Minimum sanction (₹)",
        min_value=0.0,
        value=0.0,
        step=100000.0,
    )
    max_sanction = st.number_input(
        "Maximum sanction (₹)",
        min_value=0.0,
        value=0.0,
        step=100000.0,
        help="Keep at 0 for no upper limit.",
    )

    st.divider()

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"API: {API_BASE}")

filter_params = build_filter_params(
    state,
    district,
    mp,
    constituency,
    category,
    work_status,
    risk_category,
    completion,
    search,
    min_sanction if min_sanction > 0 else None,
    max_sanction if max_sanction > 0 else None,
    risk_range,
)

try:
    filtered_summary = api_get("/api/v1/filtered-summary", filter_params)
    works_payload = api_get(
        "/api/v1/works",
        {**filter_params, "page": 1, "page_size": PRIORITY_PAGE_SIZE, "sort_by": "priority_score", "sort_order": "desc"},
    )
except Exception as error:
    st.error(f"Unable to retrieve filtered analytical data: {error}")
    st.stop()

priority_items = extract_items(works_payload)
priority_df = pd.DataFrame(priority_items)

total_matching = safe_int(
    works_payload.get("total_matching") if isinstance(works_payload, dict) else filtered_summary.get("total_works"),
    safe_int(filtered_summary.get("total_works")),
)


# ============================================================
# HEADER
# ============================================================

source = health.get("data_source", "unknown")
data_as_of = health.get("data_as_of")
loaded = safe_int(health.get("works_loaded"), safe_int(national_summary.get("total_works")))

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🏛️ MPLADS AI Monitor</div>
        <div class="hero-sub">
            Explainable, AI-assisted monitoring of public works, expenditure, execution,
            anomaly signals and investigation priorities.
        </div>
        <div class="status-strip">
            <span class="status-pill">Dataset: {loaded:,} works</span>
            <span class="status-pill">Source: {source}</span>
            <span class="status-pill">As-of: {data_as_of or 'not declared'}</span>
            <span class="status-pill">App v{APP_VERSION}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if filter_params:
    st.info(
        f"Filtered monitoring view: **{total_matching:,} matching works**. "
        f"The investigation queue displays up to **{len(priority_df):,} highest-priority works**."
    )
else:
    st.success(
        f"National monitoring view: **{total_matching:,} works** loaded. "
        f"Investigation queue displays up to **{len(priority_df):,} priority works**."
    )


# ============================================================
# SHARED KPI BLOCK
# ============================================================

st.markdown("### Portfolio snapshot")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Works", f"{safe_int(filtered_summary.get('total_works')):,}")
k2.metric("Completed", f"{safe_int(filtered_summary.get('completed_works')):,}")
k3.metric("Open", f"{safe_int(filtered_summary.get('open_works')):,}")
k4.metric("Sanctioned", inr(filtered_summary.get("sanctioned_amount")))
k5.metric("Expenditure", inr(filtered_summary.get("total_expenditure")))
k6.metric("Utilisation", pct(filtered_summary.get("portfolio_utilization_pct")))

k7, k8, k9, k10, k11, k12 = st.columns(6)
k7.metric("High / Critical", f"{safe_int(filtered_summary.get('high_or_critical')):,}")
k8.metric("Critical", f"{safe_int(filtered_summary.get('critical')):,}")
k9.metric("High/Critical rate", pct(filtered_summary.get("high_or_critical_rate_pct")))
k10.metric("Overdue open", f"{safe_int(filtered_summary.get('overdue_open_works')):,}")
k11.metric("Duplicate candidates", f"{safe_int(filtered_summary.get('duplicate_candidates')):,}")
k12.metric("Multi-signal", f"{safe_int(filtered_summary.get('multi_signal_cases')):,}")

st.divider()


# ============================================================
# DATA QUALITY / SCOPE MESSAGE
# ============================================================

national_high_rate = safe_float(national_summary.get("high_or_critical_rate_pct"))
filtered_high_rate = safe_float(filtered_summary.get("high_or_critical_rate_pct"))
rate_delta = filtered_high_rate - national_high_rate

left_scope, right_scope = st.columns([1.3, 1])
with left_scope:
    st.markdown("#### Monitoring context")
    scope_text = (
        f"The current view contains **{safe_int(filtered_summary.get('total_works')):,} works**. "
        f"{safe_int(filtered_summary.get('high_or_critical')):,} are HIGH/CRITICAL, giving a rate of "
        f"**{filtered_high_rate:,.1f}%**."
    )
    st.markdown(f'<div class="insight">{scope_text}</div>', unsafe_allow_html=True)

    if national_summary.get("total_works"):
        direction = "above" if rate_delta >= 0 else "below"
        comparison = (
            f"This is **{abs(rate_delta):,.1f} percentage points {direction}** the loaded national baseline "
            f"of {national_high_rate:,.1f}%."
        )
        st.markdown(f'<div class="insight">Benchmark: {comparison}</div>', unsafe_allow_html=True)

with right_scope:
    st.markdown("#### Exposure at a glance")
    sanctioned = safe_float(filtered_summary.get("sanctioned_amount"))
    expenditure = safe_float(filtered_summary.get("total_expenditure"))
    remaining = sanctioned - expenditure
    st.markdown(
        f'<div class="risk-card"><div class="small-muted">Sanctioned minus recorded expenditure</div>'
        f'<div class="risk-value">{inr(remaining)}</div>'
        f'<div class="small-muted">This is a calculated portfolio gap, not an official balance statement.</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# DATASETS USED THROUGHOUT THE APP
# ============================================================

try:
    risk_dist = pd.DataFrame(api_get("/api/v1/risk-distribution", {"risk_basis": "final"}))
    risk_reasons = pd.DataFrame(api_get("/api/v1/risk-reasons"))
    time_df = pd.DataFrame(api_get("/api/v1/time-series"))
    state_df = pd.DataFrame(api_get("/api/v1/state-analytics"))
    district_df = pd.DataFrame(
        api_get(
            "/api/v1/district-analytics",
            {"state": state} if state != "All" else {},
        )
    )
    sector_df = pd.DataFrame(api_get("/api/v1/sector-analytics"))
except Exception as error:
    risk_dist = pd.DataFrame()
    risk_reasons = pd.DataFrame()
    time_df = pd.DataFrame()
    state_df = pd.DataFrame()
    district_df = pd.DataFrame()
    sector_df = pd.DataFrame()
    st.warning(f"One or more analytic panels could not be loaded: {error}")


# ============================================================
# TABS
# ============================================================

overview, risk_tab, finance_tab, geography_tab, explorer_tab, methodology_tab = st.tabs(
    [
        "📊 Overview",
        "⚠️ Risk Intelligence",
        "💰 Financial & Execution",
        "🗺️ Geography",
        "🔎 Work Explorer",
        "📐 Methodology",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview:
    st.subheader("Executive monitoring view")
    st.markdown(
        "<div class='section-note'>"
        "Population-level baselines are shown separately from the investigation queue to avoid presenting the top-ranked sample as the whole dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if not risk_dist.empty and {"risk_category", "count"}.issubset(risk_dist.columns):
            r = risk_dist.copy()
            r["risk_category"] = pd.Categorical(r["risk_category"], categories=RISK_ORDER, ordered=True)
            r = r.sort_values("risk_category")
            fig = px.bar(
                r,
                x="count",
                y="risk_category",
                orientation="h",
                text="pct",
                title="Risk distribution — full loaded population",
                labels={"count": "Works", "risk_category": "Risk band", "pct": "Share"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=370, margin=dict(l=10, r=20, t=55, b=25), yaxis={"categoryorder": "array", "categoryarray": RISK_ORDER})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk distribution is unavailable.")

    with c2:
        if not time_df.empty and "financial_year" in time_df.columns:
            t = time_df.copy()
            if "sanctioned_amount" in t.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t["financial_year"],
                    y=pd.to_numeric(t["sanctioned_amount"], errors="coerce"),
                    mode="lines+markers",
                    name="Sanctioned",
                    customdata=t[["works", "completed_works", "utilization_pct"]].fillna(0).to_numpy()
                    if {"works", "completed_works", "utilization_pct"}.issubset(t.columns) else None,
                    hovertemplate=(
                        "FY %{x}<br>Sanctioned: ₹%{y:,.0f}<br>"
                        "Works: %{customdata[0]:,.0f}<br>"
                        "Completed: %{customdata[1]:,.0f}<br>"
                        "Utilisation: %{customdata[2]:.1f}%<extra></extra>"
                    ) if {"works", "completed_works", "utilization_pct"}.issubset(t.columns) else None,
                ))
                if "total_expenditure" in t.columns:
                    fig.add_trace(go.Scatter(
                        x=t["financial_year"],
                        y=pd.to_numeric(t["total_expenditure"], errors="coerce"),
                        mode="lines+markers",
                        name="Recorded expenditure",
                    ))
                fig.update_layout(
                    title="Financial flow across financial years",
                    yaxis_title="Amount (₹)",
                    height=370,
                    margin=dict(l=10, r=20, t=55, b=25),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Time-series data is unavailable.")

    # Priority queue preview
    st.subheader("Priority investigation queue")
    st.markdown(
        "<div class='section-note'>These are the highest-priority records within the active filter set. They are investigation candidates, not findings of wrongdoing.</div>",
        unsafe_allow_html=True,
    )

    if not priority_df.empty:
        preview_cols = [
            "priority_rank", "work_uid", "work", "state", "ida", "work_category",
            "sanction_amount", "total_expenditure", "final_risk_score", "risk_category",
            "confidence_score", "priority_score", "primary_risk_reason",
        ]
        preview_cols = [c for c in preview_cols if c in priority_df.columns]
        preview = priority_df[preview_cols].head(10).copy()
        st.dataframe(
            preview,
            width="stretch",
            hide_index=True,
            height=390,
            column_config={
                "priority_rank": st.column_config.NumberColumn("#", width="small"),
                "work_uid": st.column_config.TextColumn("Work ID", width="small"),
                "sanction_amount": st.column_config.NumberColumn("Sanction", format="₹ %.0f"),
                "total_expenditure": st.column_config.NumberColumn("Expenditure", format="₹ %.0f"),
                "final_risk_score": st.column_config.NumberColumn("Risk", format="%.1f"),
                "priority_score": st.column_config.NumberColumn("Priority", format="%.1f"),
                "confidence_score": st.column_config.NumberColumn("Confidence", format="%.1f"),
            },
        )
    else:
        st.info("No works match the active filters.")


# ============================================================
# RISK INTELLIGENCE
# ============================================================

with risk_tab:
    st.subheader("⚠️ Risk Intelligence")
    st.markdown(
        "<div class='section-note'>The risk system combines ML anomaly ranking with financial, execution, similarity and data-integrity evidence. It does not establish fraud or wrongdoing.</div>",
        unsafe_allow_html=True,
    )

    # Risk basis comparison — full population
    if not risk_dist.empty:
        rb1, rb2 = st.columns([1.1, 0.9])
        with rb1:
            rd = risk_dist.copy()
            rd["risk_category"] = pd.Categorical(rd["risk_category"], categories=RISK_ORDER, ordered=True)
            rd = rd.sort_values("risk_category")
            fig = px.bar(
                rd,
                x="risk_category",
                y="count",
                text="pct",
                title="Risk bands",
                labels={"risk_category": "Risk", "count": "Works", "pct": "Population share"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=25))
            st.plotly_chart(fig, use_container_width=True)

        with rb2:
            if not risk_reasons.empty and {"reason", "count"}.issubset(risk_reasons.columns):
                rr = risk_reasons.sort_values("count", ascending=True)
                fig = px.bar(
                    rr,
                    x="count",
                    y="reason",
                    orientation="h",
                    title="Detected risk-signal frequency",
                    labels={"count": "Works", "reason": "Signal"},
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=25))
                st.plotly_chart(fig, use_container_width=True)

    # Full-population score diagnostics
    if not priority_df.empty:
        d = priority_df.copy()
        xcol = "final_risk_score" if "final_risk_score" in d.columns else None
        ycol = "financial_exposure_percentile" if "financial_exposure_percentile" in d.columns else None
        if xcol and ycol:
            fig = px.scatter(
                d,
                x=xcol,
                y=ycol,
                color="risk_category" if "risk_category" in d.columns else None,
                size="sanction_amount" if "sanction_amount" in d.columns else None,
                hover_name="work_uid" if "work_uid" in d.columns else None,
                hover_data=[c for c in ["work", "state", "ida", "sanction_amount", "priority_score", "confidence_score"] if c in d.columns],
                title="Priority queue: risk × financial exposure percentile",
                labels={
                    xcol: "Final analytical risk",
                    ycol: "Financial exposure percentile",
                    "sanction_amount": "Sanction amount",
                },
            )
            fig.update_layout(height=470, margin=dict(l=10, r=10, t=55, b=25))
            st.plotly_chart(fig, use_container_width=True)

        # Component score panel
        component_cols = [
            "ml_anomaly_percentile",
            "financial_risk_score",
            "execution_risk_score",
            "duplicate_risk_score",
            "data_integrity_risk_score",
        ]
        available_components = [c for c in component_cols if c in d.columns]
        if available_components:
            comp = d[available_components].apply(pd.to_numeric, errors="coerce").median().sort_values(ascending=False)
            label_map = {
                "ml_anomaly_percentile": "ML anomaly percentile",
                "financial_risk_score": "Financial",
                "execution_risk_score": "Execution",
                "duplicate_risk_score": "Similarity",
                "data_integrity_risk_score": "Data integrity",
            }
            comp.index = [label_map.get(x, x) for x in comp.index]
            fig = px.bar(
                comp.reset_index(name="score"),
                x="score",
                y="index",
                orientation="h",
                text="score",
                title="Median risk-component intensity in the displayed priority queue",
                labels={"score": "Score (0–100)", "index": "Component"},
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=55, b=25), xaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

    # Selected high-priority work
    if not priority_df.empty and "work_uid" in priority_df.columns:
        selected_uid = st.selectbox(
            "Inspect a priority record",
            priority_df["work_uid"].astype(str).tolist(),
            key="risk_selected_work",
        )
        try:
            detail = api_get(f"/api/v1/works/{selected_uid}")
            row = detail.get("work", {}) if isinstance(detail, dict) else {}
            components = detail.get("risk_components", {}) if isinstance(detail, dict) else {}
            evidence = parse_evidence(detail.get("evidence")) if isinstance(detail, dict) else {}

            st.markdown(f"### {row.get('work', selected_uid)}")
            st.markdown(f"**Work ID:** `{selected_uid}`  ·  **{risk_badge(row.get('risk_category'))}**")

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Final risk", f"{safe_float(components.get('final_risk_score')):,.1f}")
            a2.metric("Priority", f"{safe_float(components.get('priority_score')):,.1f}")
            a3.metric("Confidence", f"{safe_float(components.get('confidence_score')):,.1f}%")
            a4.metric("Signals", str(safe_int(row.get("independent_signal_count"))))

            ec1, ec2 = st.columns([1, 1])
            with ec1:
                st.markdown("#### Component scores")
                component_rows = []
                for key, label in [
                    ("ml_anomaly_percentile", "ML anomaly"),
                    ("financial_risk_score", "Financial"),
                    ("execution_risk_score", "Execution"),
                    ("duplicate_risk_score", "Similarity"),
                    ("data_integrity_risk_score", "Data integrity"),
                ]:
                    if key in components:
                        component_rows.append({"Component": label, "Score": safe_float(components.get(key))})
                if component_rows:
                    component_frame = pd.DataFrame(component_rows)
                    fig = px.bar(component_frame, x="Score", y="Component", orientation="h", text="Score", title="Risk decomposition")
                    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=55, b=15), xaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)

            with ec2:
                st.markdown("#### Why this record is prioritised")
                explanation = row.get("risk_explanation") or "No strong rule-based signal."
                st.markdown(f'<div class="selected-work">{explanation}</div>', unsafe_allow_html=True)
                if evidence:
                    fin = evidence.get("financial", {})
                    exe = evidence.get("execution", {})
                    sim = evidence.get("duplicate", {})
                    st.write(
                        {
                            "Expenditure variance %": fin.get("overspend_pct"),
                            "Utilisation %": fin.get("utilization_pct"),
                            "Cost robust z": fin.get("cost_robust_z"),
                            "Cost peer group n": fin.get("peer_group_n"),
                            "Days open": exe.get("days_open_since_sanction"),
                            "Days beyond benchmark": exe.get("days_over_general_one_year_benchmark"),
                            "Days since last expenditure": exe.get("days_since_last_expenditure"),
                            "Potential similarity candidate": sim.get("is_candidate"),
                        }
                    )
        except Exception as error:
            st.warning(f"Selected-work evidence is unavailable: {error}")


# ============================================================
# FINANCIAL & EXECUTION
# ============================================================

with finance_tab:
    st.subheader("💰 Financial & Execution Intelligence")
    st.markdown(
        "<div class='section-note'>Financial and execution indicators are calculated analytics; they should not be interpreted as audited accounts unless the source field semantics support that conclusion.</div>",
        unsafe_allow_html=True,
    )

    if not time_df.empty and "financial_year" in time_df.columns:
        t = time_df.copy()
        f1, f2 = st.columns(2)

        with f1:
            if {"sanctioned_amount", "total_expenditure"}.issubset(t.columns):
                fig = go.Figure()
                fig.add_trace(go.Bar(x=t["financial_year"], y=t["sanctioned_amount"], name="Sanctioned"))
                fig.add_trace(go.Bar(x=t["financial_year"], y=t["total_expenditure"], name="Recorded expenditure"))
                fig.update_layout(
                    barmode="group",
                    title="Sanctioned vs recorded expenditure",
                    yaxis_title="Amount (₹)",
                    height=390,
                    margin=dict(l=10, r=10, t=55, b=25),
                )
                st.plotly_chart(fig, use_container_width=True)

        with f2:
            if "utilization_pct" in t.columns:
                fig = px.line(
                    t,
                    x="financial_year",
                    y="utilization_pct",
                    markers=True,
                    title="Portfolio utilisation trend",
                    labels={"utilization_pct": "Utilisation (%)", "financial_year": "Financial year"},
                )
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=25), yaxis_range=[0, max(100, safe_float(t["utilization_pct"].max()) * 1.1)])
                st.plotly_chart(fig, use_container_width=True)

    # Current filtered queue financial exposure
    if not priority_df.empty:
        d = priority_df.copy()
        b1, b2 = st.columns(2)
        with b1:
            if "sanction_amount" in d.columns:
                fig = px.histogram(
                    d,
                    x=pd.to_numeric(d["sanction_amount"], errors="coerce"),
                    nbins=30,
                    title="Sanction distribution — priority queue",
                    labels={"x": "Sanction amount (₹)"},
                )
                fig.update_xaxes(tickprefix="₹", tickformat="~s")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=25))
                st.plotly_chart(fig, use_container_width=True)

        with b2:
            if {"sanction_amount", "total_expenditure"}.issubset(d.columns):
                scatter = d.copy()
                scatter["sanction_amount"] = pd.to_numeric(scatter["sanction_amount"], errors="coerce")
                scatter["total_expenditure"] = pd.to_numeric(scatter["total_expenditure"], errors="coerce")
                fig = px.scatter(
                    scatter,
                    x="sanction_amount",
                    y="total_expenditure",
                    color="risk_category" if "risk_category" in scatter.columns else None,
                    hover_name="work_uid" if "work_uid" in scatter.columns else None,
                    hover_data=[c for c in ["work", "state", "ida", "utilization_pct", "final_risk_score"] if c in scatter.columns],
                    title="Sanction vs recorded expenditure",
                    labels={"sanction_amount": "Sanction amount (₹)", "total_expenditure": "Recorded expenditure (₹)"},
                )
                # 1:1 reference line
                max_val = max(
                    safe_float(scatter["sanction_amount"].max()),
                    safe_float(scatter["total_expenditure"].max()),
                    1,
                )
                fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash"))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=25))
                st.plotly_chart(fig, use_container_width=True)

    # Execution indicators
    if not priority_df.empty:
        exec_cols = [c for c in ["days_open_since_sanction", "days_since_last_expenditure", "days_sanction_to_complete"] if c in priority_df.columns]
        if exec_cols:
            long_rows = []
            labels = {
                "days_open_since_sanction": "Open age (days)",
                "days_since_last_expenditure": "Days since last expenditure",
                "days_sanction_to_complete": "Sanction-to-completion (days)",
            }
            for col in exec_cols:
                s = pd.to_numeric(priority_df[col], errors="coerce").dropna()
                for value in s:
                    long_rows.append({"Metric": labels[col], "Days": value})
            if long_rows:
                exec_frame = pd.DataFrame(long_rows)
                fig = px.box(
                    exec_frame,
                    x="Metric",
                    y="Days",
                    points="outliers",
                    title="Execution-duration distributions in the priority queue",
                )
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=35))
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# GEOGRAPHY
# ============================================================

with geography_tab:
    st.subheader("🗺️ Geographic Intelligence")
    st.markdown(
        "<div class='section-note'>National state analytics are shown from the analytical API. A work map is shown only when reliable latitude/longitude fields are present in the analytical dataset.</div>",
        unsafe_allow_html=True,
    )

    if not state_df.empty:
        g1, g2 = st.columns(2)
        with g1:
            plot_df = state_df.copy().sort_values("high_or_critical_rate_pct", ascending=False).head(20)
            if "high_or_critical_rate_pct" in plot_df.columns:
                fig = px.bar(
                    plot_df.sort_values("high_or_critical_rate_pct"),
                    x="high_or_critical_rate_pct",
                    y="state",
                    orientation="h",
                    title="Highest high/critical rates by state",
                    labels={"high_or_critical_rate_pct": "High/Critical rate (%)", "state": "State"},
                    hover_data=[c for c in ["works", "completed_works", "sanctioned_amount", "total_expenditure", "completion_rate_pct", "utilization_pct"] if c in plot_df.columns],
                )
                fig.update_layout(height=480, margin=dict(l=10, r=10, t=55, b=25))
                st.plotly_chart(fig, use_container_width=True)

        with g2:
            if {"completion_rate_pct", "utilization_pct"}.issubset(state_df.columns):
                plot_df = state_df.copy()
                plot_df["bubble"] = pd.to_numeric(plot_df["sanctioned_amount"], errors="coerce") if "sanctioned_amount" in plot_df.columns else 1
                fig = px.scatter(
                    plot_df,
                    x="completion_rate_pct",
                    y="utilization_pct",
                    size="bubble",
                    color="high_or_critical_rate_pct" if "high_or_critical_rate_pct" in plot_df.columns else None,
                    hover_name="state",
                    hover_data=[c for c in ["works", "median_risk", "high_or_critical_rate_pct", "critical_rate_pct"] if c in plot_df.columns],
                    title="State performance map: completion × utilisation",
                    labels={"completion_rate_pct": "Completion (%)", "utilization_pct": "Utilisation (%)", "bubble": "Sanctioned amount"},
                )
                fig.update_layout(height=480, margin=dict(l=10, r=10, t=55, b=25))
                st.plotly_chart(fig, use_container_width=True)

    if not district_df.empty:
        st.markdown("### District drill-down")
        district_plot = district_df.head(30).copy()
        if "high_or_critical_rate_pct" in district_plot.columns:
            fig = px.bar(
                district_plot.sort_values("high_or_critical_rate_pct"),
                x="high_or_critical_rate_pct",
                y="ida",
                orientation="h",
                title=(f"Top district risk rates — {state}" if state != "All" else "Top district risk rates"),
                labels={"high_or_critical_rate_pct": "High/Critical rate (%)", "ida": "District / IDA"},
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=25))
            st.plotly_chart(fig, use_container_width=True)

    try:
        geo = api_get(
            "/api/v1/geography-points",
            {"risk_min": risk_range[0]},
        )
        points = pd.DataFrame(geo.get("points", [])) if isinstance(geo, dict) else pd.DataFrame()
        if isinstance(geo, dict) and geo.get("available") and not points.empty:
            st.markdown("### Work-level risk map")
            if {"lat", "lon"}.issubset(points.columns):
                map_df = points.copy()
                map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
                map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
                map_df = map_df.dropna(subset=["lat", "lon"])
                st.map(map_df[["lat", "lon"]], use_container_width=True)
                st.caption(f"Showing up to {len(map_df):,} geolocated works from the current API response. Map points are not official geographic validation unless the source coordinates are verified.")
        elif isinstance(geo, dict):
            st.info(geo.get("reason", "No geospatial records available."))
    except Exception as error:
        st.warning(f"Work-level geography unavailable: {error}")


# ============================================================
# WORK EXPLORER
# ============================================================

with explorer_tab:
    st.subheader("🔎 Work Explorer / Investigation Workbench")
    st.markdown(
        f"<div class='section-note'>Showing <b>{len(priority_df):,}</b> records from <b>{total_matching:,}</b> matching works. Sort and inspect the highest-priority records first.</div>",
        unsafe_allow_html=True,
    )

    if priority_df.empty:
        st.info("No work matches the current filters.")
    else:
        display_cols = [
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
        display_cols = [c for c in display_cols if c in priority_df.columns]
        table_df = priority_df[display_cols].copy()

        event = st.dataframe(
            table_df,
            width="stretch",
            height=600,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="priority_work_table",
            column_config={
                "priority_rank": st.column_config.NumberColumn("#", width="small"),
                "work_uid": st.column_config.TextColumn("Work ID", width="small"),
                "sanction_amount": st.column_config.NumberColumn("Sanction", format="₹ %.0f"),
                "total_expenditure": st.column_config.NumberColumn("Expenditure", format="₹ %.0f"),
                "utilization_pct": st.column_config.NumberColumn("Utilisation", format="%.1f%%"),
                "days_open_since_sanction": st.column_config.NumberColumn("Open age", format="%.0f"),
                "final_risk_score": st.column_config.NumberColumn("Risk", format="%.1f"),
                "priority_score": st.column_config.NumberColumn("Priority", format="%.1f"),
                "confidence_score": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
            },
        )

        selected_rows = []
        try:
            selected_rows = event.selection.rows
        except Exception:
            pass

        selected_uid = None
        if selected_rows:
            idx = selected_rows[0]
            if 0 <= idx < len(table_df) and "work_uid" in table_df.columns:
                selected_uid = str(table_df.iloc[idx]["work_uid"])

        if selected_uid:
            try:
                detail = api_get(f"/api/v1/works/{selected_uid}")
                row = detail.get("work", {})
                components = detail.get("risk_components", {})
                st.markdown(f"### Selected: {row.get('work', selected_uid)}")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Risk", f"{safe_float(components.get('final_risk_score')):.1f}")
                r2.metric("Priority", f"{safe_float(components.get('priority_score')):.1f}")
                r3.metric("Confidence", f"{safe_float(components.get('confidence_score')):.1f}%")
                r4.metric("Sanction", inr(row.get("sanction_amount")))

                left, right = st.columns([1.2, 0.8])
                with left:
                    st.markdown("#### Explanation")
                    st.markdown(f'<div class="selected-work">{row.get("risk_explanation", "No strong rule-based signal.")}</div>', unsafe_allow_html=True)
                with right:
                    st.markdown("#### Risk components")
                    component_dict = {
                        "ML anomaly": components.get("ml_anomaly_percentile"),
                        "Financial": components.get("financial_risk_score"),
                        "Execution": components.get("execution_risk_score"),
                        "Similarity": components.get("duplicate_risk_score"),
                        "Integrity": components.get("data_integrity_risk_score"),
                    }
                    component_dict = {k: safe_float(v) for k, v in component_dict.items() if v is not None}
                    if component_dict:
                        comp = pd.DataFrame({"Component": list(component_dict), "Score": list(component_dict.values())})
                        fig = px.bar(comp, x="Score", y="Component", orientation="h", text="Score")
                        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                        fig.update_layout(height=290, margin=dict(l=10, r=10, t=15, b=15), xaxis_range=[0, 100])
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as error:
                st.warning(f"Could not load selected work: {error}")

        csv_bytes = priority_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download displayed priority queue",
            data=csv_bytes,
            file_name="mplads_priority_queue.csv",
            mime="text/csv",
        )


# ============================================================
# METHODOLOGY
# ============================================================

with methodology_tab:
    st.subheader("📐 Methodology, Interpretation & Guardrails")

    st.markdown("### Analytical hierarchy")
    st.code(
        "Official source data\n"
        "      ↓\n"
        "Data validation / feature engineering\n"
        "      ↓\n"
        "Deterministic risk signals + robust peer statistics\n"
        "      ↓\n"
        "Isolation Forest anomaly score\n"
        "      ↓\n"
        "Final analytical risk\n"
        "      ↓\n"
        "Financial exposure + evidence support\n"
        "      ↓\n"
        "Investigation priority\n"
        "      ↓\n"
        "Human review"
    )

    st.markdown("### Score semantics")
    st.markdown(
        "**ML anomaly percentile** is a population-relative abnormality ranking. "
        "It is **not** a probability of fraud. Isolation Forest is an unsupervised anomaly-detection method."
    )
    st.markdown(
        "**Final risk** is a composite analytical score. The project's weights are design parameters, not official MPLADS weights."
    )
    st.markdown(
        "**Investigation priority** is a triage measure that combines analytical risk with financial exposure and evidence support."
    )

    st.markdown("### Risk bands")
    bands = pd.DataFrame(
        [{"Band": k, "Range": RISK_RANGES[k]} for k in RISK_ORDER]
    )
    st.dataframe(bands, width="stretch", hide_index=True)

    st.markdown("### Important interpretation rules")
    st.markdown(
        "- An anomaly is a **signal for review**, not proof of wrongdoing.\n"
        "- A duplicate/similarity candidate is **not a confirmed duplicate**.\n"
        "- A data-integrity issue is a **data-quality signal**, not substantive wrongdoing.\n"
        "- A one-year timing benchmark should not be described as an automatic statutory violation without a rule-specific verification.\n"
        "- MP-level results should not imply direct operational responsibility for downstream execution when the official framework assigns execution responsibilities to district authorities."
    )

    st.markdown("### Current data/runtime status")
    status_table = pd.DataFrame(
        [
            ["API version", health.get("version")],
            ["Data source", health.get("data_source")],
            ["Works loaded", safe_int(health.get("works_loaded"))],
            ["Data as-of", health.get("data_as_of")],
            ["Dashboard version", APP_VERSION],
        ],
        columns=["Item", "Value"],
    )
    st.dataframe(status_table, width="stretch", hide_index=True)

    st.markdown("### What the dashboard deliberately does not claim")
    st.warning(
        "This application prioritizes records for human investigation. It does not independently establish fraud, corruption, legal non-compliance, financial misappropriation, or causality."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-note">MPLADS AI Monitor · Analytical decision-support interface · Risk signals are for review and prioritisation, not adjudication.</div>',
    unsafe_allow_html=True,
)
