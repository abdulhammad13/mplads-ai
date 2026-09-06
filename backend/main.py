from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text

from .database import engine


# ============================================================
# PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "final_risk_data.csv"

SERVICE_VERSION = "2.0"

DATA_SOURCE_MODE = os.getenv("MPLADS_DATA_SOURCE", "csv").strip().lower()
SQL_TABLE = os.getenv("MPLADS_SQL_TABLE", "dbo.works")

WORKS: pd.DataFrame | None = None
WORKS_SOURCE: str | None = None


# ============================================================
# DATA LOADING / VALIDATION
# ============================================================


KEY_NUMERIC_COLUMNS = [
    "sanction_amount",
    "recommended_amount",
    "completed_amount",
    "total_expenditure",
    "allocated_amount",
    "utilization_pct",
    "final_risk_score",
    "priority_score",
    "ml_anomaly_percentile",
    "rule_risk_score",
]

DATE_COLUMNS = [
    "recommended_date",
    "sanction_date",
    "completion_date",
    "first_expenditure_date",
    "last_expenditure_date",
    "monitoring_as_of_date",
]


def _read_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing final analytical dataset: {CSV_PATH}")
    frame = pd.read_csv(CSV_PATH, low_memory=False)
    return _normalize_frame(frame)


def _read_sql() -> pd.DataFrame:
    # SQL_TABLE is an environment-controlled identifier, not a user query.
    query = text(f"SELECT * FROM {SQL_TABLE}")
    with engine.connect() as connection:
        frame = pd.read_sql(query, connection)
    return _normalize_frame(frame)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)

    for column in KEY_NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    for column in [
        "is_completed",
        "is_open",
        "is_duplicate_candidate",
        "flag_disbursement_over_sanction",
        "flag_bad_dates",
        "flag_overdue_open_work",
        "flag_stalled_expenditure",
        "flag_duplicate_candidate",
        "flag_cost_outlier",
        "flag_duration_outlier",
    ]:
        if column in frame.columns:
            normalized = frame[column].astype("string").str.strip().str.lower()
            parsed = pd.Series(False, index=frame.index, dtype="boolean")
            parsed.loc[normalized.isin({"true", "1", "yes", "y", "t"})] = True
            frame[column] = parsed.fillna(False)

    if "risk_category" in frame.columns:
        frame["risk_category"] = frame["risk_category"].astype("string")

    return frame


def _has_required_schema(frame: pd.DataFrame) -> tuple[bool, list[str]]:
    required = [
        "work_uid",
        "state",
        "sanction_amount",
        "final_risk_score",
        "risk_category",
        "priority_score",
    ]
    missing = [column for column in required if column not in frame.columns]
    return len(missing) == 0, missing


def load_data_once(force_refresh: bool = False) -> pd.DataFrame:
    global WORKS, WORKS_SOURCE

    if WORKS is not None and not force_refresh:
        return WORKS

    errors: list[str] = []

    mode = DATA_SOURCE_MODE
    if mode not in {"auto", "csv", "sql"}:
        raise RuntimeError(
            "MPLADS_DATA_SOURCE must be one of: auto, csv, sql"
        )

    candidates = ["csv", "sql"] if mode == "auto" else [mode]

    for source in candidates:
        try:
            frame = _read_csv() if source == "csv" else _read_sql()
            valid, missing = _has_required_schema(frame)
            if not valid:
                errors.append(
                    f"{source}: analytical schema missing {missing}"
                )
                continue

            WORKS = frame
            WORKS_SOURCE = source
            print(f"Loaded {len(frame):,} works from {source}.")
            return WORKS
        except Exception as error:
            errors.append(f"{source}: {error}")

    raise RuntimeError(
        "No valid analytical data source could be loaded.\n"
        + "\n".join(errors)
    )


# ============================================================
# FASTAPI LIFESPAN
# ============================================================


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_data_once()
    yield


app = FastAPI(
    title="MPLADS AI Monitoring Platform",
    description=(
        "AI-assisted monitoring of MPLADS works using transparent risk signals, "
        "peer-relative anomaly detection, execution/financial indicators, "
        "duplicate candidates, and explainable investigation prioritization."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# ============================================================
# RESPONSE / FILTER HELPERS
# ============================================================


def _json_value(value: Any):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    data = frame.copy().replace([np.inf, -np.inf], np.nan)
    rows = data.to_dict(orient="records")
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in rows
    ]


def _contains_case_insensitive(series: pd.Series, value: str) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.contains(value, case=False, regex=False, na=False)
    )


def apply_filters(
    df: pd.DataFrame,
    state: str | None = None,
    district: str | None = None,
    mp: str | None = None,
    constituency: str | None = None,
    work_category: str | None = None,
    work_status: str | None = None,
    risk_category: str | None = None,
    completion_status: str | None = None,
    search: str | None = None,
    min_sanction: float | None = None,
    max_sanction: float | None = None,
    min_risk: float = 0,
    max_risk: float = 100,
) -> pd.DataFrame:
    result = df

    exact_filters = {
        "state": state,
        "ida": district,
        "mp": mp,
        "constituency": constituency,
        "work_category": work_category,
        "work_status": work_status,
        "risk_category": risk_category,
    }

    for column, value in exact_filters.items():
        if value and column in result.columns:
            result = result[
                result[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
                == value.strip().casefold()
            ]

    if completion_status == "Completed" and "is_completed" in result.columns:
        result = result[result["is_completed"].fillna(False)]
    elif completion_status == "Open" and "is_completed" in result.columns:
        result = result[~result["is_completed"].fillna(False)]

    if search:
        search_columns = [
            "work",
            "work_description",
            "mp",
            "constituency",
            "ida",
            "work_uid",
        ]
        mask = pd.Series(False, index=result.index)
        for column in search_columns:
            if column in result.columns:
                mask |= _contains_case_insensitive(result[column], search)
        result = result[mask]

    if "sanction_amount" in result.columns:
        amount = pd.to_numeric(result["sanction_amount"], errors="coerce")
        if min_sanction is not None:
            result = result[amount.ge(min_sanction)]
            amount = pd.to_numeric(result["sanction_amount"], errors="coerce")
        if max_sanction is not None:
            result = result[amount.le(max_sanction)]

    if "final_risk_score" in result.columns:
        risk = pd.to_numeric(result["final_risk_score"], errors="coerce")
        result = result[risk.between(min_risk, max_risk, inclusive="both")]

    return result


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    completed = int(frame["is_completed"].fillna(False).sum()) if "is_completed" in frame.columns else 0
    sanctioned = pd.to_numeric(frame.get("sanction_amount"), errors="coerce").sum(min_count=1)
    expenditure = pd.to_numeric(frame.get("total_expenditure"), errors="coerce").sum(min_count=1)

    utilisation = (
        expenditure / sanctioned * 100
        if pd.notna(sanctioned) and sanctioned > 0 and pd.notna(expenditure)
        else None
    )

    risk = pd.to_numeric(frame.get("final_risk_score"), errors="coerce")
    high = int(frame["risk_category"].isin(["HIGH", "CRITICAL"]).sum()) if "risk_category" in frame.columns else 0
    critical = int(frame["risk_category"].eq("CRITICAL").sum()) if "risk_category" in frame.columns else 0
    duplicates = int(frame.get("is_duplicate_candidate", pd.Series(False, index=frame.index)).fillna(False).sum())
    overdue = int(frame.get("flag_overdue_open_work", pd.Series(False, index=frame.index)).fillna(False).sum())
    multi_signal = int(frame.get("independent_signal_count", pd.Series(0, index=frame.index)).ge(2).sum())

    return {
        "total_works": total,
        "completed_works": completed,
        "open_works": total - completed,
        "completion_rate_pct": round(completed / total * 100, 2) if total else 0,
        "sanctioned_amount": None if pd.isna(sanctioned) else float(sanctioned),
        "total_expenditure": None if pd.isna(expenditure) else float(expenditure),
        "portfolio_utilization_pct": None if utilisation is None else round(float(utilisation), 2),
        "mean_risk": None if risk.dropna().empty else round(float(risk.mean()), 2),
        "median_risk": None if risk.dropna().empty else round(float(risk.median()), 2),
        "high_or_critical": high,
        "critical": critical,
        "high_or_critical_rate_pct": round(high / total * 100, 2) if total else 0,
        "duplicate_candidates": duplicates,
        "overdue_open_works": overdue,
        "multi_signal_cases": multi_signal,
    }


# ============================================================
# ROOT / HEALTH
# ============================================================


@app.get("/")
def root():
    return {
        "service": "MPLADS AI Monitoring API",
        "status": "running",
        "version": SERVICE_VERSION,
        "data_source": WORKS_SOURCE,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    df = load_data_once()
    return {
        "status": "ok",
        "service": "MPLADS AI Monitoring API",
        "version": SERVICE_VERSION,
        "works_loaded": len(df),
        "data_source": WORKS_SOURCE,
        "data_as_of": (
            str(df["monitoring_as_of_date"].dropna().max().date())
            if "monitoring_as_of_date" in df.columns and df["monitoring_as_of_date"].notna().any()
            else None
        ),
    }


@app.post("/api/v1/refresh")
def refresh_data():
    df = load_data_once(force_refresh=True)
    return {
        "status": "ok",
        "works_loaded": len(df),
        "data_source": WORKS_SOURCE,
    }


# ============================================================
# FILTER OPTIONS
# ============================================================


@app.get("/api/v1/filter-options")
def filter_options():
    df = load_data_once()

    def options(column: str) -> list[str]:
        if column not in df.columns:
            return []
        return (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )

    return {
        "states": options("state"),
        "districts": options("ida"),
        "mps": options("mp"),
        "constituencies": options("constituency"),
        "work_categories": options("work_category"),
        "work_statuses": options("work_status"),
        "risk_categories": options("risk_category"),
        "financial_years": options("financial_year"),
    }


# ============================================================
# OVERVIEW / RISK SUMMARY
# ============================================================


@app.get("/api/v1/dashboard-summary")
def dashboard_summary():
    df = load_data_once()
    result = _summary(df)

    if "risk_category" in df.columns:
        distribution = (
            df["risk_category"]
            .value_counts(dropna=False)
            .rename_axis("risk_category")
            .reset_index(name="count")
        )
        result["risk_distribution"] = to_records(distribution)

    return result


@app.get("/api/v1/filtered-summary")
def filtered_summary(
    state: str | None = None,
    district: str | None = None,
    mp: str | None = None,
    constituency: str | None = None,
    work_category: str | None = None,
    work_status: str | None = None,
    risk_category: str | None = None,
    completion_status: str | None = None,
    search: str | None = None,
    min_sanction: float | None = Query(default=None, ge=0),
    max_sanction: float | None = Query(default=None, ge=0),
    min_risk: float = Query(default=0, ge=0, le=100),
    max_risk: float = Query(default=100, ge=0, le=100),
):
    df = load_data_once()
    filtered = apply_filters(
        df,
        state=state,
        district=district,
        mp=mp,
        constituency=constituency,
        work_category=work_category,
        work_status=work_status,
        risk_category=risk_category,
        completion_status=completion_status,
        search=search,
        min_sanction=min_sanction,
        max_sanction=max_sanction,
        min_risk=min_risk,
        max_risk=max_risk,
    )
    return _summary(filtered)


@app.get("/api/v1/risk-distribution")
def risk_distribution(
    risk_basis: str = Query(default="final", pattern="^(final|rule|ml)$"),
):
    df = load_data_once()

    if risk_basis == "final":
        values = pd.to_numeric(df["final_risk_score"], errors="coerce")
        categories = pd.cut(values, bins=[-1, 25, 50, 75, 100], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    elif risk_basis == "rule":
        values = pd.to_numeric(df["rule_risk_score"], errors="coerce")
        categories = pd.cut(values, bins=[-1, 25, 50, 75, 100], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    else:
        values = pd.to_numeric(df["ml_anomaly_percentile"], errors="coerce")
        categories = pd.cut(values, bins=[-1, 25, 50, 75, 100], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    result = categories.value_counts(sort=False).rename_axis("risk_category").reset_index(name="count")
    result["pct"] = result["count"] / len(df) * 100 if len(df) else 0
    return to_records(result)


@app.get("/api/v1/risk-reasons")
def risk_reasons():
    df = load_data_once()
    definitions = [
        ("flag_disbursement_over_sanction", "Expenditure exceeds sanction"),
        ("flag_overdue_open_work", "Open beyond general one-year benchmark"),
        ("flag_stalled_expenditure", "Long expenditure gap"),
        ("flag_duplicate_candidate", "Potential duplicate/similar work"),
        ("flag_cost_outlier", "Cost outlier within category"),
        ("flag_duration_outlier", "Duration outlier within category"),
        ("flag_bad_dates", "Date integrity issue"),
    ]

    rows = []
    for column, label in definitions:
        if column not in df.columns:
            continue
        count = int(df[column].fillna(False).sum())
        rows.append({
            "reason": label,
            "count": count,
            "pct_of_all_works": round(count / len(df) * 100, 2) if len(df) else 0,
        })

    return rows


# ============================================================
# WORK EXPLORER
# ============================================================


@app.get("/api/v1/works")
def works(
    state: str | None = None,
    district: str | None = None,
    mp: str | None = None,
    constituency: str | None = None,
    work_category: str | None = None,
    work_status: str | None = None,
    risk_category: str | None = None,
    completion_status: str | None = None,
    search: str | None = None,
    min_sanction: float | None = Query(default=None, ge=0),
    max_sanction: float | None = Query(default=None, ge=0),
    min_risk: float = Query(default=0, ge=0, le=100),
    max_risk: float = Query(default=100, ge=0, le=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    sort_by: str = Query(
        default="priority_score",
        pattern="^(priority_score|final_risk_score|sanction_amount|total_expenditure|confidence_score|days_open_since_sanction)$",
    ),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
):
    df = load_data_once()
    filtered = apply_filters(
        df,
        state=state,
        district=district,
        mp=mp,
        constituency=constituency,
        work_category=work_category,
        work_status=work_status,
        risk_category=risk_category,
        completion_status=completion_status,
        search=search,
        min_sanction=min_sanction,
        max_sanction=max_sanction,
        min_risk=min_risk,
        max_risk=max_risk,
    )

    total = len(filtered)
    ascending = sort_order == "asc"
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(
            sort_by,
            ascending=ascending,
            na_position="last",
        )

    start = (page - 1) * page_size
    page_frame = filtered.iloc[start:start + page_size].copy()

    # Keep the explorer response lightweight and human-readable.
    preferred = [
        "work_uid",
        "work",
        "work_description",
        "state",
        "ida",
        "mp",
        "constituency",
        "work_category",
        "work_status",
        "financial_year",
        "sanction_amount",
        "total_expenditure",
        "utilization_pct",
        "is_completed",
        "days_open_since_sanction",
        "final_risk_score",
        "risk_category",
        "priority_score",
        "priority_rank",
        "confidence_score",
        "primary_risk_reason",
        "risk_explanation",
    ]
    selected = [c for c in preferred if c in page_frame.columns]

    return {
        "items": to_records(page_frame[selected]),
        "page": page,
        "page_size": page_size,
        "total_matching": total,
        "total_pages": int(np.ceil(total / page_size)) if total else 0,
    }


@app.get("/api/v1/works/{work_uid}")
def work_detail(work_uid: str):
    df = load_data_once()
    if "work_uid" not in df.columns:
        raise HTTPException(status_code=500, detail="work_uid is missing from analytical dataset")

    match = df[df["work_uid"].astype(str) == str(work_uid)]
    if match.empty:
        raise HTTPException(status_code=404, detail="Work not found")

    row = match.iloc[0]
    return {
        "work": to_records(pd.DataFrame([row]))[0],
        "risk_components": {
            key: _json_value(row[key])
            for key in [
                "ml_anomaly_percentile",
                "rule_risk_score",
                "financial_risk_score",
                "execution_risk_score",
                "duplicate_risk_score",
                "data_integrity_risk_score",
                "final_risk_score",
                "confidence_score",
                "priority_score",
                "priority_rank",
            ]
            if key in row.index
        },
        "evidence": row.get("evidence_json"),
    }


@app.get("/api/v1/works/{work_uid}/similar")
def work_similar(work_uid: str, limit: int = Query(default=10, ge=1, le=50)):
    df = load_data_once()
    if "work_uid" not in df.columns:
        raise HTTPException(status_code=500, detail="work_uid is missing")

    match = df[df["work_uid"].astype(str) == str(work_uid)]
    if match.empty:
        raise HTTPException(status_code=404, detail="Work not found")

    row = match.iloc[0]
    candidates = df[df["work_uid"].astype(str) != str(work_uid)].copy()

    if "is_duplicate_candidate" in row.index and bool(row.get("is_duplicate_candidate", False)):
        if "duplicate_group_count" in row.index and "work_text" in row.index:
            same_group = candidates[
                candidates.get("work_text", pd.Series("", index=candidates.index)).astype(str)
                == str(row.get("work_text", ""))
            ]
            if not same_group.empty:
                return to_records(
                    same_group.sort_values("priority_score", ascending=False).head(limit)
                )

    # No semantic engine in this endpoint yet: return conservative exact-context
    # candidates instead of fabricating similarity scores.
    keys = ["state", "ida", "constituency", "work_category"]
    available = [c for c in keys if c in candidates.columns and c in row.index]
    if available:
        mask = pd.Series(True, index=candidates.index)
        for column in available:
            value = row[column]
            if pd.notna(value):
                mask &= candidates[column].astype(str).eq(str(value))
        subset = candidates[mask]
    else:
        subset = candidates.iloc[0:0]

    cols = [
        c for c in [
            "work_uid",
            "work",
            "work_description",
            "state",
            "ida",
            "constituency",
            "work_category",
            "sanction_amount",
            "final_risk_score",
            "priority_score",
        ]
        if c in subset.columns
    ]
    return to_records(subset.sort_values("priority_score", ascending=False).head(limit)[cols])


# ============================================================
# ANALYTICS
# ============================================================


@app.get("/api/v1/state-analytics")
def state_analytics():
    df = load_data_once()
    if "state" not in df.columns:
        return []

    result = (
        df.groupby("state", dropna=False)
        .agg(
            works=("state", "size"),
            completed_works=("is_completed", "sum"),
            sanctioned_amount=("sanction_amount", "sum"),
            total_expenditure=("total_expenditure", "sum"),
            median_risk=("final_risk_score", "median"),
            high_or_critical=("risk_category", lambda s: s.isin(["HIGH", "CRITICAL"]).sum()),
            critical=("risk_category", lambda s: s.eq("CRITICAL").sum()),
            overdue_open_works=("flag_overdue_open_work", "sum"),
            duplicate_candidates=("is_duplicate_candidate", "sum"),
        )
        .reset_index()
    )

    result["completion_rate_pct"] = result["completed_works"] / result["works"] * 100
    result["utilization_pct"] = result["total_expenditure"] / result["sanctioned_amount"] * 100
    result["high_or_critical_rate_pct"] = result["high_or_critical"] / result["works"] * 100
    result["critical_rate_pct"] = result["critical"] / result["works"] * 100

    return to_records(result.sort_values("high_or_critical_rate_pct", ascending=False))


@app.get("/api/v1/district-analytics")
def district_analytics(state: str | None = None):
    df = load_data_once()
    if state and "state" in df.columns:
        df = df[df["state"].astype(str).str.casefold() == state.strip().casefold()]

    district_col = "ida"
    if district_col not in df.columns:
        return []

    result = (
        df.groupby(district_col, dropna=False)
        .agg(
            works=(district_col, "size"),
            completed_works=("is_completed", "sum"),
            sanctioned_amount=("sanction_amount", "sum"),
            total_expenditure=("total_expenditure", "sum"),
            median_risk=("final_risk_score", "median"),
            high_or_critical=("risk_category", lambda s: s.isin(["HIGH", "CRITICAL"]).sum()),
            overdue_open_works=("flag_overdue_open_work", "sum"),
            duplicate_candidates=("is_duplicate_candidate", "sum"),
        )
        .reset_index()
    )
    result["completion_rate_pct"] = result["completed_works"] / result["works"] * 100
    result["utilization_pct"] = result["total_expenditure"] / result["sanctioned_amount"] * 100
    result["high_or_critical_rate_pct"] = result["high_or_critical"] / result["works"] * 100

    return to_records(result.sort_values("high_or_critical_rate_pct", ascending=False))


@app.get("/api/v1/sector-analytics")
def sector_analytics():
    df = load_data_once()
    if "work_category" not in df.columns:
        return []

    result = (
        df.groupby("work_category", dropna=False)
        .agg(
            works=("work_category", "size"),
            completed_works=("is_completed", "sum"),
            sanctioned_amount=("sanction_amount", "sum"),
            total_expenditure=("total_expenditure", "sum"),
            median_risk=("final_risk_score", "median"),
            median_cost=("sanction_amount", "median"),
            high_or_critical=("risk_category", lambda s: s.isin(["HIGH", "CRITICAL"]).sum()),
        )
        .reset_index()
    )
    result["completion_rate_pct"] = result["completed_works"] / result["works"] * 100
    result["utilization_pct"] = result["total_expenditure"] / result["sanctioned_amount"] * 100
    result["high_or_critical_rate_pct"] = result["high_or_critical"] / result["works"] * 100

    return to_records(result.sort_values("high_or_critical_rate_pct", ascending=False))


@app.get("/api/v1/time-series")
def time_series():
    df = load_data_once()
    if "financial_year" not in df.columns:
        return []

    result = (
        df.groupby("financial_year", dropna=False)
        .agg(
            works=("financial_year", "size"),
            completed_works=("is_completed", "sum"),
            sanctioned_amount=("sanction_amount", "sum"),
            total_expenditure=("total_expenditure", "sum"),
            median_risk=("final_risk_score", "median"),
            high_or_critical=("risk_category", lambda s: s.isin(["HIGH", "CRITICAL"]).sum()),
            critical=("risk_category", lambda s: s.eq("CRITICAL").sum()),
        )
        .reset_index()
    )
    result["completion_rate_pct"] = result["completed_works"] / result["works"] * 100
    result["utilization_pct"] = result["total_expenditure"] / result["sanctioned_amount"] * 100
    result["high_or_critical_rate_pct"] = result["high_or_critical"] / result["works"] * 100
    result["critical_rate_pct"] = result["critical"] / result["works"] * 100

    return to_records(result)


@app.get("/api/v1/geography-points")
def geography_points(risk_min: float = Query(default=0, ge=0, le=100)):
    df = load_data_once()
    lat_col = next((c for c in ["latitude", "lat", "work_latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude", "lon", "lng", "work_longitude"] if c in df.columns), None)
    if not lat_col or not lon_col:
        return {"available": False, "reason": "No latitude/longitude columns in analytical dataset", "points": []}

    subset = df[pd.to_numeric(df["final_risk_score"], errors="coerce").ge(risk_min)].copy()
    subset["lat"] = pd.to_numeric(subset[lat_col], errors="coerce")
    subset["lon"] = pd.to_numeric(subset[lon_col], errors="coerce")
    subset = subset.dropna(subset=["lat", "lon"])

    preferred = [
        "work_uid",
        "work",
        "state",
        "ida",
        "constituency",
        "work_category",
        "sanction_amount",
        "final_risk_score",
        "risk_category",
        "priority_score",
    ]
    available = [c for c in preferred if c in subset.columns]
    result = subset[available + ["lat", "lon"]].head(5000).copy()
    return {"available": True, "points": to_records(result)}
