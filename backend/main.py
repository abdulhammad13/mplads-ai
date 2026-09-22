from __future__ import annotations

import math
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text

from .database import engine

# ============================================================
# MPLADS AI MONITOR — FASTAPI DATA/API LAYER
# ============================================================
# Architecture:
#   source CSV / SQL -> Polars -> FastAPI -> Dash/Plotly
#
# This module intentionally does NOT recompute feature engineering,
# ML anomaly detection, or final risk methodology. Those remain in
# the existing backend pipeline. The API exposes the resulting
# analytical dataset consistently to the dashboard.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "final_risk_data.csv"

SERVICE_VERSION = "3.0-Polars"
DATA_SOURCE_MODE = os.getenv("MPLADS_DATA_SOURCE", "csv").strip().lower()
SQL_TABLE = os.getenv("MPLADS_SQL_TABLE", "dbo.works").strip()

WORKS: pl.DataFrame | None = None
WORKS_SOURCE: str | None = None

KEY_NUMERIC_COLUMNS = (
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
    "financial_risk_score",
    "execution_risk_score",
    "duplicate_risk_score",
    "data_integrity_risk_score",
    "confidence_score",
    "priority_rank",
    "days_open_since_sanction",
    "days_rec_to_sanction",
    "days_sanction_to_complete",
    "independent_signal_count",
    "duplicate_group_count",
)

DATE_COLUMNS = (
    "recommended_date",
    "sanction_date",
    "completion_date",
    "first_expenditure_date",
    "last_expenditure_date",
    "monitoring_as_of_date",
)

BOOLEAN_COLUMNS = (
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
)

REQUIRED_COLUMNS = (
    "work_uid",
    "state",
    "sanction_amount",
    "final_risk_score",
    "risk_category",
    "priority_score",
)

SORT_COLUMNS = {
    "priority_score",
    "final_risk_score",
    "sanction_amount",
    "total_expenditure",
    "confidence_score",
    "days_open_since_sanction",
}

RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Environment-controlled SQL identifiers are validated before interpolation.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sql_table(value: str) -> str:
    parts = [part.strip() for part in value.split(".") if part.strip()]
    if not parts or len(parts) > 2 or any(not _IDENTIFIER.fullmatch(p) for p in parts):
        raise ValueError(
            "MPLADS_SQL_TABLE must be a table identifier such as 'dbo.works'."
        )
    return ".".join(f"[{part}]" for part in parts)


def _empty_series(dtype: pl.DataType = pl.String) -> pl.Series:
    return pl.Series([], dtype=dtype)


def _ensure_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Create only optional runtime columns needed by API calculations."""
    expressions: list[pl.Expr] = []

    numeric_defaults = {
        "sanction_amount": pl.Float64,
        "recommended_amount": pl.Float64,
        "completed_amount": pl.Float64,
        "total_expenditure": pl.Float64,
        "allocated_amount": pl.Float64,
        "utilization_pct": pl.Float64,
        "final_risk_score": pl.Float64,
        "priority_score": pl.Float64,
        "ml_anomaly_percentile": pl.Float64,
        "rule_risk_score": pl.Float64,
        "financial_risk_score": pl.Float64,
        "execution_risk_score": pl.Float64,
        "duplicate_risk_score": pl.Float64,
        "data_integrity_risk_score": pl.Float64,
        "confidence_score": pl.Float64,
        "priority_rank": pl.Int64,
        "days_open_since_sanction": pl.Float64,
        "days_rec_to_sanction": pl.Float64,
        "days_sanction_to_complete": pl.Float64,
        "independent_signal_count": pl.Int64,
        "duplicate_group_count": pl.Int64,
    }
    for column, dtype in numeric_defaults.items():
        if column not in frame.columns:
            expressions.append(pl.lit(None, dtype=dtype).alias(column))

    for column in BOOLEAN_COLUMNS:
        if column not in frame.columns:
            expressions.append(pl.lit(False).alias(column))

    if "risk_category" not in frame.columns:
        expressions.append(pl.lit(None, dtype=pl.String).alias("risk_category"))

    if expressions:
        frame = frame.with_columns(expressions)

    return frame


def _normalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize analytical data once at the API boundary."""
    frame = frame.clone()

    # Standardize column names without changing business semantics.
    frame = frame.rename({column: column.strip() for column in frame.columns})

    expressions: list[pl.Expr] = []

    for column in KEY_NUMERIC_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .alias(column)
            )

    for column in DATE_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .str.to_date(strict=False)
                .alias(column)
            )

    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(["true", "1", "yes", "y", "t"])
                .fill_null(False)
                .alias(column)
            )

    for column in (
        "state",
        "ida",
        "mp",
        "constituency",
        "work",
        "work_description",
        "work_category",
        "work_status",
        "financial_year",
        "primary_risk_reason",
        "risk_explanation",
        "risk_category",
        "work_uid",
        "evidence_json",
    ):
        if column in frame.columns:
            expressions.append(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias(column)
            )

    if expressions:
        frame = frame.with_columns(expressions)

    frame = _ensure_columns(frame)

    # Replace non-finite numeric values with null.
    finite_exprs = [
        pl.when(pl.col(column).is_finite())
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in KEY_NUMERIC_COLUMNS
        if column in frame.columns
    ]
    if finite_exprs:
        frame = frame.with_columns(finite_exprs)

    return frame


def _read_csv() -> pl.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing final analytical dataset: {CSV_PATH}")

    frame = pl.read_csv(
        CSV_PATH,
        infer_schema_length=10000,
        try_parse_dates=False,
        null_values=["", "NULL", "null", "NA", "N/A", "NaN"],
    )
    return _normalize_frame(frame)


def _read_sql() -> pl.DataFrame:
    table = _validate_sql_table(SQL_TABLE)
    query = text(f"SELECT * FROM {table}")

    # SQLAlchemy remains the connection layer. Polars owns the dataframe.
    with engine.connect() as connection:
        result = connection.execute(query)
        rows = result.mappings().all()
        if not rows:
            return _normalize_frame(pl.DataFrame())
        frame = pl.DataFrame([dict(row) for row in rows])

    return _normalize_frame(frame)


def _has_required_schema(frame: pl.DataFrame) -> tuple[bool, list[str]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    return not missing, missing


def load_data_once(force_refresh: bool = False) -> pl.DataFrame:
    global WORKS, WORKS_SOURCE

    if WORKS is not None and not force_refresh:
        return WORKS

    if DATA_SOURCE_MODE not in {"auto", "csv", "sql"}:
        raise RuntimeError("MPLADS_DATA_SOURCE must be one of: auto, csv, sql")

    candidates = ("csv", "sql") if DATA_SOURCE_MODE == "auto" else (DATA_SOURCE_MODE,)
    errors: list[str] = []

    for source in candidates:
        try:
            frame = _read_csv() if source == "csv" else _read_sql()
            valid, missing = _has_required_schema(frame)
            if not valid:
                errors.append(f"{source}: analytical schema missing {missing}")
                continue

            WORKS = frame
            WORKS_SOURCE = source
            print(f"Loaded {frame.height:,} works from {source}.")
            return WORKS
        except Exception as error:
            errors.append(f"{source}: {type(error).__name__}: {error}")

    raise RuntimeError("No valid analytical data source could be loaded.\n" + "\n".join(errors))


# ============================================================
# JSON / SERIALIZATION
# ============================================================


def _json_value(value: Any) -> Any:
    """Convert Polars/Python/NumPy values into strict JSON primitives."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]

    return value


def to_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dicts()
    ]


def _first_value(frame: pl.DataFrame, column: str) -> Any:
    if column not in frame.columns or frame.height == 0:
        return None
    return frame.get_column(column)[0]


def _sum(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    value = frame.select(pl.col(column).sum()).item()
    return _json_value(value)


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return _json_value(frame.select(pl.col(column).mean()).item())


def _median(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return _json_value(frame.select(pl.col(column).median()).item())


def _count_true(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    value = frame.select(pl.col(column).fill_null(False).cast(pl.Int64).sum()).item()
    return int(value or 0)


def _safe_rate(numerator: float | int | None, denominator: int | float | None) -> float:
    if numerator is None or denominator in (None, 0):
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


# ============================================================
# FILTERING
# ============================================================


def _eq_filter(column: str, value: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        == value.strip().lower()
    )


def _apply_filters_expr(
    frame: pl.DataFrame,
    *,
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
    min_risk: float = 0.0,
    max_risk: float = 100.0,
) -> pl.DataFrame:
    result = frame
    predicates: list[pl.Expr] = []

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
        if value and value != "All" and column in result.columns:
            predicates.append(_eq_filter(column, value))

    if completion_status and completion_status != "All" and "is_completed" in result.columns:
        if completion_status.casefold() == "completed":
            predicates.append(pl.col("is_completed").fill_null(False))
        elif completion_status.casefold() == "open":
            predicates.append(~pl.col("is_completed").fill_null(False))

    if search and search.strip():
        term = search.strip().lower()
        search_columns = [
            column
            for column in (
                "work",
                "work_description",
                "mp",
                "constituency",
                "ida",
                "work_uid",
            )
            if column in result.columns
        ]
        if search_columns:
            predicates.append(
                pl.any_horizontal(
                    [
                        pl.col(column)
                        .cast(pl.String, strict=False)
                        .fill_null("")
                        .str.to_lowercase()
                        .str.contains(term, literal=True)
                        for column in search_columns
                    ]
                )
            )

    if "sanction_amount" in result.columns:
        if min_sanction is not None:
            predicates.append(pl.col("sanction_amount").ge(min_sanction))
        if max_sanction is not None:
            predicates.append(pl.col("sanction_amount").le(max_sanction))

    if "final_risk_score" in result.columns:
        predicates.append(pl.col("final_risk_score").is_between(min_risk, max_risk, closed="both"))

    if predicates:
        result = result.filter(pl.all_horizontal(predicates))

    return result


# ============================================================
# SUMMARY / ANALYTICS HELPERS
# ============================================================


def _summary(frame: pl.DataFrame) -> dict[str, Any]:
    total = frame.height
    completed = _count_true(frame, "is_completed")
    sanctioned = _sum(frame, "sanction_amount")
    expenditure = _sum(frame, "total_expenditure")

    utilization = None
    if sanctioned is not None and expenditure is not None and sanctioned > 0:
        utilization = round(float(expenditure) / float(sanctioned) * 100.0, 2)

    high = 0
    critical = 0
    if "risk_category" in frame.columns:
        high = frame.select(
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum()
        ).item() or 0
        critical = frame.select(
            pl.col("risk_category").eq("CRITICAL").sum()
        ).item() or 0

    return {
        "total_works": total,
        "completed_works": completed,
        "open_works": total - completed,
        "completion_rate_pct": _safe_rate(completed, total),
        "sanctioned_amount": sanctioned,
        "total_expenditure": expenditure,
        "portfolio_utilization_pct": utilization,
        "mean_risk": _mean(frame, "final_risk_score"),
        "median_risk": _median(frame, "final_risk_score"),
        "high_or_critical": int(high),
        "critical": int(critical),
        "high_or_critical_rate_pct": _safe_rate(high, total),
        "duplicate_candidates": _count_true(frame, "is_duplicate_candidate"),
        "overdue_open_works": _count_true(frame, "flag_overdue_open_work"),
        "multi_signal_cases": int(
            frame.select(
                pl.col("independent_signal_count").fill_null(0).ge(2).sum()
            ).item()
            or 0
        ),
    }


def _risk_band_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_null()).then(None)
        .when(pl.col(column).le(25)).then(pl.lit("LOW"))
        .when(pl.col(column).le(50)).then(pl.lit("MEDIUM"))
        .when(pl.col(column).le(75)).then(pl.lit("HIGH"))
        .otherwise(pl.lit("CRITICAL"))
        .alias("risk_category")
    )


def _group_risk_stats(frame: pl.DataFrame, group_column: str) -> pl.DataFrame:
    return (
        frame.group_by(group_column, maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
            pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
            pl.col("flag_overdue_open_work").fill_null(False).cast(pl.Int64).sum().alias("overdue_open_works"),
            pl.col("is_duplicate_candidate").fill_null(False).cast(pl.Int64).sum().alias("duplicate_candidates"),
        )
        .with_columns(
            pl.when(pl.col("sanctioned_amount") > 0)
            .then(pl.col("total_expenditure") / pl.col("sanctioned_amount") * 100)
            .otherwise(None)
            .alias("utilization_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("completed_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("completion_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("high_or_critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("high_or_critical_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("critical_rate_pct"),
        )
    )


# ============================================================
# APPLICATION
# ============================================================


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_data_once()
    yield


app = FastAPI(
    title="MPLADS AI Monitoring Platform",
    description=(
        "AI-assisted monitoring of MPLADS works using transparent risk signals, "
        "peer-relative anomaly detection, financial/execution indicators, "
        "duplicate candidates and explainable investigation prioritization."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# ============================================================
# ROOT / HEALTH / REFRESH
# ============================================================


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "MPLADS AI Monitoring API",
        "status": "running",
        "version": SERVICE_VERSION,
        "data_source": WORKS_SOURCE,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    frame = load_data_once()
    as_of = None
    if "monitoring_as_of_date" in frame.columns:
        value = frame.select(pl.col("monitoring_as_of_date").max()).item()
        as_of = _json_value(value)

    return {
        "status": "ok",
        "service": "MPLADS AI Monitoring API",
        "version": SERVICE_VERSION,
        "works_loaded": frame.height,
        "data_source": WORKS_SOURCE,
        "data_as_of": as_of,
    }


@app.post("/api/v1/refresh")
def refresh_data() -> dict[str, Any]:
    frame = load_data_once(force_refresh=True)
    return {
        "status": "ok",
        "works_loaded": frame.height,
        "data_source": WORKS_SOURCE,
    }


# ============================================================
# FILTER OPTIONS
# ============================================================


@app.get("/api/v1/filter-options")
def filter_options() -> dict[str, list[str]]:
    """Return synchronized, deterministic filter values from WORKS."""
    frame = load_data_once()

    def options(column: str) -> list[str]:
        if column not in frame.columns:
            return []

        values = (
            frame
            .select(
                pl.col(column)
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .alias("_value")
            )
            .filter(
                pl.col("_value").is_not_null()
                & pl.col("_value").ne("")
            )
            .unique()
            .sort("_value")
            .get_column("_value")
            .to_list()
        )
        return [str(value) for value in values if value is not None]

    return {
        "states": options("state"),
        "districts": options("ida"),
        "mps": options("mp"),
        "constituencies": options("constituency"),
        "work_categories": options("work_category"),
        "work_statuses": options("work_status"),
        "risk_categories": [
            value
            for value in RISK_ORDER
            if value in options("risk_category")
        ],
        "financial_years": options("financial_year"),
    }


# ============================================================
# OVERVIEW / RISK
# ============================================================


@app.get("/api/v1/dashboard-summary")
def dashboard_summary() -> dict[str, Any]:
    frame = load_data_once()
    result = _summary(frame)

    if "risk_category" in frame.columns:
        distribution = (
            frame.group_by("risk_category")
            .agg(pl.len().alias("count"))
            .with_columns(
                pl.when(pl.lit(frame.height) > 0)
                .then(pl.col("count") / frame.height * 100)
                .otherwise(0)
                .alias("pct")
            )
            .sort("risk_category")
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
) -> dict[str, Any]:
    if min_sanction is not None and max_sanction is not None and min_sanction > max_sanction:
        raise HTTPException(status_code=422, detail="min_sanction cannot exceed max_sanction")
    if min_risk > max_risk:
        raise HTTPException(status_code=422, detail="min_risk cannot exceed max_risk")

    frame = _apply_filters_expr(
        load_data_once(),
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
    return _summary(frame)


@app.get("/api/v1/risk-distribution")
def risk_distribution(
    risk_basis: str = Query(default="final", pattern="^(final|rule|ml)$"),
) -> list[dict[str, Any]]:
    frame = load_data_once()
    column = {
        "final": "final_risk_score",
        "rule": "rule_risk_score",
        "ml": "ml_anomaly_percentile",
    }[risk_basis]

    if column not in frame.columns:
        return []

    result = (
        frame
        .with_columns(_risk_band_expr(column))
        .filter(pl.col("risk_category").is_not_null())
        .group_by("risk_category")
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.when(pl.lit(frame.height) > 0)
            .then(pl.col("count") / frame.height * 100)
            .otherwise(0)
            .alias("pct")
        )
    )

    return to_records(result)


@app.get("/api/v1/risk-reasons")
def risk_reasons() -> list[dict[str, Any]]:
    frame = load_data_once()
    definitions = (
        ("flag_disbursement_over_sanction", "Expenditure exceeds sanction"),
        ("flag_overdue_open_work", "Open beyond general one-year benchmark"),
        ("flag_stalled_expenditure", "Long expenditure gap"),
        ("flag_duplicate_candidate", "Potential duplicate/similar work"),
        ("flag_cost_outlier", "Cost outlier within category"),
        ("flag_duration_outlier", "Duration outlier within category"),
        ("flag_bad_dates", "Date integrity issue"),
    )

    rows = []
    for column, label in definitions:
        if column not in frame.columns:
            continue
        count = _count_true(frame, column)
        rows.append({
            "reason": label,
            "count": count,
            "pct_of_all_works": _safe_rate(count, frame.height),
        })
    return rows


@app.get("/api/v1/filtered-analytics")
def filtered_analytics(
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
) -> dict[str, Any]:
    """
    Return ALL dashboard analytics from the same filtered dataframe.

    This endpoint is intentionally separate from /api/v1/works because the
    work explorer is capped/paginated, while charts must describe the complete
    active scope. This prevents a 500-row queue from being mistaken for the
    entire filtered population.
    """
    if min_sanction is not None and max_sanction is not None and min_sanction > max_sanction:
        raise HTTPException(status_code=422, detail="min_sanction cannot exceed max_sanction")
    if min_risk > max_risk:
        raise HTTPException(status_code=422, detail="min_risk cannot exceed max_risk")

    frame = _apply_filters_expr(
        load_data_once(),
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

    def risk_dist(column: str) -> list[dict[str, Any]]:
        if column not in frame.columns:
            return []
        result = (
            frame.with_columns(_risk_band_expr(column))
            .filter(pl.col("risk_category").is_not_null())
            .group_by("risk_category")
            .agg(pl.len().alias("count"))
            .with_columns(
                pl.when(pl.lit(frame.height) > 0)
                .then(pl.col("count") / frame.height * 100)
                .otherwise(0)
                .alias("pct")
            )
        )
        return to_records(result)

    reason_defs = (
        ("flag_disbursement_over_sanction", "Expenditure exceeds sanction"),
        ("flag_overdue_open_work", "Open beyond general one-year benchmark"),
        ("flag_stalled_expenditure", "Long expenditure gap"),
        ("flag_duplicate_candidate", "Potential duplicate/similar work"),
        ("flag_cost_outlier", "Cost outlier within category"),
        ("flag_duration_outlier", "Duration outlier within category"),
        ("flag_bad_dates", "Date integrity issue"),
    )
    reasons=[]
    for column,label in reason_defs:
        if column in frame.columns:
            count=_count_true(frame,column)
            reasons.append({
                "reason": label,
                "count": count,
                "pct_of_filtered_works": _safe_rate(count, frame.height),
            })

    def group_stats(column: str) -> list[dict[str, Any]]:
        if column not in frame.columns:
            return []
        return to_records(_group_risk_stats(frame,column).sort("high_or_critical_rate_pct",descending=True,nulls_last=True))

    sector=[]
    if "work_category" in frame.columns:
        sector_frame=(
            frame.group_by("work_category",maintain_order=True)
            .agg(
                pl.len().alias("works"),
                pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
                pl.col("sanction_amount").sum().alias("sanctioned_amount"),
                pl.col("total_expenditure").sum().alias("total_expenditure"),
                pl.col("final_risk_score").median().alias("median_risk"),
                pl.col("sanction_amount").median().alias("median_cost"),
                pl.col("risk_category").is_in(["HIGH","CRITICAL"]).sum().alias("high_or_critical"),
            )
            .with_columns(
                pl.when(pl.col("sanctioned_amount")>0).then(pl.col("total_expenditure")/pl.col("sanctioned_amount")*100).otherwise(None).alias("utilization_pct"),
                pl.when(pl.col("works")>0).then(pl.col("completed_works")/pl.col("works")*100).otherwise(0).alias("completion_rate_pct"),
                pl.when(pl.col("works")>0).then(pl.col("high_or_critical")/pl.col("works")*100).otherwise(0).alias("high_or_critical_rate_pct"),
            )
            .sort("high_or_critical_rate_pct",descending=True,nulls_last=True)
        )
        sector=to_records(sector_frame)

    time=[]
    if "financial_year" in frame.columns:
        time_frame=(
            frame.filter(pl.col("financial_year").is_not_null())
            .group_by("financial_year",maintain_order=True)
            .agg(
                pl.len().alias("works"),
                pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
                pl.col("sanction_amount").sum().alias("sanctioned_amount"),
                pl.col("total_expenditure").sum().alias("total_expenditure"),
                pl.col("final_risk_score").median().alias("median_risk"),
                pl.col("risk_category").is_in(["HIGH","CRITICAL"]).sum().alias("high_or_critical"),
                pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
            )
            .with_columns(
                pl.when(pl.col("sanctioned_amount")>0).then(pl.col("total_expenditure")/pl.col("sanctioned_amount")*100).otherwise(None).alias("utilization_pct"),
                pl.when(pl.col("works")>0).then(pl.col("completed_works")/pl.col("works")*100).otherwise(0).alias("completion_rate_pct"),
                pl.when(pl.col("works")>0).then(pl.col("high_or_critical")/pl.col("works")*100).otherwise(0).alias("high_or_critical_rate_pct"),
                pl.when(pl.col("works")>0).then(pl.col("critical")/pl.col("works")*100).otherwise(0).alias("critical_rate_pct"),
            )
        )
        time=to_records(time_frame.sort("financial_year"))

    return {
        "scope": _summary(frame),
        "risk_final": risk_dist("final_risk_score"),
        "risk_rule": risk_dist("rule_risk_score"),
        "risk_ml": risk_dist("ml_anomaly_percentile"),
        "risk_reasons": reasons,
        "state": group_stats("state"),
        "district": group_stats("ida"),
        "sector": sector,
        "time": time,
    }


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
    sort_by: str = Query(default="priority_score"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    if sort_by not in SORT_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported sort_by. Allowed values: {sorted(SORT_COLUMNS)}",
        )
    if min_sanction is not None and max_sanction is not None and min_sanction > max_sanction:
        raise HTTPException(status_code=422, detail="min_sanction cannot exceed max_sanction")
    if min_risk > max_risk:
        raise HTTPException(status_code=422, detail="min_risk cannot exceed max_risk")

    frame = _apply_filters_expr(
        load_data_once(),
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

    total = frame.height
    descending = sort_order == "desc"
    if sort_by in frame.columns:
        frame = frame.sort(sort_by, descending=descending, nulls_last=True)

    start = (page - 1) * page_size
    page_frame = frame.slice(start, page_size)

    preferred = (
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
    )
    selected = [column for column in preferred if column in page_frame.columns]

    return {
        "items": to_records(page_frame.select(selected)),
        "page": page,
        "page_size": page_size,
        "total_matching": total,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


@app.get("/api/v1/works/{work_uid}")
def work_detail(work_uid: str) -> dict[str, Any]:
    frame = load_data_once()
    if "work_uid" not in frame.columns:
        raise HTTPException(status_code=500, detail="work_uid is missing from analytical dataset")

    match = frame.filter(pl.col("work_uid").cast(pl.String, strict=False) == str(work_uid))
    if match.is_empty():
        raise HTTPException(status_code=404, detail="Work not found")

    row = match.row(0, named=True)
    component_columns = (
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
    )

    return {
        "work": {key: _json_value(value) for key, value in row.items()},
        "risk_components": {
            key: _json_value(row[key])
            for key in component_columns
            if key in row
        },
        "evidence": row.get("evidence_json"),
    }


@app.get("/api/v1/works/{work_uid}/similar")
def work_similar(
    work_uid: str,
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict[str, Any]]:
    frame = load_data_once()
    if "work_uid" not in frame.columns:
        raise HTTPException(status_code=500, detail="work_uid is missing")

    match = frame.filter(pl.col("work_uid").cast(pl.String, strict=False) == str(work_uid))
    if match.is_empty():
        raise HTTPException(status_code=404, detail="Work not found")

    row = match.row(0, named=True)
    candidates = frame.filter(pl.col("work_uid").cast(pl.String, strict=False) != str(work_uid))

    # Conservative exact-context matching only. No fabricated semantic score.
    keys = ["state", "ida", "constituency", "work_category"]
    predicates = []
    for column in keys:
        value = row.get(column)
        if column in candidates.columns and value not in (None, ""):
            predicates.append(
                pl.col(column).cast(pl.String, strict=False) == str(value)
            )

    if predicates:
        subset = candidates.filter(pl.all_horizontal(predicates))
    else:
        subset = candidates.head(0)

    columns = [
        column
        for column in (
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
        )
        if column in subset.columns
    ]

    if "priority_score" in subset.columns:
        subset = subset.sort("priority_score", descending=True, nulls_last=True)
    return to_records(subset.head(limit).select(columns))


# ============================================================
# AGGREGATED ANALYTICS
# ============================================================


@app.get("/api/v1/state-analytics")
def state_analytics() -> list[dict[str, Any]]:
    frame = load_data_once()
    if "state" not in frame.columns:
        return []

    result = _group_risk_stats(frame, "state")
    return to_records(result.sort("high_or_critical_rate_pct", descending=True, nulls_last=True))


@app.get("/api/v1/district-analytics")
def district_analytics(state: str | None = None) -> list[dict[str, Any]]:
    frame = load_data_once()
    if state and state != "All" and "state" in frame.columns:
        frame = frame.filter(_eq_filter("state", state))

    if "ida" not in frame.columns:
        return []

    result = _group_risk_stats(frame, "ida")
    return to_records(result.sort("high_or_critical_rate_pct", descending=True, nulls_last=True))


@app.get("/api/v1/sector-analytics")
def sector_analytics() -> list[dict[str, Any]]:
    frame = load_data_once()
    if "work_category" not in frame.columns:
        return []

    result = (
        frame.group_by("work_category", maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("sanction_amount").median().alias("median_cost"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
        )
        .with_columns(
            pl.when(pl.col("sanctioned_amount") > 0)
            .then(pl.col("total_expenditure") / pl.col("sanctioned_amount") * 100)
            .otherwise(None)
            .alias("utilization_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("completed_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("completion_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("high_or_critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("high_or_critical_rate_pct"),
        )
        .sort("high_or_critical_rate_pct", descending=True, nulls_last=True)
    )
    return to_records(result)


@app.get("/api/v1/time-series")
def time_series() -> list[dict[str, Any]]:
    frame = load_data_once()
    if "financial_year" not in frame.columns:
        return []

    result = (
        frame.filter(pl.col("financial_year").is_not_null())
        .group_by("financial_year", maintain_order=True)
        .agg(
            pl.len().alias("works"),
            pl.col("is_completed").fill_null(False).cast(pl.Int64).sum().alias("completed_works"),
            pl.col("sanction_amount").sum().alias("sanctioned_amount"),
            pl.col("total_expenditure").sum().alias("total_expenditure"),
            pl.col("final_risk_score").median().alias("median_risk"),
            pl.col("risk_category").is_in(["HIGH", "CRITICAL"]).sum().alias("high_or_critical"),
            pl.col("risk_category").eq("CRITICAL").sum().alias("critical"),
        )
        .with_columns(
            pl.when(pl.col("sanctioned_amount") > 0)
            .then(pl.col("total_expenditure") / pl.col("sanctioned_amount") * 100)
            .otherwise(None)
            .alias("utilization_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("completed_works") / pl.col("works") * 100)
            .otherwise(0)
            .alias("completion_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("high_or_critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("high_or_critical_rate_pct"),
            pl.when(pl.col("works") > 0)
            .then(pl.col("critical") / pl.col("works") * 100)
            .otherwise(0)
            .alias("critical_rate_pct"),
        )
    )

    # Prefer chronological FY ordering when labels are YYYY-YY.
    return to_records(result.sort("financial_year"))


@app.get("/api/v1/geography-points")
def geography_points(
    risk_min: float = Query(default=0, ge=0, le=100),
) -> dict[str, Any]:
    frame = load_data_once()
    lat_col = next(
        (column for column in ("latitude", "lat", "work_latitude") if column in frame.columns),
        None,
    )
    lon_col = next(
        (column for column in ("longitude", "lon", "lng", "work_longitude") if column in frame.columns),
        None,
    )

    if not lat_col or not lon_col:
        return {
            "available": False,
            "reason": "No latitude/longitude columns in analytical dataset",
            "points": [],
        }

    subset = (
        frame
        .filter(pl.col("final_risk_score").ge(risk_min))
        .with_columns(
            pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
            pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon"),
        )
        .filter(
            pl.col("lat").is_not_null()
            & pl.col("lon").is_not_null()
            & pl.col("lat").is_between(-90, 90, closed="both")
            & pl.col("lon").is_between(-180, 180, closed="both")
        )
        .head(5000)
    )

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
        "lat",
        "lon",
    ]
    selected = [column for column in preferred if column in subset.columns]
    return {"available": True, "points": to_records(subset.select(selected))}


if __name__ == "__main__":
    # Direct execution is intentionally simple; use Uvicorn for the server.
    print("MPLADS AI MONITOR — FastAPI")
    print("API: http://127.0.0.1:8000")
    print("Docs: http://127.0.0.1:8000/docs")
