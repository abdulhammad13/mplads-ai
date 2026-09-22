from __future__ import annotations

import hashlib
import os
from datetime import date, datetime
from pathlib import Path
from typing import Final

import polars as pl

from .data_loader import load_data


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

PROCESSED_DIR: Final[Path] = (
    PROJECT_ROOT / "data" / "processed"
)

MASTER_PATH: Final[Path] = (
    PROCESSED_DIR / "master_works.csv"
)

EXPENDITURE_PATH: Final[Path] = (
    PROCESSED_DIR / "expenditure_summary.csv"
)


# ------------------------------------------------------------
# Pipeline version
# ------------------------------------------------------------
#
# 4.0 was the previous Pandas implementation.
# 5.0 represents the Polars-native implementation while
# preserving the analytical methodology and output fields.
#

PIPELINE_VERSION: Final[str] = "5.0"


# ============================================================
# ANALYTICAL CONFIGURATION
# ============================================================

GENERAL_ONE_YEAR_BENCHMARK_DAYS: Final[int] = 365

MIN_DUPLICATE_TEXT_LENGTH: Final[int] = 8


# ============================================================
# SOURCE COLUMN DEFINITIONS
# ============================================================

FINANCIAL_COLUMNS: Final[tuple[str, ...]] = (
    "sanction_amount",
    "recommended_amount",
    "completed_amount",
    "allocated_amount",
    "total_expenditure",
)


DATE_COLUMNS: Final[tuple[str, ...]] = (
    "recommended_date",
    "sanction_date",
    "completion_date",
    "expenditure_date",
)


TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "state",
    "district",
    "mp",
    "constituency",
    "ida",
    "work",
    "work_description",
    "work_category",
    "work_status",
    "vendor_name",
    "payment_status",
)


# ============================================================
# DATE PARSING
# ============================================================

DATE_FORMATS: Final[tuple[str, ...]] = (
    # Date only
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%m/%d/%Y",

    # Date + time
    "%d/%m/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",

    # Fractional seconds
    "%d/%m/%Y %H:%M:%S%.f",
    "%d-%m-%Y %H:%M:%S%.f",
    "%d.%m.%Y %H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%m/%d/%Y %H:%M:%S%.f",

    # Month names
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d-%b-%Y %H:%M:%S",
    "%d-%B-%Y %H:%M:%S",
)


# ============================================================
# GENERIC POLARS HELPERS
# ============================================================

def numeric_expr(column: str) -> pl.Expr:
    """
    Convert an arbitrary source column into Float64.

    Invalid values become null.

    This replaces the previous Pandas:
        pd.to_numeric(..., errors="coerce")
    """

    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.replace_all(",", "")
        .str.replace_all("₹", "")
        .str.replace_all(
            r"(?i)\brs\.?\b",
            "",
        )
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(column)
    )


def date_expr(column: str) -> pl.Expr:
    """
    Parse a source date column using explicit formats.

    Invalid values become null.

    The result is normalized to Polars Date because all current
    feature calculations operate at calendar-day precision.
    """

    value = (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
    )

    expressions = [
        value.str.strptime(
            pl.Datetime,
            format=fmt,
            strict=False,
        )
        for fmt in DATE_FORMATS
    ]

    return (
        pl.coalesce(expressions)
        .cast(pl.Date, strict=False)
        .alias(column)
    )


def clean_text_expr(column: str) -> pl.Expr:
    """
    Deterministically normalize text for matching/search.

    Null values become empty strings.
    """

    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias(column)
    )


def clean_optional_text_expr(column: str) -> pl.Expr:
    """
    Normalize optional text while retaining nulls.
    """

    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias(column)
    )


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def ensure_columns(
    df: pl.DataFrame,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    """
    Create absent analytical columns as null values.

    Existing columns are never overwritten.
    """

    expressions: list[pl.Expr] = []

    for column in columns:

        if column not in df.columns:
            expressions.append(
                pl.lit(None)
                .cast(pl.String)
                .alias(column)
            )

    if expressions:
        df = df.with_columns(expressions)

    return df


def ensure_typed_columns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Ensure all required analytical columns exist with useful
    Polars dtypes.

    This is intentionally explicit so downstream modules receive
    predictable schemas.
    """

    # --------------------------------------------------------
    # Text columns
    # --------------------------------------------------------

    for column in (
        "work",
        "work_description",
        "state",
        "district",
        "mp",
        "constituency",
        "ida",
        "work_category",
        "work_status",
        "vendor_name",
        "payment_status",
        "financial_year",
        "link_key",
        "work_text",
        "work_uid",
        "expenditure_source",
        "pipeline_version",
    ):
        if column not in df.columns:
            df = df.with_columns(
                pl.lit(None)
                .cast(pl.String)
                .alias(column)
            )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in (
        "sanction_amount",
        "recommended_amount",
        "completed_amount",
        "allocated_amount",
        "total_expenditure",
        "expenditure_variance_amount",
        "expenditure_variance_pct",
        "utilization_pct",
        "overspend_pct",
        "overrun_pct",
        "disbursement_to_sanction_ratio",
        "days_rec_to_sanction",
        "days_sanction_to_complete",
        "days_open_since_sanction",
        "days_since_last_expenditure",
        "days_over_general_one_year_benchmark",
        "duplicate_group_count",
        "num_transactions",
        "num_vendors",
        "data_quality_issue_count",
    ):
        if column not in df.columns:
            df = df.with_columns(
                pl.lit(None)
                .cast(pl.Float64)
                .alias(column)
            )

    # --------------------------------------------------------
    # Boolean columns
    # --------------------------------------------------------

    for column in (
        "is_completed",
        "is_open",
        "is_duplicate_candidate",
        "bad_recommendation_date",
        "bad_completion_date",
        "future_sanction_date",
        "future_completion_date",
        "negative_sanction_amount",
        "negative_expenditure",
        "missing_sanction_amount",
        "missing_sanction_date",
    ):
        if column not in df.columns:
            df = df.with_columns(
                pl.lit(False).alias(column)
            )

    # --------------------------------------------------------
    # Date columns
    # --------------------------------------------------------

    for column in (
        "recommended_date",
        "sanction_date",
        "completion_date",
        "first_expenditure_date",
        "last_expenditure_date",
        "monitoring_as_of_date",
    ):
        if column not in df.columns:
            df = df.with_columns(
                pl.lit(None)
                .cast(pl.Date)
                .alias(column)
            )

    return df


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(
    df: pl.DataFrame,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    """
    Normalize all available date columns to Polars Date.
    """

    expressions = [
        date_expr(column)
        for column in columns
        if column in df.columns
    ]

    if expressions:
        df = df.with_columns(expressions)

    return df


# ============================================================
# LINK KEY
# ============================================================

def make_link_key(df: pl.DataFrame) -> pl.Series:
    """
    Create the backward-compatible work linkage key.

    Existing MPLADS source design uses:

        Work + IDA

    The linkage is normalized to:

        normalized_work|normalized_ida

    When both Work and IDA are missing, the resulting key is null.
    """

    height = df.height

    # --------------------------------------------------------
    # Work
    # --------------------------------------------------------

    if "work" in df.columns:
        work = (
            df["work"]
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.strip_chars()
            .str.to_lowercase()
        )
    else:
        work = pl.Series(
            "work",
            [""] * height,
            dtype=pl.String,
        )

    # --------------------------------------------------------
    # IDA
    # --------------------------------------------------------

    if "ida" in df.columns:
        ida = (
            df["ida"]
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.strip_chars()
            .str.to_lowercase()
        )
    else:
        ida = pl.Series(
            "ida",
            [""] * height,
            dtype=pl.String,
        )

    # --------------------------------------------------------
    # Build linkage key
    # --------------------------------------------------------

    key = (
        work
        + "|"
        + ida
    )

    # --------------------------------------------------------
    # Do not create a misleading key when both source fields
    # are actually missing.
    # --------------------------------------------------------

    if "work" in df.columns and "ida" in df.columns:

        both_missing = (
            df["work"].is_null()
            & df["ida"].is_null()
        )

        key = key.zip_with(
            ~both_missing,
            pl.Series(
                "null_key",
                [None] * height,
                dtype=pl.String,
            ),
        )

    # --------------------------------------------------------
    # Return a correctly named Series.
    #
    # Series.rename() returns a new Series; it does not mutate
    # the existing Series name.
    # --------------------------------------------------------

    return key.rename("link_key")


# ============================================================
# STABLE WORK UID
# ============================================================

def _stable_hash(value: str) -> str:
    """
    Generate a deterministic SHA-1 identifier.

    SHA-1 is used here only as a deterministic record fingerprint,
    not for security.
    """

    return hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:20]


def _stable_value(value: object) -> str:
    """
    Convert a value into a deterministic normalized string.
    """

    if value is None:
        return ""

    text = str(value).strip().lower()

    return text


def stable_work_uid(
    row: dict[str, object],
) -> str:
    """
    Create a stable work identifier.

    Priority:

        1. work_id
        2. id
        3. work_code
        4. link_key
        5. deterministic business-attribute hash

    The same strategy as the existing implementation is retained.
    """

    for candidate in (
        "work_id",
        "id",
        "work_code",
        "link_key",
    ):

        value = row.get(candidate)

        if value is not None:

            normalized = _stable_value(value)

            if normalized:
                return normalized

    fields = [
        row.get("state"),
        row.get("ida"),
        row.get("mp"),
        row.get("constituency"),
        row.get("work"),
        row.get("recommended_date"),
        row.get("sanction_date"),
        row.get("sanction_amount"),
    ]

    payload = "|".join(
        _stable_value(value)
        for value in fields
    )

    return _stable_hash(payload)


def add_stable_work_uid(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add deterministic work_uid values.
    """

    # Convert rows only for this identifier-generation operation.
    #
    # The analytical dataframe remains entirely Polars-based.
    #

    rows = df.select(
        [
            column
            for column in (
                "work_id",
                "id",
                "work_code",
                "link_key",
                "state",
                "ida",
                "mp",
                "constituency",
                "work",
                "recommended_date",
                "sanction_date",
                "sanction_amount",
            )
            if column in df.columns
        ]
    ).to_dicts()

    uids = [
        stable_work_uid(row)
        for row in rows
    ]

    return df.with_columns(
        pl.Series(
            "work_uid",
            uids,
            dtype=pl.String,
        )
    )


# ============================================================
# WORK UID VALIDATION
# ============================================================

def validate_unique_work_uid(
    df: pl.DataFrame,
) -> None:
    """
    Fail early if stable work identifiers are duplicated.
    """

    if "work_uid" not in df.columns:
        raise ValueError(
            "Missing required column: work_uid"
        )

    duplicate_mask = (
        df["work_uid"]
        .is_duplicated()
        & df["work_uid"].is_not_null()
    )

    if not duplicate_mask.any():
        return

    examples = (
        df
        .filter(duplicate_mask)
        .select("work_uid")
        .head(10)
        .to_series()
        .to_list()
    )

    raise ValueError(
        "Duplicate work_uid values detected. "
        f"Examples: {examples}"
    )


# ============================================================
# FINANCIAL YEAR
# ============================================================

def financial_year_expr(
    date_column: str,
) -> pl.Expr:
    """
    Derive Indian financial-year labels.

    Examples:

        2025-03-31 -> 2024-25
        2025-04-01 -> 2025-26
    """

    return (
        pl.when(pl.col(date_column).is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(pl.col(date_column).dt.month() >= 4)
        .then(
            pl.concat_str(
                [
                    pl.col(date_column)
                    .dt.year()
                    .cast(pl.String),

                    pl.lit("-"),

                    (
                        pl.col(date_column)
                        .dt.year()
                        .add(1)
                        .mod(100)
                        .cast(pl.String)
                        .str.zfill(2)
                    ),
                ]
            )
        )
        .otherwise(
            pl.concat_str(
                [
                    (
                        pl.col(date_column)
                        .dt.year()
                        .sub(1)
                        .cast(pl.String)
                    ),

                    pl.lit("-"),

                    (
                        pl.col(date_column)
                        .dt.year()
                        .mod(100)
                        .cast(pl.String)
                        .str.zfill(2)
                    ),
                ]
            )
        )
        .alias("derived_financial_year")
    )


# ============================================================
# MONITORING SNAPSHOT
# ============================================================

def get_as_of_date() -> date:
    """
    Resolve the monitoring snapshot date.

    Production/batch jobs should set:

        MPLADS_AS_OF_DATE=YYYY-MM-DD

    so that monitoring results are reproducible.

    If it is not supplied, today's local system date is used.
    """

    raw = os.getenv(
        "MPLADS_AS_OF_DATE",
        "",
    ).strip()

    if raw:

        try:
            return datetime.strptime(
                raw,
                "%Y-%m-%d",
            ).date()

        except ValueError as exc:

            raise ValueError(
                f"Invalid MPLADS_AS_OF_DATE: {raw!r}. "
                "Expected YYYY-MM-DD."
            ) from exc

    return datetime.now().date()


# ============================================================
# EXPENDITURE SUMMARY
# ============================================================

def build_expenditure_summary(
    data: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    Aggregate expenditure transactions by link_key.

    Outputs:

        link_key
        total_expenditure
        num_transactions
        num_vendors
        first_expenditure_date
        last_expenditure_date

    Transaction-level expenditure remains authoritative for
    financial totals.
    """

    expenditure = data.get(
        "expenditure"
    )

    if expenditure is None:
        return pl.DataFrame(
            schema={
                "link_key": pl.String,
                "total_expenditure": pl.Float64,
                "num_transactions": pl.Int64,
                "num_vendors": pl.Int64,
                "first_expenditure_date": pl.Date,
                "last_expenditure_date": pl.Date,
            }
        )

    if expenditure.is_empty():

        return pl.DataFrame(
            schema={
                "link_key": pl.String,
                "total_expenditure": pl.Float64,
                "num_transactions": pl.Int64,
                "num_vendors": pl.Int64,
                "first_expenditure_date": pl.Date,
                "last_expenditure_date": pl.Date,
            }
        )

    # --------------------------------------------------------
    # Normalize source fields.
    # --------------------------------------------------------

    expenditure = normalize_dates(
        expenditure,
        ("expenditure_date",),
    )

    if "fund_disbursed_amount" in expenditure.columns:

        expenditure = expenditure.with_columns(
            numeric_expr(
                "fund_disbursed_amount"
            )
        )

    # --------------------------------------------------------
    # Link key.
    # --------------------------------------------------------

    expenditure = expenditure.with_columns(
        make_link_key(expenditure)
    )

    expenditure = expenditure.filter(
        pl.col("link_key").is_not_null()
    )

    if expenditure.is_empty():

        return pl.DataFrame(
            schema={
                "link_key": pl.String,
                "total_expenditure": pl.Float64,
                "num_transactions": pl.Int64,
                "num_vendors": pl.Int64,
                "first_expenditure_date": pl.Date,
                "last_expenditure_date": pl.Date,
            }
        )

    # --------------------------------------------------------
    # Transaction count.
    # --------------------------------------------------------

    if "work_id" in expenditure.columns:

        transaction_count = (
            pl.col("work_id")
            .count()
            .alias("num_transactions")
        )

    else:

        transaction_count = (
            pl.len()
            .alias("num_transactions")
        )

    # --------------------------------------------------------
    # Vendor count.
    # --------------------------------------------------------

    if "vendor_name" in expenditure.columns:

        vendor_count = (
            pl.col("vendor_name")
            .cast(pl.String, strict=False)
            .str.strip_chars()
            .filter(
                pl.col("vendor_name")
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .ne("")
            )
            .n_unique()
            .alias("num_vendors")
        )

    else:

        vendor_count = (
            pl.lit(0)
            .cast(pl.Int64)
            .alias("num_vendors")
        )

    # --------------------------------------------------------
    # Aggregate.
    # --------------------------------------------------------

    summary = (
        expenditure
        .group_by("link_key")
        .agg(
            pl.col("fund_disbursed_amount")
            .sum()
            .alias("total_expenditure"),

            transaction_count,

            vendor_count,

            pl.col("expenditure_date")
            .min()
            .alias("first_expenditure_date"),

            pl.col("expenditure_date")
            .max()
            .alias("last_expenditure_date"),
        )
    )

    return summary.with_columns(
        pl.col("total_expenditure")
        .cast(pl.Float64, strict=False)
    )


# ============================================================
# MASTER WORK TABLE
# ============================================================

def build_master(
    data: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    Build the master analytical work table.

    Master grain:

        One row per linked sanctioned work.

    Existing analytical methodology is preserved:

        sanctioned
            ↓
        recommended metadata
            ↓
        completed metadata
            ↓
        expenditure transaction summary
            ↓
        allocation
            ↓
        financial metrics
            ↓
        time metrics
            ↓
        duplicate indicators
            ↓
        data-quality indicators
            ↓
        stable work UID
    """

    # ========================================================
    # SOURCE DATA
    # ========================================================

    required_sources = (
        "sanctioned",
        "recommended",
        "completed",
        "allocation",
    )

    missing_sources = [
        source
        for source in required_sources
        if source not in data
    ]

    if missing_sources:

        raise KeyError(
            "Missing required datasets: "
            f"{missing_sources}"
        )

    sanctioned = data["sanctioned"]
    recommended = data["recommended"]
    completed = data["completed"]
    allocation = data["allocation"]

    expenditure = data.get(
        "expenditure",
        pl.DataFrame(),
    )

    # ========================================================
    # SOURCE NORMALIZATION
    # ========================================================

    source_frames = {
        "sanctioned": sanctioned,
        "recommended": recommended,
        "completed": completed,
        "allocation": allocation,
        "expenditure": expenditure,
    }

    numeric_map = {
        "sanctioned": (
            "sanction_amount",
        ),
        "recommended": (
            "recommended_amount",
        ),
        "completed": (
            "amount_disbursed",
        ),
        "allocation": (
            "allocated_amount",
        ),
        "expenditure": (
            "fund_disbursed_amount",
        ),
    }

    date_map = {
        "sanctioned": (
            "recommended_date",
            "sanction_date",
        ),
        "recommended": (
            "recommended_date",
            "sanction_date",
        ),
        "completed": (
            "completion_date",
        ),
        "expenditure": (
            "expenditure_date",
        ),
        "allocation": (),
    }

    normalized_sources: dict[
        str,
        pl.DataFrame,
    ] = {}

    for name, frame in source_frames.items():

        if frame is None or frame.is_empty():

            normalized_sources[name] = frame

            continue

        expressions: list[pl.Expr] = []

        for column in numeric_map.get(
            name,
            (),
        ):

            if column in frame.columns:
                expressions.append(
                    numeric_expr(column)
                )

        for column in date_map.get(
            name,
            (),
        ):

            if column in frame.columns:
                expressions.append(
                    date_expr(column)
                )

        if expressions:
            frame = frame.with_columns(
                expressions
            )

        normalized_sources[name] = frame

    sanctioned = normalized_sources["sanctioned"]
    recommended = normalized_sources["recommended"]
    completed = normalized_sources["completed"]
    allocation = normalized_sources["allocation"]
    expenditure = normalized_sources["expenditure"]

    # ========================================================
    # LINK KEYS
    # ========================================================

    sanctioned = sanctioned.with_columns(
        make_link_key(sanctioned)
    )

    recommended = recommended.with_columns(
        make_link_key(recommended)
    )

    completed = completed.with_columns(
        make_link_key(completed)
    )

    if expenditure is not None and not expenditure.is_empty():

        expenditure = expenditure.with_columns(
            make_link_key(expenditure)
        )

    # ========================================================
    # MASTER = SANCTIONED WORKS
    # ========================================================

    sanctioned = sanctioned.filter(
        pl.col("link_key").is_not_null()
    )

    if sanctioned.is_empty():

        raise ValueError(
            "No valid sanctioned works remain after "
            "link-key generation."
        )

    # --------------------------------------------------------
    # One row per link_key.
    #
    # Latest sanction_date wins when duplicates exist.
    # --------------------------------------------------------

    if "sanction_date" in sanctioned.columns:

        sanctioned = (
            sanctioned
            .sort(
                [
                    "link_key",
                    "sanction_date",
                ],
                nulls_last=True,
            )
            .unique(
                subset=["link_key"],
                keep="last",
                maintain_order=True,
            )
        )

    else:

        sanctioned = sanctioned.unique(
            subset=["link_key"],
            keep="last",
            maintain_order=True,
        )

    master = sanctioned

    # ========================================================
    # RECOMMENDED AMOUNT
    # ========================================================

    if (
        not recommended.is_empty()
        and "recommended_amount"
        in recommended.columns
    ):

        recommended_small = recommended.filter(
            pl.col("link_key").is_not_null()
        )

        if "recommended_date" in recommended_small.columns:

            recommended_small = (
                recommended_small
                .sort(
                    [
                        "link_key",
                        "recommended_date",
                    ],
                    nulls_last=True,
                )
                .unique(
                    subset=["link_key"],
                    keep="last",
                    maintain_order=True,
                )
            )

        else:

            recommended_small = (
                recommended_small
                .unique(
                    subset=["link_key"],
                    keep="last",
                    maintain_order=True,
                )
            )

        recommended_small = recommended_small.select(
            [
                "link_key",
                "recommended_amount",
            ]
        )

        master = master.join(
            recommended_small,
            on="link_key",
            how="left",
            validate="1:1",
        )

    else:

        master = master.with_columns(
            pl.lit(None)
            .cast(pl.Float64)
            .alias("recommended_amount")
        )

    # ========================================================
    # COMPLETED METADATA
    # ========================================================

    if not completed.is_empty():

        completed_small = completed.filter(
            pl.col("link_key").is_not_null()
        )

        if "completion_date" in completed_small.columns:

            completed_small = (
                completed_small
                .sort(
                    [
                        "link_key",
                        "completion_date",
                    ],
                    nulls_last=True,
                )
                .unique(
                    subset=["link_key"],
                    keep="last",
                    maintain_order=True,
                )
            )

        else:

            completed_small = (
                completed_small
                .unique(
                    subset=["link_key"],
                    keep="last",
                    maintain_order=True,
                )
            )

        selected = ["link_key"]

        if "completion_date" in completed_small.columns:
            selected.append("completion_date")

        if "amount_disbursed" in completed_small.columns:
            selected.append("amount_disbursed")

        completed_small = completed_small.select(
            selected
        )

        if "amount_disbursed" in completed_small.columns:

            completed_small = completed_small.rename(
                {
                    "amount_disbursed":
                    "completed_amount"
                }
            )

        else:

            completed_small = completed_small.with_columns(
                pl.lit(None)
                .cast(pl.Float64)
                .alias("completed_amount")
            )

        master = master.join(
            completed_small,
            on="link_key",
            how="left",
            validate="1:1",
        )

    else:

        master = master.with_columns(
            [
                pl.lit(None)
                .cast(pl.Date)
                .alias("completion_date"),

                pl.lit(None)
                .cast(pl.Float64)
                .alias("completed_amount"),
            ]
        )

    # ========================================================
    # EXPENDITURE SUMMARY
    # ========================================================

    expenditure_summary = (
        build_expenditure_summary(
            {
                "expenditure": expenditure
            }
        )
    )

    if not expenditure_summary.is_empty():

        master = master.join(
            expenditure_summary,
            on="link_key",
            how="left",
            validate="1:1",
        )

    else:

        master = master.with_columns(
            [
                pl.lit(None)
                .cast(pl.Float64)
                .alias("total_expenditure"),

                pl.lit(None)
                .cast(pl.Int64)
                .alias("num_transactions"),

                pl.lit(None)
                .cast(pl.Int64)
                .alias("num_vendors"),

                pl.lit(None)
                .cast(pl.Date)
                .alias("first_expenditure_date"),

                pl.lit(None)
                .cast(pl.Date)
                .alias("last_expenditure_date"),
            ]
        )

    # ========================================================
    # ALLOCATION
    # ========================================================

    allocation_columns = [
        column
        for column in (
            "state",
            "mp",
            "constituency",
        )
        if column in allocation.columns
        and column in master.columns
    ]

    if "financial_year" in allocation.columns:
        allocation_columns.append(
            "financial_year"
        )

    if allocation_columns:

        allocation = allocation.filter(
            pl.all_horizontal(
                [
                    pl.col(column)
                    .is_not_null()
                    for column in allocation_columns
                ]
            )
        )

    if (
        not allocation.is_empty()
        and allocation_columns
        and "allocated_amount"
        in allocation.columns
    ):

        allocation_small = (
            allocation
            .group_by(
                allocation_columns
            )
            .agg(
                pl.col("allocated_amount")
                .max()
                .alias("allocated_amount")
            )
        )

        join_columns = [
            column
            for column in allocation_columns
            if column in master.columns
        ]

        if join_columns:

            master = master.join(
                allocation_small,
                on=join_columns,
                how="left",
                validate="m:1",
            )

        else:

            master = master.with_columns(
                pl.lit(None)
                .cast(pl.Float64)
                .alias("allocated_amount")
            )

    else:

        master = master.with_columns(
            pl.lit(None)
            .cast(pl.Float64)
            .alias("allocated_amount")
        )

    # ========================================================
    # REQUIRED ANALYTICAL COLUMNS
    # ========================================================

    required_columns = (
        "work",
        "work_description",
        "state",
        "district",
        "mp",
        "constituency",
        "ida",
        "work_category",
        "work_status",
        "recommended_date",
        "sanction_date",
        "completion_date",
        "sanction_amount",
        "recommended_amount",
        "completed_amount",
        "allocated_amount",
        "total_expenditure",
        "first_expenditure_date",
        "last_expenditure_date",
    )

    # Add missing columns with nulls.
    for column in required_columns:

        if column not in master.columns:

            if column.endswith("_date"):

                dtype = pl.Date

            elif column in FINANCIAL_COLUMNS:

                dtype = pl.Float64

            else:

                dtype = pl.String

            master = master.with_columns(
                pl.lit(None)
                .cast(dtype)
                .alias(column)
            )

    # ========================================================
    # FINAL TYPE NORMALIZATION
    # ========================================================

    expressions: list[pl.Expr] = []

    for column in FINANCIAL_COLUMNS:

        if column in master.columns:

            expressions.append(
                numeric_expr(column)
            )

    for column in (
        "recommended_date",
        "sanction_date",
        "completion_date",
        "first_expenditure_date",
        "last_expenditure_date",
    ):

        if column in master.columns:

            expressions.append(
                pl.col(column)
                .cast(pl.Date, strict=False)
                .alias(column)
            )

    if expressions:
        master = master.with_columns(
            expressions
        )

    # ========================================================
    # COMPLETION STATUS
    # ========================================================

    master = master.with_columns(
        [
            pl.col("completion_date")
            .is_not_null()
            .alias("is_completed"),

            pl.col("completion_date")
            .is_null()
            .alias("is_open"),
        ]
    )

    # ========================================================
    # EXPENDITURE SOURCE
    # ========================================================

    master = master.with_columns(
        pl.when(
            pl.col("total_expenditure")
            .is_not_null()
        )
        .then(
            pl.lit("transaction_sum")
        )
        .when(
            pl.col("completed_amount")
            .is_not_null()
        )
        .then(
            pl.lit("completed_amount_fallback")
        )
        .otherwise(
            pl.lit("missing")
        )
        .alias("expenditure_source")
    )

    # --------------------------------------------------------
    # Transaction sum is authoritative whenever available.
    # Otherwise fall back to completed amount.
    # --------------------------------------------------------

    master = master.with_columns(
        pl.coalesce(
            [
                pl.col("total_expenditure"),
                pl.col("completed_amount"),
            ]
        )
        .alias("total_expenditure")
    )

    # ========================================================
    # FINANCIAL METRICS
    # ========================================================

    valid_sanction = (
        pl.col("sanction_amount")
        .is_not_null()
        & pl.col("sanction_amount")
        .gt(0)
    )

    master = master.with_columns(
        [
            pl.when(valid_sanction)
            .then(
                pl.col("total_expenditure")
                - pl.col("sanction_amount")
            )
            .otherwise(None)
            .alias(
                "expenditure_variance_amount"
            ),

            pl.when(valid_sanction)
            .then(
                (
                    (
                        pl.col("total_expenditure")
                        - pl.col("sanction_amount")
                    )
                    / pl.col("sanction_amount")
                    * 100.0
                )
            )
            .otherwise(None)
            .alias(
                "expenditure_variance_pct"
            ),

            pl.when(valid_sanction)
            .then(
                (
                    pl.col("total_expenditure")
                    / pl.col("sanction_amount")
                    * 100.0
                )
            )
            .otherwise(None)
            .alias(
                "utilization_pct"
            ),
        ]
    )

    # Positive expenditure variance only.
    master = master.with_columns(
        pl.col("expenditure_variance_pct")
        .clip(lower_bound=0.0)
        .alias("overspend_pct")
    )

    # ========================================================
    # LEGACY MODEL FEATURE
    # ========================================================
    #
    # The existing Isolation Forest historically used the
    # feature name "overrun_pct".
    #
    # Preserve the alias so the existing model remains
    # compatible.
    #

    master = master.with_columns(
        pl.col("overspend_pct")
        .alias("overrun_pct")
    )

    # ========================================================
    # DISBURSEMENT / SANCTION RATIO
    # ========================================================

    master = master.with_columns(
        pl.when(valid_sanction)
        .then(
            pl.col("total_expenditure")
            / pl.col("sanction_amount")
        )
        .otherwise(None)
        .alias(
            "disbursement_to_sanction_ratio"
        )
    )

    # ========================================================
    # TIME FEATURES
    # ========================================================

    master = master.with_columns(
        [
            (
                pl.col("sanction_date")
                - pl.col("recommended_date")
            )
            .dt.total_days()
            .cast(pl.Float64)
            .alias(
                "days_rec_to_sanction"
            ),

            (
                pl.col("completion_date")
                - pl.col("sanction_date")
            )
            .dt.total_days()
            .cast(pl.Float64)
            .alias(
                "days_sanction_to_complete"
            ),
        ]
    )

    # ========================================================
    # MONITORING SNAPSHOT
    # ========================================================

    as_of_date = get_as_of_date()

    master = master.with_columns(
        pl.lit(as_of_date)
        .cast(pl.Date)
        .alias("monitoring_as_of_date")
    )

    # --------------------------------------------------------
    # Days open since sanction.
    #
    # Only open works with a valid sanction date are measured.
    # Negative values are clipped to zero.
    # --------------------------------------------------------

    master = master.with_columns(
        pl.when(
            pl.col("is_open")
            & pl.col("sanction_date").is_not_null()
        )
        .then(
            (
                pl.lit(as_of_date)
                - pl.col("sanction_date")
            )
            .dt.total_days()
            .clip(lower_bound=0)
            .cast(pl.Float64)
        )
        .otherwise(None)
        .alias(
            "days_open_since_sanction"
        )
    )

    # --------------------------------------------------------
    # Days since last expenditure.
    # --------------------------------------------------------

    master = master.with_columns(
        pl.when(
            pl.col("is_open")
            & pl.col(
                "last_expenditure_date"
            ).is_not_null()
        )
        .then(
            (
                pl.lit(as_of_date)
                - pl.col(
                    "last_expenditure_date"
                )
            )
            .dt.total_days()
            .clip(lower_bound=0)
            .cast(pl.Float64)
        )
        .otherwise(None)
        .alias(
            "days_since_last_expenditure"
        )
    )

    # --------------------------------------------------------
    # Analytical one-year benchmark.
    #
    # This is an analytical benchmark, not itself a legal
    # violation flag.
    # --------------------------------------------------------

    master = master.with_columns(
        pl.col(
            "days_open_since_sanction"
        )
        .sub(
            GENERAL_ONE_YEAR_BENCHMARK_DAYS
        )
        .clip(lower_bound=0)
        .alias(
            "days_over_general_one_year_benchmark"
        )
    )

    # ========================================================
    # FINANCIAL YEAR
    # ========================================================

    derived_fy = (
        financial_year_expr(
            "sanction_date"
        )
    )

    if "financial_year" in master.columns:

        supplied_fy = (
            pl.col("financial_year")
            .cast(pl.String, strict=False)
            .str.strip_chars()
        )

        master = master.with_columns(
            derived_fy
        )

        master = master.with_columns(
            pl.when(
                supplied_fy.is_not_null()
                & supplied_fy.ne("")
            )
            .then(supplied_fy)
            .otherwise(
                pl.col(
                    "derived_financial_year"
                )
            )
            .alias("financial_year")
        )

        master = master.drop(
            "derived_financial_year"
        )

    else:

        master = master.with_columns(
            derived_fy
        )

        master = master.rename(
            {
                "derived_financial_year":
                "financial_year"
            }
        )

    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================

    if "work_description" in master.columns:

        work_description = (
            pl.col("work_description")
            .cast(
                pl.String,
                strict=False,
            )
        )

    else:

        work_description = (
            pl.lit(None)
            .cast(pl.String)
        )

    if "work" in master.columns:

        work = (
            pl.col("work")
            .cast(
                pl.String,
                strict=False,
            )
        )

    else:

        work = (
            pl.lit(None)
            .cast(pl.String)
        )

    master = master.with_columns(
        pl.coalesce(
            [
                work_description,
                work,
            ]
        )
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(
            r"\s+",
            " ",
        )
        .str.strip_chars()
        .alias("work_text")
    )

    # ========================================================
    # EXACT DUPLICATE CANDIDATES
    # ========================================================

    duplicate_keys = [
        column
        for column in (
            "state",
            "constituency",
            "ida",
            "work_text",
        )
        if column in master.columns
    ]

    if duplicate_keys:

        master = master.with_columns(
            pl.when(
                pl.col("work_text")
                .str.len_chars()
                .ge(
                    MIN_DUPLICATE_TEXT_LENGTH
                )
            )
            .then(
                pl.col("work_text")
                .count()
                .over(duplicate_keys)
            )
            .otherwise(
                pl.lit(1)
            )
            .cast(pl.Int64)
            .alias(
                "duplicate_group_count"
            )
        )

        master = master.with_columns(
            (
                pl.col("work_text")
                .str.len_chars()
                .ge(
                    MIN_DUPLICATE_TEXT_LENGTH
                )
                & pl.col(
                    "duplicate_group_count"
                ).gt(1)
            )
            .alias(
                "is_duplicate_candidate"
            )
        )

    else:

        master = master.with_columns(
            [
                pl.lit(1)
                .cast(pl.Int64)
                .alias(
                    "duplicate_group_count"
                ),

                pl.lit(False)
                .alias(
                    "is_duplicate_candidate"
                ),
            ]
        )

    # ========================================================
    # DATE INTEGRITY
    # ========================================================

    master = master.with_columns(
        [
            (
                pl.col("recommended_date")
                .is_not_null()
                & pl.col("sanction_date")
                .is_not_null()
                & pl.col("sanction_date")
                .lt(
                    pl.col("recommended_date")
                )
            )
            .alias(
                "bad_recommendation_date"
            ),

            (
                pl.col("sanction_date")
                .is_not_null()
                & pl.col("completion_date")
                .is_not_null()
                & pl.col("completion_date")
                .lt(
                    pl.col("sanction_date")
                )
            )
            .alias(
                "bad_completion_date"
            ),

            (
                pl.col("sanction_date")
                .is_not_null()
                & pl.col("sanction_date")
                .gt(
                    pl.lit(as_of_date)
                )
            )
            .alias(
                "future_sanction_date"
            ),

            (
                pl.col("completion_date")
                .is_not_null()
                & pl.col("completion_date")
                .gt(
                    pl.lit(as_of_date)
                )
            )
            .alias(
                "future_completion_date"
            ),
        ]
    )

    # ========================================================
    # VALUE INTEGRITY
    # ========================================================

    master = master.with_columns(
        [
            (
                pl.col("sanction_amount")
                .is_not_null()
                & pl.col("sanction_amount")
                .lt(0)
            )
            .alias(
                "negative_sanction_amount"
            ),

            (
                pl.col("total_expenditure")
                .is_not_null()
                & pl.col("total_expenditure")
                .lt(0)
            )
            .alias(
                "negative_expenditure"
            ),

            pl.col("sanction_amount")
            .is_null()
            .alias(
                "missing_sanction_amount"
            ),

            pl.col("sanction_date")
            .is_null()
            .alias(
                "missing_sanction_date"
            ),
        ]
    )

    # ========================================================
    # DATA QUALITY ISSUE COUNT
    # ========================================================

    integrity_flags = (
        "bad_recommendation_date",
        "bad_completion_date",
        "future_sanction_date",
        "future_completion_date",
        "negative_sanction_amount",
        "negative_expenditure",
        "missing_sanction_amount",
        "missing_sanction_date",
    )

    master = master.with_columns(
        pl.sum_horizontal(
            [
                pl.col(column)
                .cast(pl.UInt8)
                for column in integrity_flags
            ]
        )
        .cast(pl.Int64)
        .alias(
            "data_quality_issue_count"
        )
    )

    # ========================================================
    # DATA QUALITY STATUS
    # ========================================================

    master = master.with_columns(
        pl.when(
            pl.col(
                "data_quality_issue_count"
            ).eq(0)
        )
        .then(pl.lit("GOOD"))
        .when(
            pl.col(
                "data_quality_issue_count"
            ).eq(1)
        )
        .then(pl.lit("WATCH"))
        .when(
            pl.col(
                "data_quality_issue_count"
            ).eq(2)
        )
        .then(pl.lit("POOR"))
        .otherwise(
            pl.lit("CRITICAL")
        )
        .alias(
            "data_quality_status"
        )
    )

    # ========================================================
    # STABLE WORK UID
    # ========================================================

    master = add_stable_work_uid(
        master
    )

    validate_unique_work_uid(
        master
    )

    # ========================================================
    # FINAL METADATA
    # ========================================================

    master = master.with_columns(
        pl.lit(PIPELINE_VERSION)
        .alias("pipeline_version")
    )

    # ========================================================
    # FINAL SORT
    # ========================================================

    sort_columns = [
        column
        for column in (
            "state",
            "district",
            "ida",
            "work_uid",
        )
        if column in master.columns
    ]

    if sort_columns:

        master = master.sort(
            sort_columns,
            nulls_last=True,
        )

    return master


# ============================================================
# SAVE MASTER / SUMMARY
# ============================================================

def save_outputs(
    master: pl.DataFrame,
    expenditure_summary: pl.DataFrame,
) -> None:
    """
    Save analytical outputs as CSV files.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.write_csv(
        MASTER_PATH,
        include_bom=False,
        null_value="",
    )

    expenditure_summary.write_csv(
        EXPENDITURE_PATH,
        include_bom=False,
        null_value="",
    )


# ============================================================
# PIPELINE SUMMARY
# ============================================================

def print_pipeline_summary(
    master: pl.DataFrame,
    expenditure_summary: pl.DataFrame,
) -> None:
    """
    Print a concise but useful pipeline summary.
    """

    print(
        "\n========================================"
    )
    print(
        "FEATURE ENGINEERING COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        "Pipeline version:",
        PIPELINE_VERSION,
    )

    snapshot = "N/A"

    if (
        "monitoring_as_of_date"
        in master.columns
        and master.height > 0
    ):

        snapshot = (
            master["monitoring_as_of_date"]
            .head(1)
            .item()
        )

    print(
        "Monitoring as-of:",
        snapshot,
    )

    print(
        "Master shape:",
        master.shape,
    )

    print(
        "Expenditure summary shape:",
        expenditure_summary.shape,
    )

    # --------------------------------------------------------
    # Work status
    # --------------------------------------------------------

    if "is_completed" in master.columns:

        completed_count = (
            master["is_completed"]
            .sum()
        )

        print(
            "Completed works:",
            int(completed_count),
        )

    if "is_open" in master.columns:

        open_count = (
            master["is_open"]
            .sum()
        )

        print(
            "Open works:",
            int(open_count),
        )

    # --------------------------------------------------------
    # Duplicate candidates
    # --------------------------------------------------------

    if (
        "is_duplicate_candidate"
        in master.columns
    ):

        duplicate_count = (
            master[
                "is_duplicate_candidate"
            ]
            .sum()
        )

        print(
            "Duplicate candidates:",
            int(duplicate_count),
        )

    # --------------------------------------------------------
    # Date-quality indicators
    # --------------------------------------------------------

    for column, label in (
        (
            "bad_recommendation_date",
            "Bad recommendation dates",
        ),
        (
            "bad_completion_date",
            "Bad completion dates",
        ),
    ):

        if column in master.columns:

            count = (
                master[column]
                .sum()
            )

            print(
                f"{label}:",
                int(count),
            )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    if (
        "data_quality_issue_count"
        in master.columns
    ):

        issue_rows = (
            master
            .filter(
                pl.col(
                    "data_quality_issue_count"
                ).gt(0)
            )
            .height
        )

        print(
            "Data-quality issue rows:",
            int(issue_rows),
        )

    # --------------------------------------------------------
    # Financial columns
    # --------------------------------------------------------

    print(
        "\nFinancial columns:"
    )

    financial_columns = (
        "sanction_amount",
        "total_expenditure",
        "expenditure_variance_amount",
        "expenditure_variance_pct",
        "utilization_pct",
        "overspend_pct",
    )

    for column in financial_columns:

        if column in master.columns:

            print(
                f"  {column}: "
                f"{master.schema[column]}"
            )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print(
        "\nSaved:"
    )

    print(
        MASTER_PATH
    )

    print(
        EXPENDITURE_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Execute the complete feature-engineering pipeline.
    """

    print(
        "Loading official MPLADS source datasets..."
    )

    data = load_data()

    print(
        "Building master work table..."
    )

    master = build_master(
        data
    )

    print(
        "Building expenditure summary..."
    )

    expenditure_summary = (
        build_expenditure_summary(
            {
                "expenditure":
                data.get(
                    "expenditure",
                    pl.DataFrame(),
                )
            }
        )
    )

    print(
        "Saving analytical outputs..."
    )

    save_outputs(
        master,
        expenditure_summary,
    )

    print_pipeline_summary(
        master,
        expenditure_summary,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()