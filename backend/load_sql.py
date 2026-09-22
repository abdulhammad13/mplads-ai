from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl
from sqlalchemy import text

from .database import engine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_risk_data.csv"
)

TABLE_NAME = "works"
SCHEMA_NAME = "dbo"


# ============================================================
# SQL LOAD CONFIGURATION
# ============================================================

# IMPORTANT:
# Do NOT use method="multi" for this SQL Server load.
#
# SQL Server has a 2,100-parameter limit per statement.
# method=None uses executemany rather than constructing one
# enormous multi-row INSERT statement.
#
# 1,000 rows is normally safe with executemany because each
# execution contains parameters for one row rather than:
#
#     1000 × number_of_columns
#
SQL_CHUNKSIZE = 1_000


# ============================================================
# REQUIRED ML / RISK COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "work_uid",
    "ml_anomaly_score",
    "ml_anomaly_percentile",
    "final_risk_score",
    "risk_category",
    "priority_score",
    "priority_rank",
    "confidence_score",
]


# ============================================================
# OPTIONAL RISK COLUMNS
# ============================================================

OPTIONAL_RISK_COLUMNS = [
    "financial_risk_score",
    "execution_risk_score",
    "duplicate_risk_score",
    "data_integrity_risk_score",
    "rule_risk_score",
]


# ============================================================
# IMPORTANT IDENTIFIER COLUMNS
# ============================================================

IDENTIFIER_COLUMNS = [
    "work_uid",
    "link_key",
]


# ============================================================
# NUMERIC COLUMNS
# ============================================================

FLOAT_COLUMNS = [
    "ml_anomaly_score",
    "ml_anomaly_percentile",
    "ml_raw_decision_score",
    "ml_raw_anomaly_score",
    "final_risk_score",
    "priority_score",
    "confidence_score",
    "financial_risk_score",
    "execution_risk_score",
    "duplicate_risk_score",
    "data_integrity_risk_score",
    "rule_risk_score",
    "financial_exposure_percentile",
    "model_input_missing_pct",
    "cost_robust_z",
    "overspend_pct",
    "utilization_pct",
    "cost_peer_group_n",
    "days_open_since_sanction",
    "days_over_general_one_year_benchmark",
    "days_since_last_expenditure",
    "sanction_amount",
    "allocated_amount",
    "expenditure_amount",
    "work_amount",
]


# ============================================================
# INTEGER-LIKE COLUMNS
# ============================================================

INTEGER_COLUMNS = [
    "priority_rank",
    "duplicate_group_count",
    "data_quality_issue_count",
    "is_duplicate_candidate",
]


# ============================================================
# DATE COLUMNS
# ============================================================

DATE_COLUMNS = [
    "recommended_date",
    "sanction_date",
    "completion_date",
    "first_expenditure_date",
    "last_expenditure_date",
    "monitoring_as_of_date",
]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize column names consistently.

    Only formatting is changed; business meaning is preserved.
    """

    renamed: dict[str, str] = {}

    for column in df.columns:

        normalized = (
            column
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        renamed[column] = normalized

    if any(
        old != new
        for old, new in renamed.items()
    ):
        df = df.rename(renamed)

    return df


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def normalize_legacy_columns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Handle known historical column names without changing
    the current risk methodology.
    """

    if (
        "final_risk_category" in df.columns
        and "risk_category" not in df.columns
    ):

        df = df.rename(
            {
                "final_risk_category": "risk_category"
            }
        )

        print(
            "Renamed legacy column:"
            " final_risk_category → risk_category"
        )

    return df


# ============================================================
# TYPE NORMALIZATION
# ============================================================

def normalize_numeric_columns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Convert known numerical fields to Float64.

    Invalid numerical values become null instead of causing
    silent string storage in SQL Server.
    """

    expressions = []

    for column in FLOAT_COLUMNS:

        if column not in df.columns:
            continue

        expressions.append(
            pl.col(column)
            .cast(
                pl.Float64,
                strict=False,
            )
            .alias(column)
        )

    if expressions:
        df = df.with_columns(expressions)

    return df


def normalize_integer_columns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Normalize integer-like fields.

    Boolean duplicate flags are converted to integers because
    SQL Server storage is more predictable this way.
    """

    expressions = []

    for column in INTEGER_COLUMNS:

        if column not in df.columns:
            continue

        expressions.append(
            pl.col(column)
            .cast(
                pl.Int64,
                strict=False,
            )
            .alias(column)
        )

    if expressions:
        df = df.with_columns(expressions)

    return df


def normalize_date_columns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Normalize date columns to Polars Date where possible.

    We deliberately use strict=False so one malformed date
    becomes null rather than killing the complete SQL load.
    """

    expressions = []

    for column in DATE_COLUMNS:

        if column not in df.columns:
            continue

        expressions.append(
            pl.col(column)
            .cast(
                pl.String,
                strict=False,
            )
            .str.strip_chars()
            .str.strptime(
                pl.Date,
                format=None,
                strict=False,
            )
            .alias(column)
        )

    if expressions:
        df = df.with_columns(expressions)

    return df


def normalize_string_columns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Clean identifier/category fields without destroying
    free-text information.
    """

    columns = [
        "work_uid",
        "link_key",
        "risk_category",
        "risk_explanation",
        "ml_score_method",
        "model_version",
        "final_risk_version",
    ]

    expressions = []

    for column in columns:

        if column not in df.columns:
            continue

        expressions.append(
            pl.col(column)
            .cast(
                pl.String,
                strict=False,
            )
            .str.strip_chars()
            .alias(column)
        )

    if expressions:
        df = df.with_columns(expressions)

    return df


# ============================================================
# DATA QUALITY
# ============================================================

def validate_required_columns(
    df: pl.DataFrame,
) -> None:
    """
    Ensure all columns required by downstream API/dashboard
    layers are present.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if not missing:
        return

    print(
        "\n❌ REQUIRED ML/RISK COLUMNS ARE MISSING:"
    )

    for column in missing:
        print(
            f"   - {column}"
        )

    print(
        "\nAvailable risk-related columns:"
    )

    risk_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in (
                "risk",
                "anomaly",
                "score",
                "priority",
                "confidence",
                "flag",
            )
        )
    ]

    for column in risk_columns:
        print(
            f"   - {column}"
        )

    raise ValueError(
        "\nfinal_risk_data.csv does not contain the "
        "complete required ML/risk schema."
    )


def validate_work_uid(
    df: pl.DataFrame,
) -> None:
    """
    Validate work_uid because it is the principal work-level
    identifier used by the monitoring system.
    """

    if "work_uid" not in df.columns:
        return

    null_count = df.select(
        pl.col("work_uid")
        .is_null()
        .sum()
    ).item()

    if null_count:
        raise ValueError(
            f"work_uid contains {null_count:,} null records."
        )

    duplicate_count = (
        df.height
        - df.select(
            pl.col("work_uid")
            .n_unique()
        ).item()
    )

    if duplicate_count:

        print(
            "\n⚠️ WARNING:"
        )

        print(
            f"work_uid has {duplicate_count:,} "
            "duplicate rows."
        )

        print(
            "The loader will preserve the records rather "
            "than silently deleting them."
        )


def validate_risk_values(
    df: pl.DataFrame,
) -> None:
    """
    Validate ranges of the final scoring fields.

    These checks detect corrupt pipeline outputs without
    altering legitimate source data.
    """

    range_checks = {
        "ml_anomaly_percentile": (0.0, 100.0),
        "final_risk_score": (0.0, 100.0),
        "priority_score": (0.0, 100.0),
        "confidence_score": (0.0, 100.0),
    }

    for column, (
        minimum,
        maximum,
    ) in range_checks.items():

        if column not in df.columns:
            continue

        invalid_count = df.select(
            (
                pl.col(column).is_not_null()
                & (
                    (pl.col(column) < minimum)
                    | (pl.col(column) > maximum)
                )
            ).sum()
        ).item()

        if invalid_count:

            raise ValueError(
                f"{column} contains "
                f"{invalid_count:,} values outside "
                f"[{minimum}, {maximum}]."
            )


# ============================================================
# LOAD CSV
# ============================================================

def load_final_dataset() -> pl.DataFrame:
    """
    Load final_risk_data.csv using Polars.
    """

    print(
        "\n========================================"
    )
    print(
        "READING FINAL RISK DATASET"
    )
    print(
        "========================================"
    )

    print(
        f"\nInput:\n{INPUT}"
    )

    if not INPUT.exists():

        raise FileNotFoundError(
            f"\nFinal risk dataset not found:\n{INPUT}"
        )

    df = pl.read_csv(
        INPUT,
        infer_schema_length=10_000,
        try_parse_dates=True,
        ignore_errors=False,
    )

    if df.is_empty():

        raise ValueError(
            "final_risk_data.csv exists but contains "
            "zero records."
        )

    print(
        f"\nRows    : {df.height:,}"
    )

    print(
        f"Columns : {df.width:,}"
    )

    # --------------------------------------------------------
    # Normalize schema
    # --------------------------------------------------------

    df = normalize_column_names(df)

    df = normalize_legacy_columns(df)

    # --------------------------------------------------------
    # Normalize types
    # --------------------------------------------------------

    df = normalize_numeric_columns(df)

    df = normalize_integer_columns(df)

    df = normalize_date_columns(df)

    df = normalize_string_columns(df)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_required_columns(df)

    validate_work_uid(df)

    validate_risk_values(df)

    return df


# ============================================================
# SUMMARY
# ============================================================

def print_risk_summary(
    df: pl.DataFrame,
) -> None:
    """
    Display important statistics before SQL insertion.
    """

    print(
        "\n========================================"
    )
    print(
        "FINAL RISK DATASET SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"\nRows    : {df.height:,}"
    )

    print(
        f"Columns : {df.width:,}"
    )

    if "risk_category" in df.columns:

        print(
            "\nRisk category distribution:"
        )

        distribution = (
            df
            .group_by(
                "risk_category",
                maintain_order=True,
            )
            .agg(
                pl.len().alias("count")
            )
            .sort(
                "count",
                descending=True,
            )
        )

        print(
            distribution
        )

    score_columns = [
        column
        for column in (
            "ml_anomaly_score",
            "ml_anomaly_percentile",
            "final_risk_score",
            "priority_score",
            "confidence_score",
        )
        if column in df.columns
    ]

    if score_columns:

        print(
            "\nScore statistics:"
        )

        print(
            df.select(
                score_columns
            ).describe()
        )


# ============================================================
# POLARS → PANDAS SQL BOUNDARY
# ============================================================

def polars_to_pandas(
    df: pl.DataFrame,
) -> pd.DataFrame:
    """
    Convert to Pandas only at the SQLAlchemy boundary.

    The rest of the data pipeline remains Polars-based.
    """

    pandas_df = df.to_pandas()

    # --------------------------------------------------------
    # Normalize Pandas datetime columns
    # --------------------------------------------------------

    for column in DATE_COLUMNS:

        if column not in pandas_df.columns:
            continue

        pandas_df[column] = pd.to_datetime(
            pandas_df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Convert Polars null/object inconsistencies
    # --------------------------------------------------------

    pandas_df = pandas_df.replace(
        {
            pd.NA: None,
        }
    )

    return pandas_df


# ============================================================
# SQL LOAD
# ============================================================

def load_into_sql(
    df: pl.DataFrame,
) -> None:
    """
    Replace dbo.works using SQLAlchemy/pyodbc executemany.

    CRITICAL:
        method=None

    We intentionally do NOT use:
        method="multi"

    because SQL Server has a 2,100-parameter limit and
    this dataset contains many columns.
    """

    print(
        "\n========================================"
    )
    print(
        "LOADING DATA INTO SQL SERVER"
    )
    print(
        "========================================"
    )

    print(
        f"\nTarget: {SCHEMA_NAME}.{TABLE_NAME}"
    )

    print(
        f"Rows to load: {df.height:,}"
    )

    print(
        f"Columns to load: {df.width:,}"
    )

    pandas_df = polars_to_pandas(df)

    print(
        "\nUsing SQLAlchemy executemany..."
    )

    print(
        "method     : None"
    )

    print(
        f"chunksize  : {SQL_CHUNKSIZE:,}"
    )

    print(
        "fast_executemany: enabled at connection level"
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # execution_options(fast_executemany=True)
    # lets SQLAlchemy/pyodbc use efficient executemany
    # operations without creating a giant multi-row INSERT.
    # --------------------------------------------------------

    sql_engine = engine.execution_options(
        fast_executemany=True
    )

    pandas_df.to_sql(
        name=TABLE_NAME,
        con=sql_engine,
        schema=SCHEMA_NAME,
        if_exists="replace",
        index=False,

        # CRITICAL:
        # Do NOT change this to "multi".
        method=None,

        chunksize=SQL_CHUNKSIZE,
    )

    print(
        "\nSQL insertion completed."
    )


# ============================================================
# SQL VERIFICATION
# ============================================================

def verify_sql_load(
    df: pl.DataFrame,
) -> None:
    """
    Verify row count, columns, and basic risk-data integrity
    after SQL Server insertion.
    """

    print(
        "\n========================================"
    )
    print(
        "VERIFYING SQL SERVER"
    )
    print(
        "========================================"
    )

    with engine.connect() as connection:

        # ----------------------------------------------------
        # Row count
        # ----------------------------------------------------

        row_count = connection.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {SCHEMA_NAME}.{TABLE_NAME}
                """
            )
        ).scalar_one()

        # ----------------------------------------------------
        # SQL columns
        # ----------------------------------------------------

        columns = connection.execute(
            text(
                """
                SELECT
                    COLUMN_NAME,
                    DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                ORDER BY ORDINAL_POSITION
                """
            ),
            {
                "schema": SCHEMA_NAME,
                "table": TABLE_NAME,
            },
        ).fetchall()

        # ----------------------------------------------------
        # Null work_uid count
        # ----------------------------------------------------

        null_uid_count = connection.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {SCHEMA_NAME}.{TABLE_NAME}
                WHERE work_uid IS NULL
                """
            )
        ).scalar_one()

        # ----------------------------------------------------
        # Risk category counts
        # ----------------------------------------------------

        risk_rows = connection.execute(
            text(
                f"""
                SELECT
                    risk_category,
                    COUNT(*) AS row_count
                FROM {SCHEMA_NAME}.{TABLE_NAME}
                GROUP BY risk_category
                ORDER BY row_count DESC
                """
            )
        ).fetchall()

    sql_columns = {
        row[0]
        for row in columns
    }

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\nCSV rows : "
        f"{df.height:,}"
    )

    print(
        "SQL rows : "
        f"{row_count:,}"
    )

    print(
        "CSV cols : "
        f"{df.width:,}"
    )

    print(
        "SQL cols : "
        f"{len(sql_columns):,}"
    )

    print(
        "\nRequired ML / Risk columns:"
    )

    for column in REQUIRED_COLUMNS:

        status = (
            "✅"
            if column in sql_columns
            else "❌"
        )

        print(
            f"   {status} {column}"
        )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    print(
        "\nRisk categories in SQL:"
    )

    for category, count in risk_rows:

        print(
            f"   {str(category):<12} "
            f"{count:,}"
        )

    # ========================================================
    # HARD VALIDATION
    # ========================================================

    if row_count != df.height:

        raise RuntimeError(
            "\nRow-count mismatch!\n"
            f"CSV : {df.height:,}\n"
            f"SQL : {row_count:,}"
        )

    if len(sql_columns) != df.width:

        raise RuntimeError(
            "\nColumn-count mismatch!\n"
            f"CSV : {df.width:,}\n"
            f"SQL : {len(sql_columns):,}"
        )

    missing_sql_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in sql_columns
    ]

    if missing_sql_columns:

        raise RuntimeError(
            "\nRequired columns missing from SQL Server:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_sql_columns
            )
        )

    if null_uid_count:

        raise RuntimeError(
            f"\nSQL Server contains "
            f"{null_uid_count:,} NULL work_uid values."
        )

    print(
        "\n========================================"
    )
    print(
        "✅ SQL SERVER VERIFICATION PASSED"
    )
    print(
        "========================================"
    )

    print(
        "\nCSV and SQL Server are synchronized."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "\n"
        "========================================\n"
        " MPLADS AI MONITOR — SQL DATA LOADER\n"
        "========================================"
    )

    # --------------------------------------------------------
    # Stage 1 — Load final risk data
    # --------------------------------------------------------

    df = load_final_dataset()

    # --------------------------------------------------------
    # Stage 2 — Show summary
    # --------------------------------------------------------

    print_risk_summary(df)

    # --------------------------------------------------------
    # Stage 3 — Replace SQL table
    # --------------------------------------------------------

    load_into_sql(df)

    # --------------------------------------------------------
    # Stage 4 — Verify SQL
    # --------------------------------------------------------

    verify_sql_load(df)

    print(
        "\n✅ COMPLETE SQL LOAD PIPELINE FINISHED."
    )


if __name__ == "__main__":
    main()