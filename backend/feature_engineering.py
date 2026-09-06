from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_loader import load_data


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_PATH = PROCESSED_DIR / "master_works.csv"
EXPENDITURE_PATH = PROCESSED_DIR / "expenditure_summary.csv"

PIPELINE_VERSION = "4.0"

# Reproducible monitoring snapshot:
# set MPLADS_AS_OF_DATE=YYYY-MM-DD for a fixed batch.
DEFAULT_AS_OF_DATE = pd.Timestamp.today().normalize()


# ============================================================
# GENERIC TYPE-SAFE HELPERS
# ============================================================

def numeric(series: pd.Series | None) -> pd.Series:
    """
    Convert arbitrary input to float64 safely.

    Never coerce analytical values to integer here:
    amounts, percentages, ratios and statistical measures may be fractional.
    Invalid values become NaN.
    """
    if series is None:
        return pd.Series(dtype="float64")

    return pd.to_numeric(
        series.astype("string").str.strip(),
        errors="coerce",
    ).astype("float64")


def parse_dates(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Parse available date fields; malformed dates become NaT."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )
    return df


def clean_text(series: pd.Series) -> pd.Series:
    """Deterministically normalize free text for matching/search."""
    return (
        series.fillna("")
        .astype("string")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_link_key(df: pd.DataFrame) -> pd.Series:
    """
    Backward-compatible work linkage key.

    Existing source design uses Work + IDA. We preserve that linkage key but
    create a separate stable work_uid for downstream API/dashboard identity.
    """
    work = df.get(
        "work",
        pd.Series(index=df.index, dtype="string"),
    )
    ida = df.get(
        "ida",
        pd.Series(index=df.index, dtype="string"),
    )

    key = (
        work.fillna("").astype("string").str.strip().str.lower()
        + "|"
        + ida.fillna("").astype("string").str.strip().str.lower()
    )

    if "work" in df.columns and "ida" in df.columns:
        both_missing = df["work"].isna() & df["ida"].isna()
        key.loc[both_missing] = pd.NA

    return key


def stable_work_uid(row: pd.Series) -> str:
    """
    Create a stable record identifier.

    Prefer a source-provided ID. Otherwise hash stable business attributes.
    """
    for candidate in ("work_id", "id", "work_code", "link_key"):
        value = row.get(candidate)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()

    fields = [
        row.get("state", ""),
        row.get("ida", ""),
        row.get("mp", ""),
        row.get("constituency", ""),
        row.get("work", ""),
        row.get("recommended_date", ""),
        row.get("sanction_date", ""),
        row.get("sanction_amount", ""),
    ]
    payload = "|".join(
        "" if pd.isna(value) else str(value).strip().lower()
        for value in fields
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def financial_year_from_date(series: pd.Series) -> pd.Series:
    """Return Indian financial-year labels such as 2025-26."""
    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="string",
    )
    valid = series.notna()

    years = series.loc[valid].dt.year
    months = series.loc[valid].dt.month
    start_year = np.where(months >= 4, years, years - 1)

    result.loc[valid] = [
        f"{year}-{str(year + 1)[-2:]}"
        for year in start_year
    ]
    return result


def get_as_of_date() -> pd.Timestamp:
    """
    Resolve the monitoring snapshot date.

    Production/batch jobs should set MPLADS_AS_OF_DATE explicitly so results
    are reproducible across machines and reruns.
    """
    raw = os.getenv("MPLADS_AS_OF_DATE", "").strip()

    if raw:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()
        raise ValueError(
            f"Invalid MPLADS_AS_OF_DATE: {raw!r}. "
            "Expected YYYY-MM-DD."
        )

    return DEFAULT_AS_OF_DATE


def ensure_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Create absent optional columns as NA."""
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def validate_unique_work_uid(df: pd.DataFrame) -> None:
    """Fail early if stable work identifiers are duplicated."""
    if "work_uid" not in df.columns:
        raise ValueError("Missing required column: work_uid")

    duplicated = df["work_uid"].duplicated(keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, "work_uid"]
            .astype("string")
            .dropna()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Duplicate work_uid values detected. "
            f"Examples: {examples}"
        )


# ============================================================
# MASTER WORK TABLE
# ============================================================

def build_master(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sanctioned = data["sanctioned"].copy()
    recommended = data["recommended"].copy()
    completed = data["completed"].copy()
    allocation = data["allocation"].copy()
    expenditure = data.get(
        "expenditure",
        pd.DataFrame(),
    ).copy()

    # --------------------------------------------------------
    # Standardize source-level numeric fields BEFORE merges.
    # --------------------------------------------------------

    numeric_map = {
        "sanctioned": ["sanction_amount"],
        "recommended": ["recommended_amount"],
        "completed": ["amount_disbursed"],
        "allocation": ["allocated_amount"],
        "expenditure": ["fund_disbursed_amount"],
    }

    source_frames = {
        "sanctioned": sanctioned,
        "recommended": recommended,
        "completed": completed,
        "allocation": allocation,
        "expenditure": expenditure,
    }

    for name, columns in numeric_map.items():
        frame = source_frames[name]
        for column in columns:
            if column in frame.columns:
                frame[column] = numeric(frame[column])

    # --------------------------------------------------------
    # Standardize source dates BEFORE merges.
    # --------------------------------------------------------

    date_columns = [
        "recommended_date",
        "sanction_date",
        "completion_date",
        "expenditure_date",
    ]

    for frame in (
        sanctioned,
        recommended,
        completed,
        expenditure,
    ):
        parse_dates(frame, date_columns)

    # --------------------------------------------------------
    # Link keys.
    # --------------------------------------------------------

    sanctioned["link_key"] = make_link_key(sanctioned)
    recommended["link_key"] = make_link_key(recommended)
    completed["link_key"] = make_link_key(completed)

    if not expenditure.empty:
        expenditure["link_key"] = make_link_key(expenditure)

    # --------------------------------------------------------
    # MASTER = sanctioned works.
    # One row per linked work.
    # --------------------------------------------------------

    sanctioned = sanctioned.dropna(subset=["link_key"]).copy()

    if "sanction_date" in sanctioned.columns:
        sanctioned = (
            sanctioned
            .sort_values(
                ["link_key", "sanction_date"],
                na_position="last",
            )
            .drop_duplicates(
                "link_key",
                keep="last",
            )
        )
    else:
        sanctioned = sanctioned.drop_duplicates(
            "link_key",
            keep="last",
        )

    master = sanctioned.copy()

    # --------------------------------------------------------
    # Recommended amount.
    # Do not merge recommended_date again when it already exists in
    # the sanctioned source schema.
    # --------------------------------------------------------

    recommended_small = (
        recommended.dropna(subset=["link_key"])
        .sort_values(
            ["link_key", "recommended_date"],
            na_position="last",
        )
        .drop_duplicates(
            "link_key",
            keep="last",
        )
        [["link_key", "recommended_amount"]]
        .copy()
    )

    master = master.merge(
        recommended_small,
        on="link_key",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Completed metadata.
    # Financial totals are calculated independently from expenditure
    # transactions, not from the latest completion row.
    # --------------------------------------------------------

    completed_small = (
        completed.dropna(subset=["link_key"])
        .sort_values(
            ["link_key", "completion_date"],
            na_position="last",
        )
        .drop_duplicates(
            "link_key",
            keep="last",
        )
        [["link_key", "completion_date", "amount_disbursed"]]
        .rename(
            columns={
                "amount_disbursed": "completed_amount",
            }
        )
        .copy()
    )

    master = master.merge(
        completed_small,
        on="link_key",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Expenditure transaction summary.
    # --------------------------------------------------------

    expenditure_summary = build_expenditure_summary(
        {"expenditure": expenditure}
    )

    if not expenditure_summary.empty:
        master = master.merge(
            expenditure_summary,
            on="link_key",
            how="left",
            validate="one_to_one",
        )
    else:
        for column in (
            "total_expenditure",
            "num_transactions",
            "num_vendors",
            "first_expenditure_date",
            "last_expenditure_date",
        ):
            master[column] = pd.NA

    # --------------------------------------------------------
    # Allocation.
    # If financial_year exists in allocation data, preserve it.
    # --------------------------------------------------------

    allocation_columns = [
        column
        for column in (
            "state",
            "mp",
            "constituency",
        )
        if column in allocation.columns
    ]

    if "financial_year" in allocation.columns:
        allocation_columns.append("financial_year")

    if allocation_columns:
        allocation = allocation.dropna(
            subset=allocation_columns
        )

    if allocation.empty or not allocation_columns:
        master["allocated_amount"] = pd.NA
    else:
        allocation_small = (
            allocation
            .groupby(
                allocation_columns,
                dropna=False,
            )["allocated_amount"]
            .max()
            .reset_index()
        )

        join_cols = [
            column
            for column in allocation_columns
            if column in master.columns
        ]

        if join_cols:
            master = master.merge(
                allocation_small,
                on=join_cols,
                how="left",
                validate="many_to_one",
            )
        else:
            master["allocated_amount"] = pd.NA

    # --------------------------------------------------------
    # Required analytical columns.
    # --------------------------------------------------------

    ensure_columns(
        master,
        [
            "work",
            "work_description",
            "state",
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
        ],
    )

    for column in (
        "sanction_amount",
        "recommended_amount",
        "completed_amount",
        "allocated_amount",
        "total_expenditure",
    ):
        master[column] = numeric(master[column])

    # --------------------------------------------------------
    # Completion / financial metrics.
    # --------------------------------------------------------

    master["is_completed"] = master["completion_date"].notna()
    master["is_open"] = ~master["is_completed"]

    # Transaction sum is authoritative whenever available.
    master["expenditure_source"] = np.where(
        master["total_expenditure"].notna(),
        "transaction_sum",
        np.where(
            master["completed_amount"].notna(),
            "completed_amount_fallback",
            "missing",
        ),
    )

    master["total_expenditure"] = (
        master["total_expenditure"]
        .fillna(master["completed_amount"])
    )

    master["expenditure_variance_amount"] = (
        master["total_expenditure"]
        - master["sanction_amount"]
    )

    master["expenditure_variance_pct"] = (
        master["expenditure_variance_amount"]
        .div(master["sanction_amount"])
        .mul(100.0)
    )

    master["utilization_pct"] = (
        master["total_expenditure"]
        .div(master["sanction_amount"])
        .mul(100.0)
    )

    # Positive expenditure variance only.
    master["overspend_pct"] = (
        master["expenditure_variance_pct"]
        .clip(lower=0.0)
    )
    # --------------------------------------------------------
# Backward-compatible legacy model feature.
#
# IMPORTANT:
# This alias is retained only because the existing trained
# Isolation Forest was fitted on the historical name
# "overrun_pct".
#
# Canonical analytical fields remain:
#   expenditure_variance_pct
#   overspend_pct
# --------------------------------------------------------

    master["overrun_pct"] = (
    master["overspend_pct"]
    )

    invalid_sanction = (
        master["sanction_amount"].isna()
        | master["sanction_amount"].le(0)
    )

    for column in (
        "expenditure_variance_pct",
        "utilization_pct",
        "overspend_pct",
    ):
        master.loc[invalid_sanction, column] = np.nan

    master["disbursement_to_sanction_ratio"] = (
        master["utilization_pct"] / 100.0
    )

    # --------------------------------------------------------
    # Time features.
    # --------------------------------------------------------

    master["days_rec_to_sanction"] = (
        master["sanction_date"]
        - master["recommended_date"]
    ).dt.days

    master["days_sanction_to_complete"] = (
        master["completion_date"]
        - master["sanction_date"]
    ).dt.days

    # --------------------------------------------------------
    # Stable monitoring snapshot.
    # --------------------------------------------------------

    as_of_date = get_as_of_date()

    master["monitoring_as_of_date"] = as_of_date

    master["days_open_since_sanction"] = np.where(
        master["is_open"] & master["sanction_date"].notna(),
        (as_of_date - master["sanction_date"]).dt.days,
        np.nan,
    )

    master["days_open_since_sanction"] = (
        pd.to_numeric(
            master["days_open_since_sanction"],
            errors="coerce",
        )
        .clip(lower=0)
    )

    master["days_since_last_expenditure"] = np.where(
        master["is_open"]
        & master["last_expenditure_date"].notna(),
        (
            as_of_date
            - master["last_expenditure_date"]
        ).dt.days,
        np.nan,
    )

    master["days_since_last_expenditure"] = (
        pd.to_numeric(
            master["days_since_last_expenditure"],
            errors="coerce",
        )
        .clip(lower=0)
    )

    # Analytical one-year benchmark; not itself a legal violation flag.
    master["days_over_general_one_year_benchmark"] = (
        master["days_open_since_sanction"]
        - 365
    ).clip(lower=0)

    # --------------------------------------------------------
    # Financial year.
    # --------------------------------------------------------

    if "financial_year" in master.columns:
        supplied_fy = (
            master["financial_year"]
            .astype("string")
            .str.strip()
        )

        derived_fy = financial_year_from_date(
            master["sanction_date"]
        )

        master["financial_year"] = supplied_fy.where(
            supplied_fy.notna()
            & supplied_fy.ne(""),
            derived_fy,
        )
    else:
        master["financial_year"] = (
            financial_year_from_date(
                master["sanction_date"]
            )
        )

    # --------------------------------------------------------
    # Text + conservative exact duplicate candidates.
    # --------------------------------------------------------

    master["work_text"] = clean_text(
        master["work_description"].fillna(
            master["work"]
        )
    )

    duplicate_keys = [
        "state",
        "constituency",
        "ida",
        "work_text",
    ]

    valid_duplicate = (
        master["work_text"].str.len().ge(8)
    )

    master["duplicate_group_count"] = 1

    if valid_duplicate.any():
        master.loc[valid_duplicate, "duplicate_group_count"] = (
            master.loc[valid_duplicate]
            .groupby(
                duplicate_keys,
                dropna=False,
            )["link_key"]
            .transform("size")
        )

    master["is_duplicate_candidate"] = (
        valid_duplicate
        & master["duplicate_group_count"].gt(1)
    )

    # --------------------------------------------------------
    # Date integrity.
    # --------------------------------------------------------

    master["bad_recommendation_date"] = (
        master["recommended_date"].notna()
        & master["sanction_date"].notna()
        & master["sanction_date"].lt(
            master["recommended_date"]
        )
    )

    master["bad_completion_date"] = (
        master["sanction_date"].notna()
        & master["completion_date"].notna()
        & master["completion_date"].lt(
            master["sanction_date"]
        )
    )

    master["future_sanction_date"] = (
        master["sanction_date"].notna()
        & master["sanction_date"].gt(as_of_date)
    )

    master["future_completion_date"] = (
        master["completion_date"].notna()
        & master["completion_date"].gt(as_of_date)
    )

    # --------------------------------------------------------
    # Basic value-integrity / missingness indicators.
    # --------------------------------------------------------

    master["negative_sanction_amount"] = (
        master["sanction_amount"].lt(0)
    )

    master["negative_expenditure"] = (
        master["total_expenditure"].lt(0)
    )

    master["missing_sanction_amount"] = (
        master["sanction_amount"].isna()
    )

    master["missing_sanction_date"] = (
        master["sanction_date"].isna()
    )

    integrity_flags = [
        "bad_recommendation_date",
        "bad_completion_date",
        "future_sanction_date",
        "future_completion_date",
        "negative_sanction_amount",
        "negative_expenditure",
        "missing_sanction_amount",
        "missing_sanction_date",
    ]

    master["data_quality_issue_count"] = (
        master[integrity_flags]
        .fillna(False)
        .astype(int)
        .sum(axis=1)
    )

    master["data_quality_status"] = pd.cut(
        master["data_quality_issue_count"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=[
            "GOOD",
            "WATCH",
            "POOR",
            "CRITICAL",
        ],
    )

    # --------------------------------------------------------
    # Stable work UID.
    # --------------------------------------------------------

    master["work_uid"] = master.apply(
        stable_work_uid,
        axis=1,
    )

    validate_unique_work_uid(master)

    # --------------------------------------------------------
    # Final metadata.
    # --------------------------------------------------------

    master["pipeline_version"] = PIPELINE_VERSION

    master = (
        master
        .sort_values(
            [
                "state",
                "district" if "district" in master.columns else "ida",
                "work_uid",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return master


# ============================================================
# EXPENDITURE SUMMARY
# ============================================================

def build_expenditure_summary(
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    expenditure = data.get(
        "expenditure",
        pd.DataFrame(),
    ).copy()

    if expenditure.empty:
        return pd.DataFrame(
            columns=[
                "link_key",
                "total_expenditure",
                "num_transactions",
                "num_vendors",
                "first_expenditure_date",
                "last_expenditure_date",
            ]
        )

    parse_dates(
        expenditure,
        ["expenditure_date"],
    )

    if "fund_disbursed_amount" in expenditure.columns:
        expenditure["fund_disbursed_amount"] = numeric(
            expenditure["fund_disbursed_amount"]
        )

    expenditure["link_key"] = make_link_key(
        expenditure
    )

    expenditure = expenditure.dropna(
        subset=["link_key"]
    )

    count_column = (
        "work_id"
        if "work_id" in expenditure.columns
        else "link_key"
    )

    agg_map = {
        "total_expenditure": (
            "fund_disbursed_amount",
            "sum",
        ),
        "num_transactions": (
            count_column,
            "count",
        ),
        "first_expenditure_date": (
            "expenditure_date",
            "min",
        ),
        "last_expenditure_date": (
            "expenditure_date",
            "max",
        ),
    }

    if "vendor_name" in expenditure.columns:
        agg_map["num_vendors"] = (
            "vendor_name",
            "nunique",
        )

    summary = (
        expenditure
        .groupby(
            "link_key",
            dropna=False,
        )
        .agg(**agg_map)
        .reset_index()
    )

    if "num_vendors" not in summary.columns:
        summary["num_vendors"] = 0

    summary["total_expenditure"] = numeric(
        summary["total_expenditure"]
    )

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading official MPLADS source datasets...")
    data = load_data()

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Building master work table...")
    master = build_master(data)

    print("Building expenditure summary...")
    expenditure_summary = build_expenditure_summary(
        {"expenditure": data.get("expenditure", pd.DataFrame())}
    )

    master.to_csv(
        MASTER_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    expenditure_summary.to_csv(
        EXPENDITURE_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    print("\n========================================")
    print("FEATURE ENGINEERING COMPLETE")
    print("========================================")
    print("Pipeline version:", PIPELINE_VERSION)

    snapshot = (
        master["monitoring_as_of_date"].iloc[0]
        if len(master)
        else "N/A"
    )
    print("Monitoring as-of:", snapshot)
    print("Master shape:", master.shape)
    print(
        "Expenditure summary shape:",
        expenditure_summary.shape,
    )
    print(
        "Completed works:",
        int(master["is_completed"].sum()),
    )
    print(
        "Open works:",
        int(master["is_open"].sum()),
    )
    print(
        "Duplicate candidates:",
        int(master["is_duplicate_candidate"].sum()),
    )
    print(
        "Bad recommendation dates:",
        int(master["bad_recommendation_date"].sum()),
    )
    print(
        "Bad completion dates:",
        int(master["bad_completion_date"].sum()),
    )
    print(
        "Data-quality issue rows:",
        int(
            master["data_quality_issue_count"].gt(0).sum()
        ),
    )

    print("\nFinancial columns:")
    financial_columns = [
        "sanction_amount",
        "total_expenditure",
        "expenditure_variance_amount",
        "expenditure_variance_pct",
        "utilization_pct",
        "overspend_pct",
    ]
    for column in financial_columns:
        if column in master.columns:
            print(
                f"  {column}: "
                f"{master[column].dtype}"
            )

    print("\nSaved:")
    print(MASTER_PATH)
    print(EXPENDITURE_PATH)
