from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_works.csv"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk_scored_works.csv"
)

RISK_ENGINE_VERSION = "4.0"

# These are analytical monitoring parameters, not legal conclusions.
GENERAL_COMPLETION_BENCHMARK_DAYS = 365
STALLED_EXPENDITURE_DAYS = 180

MIN_PEER_GROUP_SIZE = 8


# ============================================================
# TYPE-SAFE HELPERS
# ============================================================

def safe_nullable_integer(series: pd.Series) -> pd.Series:
    """
    Convert numeric count-like data to nullable Int64.

    Counts are mathematically integral. We explicitly round only here,
    never for amounts, percentages or statistical scores.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.where(
        np.isfinite(values),
        np.nan,
    )

    return values.round().astype("Int64")


def numeric(series: pd.Series) -> pd.Series:
    """Convert arbitrary input to float64 safely."""
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("float64")


def exponential_severity(
    excess: pd.Series,
    scale: float,
) -> pd.Series:
    """
    Bounded monotonic severity:

        f(x) = 100 * (1 - exp(-x / scale))

    x must be non-negative.

    The scale is an analytical parameter and should be sensitivity-tested.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")

    x = numeric(excess).clip(lower=0)

    score = 100.0 * (
        1.0 - np.exp(
            -x / float(scale)
        )
    )

    return pd.Series(
        score,
        index=excess.index,
    ).fillna(0.0).clip(0.0, 100.0)


def robust_group_zscore(
    values: pd.Series,
    groups: pd.Series,
    min_group_size: int = MIN_PEER_GROUP_SIZE,
) -> tuple[pd.Series, pd.Series]:
    """
    Robust within-group z-score:

        z_R = 0.6745 * (x - median) / MAD

    The calculation is intentionally conservative:
    - insufficient peer groups receive 0
    - MAD = 0 receives 0
    - ties therefore do not create artificial anomalies
    """
    x = numeric(values)

    counts = (
        x.groupby(
            groups,
            dropna=False,
        )
        .transform("count")
    )

    median = (
        x.groupby(
            groups,
            dropna=False,
        )
        .transform("median")
    )

    absolute_deviation = (
        x - median
    ).abs()

    mad = (
        absolute_deviation
        .groupby(
            groups,
            dropna=False,
        )
        .transform("median")
    )

    z = pd.Series(
        0.0,
        index=x.index,
        dtype="float64",
    )

    valid = (
        x.notna()
        & median.notna()
        & mad.notna()
        & counts.ge(min_group_size)
        & mad.gt(0)
    )

    z.loc[valid] = (
        0.6745
        * (
            x.loc[valid]
            - median.loc[valid]
        )
        / mad.loc[valid]
    )

    return (
        z.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0),
        counts.fillna(0.0),
    )


def robust_outlier_score(
    z: pd.Series,
    scale: float = 2.0,
) -> pd.Series:
    """
    Convert |robust-z| to bounded 0-100 severity.
    """
    magnitude = (
        numeric(z)
        .abs()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return (
        100.0
        * (
            1.0
            - np.exp(
                -magnitude / float(scale)
            )
        )
    ).clip(0.0, 100.0)


def exact_iqr_outlier(
    values: pd.Series,
    groups: pd.Series,
    min_group_size: int = MIN_PEER_GROUP_SIZE,
) -> pd.Series:
    """
    Classical Tukey 1.5*IQR outlier flag within peer groups.
    """
    x = numeric(values)

    counts = (
        x.groupby(
            groups,
            dropna=False,
        )
        .transform("count")
    )

    q1 = (
        x.groupby(
            groups,
            dropna=False,
        )
        .transform(
            lambda s: s.quantile(0.25)
        )
    )

    q3 = (
        x.groupby(
            groups,
            dropna=False,
        )
        .transform(
            lambda s: s.quantile(0.75)
        )
    )

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (
        x.notna()
        & counts.ge(min_group_size)
        & iqr.gt(0)
        & (
            (x < lower)
            | (x > upper)
        )
    )


def normalize_weighted_scores(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Weighted mean of available component scores.

    Missing components are excluded from that row's denominator, rather
    than being treated as zero evidence.
    """
    numerator = pd.Series(
        0.0,
        index=frame.index,
    )

    denominator = pd.Series(
        0.0,
        index=frame.index,
    )

    for column, weight in weights.items():
        if column not in frame.columns:
            continue

        component = numeric(
            frame[column]
        )

        valid = component.notna()

        numerator.loc[valid] += (
            component.loc[valid]
            * float(weight)
        )

        denominator.loc[valid] += float(weight)

    result = numerator.div(
        denominator.replace(
            0.0,
            np.nan,
        )
    )

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(0.0, 100.0)
    )


def to_bool(series: pd.Series) -> pd.Series:
    """
    Robustly parse booleans from CSV/SQL values.

    Important: bool("False") is True in Python, so astype(bool) is unsafe
    for textual boolean data.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    result = pd.Series(
        False,
        index=series.index,
        dtype="boolean",
    )

    result.loc[
        normalized.isin(
            {"true", "1", "yes", "y", "t"}
        )
    ] = True

    result.loc[
        normalized.isin(
            {"false", "0", "no", "n", "f", ""}
        )
    ] = False

    return result.fillna(False).astype(bool)


def ensure_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Create missing source columns as NA for stable downstream logic."""
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


# ============================================================
# RISK FEATURE / SIGNAL ENGINE
# ============================================================

def add_risk_features(
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    master = master.copy()

    required = [
        "sanction_amount",
        "total_expenditure",
        "overspend_pct",
        "days_open_since_sanction",
        "days_since_last_expenditure",
        "days_sanction_to_complete",
        "work_category",
        "is_completed",
        "is_duplicate_candidate",
        "bad_recommendation_date",
        "bad_completion_date",
    ]

    master = ensure_columns(
        master,
        required,
    )

    # --------------------------------------------------------
    # Safe boolean normalization.
    # --------------------------------------------------------

    boolean_columns = [
        "is_completed",
        "is_duplicate_candidate",
        "bad_recommendation_date",
        "bad_completion_date",
        "future_sanction_date",
        "future_completion_date",
        "negative_sanction_amount",
        "negative_expenditure",
    ]

    for column in boolean_columns:
        if column in master.columns:
            master[column] = to_bool(
                master[column]
            )

    # --------------------------------------------------------
    # Financial signals.
    # --------------------------------------------------------

    sanction = numeric(
        master["sanction_amount"]
    )

    expenditure = numeric(
        master["total_expenditure"]
    )

    # Reconstruct variance from authoritative analytical columns.
    derived_variance_pct = (
        (expenditure - sanction)
        .div(sanction)
        .mul(100.0)
    )

    stored_overspend = numeric(
        master["overspend_pct"]
    )

    # Prefer the stored feature when valid; otherwise reconstruct it.
    overspend = stored_overspend.where(
        stored_overspend.notna(),
        derived_variance_pct.clip(lower=0.0),
    )

    overspend.loc[
        sanction.isna()
        | sanction.le(0)
    ] = np.nan

    master["overspend_pct"] = overspend

    master["financial_overspend_severity"] = (
        exponential_severity(
            overspend,
            scale=10.0,
        )
    )

    # Cost outlier: robust within category.
    category_group = (
        master["work_category"]
        .astype("string")
        .fillna("__MISSING__")
    )

    cost_z, cost_group_n = robust_group_zscore(
        master["sanction_amount"],
        category_group,
    )

    master["cost_robust_z"] = (
        cost_z.round(4)
    )

    master["cost_peer_group_n"] = (
        safe_nullable_integer(
            cost_group_n
        )
    )

    master["cost_outlier_score"] = (
        robust_outlier_score(
            cost_z
        )
    )

    master["flag_disbursement_over_sanction"] = (
        overspend.gt(0).fillna(False)
    )

    master["flag_cost_outlier"] = (
        exact_iqr_outlier(
            master["sanction_amount"],
            category_group,
        )
    )

    master["financial_risk_score"] = (
        normalize_weighted_scores(
            master,
            {
                "financial_overspend_severity": 0.60,
                "cost_outlier_score": 0.40,
            },
        )
        .round(2)
    )

    # --------------------------------------------------------
    # Execution / ageing signals.
    # --------------------------------------------------------

    days_open = numeric(
        master["days_open_since_sanction"]
    )

    days_over = (
        days_open
        - GENERAL_COMPLETION_BENCHMARK_DAYS
    ).clip(lower=0.0)

    master[
        "days_over_general_one_year_benchmark"
    ] = days_over

    master["overdue_open_score"] = (
        exponential_severity(
            days_over,
            scale=180.0,
        )
    )

    days_since_expenditure = numeric(
        master["days_since_last_expenditure"]
    )

    stalled_excess = (
        days_since_expenditure
        - STALLED_EXPENDITURE_DAYS
    ).clip(lower=0.0)

    master["stalled_expenditure_score"] = (
        exponential_severity(
            stalled_excess,
            scale=180.0,
        )
    )

    duration_z, duration_group_n = (
        robust_group_zscore(
            master["days_sanction_to_complete"],
            category_group,
        )
    )

    master["duration_robust_z"] = (
        duration_z.round(4)
    )

    master["duration_peer_group_n"] = (
        safe_nullable_integer(
            duration_group_n
        )
    )

    master["duration_outlier_score"] = (
        robust_outlier_score(
            duration_z
        )
    )

    master["flag_overdue_open_work"] = (
        master["is_completed"].eq(False)
        & days_open.notna()
        & days_over.gt(0)
    )

    master["flag_stalled_expenditure"] = (
        master["is_completed"].eq(False)
        & days_since_expenditure.notna()
        & days_since_expenditure.gt(
            STALLED_EXPENDITURE_DAYS
        )
    )

    master["flag_duration_outlier"] = (
        exact_iqr_outlier(
            master["days_sanction_to_complete"],
            category_group,
        )
        & master["is_completed"]
    )

    master["execution_risk_score"] = (
        normalize_weighted_scores(
            master,
            {
                "overdue_open_score": 0.50,
                "stalled_expenditure_score": 0.25,
                "duration_outlier_score": 0.25,
            },
        )
        .round(2)
    )

    # --------------------------------------------------------
    # Duplicate / similarity signals.
    # --------------------------------------------------------

    duplicate_candidate = (
        master["is_duplicate_candidate"]
        .fillna(False)
    )

    master["duplicate_risk_score"] = (
        duplicate_candidate.astype(float)
        * 100.0
    )

    # Optional future similarity columns are accepted without making
    # semantic/spatial similarity a claim when those fields do not exist.
    similarity_columns = [
        "semantic_similarity",
        "text_similarity",
        "spatial_similarity",
        "duplicate_similarity_score",
    ]

    available_similarity = [
        column
        for column in similarity_columns
        if column in master.columns
    ]

    if available_similarity:
        similarity = (
            master[available_similarity]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        for column in available_similarity:
            finite = similarity[column].dropna()

            if (
                not finite.empty
                and finite.max() <= 1.0
            ):
                similarity[column] = (
                    similarity[column] * 100.0
                )

        semantic_score = (
            similarity
            .clip(0.0, 100.0)
            .mean(axis=1)
        )

        master["duplicate_risk_score"] = (
            np.maximum(
                master[
                    "duplicate_risk_score"
                ],
                semantic_score.fillna(0.0),
            )
        )

    master["duplicate_risk_score"] = (
        numeric(
            master["duplicate_risk_score"]
        )
        .clip(0.0, 100.0)
        .round(2)
    )

    master["flag_duplicate_candidate"] = (
        duplicate_candidate.astype(bool)
    )

    # --------------------------------------------------------
    # Data / date integrity.
    #
    # This is deliberately kept OUT of substantive final risk.
    # It influences confidence / data-quality monitoring instead.
    # --------------------------------------------------------

    future_sanction = (
        master["future_sanction_date"]
        if "future_sanction_date" in master.columns
        else pd.Series(False, index=master.index)
    )

    future_completion = (
        master["future_completion_date"]
        if "future_completion_date" in master.columns
        else pd.Series(False, index=master.index)
    )

    negative_sanction = (
        master["negative_sanction_amount"]
        if "negative_sanction_amount" in master.columns
        else pd.Series(False, index=master.index)
    )

    negative_expenditure = (
        master["negative_expenditure"]
        if "negative_expenditure" in master.columns
        else pd.Series(False, index=master.index)
    )

    date_issue = (
        master["bad_recommendation_date"].fillna(False)
        | master["bad_completion_date"].fillna(False)
        | to_bool(future_sanction)
        | to_bool(future_completion)
    )

    numeric_issue = (
        to_bool(negative_sanction)
        | to_bool(negative_expenditure)
    )

    master["data_integrity_risk_score"] = (
        (
            date_issue.astype(int) * 70.0
            + numeric_issue.astype(int) * 70.0
        )
        .clip(0.0, 100.0)
        .round(2)
    )

    master["flag_bad_dates"] = (
        date_issue.astype(bool)
    )

    # --------------------------------------------------------
    # Transparent rule-based baseline.
    #
    # This is deliberately separate from ML final risk.
    # --------------------------------------------------------

    rule_components = {
        "flag_disbursement_over_sanction": 18.0,
        "flag_bad_dates": 15.0,
        "flag_overdue_open_work": 18.0,
        "flag_stalled_expenditure": 12.0,
        "flag_duplicate_candidate": 14.0,
        "flag_cost_outlier": 10.0,
        "flag_duration_outlier": 13.0,
    }

    available_flags = [
        column
        for column in rule_components
        if column in master.columns
    ]

    rule_denominator = sum(
        rule_components[column]
        for column in available_flags
    ) or 1.0

    weighted_flags = pd.Series(
        0.0,
        index=master.index,
    )

    for column in available_flags:
        weighted_flags += (
            master[column]
            .fillna(False)
            .astype(int)
            * rule_components[column]
        )

    master["rule_risk_score"] = (
        weighted_flags
        .div(rule_denominator)
        .mul(100.0)
        .round(2)
        .clip(0.0, 100.0)
    )

    # --------------------------------------------------------
    # Rule signal count.
    # --------------------------------------------------------

    master["signal_count"] = (
        master[available_flags]
        .fillna(False)
        .astype(int)
        .sum(axis=1)
    )

    # Independent substantive signals:
    # integrity is intentionally excluded because bad data is not itself
    # evidence that the work is substantively anomalous.
    substantive_components = [
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
    ]

    high_signal = (
        master[substantive_components]
        .ge(60.0)
        .sum(axis=1)
    )

    master["independent_signal_count"] = (
        high_signal.astype(int)
    )

    # --------------------------------------------------------
    # Rule-only category.
    # Final category is assigned by final_risk.py after ML scoring.
    # --------------------------------------------------------

    master["rule_risk_category"] = pd.cut(
        master["rule_risk_score"],
        bins=[
            -np.inf,
            25.0,
            50.0,
            75.0,
            np.inf,
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ],
        right=False,
    )

    # --------------------------------------------------------
    # Human-readable explanation.
    # --------------------------------------------------------

    reason_map = {
        "financial_risk_score":
            "Financial deviation / cost anomaly",
        "execution_risk_score":
            "Execution delay / stalled spending",
        "duplicate_risk_score":
            "Potential duplicate / similar work",
    }

    score_frame = master[
        list(reason_map.keys())
    ]

    max_component = score_frame.idxmax(
        axis=1
    )

    master["primary_risk_reason"] = (
        max_component.map(reason_map)
    )

    def build_reason(row: pd.Series) -> str:
        reasons: list[str] = []

        if bool(
            row.get(
                "flag_disbursement_over_sanction",
                False,
            )
        ):
            pct = row.get("overspend_pct")

            if pd.notna(pct):
                reasons.append(
                    "Expenditure exceeds sanction by "
                    f"{float(pct):.1f}%"
                )
            else:
                reasons.append(
                    "Expenditure exceeds sanction"
                )

        if bool(
            row.get(
                "flag_overdue_open_work",
                False,
            )
        ):
            days = row.get(
                "days_over_general_one_year_benchmark"
            )

            if pd.notna(days):
                reasons.append(
                    "Open beyond the general "
                    "one-year benchmark by "
                    f"{int(days):,} days"
                )
            else:
                reasons.append(
                    "Open beyond the general "
                    "one-year benchmark"
                )

        if bool(
            row.get(
                "flag_stalled_expenditure",
                False,
            )
        ):
            days = row.get(
                "days_since_last_expenditure"
            )

            if pd.notna(days):
                reasons.append(
                    "No recorded expenditure for "
                    f"{int(days):,} days"
                )
            else:
                reasons.append(
                    "Long expenditure gap"
                )

        if bool(
            row.get(
                "flag_duplicate_candidate",
                False,
            )
        ):
            reasons.append(
                "Potential duplicate/similar work candidate"
            )

        if bool(
            row.get(
                "flag_cost_outlier",
                False,
            )
        ):
            reasons.append(
                "Sanction amount is an outlier "
                "within the work category"
            )

        if bool(
            row.get(
                "flag_duration_outlier",
                False,
            )
        ):
            reasons.append(
                "Completion duration is an outlier "
                "within the work category"
            )

        if bool(
            row.get(
                "flag_bad_dates",
                False,
            )
        ):
            reasons.append(
                "Inconsistent or future date sequence"
            )

        if not reasons:
            return (
                "No strong transparent rule-based signal"
            )

        return " • ".join(reasons)

    master["risk_explanation"] = master.apply(
        build_reason,
        axis=1,
    )

    master["risk_engine_version"] = (
        RISK_ENGINE_VERSION
    )

    return master, available_flags


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}\n"
            "Run backend.feature_engineering first."
        )

    print("Loading master analytical dataset...")
    master = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    date_columns = [
        "recommended_date",
        "sanction_date",
        "completion_date",
        "first_expenditure_date",
        "last_expenditure_date",
        "monitoring_as_of_date",
    ]

    for column in date_columns:
        if column in master.columns:
            master[column] = pd.to_datetime(
                master[column],
                errors="coerce",
            )

    print("Generating transparent risk signals...")
    master, flags = add_risk_features(
        master
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    print("\n========================================")
    print("RISK SIGNAL ENGINE COMPLETE")
    print("========================================")
    print("Risk engine version:", RISK_ENGINE_VERSION)
    print("Rows:", len(master))
    print("Rule signals:", len(flags))

    print(
        "Multi-signal records (>=2):",
        int(
            master[
                "independent_signal_count"
            ].ge(2).sum()
        ),
    )

    print(
        "Potential duplicate candidates:",
        int(
            master[
                "flag_duplicate_candidate"
            ].sum()
        ),
    )

    print(
        "Financial signal records:",
        int(
            master[
                "financial_risk_score"
            ].ge(60).sum()
        ),
    )

    print(
        "Execution signal records:",
        int(
            master[
                "execution_risk_score"
            ].ge(60).sum()
        ),
    )

    print("\nRule-risk distribution:")
    print(
        master[
            "rule_risk_category"
        ]
        .value_counts(
            dropna=False,
            sort=False,
        )
    )

    print("\nKey score diagnostics:")
    diagnostic_columns = [
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
        "data_integrity_risk_score",
        "rule_risk_score",
    ]

    print(
        master[
            diagnostic_columns
        ]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )

    print("\nSaved:")
    print(OUTPUT)
