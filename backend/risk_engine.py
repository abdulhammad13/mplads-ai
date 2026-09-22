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

RISK_ENGINE_VERSION = "4.1"

# These are analytical monitoring parameters,
# not legal conclusions.
GENERAL_COMPLETION_BENCHMARK_DAYS = 365
STALLED_EXPENDITURE_DAYS = 180

MIN_PEER_GROUP_SIZE = 8


# ============================================================
# CONSTANTS
# ============================================================

RISK_BINS = [
    -np.inf,
    25.0,
    50.0,
    75.0,
    np.inf,
]

RISK_LABELS = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]

BOOLEAN_TRUE_VALUES = {
    "true",
    "1",
    "yes",
    "y",
    "t",
}

BOOLEAN_FALSE_VALUES = {
    "false",
    "0",
    "no",
    "n",
    "f",
    "",
    "none",
    "nan",
    "null",
}


# ============================================================
# TYPE-SAFE HELPERS
# ============================================================

def numeric(series: pd.Series) -> pd.Series:
    """
    Convert arbitrary input to float64 safely.

    Invalid values become NaN.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("float64")


def safe_nullable_integer(series: pd.Series) -> pd.Series:
    """
    Convert count-like values to nullable Int64.

    Counts are mathematically integral, so rounding is appropriate
    only for count-like fields.
    """
    values = numeric(series)

    values = values.where(
        np.isfinite(values),
        np.nan,
    )

    return values.round().astype("Int64")


def exponential_severity(
    excess: pd.Series,
    scale: float,
) -> pd.Series:
    """
    Convert a non-negative excess value into bounded 0-100 severity.

        f(x) = 100 * (1 - exp(-x / scale))

    Properties:
    - x < 0 is clipped to zero
    - result is bounded between 0 and 100
    - larger excess produces larger severity
    """
    if scale <= 0:
        raise ValueError(
            "exponential_severity scale must be > 0"
        )

    x = numeric(excess).clip(lower=0.0)

    score = 100.0 * (
        1.0
        - np.exp(
            -x / float(scale)
        )
    )

    score = pd.Series(
        score,
        index=excess.index,
        dtype="float64",
    )

    return (
        score
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(0.0, 100.0)
    )


def robust_group_zscore(
    values: pd.Series,
    groups: pd.Series,
    min_group_size: int = MIN_PEER_GROUP_SIZE,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate a robust within-group z-score:

        z_R = 0.6745 * (x - median) / MAD

    Conservative behavior:
    - insufficient peer groups -> 0
    - missing median -> 0
    - MAD == 0 -> 0
    - ties do not create artificial anomalies

    Returns:
        z_score
        peer_group_count
    """
    x = numeric(values)

    group_series = (
        groups
        .astype("string")
        .fillna("__MISSING__")
    )

    grouped = x.groupby(
        group_series,
        dropna=False,
        sort=False,
    )

    counts = grouped.transform("count")
    median = grouped.transform("median")

    absolute_deviation = (
        x - median
    ).abs()

    mad = (
        absolute_deviation
        .groupby(
            group_series,
            dropna=False,
            sort=False,
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
        & mad.gt(0.0)
    )

    z.loc[valid] = (
        0.6745
        * (
            x.loc[valid]
            - median.loc[valid]
        )
        / mad.loc[valid]
    )

    z = (
        z
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return (
        z,
        counts.fillna(0.0),
    )


def robust_outlier_score(
    z: pd.Series,
    scale: float = 2.0,
) -> pd.Series:
    """
    Convert absolute robust-z magnitude into bounded 0-100 severity.
    """
    if scale <= 0:
        raise ValueError(
            "robust_outlier_score scale must be > 0"
        )

    magnitude = (
        numeric(z)
        .abs()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    score = (
        100.0
        * (
            1.0
            - np.exp(
                -magnitude / float(scale)
            )
        )
    )

    return (
        score
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(0.0, 100.0)
    )


def exact_iqr_outlier(
    values: pd.Series,
    groups: pd.Series,
    min_group_size: int = MIN_PEER_GROUP_SIZE,
) -> pd.Series:
    """
    Classical Tukey 1.5*IQR outlier flag within peer groups.
    """
    x = numeric(values)

    group_series = (
        groups
        .astype("string")
        .fillna("__MISSING__")
    )

    grouped = x.groupby(
        group_series,
        dropna=False,
        sort=False,
    )

    counts = grouped.transform("count")
    q1 = grouped.transform(
        lambda s: s.quantile(0.25)
    )
    q3 = grouped.transform(
        lambda s: s.quantile(0.75)
    )

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (
        x.notna()
        & counts.ge(min_group_size)
        & iqr.gt(0.0)
        & (
            (x < lower)
            | (x > upper)
        )
    ).fillna(False)


def normalize_weighted_scores(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Calculate a weighted mean of available component scores.

    Missing components are excluded from the row denominator rather
    than treated as zero evidence.
    """
    numerator = pd.Series(
        0.0,
        index=frame.index,
        dtype="float64",
    )

    denominator = pd.Series(
        0.0,
        index=frame.index,
        dtype="float64",
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

    Important:
        bool("False") == True

    Therefore textual booleans must never be converted using
    astype(bool) directly.
    """
    if pd.api.types.is_bool_dtype(series):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    result = pd.Series(
        False,
        index=series.index,
        dtype="bool",
    )

    result.loc[
        normalized.isin(
            BOOLEAN_TRUE_VALUES
        )
    ] = True

    result.loc[
        normalized.isin(
            BOOLEAN_FALSE_VALUES
        )
    ] = False

    return result.fillna(False).astype(bool)


def ensure_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Create missing source columns as NA.

    This keeps the downstream analytical pipeline stable when
    optional source fields are absent.
    """
    result = df.copy()

    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA

    return result


# ============================================================
# SIGNAL ENGINE
# ============================================================

def add_risk_features(
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate transparent rule-based risk features.

    The methodology intentionally separates:

    1. Financial signals
    2. Execution signals
    3. Duplicate/similarity signals
    4. Data-integrity monitoring
    5. Transparent rule-risk baseline

    Data-integrity signals are NOT included directly in substantive
    rule risk. They are intended for confidence/data-quality monitoring.
    """

    if not isinstance(master, pd.DataFrame):
        raise TypeError(
            "add_risk_features expects a pandas DataFrame"
        )

    master = master.copy()

    # --------------------------------------------------------
    # Required analytical columns.
    # --------------------------------------------------------

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
    # Boolean normalization.
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

    # Reconstruct expenditure variance from authoritative
    # analytical fields.
    valid_sanction = (
        sanction.notna()
        & sanction.gt(0.0)
    )

    derived_variance_pct = pd.Series(
        np.nan,
        index=master.index,
        dtype="float64",
    )

    derived_variance_pct.loc[
        valid_sanction
    ] = (
        (
            expenditure.loc[valid_sanction]
            - sanction.loc[valid_sanction]
        )
        .div(
            sanction.loc[valid_sanction]
        )
        .mul(100.0)
    )

    stored_overspend = numeric(
        master["overspend_pct"]
    )

    # Prefer an existing validated analytical feature.
    # Otherwise reconstruct it.
    overspend = stored_overspend.where(
        stored_overspend.notna(),
        derived_variance_pct.clip(
            lower=0.0
        ),
    )

    overspend.loc[
        ~valid_sanction
    ] = np.nan

    master["overspend_pct"] = overspend

    # --------------------------------------------------------
    # Overspend severity.
    # --------------------------------------------------------

    master[
        "financial_overspend_severity"
    ] = exponential_severity(
        overspend,
        scale=10.0,
    )

    # --------------------------------------------------------
    # Cost outlier analysis.
    # --------------------------------------------------------

    category_group = (
        master["work_category"]
        .astype("string")
        .str.strip()
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

    master[
        "flag_disbursement_over_sanction"
    ] = (
        overspend
        .gt(0.0)
        .fillna(False)
        .astype(bool)
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
            master[
                "days_sanction_to_complete"
            ],
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

    master[
        "flag_overdue_open_work"
    ] = (
        master["is_completed"].eq(False)
        & days_open.notna()
        & days_over.gt(0.0)
    )

    master[
        "flag_stalled_expenditure"
    ] = (
        master["is_completed"].eq(False)
        & days_since_expenditure.notna()
        & days_since_expenditure.gt(
            STALLED_EXPENDITURE_DAYS
        )
    )

    master[
        "flag_duration_outlier"
    ] = (
        exact_iqr_outlier(
            master[
                "days_sanction_to_complete"
            ],
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
        master[
            "is_duplicate_candidate"
        ]
        .fillna(False)
        .astype(bool)
    )

    master[
        "duplicate_risk_score"
    ] = (
        duplicate_candidate.astype(float)
        * 100.0
    )

    # Optional similarity features.
    #
    # These are only used if they genuinely exist in the dataset.
    # No semantic/spatial similarity is invented when unavailable.
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

        similarity = master[
            available_similarity
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Normalize 0-1 similarity scores to 0-100.
        for column in available_similarity:

            finite = similarity[
                column
            ].dropna()

            if (
                not finite.empty
                and finite.max() <= 1.0
            ):
                similarity[column] = (
                    similarity[column]
                    * 100.0
                )

        similarity = similarity.clip(
            lower=0.0,
            upper=100.0,
        )

        similarity_score = (
            similarity.mean(
                axis=1,
                skipna=True,
            )
            .fillna(0.0)
        )

        master[
            "duplicate_risk_score"
        ] = np.maximum(
            numeric(
                master[
                    "duplicate_risk_score"
                ]
            ).fillna(0.0),
            similarity_score,
        )

    master[
        "duplicate_risk_score"
    ] = (
        numeric(
            master[
                "duplicate_risk_score"
            ]
        )
        .clip(0.0, 100.0)
        .round(2)
    )

    master[
        "flag_duplicate_candidate"
    ] = duplicate_candidate

    # --------------------------------------------------------
    # Data / date integrity.
    #
    # Intentionally excluded from substantive final risk.
    # --------------------------------------------------------

    if "future_sanction_date" in master.columns:
        future_sanction = to_bool(
            master["future_sanction_date"]
        )
    else:
        future_sanction = pd.Series(
            False,
            index=master.index,
            dtype=bool,
        )

    if "future_completion_date" in master.columns:
        future_completion = to_bool(
            master["future_completion_date"]
        )
    else:
        future_completion = pd.Series(
            False,
            index=master.index,
            dtype=bool,
        )

    if "negative_sanction_amount" in master.columns:
        negative_sanction = to_bool(
            master[
                "negative_sanction_amount"
            ]
        )
    else:
        negative_sanction = pd.Series(
            False,
            index=master.index,
            dtype=bool,
        )

    if "negative_expenditure" in master.columns:
        negative_expenditure = to_bool(
            master["negative_expenditure"]
        )
    else:
        negative_expenditure = pd.Series(
            False,
            index=master.index,
            dtype=bool,
        )

    date_issue = (
        master[
            "bad_recommendation_date"
        ]
        .fillna(False)
        .astype(bool)
        |
        master[
            "bad_completion_date"
        ]
        .fillna(False)
        .astype(bool)
        |
        future_sanction
        |
        future_completion
    )

    numeric_issue = (
        negative_sanction
        |
        negative_expenditure
    )

    master[
        "data_integrity_risk_score"
    ] = (
        (
            date_issue.astype(int)
            * 70.0
        )
        +
        (
            numeric_issue.astype(int)
            * 70.0
        )
    ).clip(
        0.0,
        100.0,
    ).round(2)

    master["flag_bad_dates"] = (
        date_issue.astype(bool)
    )

    # --------------------------------------------------------
    # Transparent rule-based baseline.
    #
    # This remains separate from the ML final risk score.
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

    rule_denominator = (
        sum(
            rule_components[column]
            for column in available_flags
        )
        or 1.0
    )

    weighted_flags = pd.Series(
        0.0,
        index=master.index,
        dtype="float64",
    )

    for column in available_flags:

        weighted_flags += (
            master[column]
            .fillna(False)
            .astype(bool)
            .astype(int)
            * rule_components[column]
        )

    master["rule_risk_score"] = (
        weighted_flags
        .div(rule_denominator)
        .mul(100.0)
        .clip(0.0, 100.0)
        .round(2)
    )

    # --------------------------------------------------------
    # Rule signal count.
    # --------------------------------------------------------

    if available_flags:
        master["signal_count"] = (
            master[available_flags]
            .fillna(False)
            .astype(bool)
            .astype(int)
            .sum(axis=1)
            .astype("int64")
        )
    else:
        master["signal_count"] = 0

    # --------------------------------------------------------
    # Independent substantive signals.
    #
    # Data integrity is deliberately excluded.
    # --------------------------------------------------------

    substantive_components = [
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
    ]

    high_signal = (
        master[substantive_components]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .ge(60.0)
        .sum(axis=1)
    )

    master[
        "independent_signal_count"
    ] = (
        high_signal
        .astype("int64")
    )

    # --------------------------------------------------------
    # Rule-only category.
    #
    # right=True keeps boundaries consistent with final_risk.py:
    #
    #   <=25       LOW
    #   >25–50     MEDIUM
    #   >50–75     HIGH
    #   >75        CRITICAL
    #
    # Final category is still assigned by final_risk.py after
    # ML scoring.
    # --------------------------------------------------------

    master[
        "rule_risk_category"
    ] = pd.cut(
        master[
            "rule_risk_score"
        ],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True,
        right=True,
    )

    # --------------------------------------------------------
    # Primary risk reason.
    # --------------------------------------------------------

    reason_map = {
        "financial_risk_score":
            "Financial deviation / cost anomaly",

        "execution_risk_score":
            "Execution delay / stalled spending",

        "duplicate_risk_score":
            "Potential duplicate / similar work",
    }

    score_columns = list(
        reason_map.keys()
    )

    score_frame = (
        master[score_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )

    max_component = score_frame.idxmax(
        axis=1
    )

    master[
        "primary_risk_reason"
    ] = max_component.map(
        reason_map
    )

    # --------------------------------------------------------
    # Human-readable explanation.
    # --------------------------------------------------------

    def build_reason(
        row: pd.Series,
    ) -> str:

        reasons: list[str] = []

        # Financial deviation.
        if bool(
            row.get(
                "flag_disbursement_over_sanction",
                False,
            )
        ):

            pct = row.get(
                "overspend_pct"
            )

            if pd.notna(pct):
                reasons.append(
                    "Expenditure exceeds sanction by "
                    f"{float(pct):.1f}%"
                )
            else:
                reasons.append(
                    "Expenditure exceeds sanction"
                )

        # Open work beyond benchmark.
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

        # Stalled expenditure.
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

        # Duplicate candidate.
        if bool(
            row.get(
                "flag_duplicate_candidate",
                False,
            )
        ):
            reasons.append(
                "Potential duplicate/similar "
                "work candidate"
            )

        # Cost outlier.
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

        # Duration outlier.
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

        # Date integrity.
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
                "No strong transparent "
                "rule-based signal"
            )

        return " • ".join(
            reasons
        )

    master[
        "risk_explanation"
    ] = master.apply(
        build_reason,
        axis=1,
    )

    # --------------------------------------------------------
    # Versioning.
    # --------------------------------------------------------

    master[
        "risk_engine_version"
    ] = RISK_ENGINE_VERSION

    return (
        master,
        available_flags,
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(
    master: pd.DataFrame,
    flags: list[str],
) -> None:
    """
    Print concise diagnostics for command-line validation.
    """

    print()
    print("========================================")
    print("RISK SIGNAL ENGINE COMPLETE")
    print("========================================")

    print(
        "Risk engine version:",
        RISK_ENGINE_VERSION,
    )

    print(
        "Rows:",
        len(master),
    )

    print(
        "Rule signals:",
        len(flags),
    )

    print(
        "Multi-signal records (>=2):",
        int(
            master[
                "independent_signal_count"
            ]
            .ge(2)
            .sum()
        ),
    )

    print(
        "Potential duplicate candidates:",
        int(
            master[
                "flag_duplicate_candidate"
            ]
            .sum()
        ),
    )

    print(
        "Financial signal records:",
        int(
            master[
                "financial_risk_score"
            ]
            .ge(60.0)
            .sum()
        ),
    )

    print(
        "Execution signal records:",
        int(
            master[
                "execution_risk_score"
            ]
            .ge(60.0)
            .sum()
        ),
    )

    print()
    print("Rule-risk distribution:")

    distribution = (
        master[
            "rule_risk_category"
        ]
        .value_counts(
            dropna=False,
            sort=False,
        )
    )

    print(distribution)

    print()
    print("Key score diagnostics:")

    diagnostic_columns = [
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
        "data_integrity_risk_score",
        "rule_risk_score",
    ]

    available_diagnostics = [
        column
        for column in diagnostic_columns
        if column in master.columns
    ]

    if available_diagnostics:
        print(
            master[
                available_diagnostics
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT}\n\n"
            "Run backend.feature_engineering first."
        )

    print(
        "Loading master analytical dataset..."
    )

    master = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    if master.empty:
        raise ValueError(
            f"Input dataset is empty:\n{INPUT}"
        )

    print(
        "Master shape:",
        master.shape,
    )

    # --------------------------------------------------------
    # Date normalization.
    # --------------------------------------------------------

    date_columns = [
        "recommended_date",
        "sanction_date",
        "completion_date",
        "first_expenditure_date",
        "last_expenditure_date",
        "monitoring_as_of_date",
    ]

    for column in date_columns:

        if column not in master.columns:
            continue

        master[column] = pd.to_datetime(
            master[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Generate signals.
    # --------------------------------------------------------

    print(
        "Generating transparent risk signals..."
    )

    master, flags = add_risk_features(
        master
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------

    print_diagnostics(
        master,
        flags,
    )

    print()
    print("Saved:")
    print(OUTPUT)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()