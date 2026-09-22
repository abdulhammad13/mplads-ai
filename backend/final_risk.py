from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import polars as pl

from .risk_engine import add_risk_features


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[1]
)

PROCESSED_DIR: Final[Path] = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT: Final[Path] = (
    PROCESSED_DIR / "master_works.csv"
)

MODEL_PATH: Final[Path] = (
    PROJECT_ROOT / "models" / "isolation_forest.pkl"
)

OUTPUT: Final[Path] = (
    PROCESSED_DIR / "final_risk_data.csv"
)

LEGACY_OUTPUT: Final[Path] = (
    PROCESSED_DIR / "risk_scored_works.csv"
)


# ============================================================
# VERSIONING
# ============================================================

FINAL_RISK_VERSION: Final[str] = "4.0"


# ============================================================
# FINAL RISK WEIGHTS
# ============================================================
#
# These are analytical design weights.
#
# They are NOT official MPLADS weights.
#
# Data quality affects confidence/priority, not substantive
# risk itself.
#

FINAL_WEIGHTS: Final[dict[str, float]] = {
    "ml_anomaly_percentile": 0.35,
    "financial_risk_score": 0.30,
    "execution_risk_score": 0.25,
    "duplicate_risk_score": 0.10,
}


# ============================================================
# RISK BINS
# ============================================================

RISK_BINS: Final[list[float]] = [
    -np.inf,
    25.0,
    50.0,
    75.0,
    np.inf,
]

RISK_LABELS: Final[list[str]] = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]


# ============================================================
# REQUIRED MODEL PACKAGE KEYS
# ============================================================

MODEL_KEY: Final[str] = "model"
FEATURES_KEY: Final[str] = "features"


# ============================================================
# DATE COLUMNS
# ============================================================

DATE_COLUMNS: Final[tuple[str, ...]] = (
    "recommended_date",
    "sanction_date",
    "completion_date",
    "first_expenditure_date",
    "last_expenditure_date",
    "monitoring_as_of_date",
)


# ============================================================
# MODEL SCORE HELPERS
# ============================================================

def _extract_reference_scores(
    package: dict,
) -> np.ndarray | None:
    """
    Extract an optional stable reference anomaly-score
    distribution from the trained model package.

    Supported package keys:

        reference_raw_anomaly_scores
        reference_scores
        training_raw_anomaly_scores

    The reference values must already be in anomaly direction:

        larger value = more anomalous

    At least 20 finite observations are required.
    """

    candidates = (
        package.get(
            "reference_raw_anomaly_scores"
        ),
        package.get(
            "reference_scores"
        ),
        package.get(
            "training_raw_anomaly_scores"
        ),
    )

    for candidate in candidates:

        if candidate is None:
            continue

        try:
            array = np.asarray(
                candidate,
                dtype=float,
            ).ravel()
        except (TypeError, ValueError):
            continue

        array = array[
            np.isfinite(array)
        ]

        if array.size >= 20:

            return np.sort(array)

    return None


def raw_to_percentile(
    raw_anomaly: pd.Series,
    reference: np.ndarray | None,
) -> tuple[pd.Series, str]:
    """
    Convert Isolation Forest anomaly-direction scores into
    a 0-100 percentile.

    IsolationForest.decision_function():

        higher = more normal
        lower  = more abnormal

    Therefore:

        abnormality = -decision_function()

    Larger final percentile values indicate more anomalous
    observations.
    """

    raw = pd.to_numeric(
        raw_anomaly,
        errors="coerce",
    )

    abnormality = -raw

    # --------------------------------------------------------
    # Stable reference population
    # --------------------------------------------------------

    if reference is not None:

        values = (
            abnormality
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        counts = np.searchsorted(
            reference,
            values,
            side="right",
        )

        percentile = (
            100.0
            * counts
            / len(reference)
        )

        score = pd.Series(
            percentile,
            index=raw.index,
            dtype="float64",
        )

        return (
            score.clip(0.0, 100.0),
            "reference_percentile",
        )

    # --------------------------------------------------------
    # Current scoring population fallback
    # --------------------------------------------------------

    score = (
        abnormality
        .rank(
            method="average",
            pct=True,
        )
        * 100.0
    )

    return (
        score
        .fillna(0.0)
        .clip(0.0, 100.0),
        "current_population_percentile",
    )


# ============================================================
# WEIGHTED MEAN
# ============================================================

def weighted_mean(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Calculate a weighted mean using only available dimensions.

    Missing dimensions do not automatically force the final
    score to zero or 100.

    The denominator is adjusted to the weights actually
    available for each observation.
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

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        valid = values.notna()

        numerator.loc[valid] += (
            values.loc[valid]
            * weight
        )

        denominator.loc[valid] += weight

    score = numerator.div(
        denominator.replace(
            0,
            np.nan,
        )
    )

    return (
        score
        .fillna(0.0)
        .clip(0.0, 100.0)
    )


# ============================================================
# EMPIRICAL PERCENTILE
# ============================================================

def empirical_percentile(
    values: pd.Series,
) -> pd.Series:
    """
    Convert numeric values into their empirical percentile.

    Used for financial exposure prioritization only.
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    return (
        numeric
        .rank(
            method="average",
            pct=True,
        )
        * 100.0
    ).fillna(0.0).clip(0.0, 100.0)


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    df: pd.DataFrame,
    model_features: list[str],
) -> pd.Series:
    """
    Calculate analytical signal confidence.

    Confidence reflects data support.

    It is NOT:

        - probability of wrongdoing
        - probability of fraud
        - probability that the risk score is correct

    Components:

        50% key-data completeness
        30% model-input completeness
        20% peer-group support
    """

    critical_fields = [
        "sanction_amount",
        "sanction_date",
        "work_category",
    ]

    available_critical = [
        column
        for column in critical_fields
        if column in df.columns
    ]

    if available_critical:

        critical_missing = (
            df[available_critical]
            .isna()
            .mean(axis=1)
        )

        data_completeness = (
            1.0 - critical_missing
        )

    else:

        data_completeness = pd.Series(
            0.5,
            index=df.index,
            dtype="float64",
        )

    model_available = [
        column
        for column in model_features
        if column in df.columns
    ]

    if model_available:

        model_missing = (
            df[model_available]
            .isna()
            .mean(axis=1)
        )

        model_completeness = (
            1.0 - model_missing
        )

    else:

        model_completeness = pd.Series(
            0.0,
            index=df.index,
            dtype="float64",
        )

    if "cost_peer_group_n" in df.columns:

        peer_counts = pd.to_numeric(
            df["cost_peer_group_n"],
            errors="coerce",
        ).fillna(0.0)

    else:

        peer_counts = pd.Series(
            0.0,
            index=df.index,
            dtype="float64",
        )

    peer_support = (
        peer_counts
        .div(50.0)
        .clip(0.0, 1.0)
    )

    confidence = 100.0 * (
        0.50 * data_completeness
        + 0.30 * model_completeness
        + 0.20 * peer_support
    )

    return (
        confidence
        .clip(0.0, 100.0)
        .round(2)
    )


# ============================================================
# SAFE JSON VALUE
# ============================================================

def _json_number(
    value: object,
    integer: bool = False,
) -> int | float | None:
    """
    Convert Pandas/NumPy values into JSON-safe numbers.
    """

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except (TypeError, ValueError):
        pass

    try:

        if integer:
            return int(value)

        return float(value)

    except (TypeError, ValueError):
        return None


# ============================================================
# EVIDENCE JSON
# ============================================================

def build_evidence_json(
    row: pd.Series,
) -> str:
    """
    Build a compact machine-readable evidence object.

    Evidence is descriptive and traceable to the underlying
    analytical dimensions.
    """

    evidence = {
        "financial": {
            "score": _json_number(
                row.get(
                    "financial_risk_score",
                    0,
                )
            ) or 0.0,

            "overspend_pct": _json_number(
                row.get(
                    "overspend_pct"
                )
            ),

            "utilization_pct": _json_number(
                row.get(
                    "utilization_pct"
                )
            ),

            "cost_robust_z": _json_number(
                row.get(
                    "cost_robust_z"
                )
            ),

            "peer_group_n": _json_number(
                row.get(
                    "cost_peer_group_n"
                ),
                integer=True,
            ),
        },

        "execution": {
            "score": _json_number(
                row.get(
                    "execution_risk_score",
                    0,
                )
            ) or 0.0,

            "days_open_since_sanction":
                _json_number(
                    row.get(
                        "days_open_since_sanction"
                    ),
                    integer=True,
                ),

            "days_over_general_one_year_benchmark":
                _json_number(
                    row.get(
                        "days_over_general_one_year_benchmark"
                    ),
                    integer=True,
                ),

            "days_since_last_expenditure":
                _json_number(
                    row.get(
                        "days_since_last_expenditure"
                    ),
                    integer=True,
                ),
        },

        "duplicate": {
            "score": _json_number(
                row.get(
                    "duplicate_risk_score",
                    0,
                )
            ) or 0.0,

            "is_candidate": bool(
                row.get(
                    "is_duplicate_candidate",
                    False,
                )
            ),

            "group_count": _json_number(
                row.get(
                    "duplicate_group_count"
                ),
                integer=True,
            ),
        },

        "data_integrity": {
            "score": _json_number(
                row.get(
                    "data_integrity_risk_score",
                    0,
                )
            ) or 0.0,

            "issue_count": _json_number(
                row.get(
                    "data_quality_issue_count",
                    0,
                ),
                integer=True,
            ) or 0,
        },

        "model": {
            "ml_anomaly_percentile":
                _json_number(
                    row.get(
                        "ml_anomaly_percentile",
                        0,
                    )
                ) or 0.0,
        },
    }

    return json.dumps(
        evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ============================================================
# MODEL PACKAGE VALIDATION
# ============================================================

def load_model_package() -> dict:
    """
    Load and validate the Isolation Forest package.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Missing Isolation Forest model:\n"
            f"{MODEL_PATH}\n\n"
            "Train/export the model first."
        )

    package = joblib.load(
        MODEL_PATH
    )

    if not isinstance(package, dict):

        raise TypeError(
            "isolation_forest.pkl must contain a "
            "dictionary with at least:\n"
            "  - model\n"
            "  - features\n"
            "and optionally:\n"
            "  - imputer\n"
            "  - preprocessor\n"
            "  - reference_raw_anomaly_scores"
        )

    if package.get(MODEL_KEY) is None:

        raise KeyError(
            "Model package is missing key: 'model'"
        )

    if package.get(FEATURES_KEY) is None:

        raise KeyError(
            "Model package is missing key: 'features'"
        )

    return package


# ============================================================
# MODEL FEATURE VALIDATION
# ============================================================

def validate_model_features(
    df: pd.DataFrame,
    features: list[str],
) -> None:
    """
    Verify that the master dataset contains exactly the inputs
    required by the trained model.
    """

    missing_features = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "The trained model expects features that are "
            "absent from master_works.csv:\n\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_features
            )
            + "\n\n"
            "Retrain the model against the current feature "
            "schema or restore the missing columns."
        )

    # --------------------------------------------------------
    # Circular-leakage protection
    # --------------------------------------------------------

    forbidden = [
        column
        for column in features
        if (
            column.startswith("final_risk")
            or column.startswith("priority")
            or column == "risk_category"
            or column.startswith("ml_anomaly")
        )
    ]

    if forbidden:

        raise ValueError(
            "Model feature leakage detected.\n"
            "The trained model contains final-risk/output "
            "columns as inputs:\n\n"
            + "\n".join(
                f"  - {column}"
                for column in forbidden
            )
            + "\n\n"
            "Retrain the model without these columns."
        )


# ============================================================
# MODEL INPUT TRANSFORMATION
# ============================================================

def transform_model_input(
    df: pd.DataFrame,
    features: list[str],
    package: dict,
) -> np.ndarray:
    """
    Transform model features using the preprocessing artifact
    stored in the model package.

    Supported configurations:

        1. preprocessor
        2. imputer
        3. no preprocessing

    NumPy is intentionally used at the scikit-learn boundary.
    """

    X = df[features].copy()

    preprocessor = package.get(
        "preprocessor"
    )

    imputer = package.get(
        "imputer"
    )

    # --------------------------------------------------------
    # Complete preprocessing pipeline
    # --------------------------------------------------------

    if preprocessor is not None:

        return np.asarray(
            preprocessor.transform(X)
        )

    # --------------------------------------------------------
    # Numeric imputer
    # --------------------------------------------------------

    if imputer is not None:

        X_numeric = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        return np.asarray(
            imputer.transform(
                X_numeric
            )
        )

    # --------------------------------------------------------
    # No preprocessing artifact.
    # --------------------------------------------------------

    X_numeric = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    return X_numeric.fillna(
        0.0
    ).to_numpy()


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate final analytical risk scores.

    The calculation has five conceptual stages:

        1. Isolation Forest anomaly score
        2. Rule-engine risk dimensions
        3. Weighted final risk score
        4. Confidence/data-support score
        5. Investigation-priority score

    The existing analytical methodology is retained.
    """

    if df.empty:

        raise ValueError(
            "Cannot calculate risk on an empty dataframe."
        )

    # --------------------------------------------------------
    # Copy input so caller's dataframe is not modified.
    # --------------------------------------------------------

    df = df.copy()

    # --------------------------------------------------------
    # Load model package.
    # --------------------------------------------------------

    package = load_model_package()

    model = package.get(
        "model"
    )

    features = list(
        package.get(
            "features"
        )
    )

    validate_model_features(
        df,
        features,
    )

    # --------------------------------------------------------
    # Model input.
    # --------------------------------------------------------

    X_transformed = (
        transform_model_input(
            df,
            features,
            package,
        )
    )

    # --------------------------------------------------------
    # Isolation Forest.
    #
    # decision_function:
    #
    #     higher -> more normal
    #     lower  -> more abnormal
    #
    # Therefore we invert it to obtain anomaly direction.
    # --------------------------------------------------------

    decision = model.decision_function(
        X_transformed
    )

    decision = np.asarray(
        decision,
        dtype=float,
    )

    if decision.ndim != 1:

        decision = decision.ravel()

    if len(decision) != len(df):

        raise ValueError(
            "Isolation Forest returned an unexpected "
            "number of scores.\n"
            f"Rows: {len(df)}\n"
            f"Scores: {len(decision)}"
        )

    df["ml_raw_decision_score"] = decision

    df["ml_raw_anomaly_score"] = (
        -decision
    )

    # --------------------------------------------------------
    # Convert to 0-100 anomaly percentile.
    # --------------------------------------------------------

    reference = (
        _extract_reference_scores(
            package
        )
    )

    (
        df["ml_anomaly_percentile"],
        score_method,
    ) = raw_to_percentile(
        df["ml_raw_anomaly_score"],
        reference,
    )

    df["ml_anomaly_score"] = (
        df["ml_anomaly_percentile"]
        .round(2)
    )

    df["ml_score_method"] = (
        score_method
    )

    # ========================================================
    # RULE ENGINE
    # ========================================================
    #
    # The risk_engine remains the authoritative implementation
    # of the transparent financial/execution/duplicate signals.
    #
    # We deliberately keep the Pandas boundary here rather than
    # duplicating its business logic.
    #

    df, _ = add_risk_features(
        df
    )

    # ========================================================
    # FINAL RISK SCORE
    # ========================================================

    df["final_risk_score"] = (
        weighted_mean(
            df,
            FINAL_WEIGHTS,
        )
        .round(2)
    )

    # --------------------------------------------------------
    # Risk category.
    # --------------------------------------------------------

    df["risk_category"] = pd.cut(
        df["final_risk_score"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True,
    )

    # Convert categorical values to strings so the output is
    # deterministic and JSON/API friendly.
    df["risk_category"] = (
        df["risk_category"]
        .astype("string")
    )

    # ========================================================
    # MODEL INPUT COMPLETENESS
    # ========================================================

    df["model_input_missing_pct"] = (
        df[features]
        .isna()
        .mean(axis=1)
        * 100.0
    ).round(2)

    # ========================================================
    # CONFIDENCE
    # ========================================================

    df["confidence_score"] = (
        calculate_confidence(
            df,
            features,
        )
    )

    # ========================================================
    # FINANCIAL EXPOSURE
    # ========================================================
    #
    # This is a prioritization metric.
    #
    # It is NOT itself a risk score.
    #

    if "sanction_amount" in df.columns:

        df[
            "financial_exposure_percentile"
        ] = (
            empirical_percentile(
                df["sanction_amount"]
            )
            .round(2)
        )

    else:

        df[
            "financial_exposure_percentile"
        ] = 0.0

    # ========================================================
    # INVESTIGATION PRIORITY
    # ========================================================
    #
    # Priority combines:
    #
    #     final risk
    #     financial exposure
    #     confidence/data support
    #
    # This is a triage mechanism and is not an official
    # MPLADS score.
    #

    risk = (
        pd.to_numeric(
            df["final_risk_score"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    exposure = (
        pd.to_numeric(
            df[
                "financial_exposure_percentile"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        / 100.0
    )

    confidence = (
        pd.to_numeric(
            df["confidence_score"],
            errors="coerce",
        )
        .fillna(0.0)
        / 100.0
    )

    df["priority_score"] = (
        risk
        * (
            0.50
            + 0.50 * exposure
        )
        * (
            0.70
            + 0.30 * confidence
        )
    ).clip(
        0.0,
        100.0,
    ).round(2)

    # ========================================================
    # PRIORITY RANK
    # ========================================================

    df["priority_rank"] = (
        df["priority_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    # ========================================================
    # RISK EXPLANATION
    # ========================================================

    if "risk_explanation" not in df.columns:

        df["risk_explanation"] = (
            "No strong rule-based signal"
        )

    df["risk_explanation"] = (
        df["risk_explanation"]
        .fillna(
            "No strong rule-based signal"
        )
        .astype(str)
    )

    # ========================================================
    # MACHINE-READABLE EVIDENCE
    # ========================================================

    df["evidence_json"] = [
        build_evidence_json(
            row
        )
        for _, row in df.iterrows()
    ]

    # ========================================================
    # VERSION METADATA
    # ========================================================

    df["final_risk_version"] = (
        FINAL_RISK_VERSION
    )

    df["model_version"] = str(
        package.get(
            "model_version",
            "unversioned",
        )
    )

    # ========================================================
    # DETERMINISTIC SORT
    # ========================================================

    sort_columns = [
        column
        for column in (
            "priority_score",
            "final_risk_score",
            "sanction_amount",
            "work_uid",
        )
        if column in df.columns
    ]

    ascending = []

    for column in sort_columns:

        if column in (
            "priority_score",
            "final_risk_score",
            "sanction_amount",
        ):
            ascending.append(False)

        else:
            ascending.append(True)

    if sort_columns:

        df = (
            df
            .sort_values(
                sort_columns,
                ascending=ascending,
                na_position="last",
            )
            .reset_index(drop=True)
        )

    return df


# ============================================================
# POLARS → PANDAS BOUNDARY
# ============================================================

def polars_to_model_dataframe(
    df: pl.DataFrame,
) -> pd.DataFrame:
    """
    Convert Polars data to Pandas at the ML/risk-engine boundary.

    Pandas is intentionally retained here because:

        - scikit-learn preprocessing artifacts may expect it
        - risk_engine.add_risk_features() is currently Pandas-based
        - model compatibility is more important than forcing a
          premature rewrite of the risk engine
    """

    return df.to_pandas()


# ============================================================
# PANDAS → POLARS BOUNDARY
# ============================================================

def pandas_to_polars(
    df: pd.DataFrame,
) -> pl.DataFrame:
    """
    Convert the scored Pandas dataframe back into Polars.

    The rest of the pipeline can therefore remain Polars-native.
    """

    return pl.from_pandas(
        df,
        include_index=False,
    )


# ============================================================
# LOAD MASTER WITH POLARS
# ============================================================

def load_master() -> pl.DataFrame:
    """
    Load master_works.csv using Polars.

    This avoids Pandas CSV parsing for the main analytical
    dataset.
    """

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing master dataset:\n"
            f"{INPUT}\n\n"
            "Run feature_engineering.py first."
        )

    df = pl.read_csv(
        INPUT,
        infer_schema_length=10000,
        try_parse_dates=True,
        ignore_errors=False,
    )

    if df.is_empty():

        raise ValueError(
            f"Master dataset is empty:\n{INPUT}"
        )

    return df


# ============================================================
# NORMALIZE OUTPUT TYPES
# ============================================================

def normalize_output_types(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Normalize final output columns for stable CSV/API use.
    """

    expressions: list[pl.Expr] = []

    # --------------------------------------------------------
    # Numeric fields.
    # --------------------------------------------------------

    numeric_columns = (
        "ml_raw_decision_score",
        "ml_raw_anomaly_score",
        "ml_anomaly_percentile",
        "ml_anomaly_score",
        "final_risk_score",
        "model_input_missing_pct",
        "confidence_score",
        "financial_exposure_percentile",
        "priority_score",
        "financial_risk_score",
        "execution_risk_score",
        "duplicate_risk_score",
        "data_integrity_risk_score",
    )

    for column in numeric_columns:

        if column in df.columns:

            expressions.append(
                pl.col(column)
                .cast(
                    pl.Float64,
                    strict=False,
                )
                .alias(column)
            )

    # --------------------------------------------------------
    # Integer-like fields.
    # --------------------------------------------------------

    integer_columns = (
        "priority_rank",
        "data_quality_issue_count",
        "duplicate_group_count",
        "num_transactions",
        "num_vendors",
    )

    for column in integer_columns:

        if column in df.columns:

            expressions.append(
                pl.col(column)
                .cast(
                    pl.Int64,
                    strict=False,
                )
                .alias(column)
            )

    # --------------------------------------------------------
    # Dates.
    # --------------------------------------------------------

    for column in DATE_COLUMNS:

        if column in df.columns:

            expressions.append(
                pl.col(column)
                .cast(
                    pl.Date,
                    strict=False,
                )
                .alias(column)
            )

    # --------------------------------------------------------
    # Strings.
    # --------------------------------------------------------

    string_columns = (
        "risk_category",
        "ml_score_method",
        "risk_explanation",
        "evidence_json",
        "final_risk_version",
        "model_version",
    )

    for column in string_columns:

        if column in df.columns:

            expressions.append(
                pl.col(column)
                .cast(
                    pl.String,
                    strict=False,
                )
                .alias(column)
            )

    if expressions:

        df = df.with_columns(
            expressions
        )

    return df


# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

def save_final_output(
    df: pl.DataFrame,
) -> None:
    """
    Save the authoritative final risk dataset and the legacy
    compatibility mirror.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Authoritative output.
    # --------------------------------------------------------

    df.write_csv(
        OUTPUT,
        include_bom=False,
        null_value="",
    )

    # --------------------------------------------------------
    # Legacy compatibility output.
    #
    # Keep this until all downstream modules have migrated to
    # final_risk_data.csv.
    # --------------------------------------------------------

    df.write_csv(
        LEGACY_OUTPUT,
        include_bom=False,
        null_value="",
    )


# ============================================================
# PIPELINE SUMMARY
# ============================================================

def print_summary(
    df: pl.DataFrame,
) -> None:
    """
    Print a concise validation summary.
    """

    print()
    print(
        "========================================"
    )
    print(
        "FINAL RISK SCORING COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        "Final risk version:",
        FINAL_RISK_VERSION,
    )

    print(
        "Records:",
        df.height,
    )

    # --------------------------------------------------------
    # Mean / median.
    # --------------------------------------------------------

    if "final_risk_score" in df.columns:

        mean_risk = (
            df["final_risk_score"]
            .mean()
        )

        median_risk = (
            df["final_risk_score"]
            .median()
        )

        print(
            "Mean risk:",
            (
                round(float(mean_risk), 2)
                if mean_risk is not None
                else "N/A"
            ),
        )

        print(
            "Median risk:",
            (
                round(float(median_risk), 2)
                if median_risk is not None
                else "N/A"
            ),
        )

    # --------------------------------------------------------
    # Risk categories.
    # --------------------------------------------------------

    if "risk_category" in df.columns:

        high_critical = (
            df
            .filter(
                pl.col("risk_category")
                .is_in(
                    [
                        "HIGH",
                        "CRITICAL",
                    ]
                )
            )
            .height
        )

        critical = (
            df
            .filter(
                pl.col("risk_category")
                == "CRITICAL"
            )
            .height
        )

        print(
            "High + Critical:",
            high_critical,
        )

        print(
            "Critical:",
            critical,
        )

        print()
        print(
            "Risk distribution:"
        )

        distribution = (
            df
            .group_by(
                "risk_category"
            )
            .agg(
                pl.len()
                .alias("count")
            )
            .sort(
                "risk_category"
            )
        )

        for row in distribution.iter_rows(
            named=True
        ):

            print(
                f"  {row['risk_category']}: "
                f"{row['count']}"
            )

    # --------------------------------------------------------
    # Multi-signal count.
    # --------------------------------------------------------

    if "independent_signal_count" in df.columns:

        multi_signal = (
            df
            .filter(
                pl.col(
                    "independent_signal_count"
                ).ge(2)
            )
            .height
        )

        print(
            "Multi-signal >=2:",
            multi_signal,
        )

    # --------------------------------------------------------
    # Confidence.
    # --------------------------------------------------------

    if "confidence_score" in df.columns:

        mean_confidence = (
            df["confidence_score"]
            .mean()
        )

        if mean_confidence is not None:

            print(
                "Mean confidence:",
                round(
                    float(mean_confidence),
                    2,
                ),
            )

    print()
    print(
        "Saved authoritative output:",
        OUTPUT,
    )

    print(
        "Saved compatibility mirror:",
        LEGACY_OUTPUT,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Execute the final-risk pipeline.

    Architecture:

        master_works.csv
              ↓
        Polars loading
              ↓
        Pandas ML boundary
              ↓
        Isolation Forest
              ↓
        Pandas risk engine
              ↓
        Polars output
              ↓
        final_risk_data.csv
    """

    print(
        "Loading master dataset with Polars..."
    )

    master_polars = load_master()

    print(
        "Master shape:",
        master_polars.shape,
    )

    print(
        "Converting to Pandas at ML/risk-engine boundary..."
    )

    master_pandas = (
        polars_to_model_dataframe(
            master_polars
        )
    )

    print(
        "Calculating Isolation Forest + rule-based risk..."
    )

    scored_pandas = calculate_risk(
        master_pandas
    )

    print(
        "Converting scored data back to Polars..."
    )

    scored_polars = (
        pandas_to_polars(
            scored_pandas
        )
    )

    print(
        "Normalizing final output schema..."
    )

    scored_polars = (
        normalize_output_types(
            scored_polars
        )
    )

    print(
        "Saving final risk datasets..."
    )

    save_final_output(
        scored_polars
    )

    print_summary(
        scored_polars
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()