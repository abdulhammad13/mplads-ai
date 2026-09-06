from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .risk_engine import add_risk_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "processed" / "master_works.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.pkl"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "final_risk_data.csv"
LEGACY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "risk_scored_works.csv"

FINAL_RISK_VERSION = "3.0"

# Final risk weights. These are analytical design weights, not official MPLADS
# weights. Keep them documented and versioned so results remain auditable.
# Substantive risk only. Data quality affects confidence/priority, not
# substantive risk itself, because missing data is not evidence of wrongdoing.
FINAL_WEIGHTS = {
    "ml_anomaly_percentile": 0.35,
    "financial_risk_score": 0.30,
    "execution_risk_score": 0.25,
    "duplicate_risk_score": 0.10,
}

RISK_BINS = [-np.inf, 25.0, 50.0, 75.0, np.inf]
RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ============================================================
# MODEL SCORE HELPERS
# ============================================================


def _extract_reference_scores(package: dict) -> np.ndarray | None:
    """
    Optional stable reference distribution for model scoring.

    Recommended package key:
        reference_raw_anomaly_scores

    It should contain raw anomaly-direction scores on a representative training
   /reference population. If absent, the pipeline falls back to the current
    scoring population's empirical percentile.
    """
    candidates = [
        package.get("reference_raw_anomaly_scores"),
        package.get("reference_scores"),
        package.get("training_raw_anomaly_scores"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        arr = np.asarray(candidate, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size >= 20:
            return np.sort(arr)
    return None


def raw_to_percentile(raw_anomaly: pd.Series, reference: np.ndarray | None) -> tuple[pd.Series, str]:
    """
    Convert anomaly-direction score to 0-100 where larger = more anomalous.

    IsolationForest decision_function: lower values are more abnormal. We first
    invert the sign, then map to a percentile. This preserves monotonicity.
    """
    raw = pd.to_numeric(raw_anomaly, errors="coerce")
    abnormality = -raw

    if reference is not None:
        # Reference scores must also be in anomaly direction (larger = more abnormal).
        counts = np.searchsorted(reference, abnormality.fillna(0).to_numpy(), side="right")
        percentile = 100.0 * counts / len(reference)
        score = pd.Series(percentile, index=raw.index)
        return score.clip(0, 100), "reference_percentile"

    # Fallback: empirical percentile within the current full scoring population.
    score = abnormality.rank(method="average", pct=True) * 100
    return score.fillna(0).clip(0, 100), "current_population_percentile"


def weighted_mean(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)

    for column, weight in weights.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        numerator.loc[valid] += values.loc[valid] * weight
        denominator.loc[valid] += weight

    return numerator.div(denominator.replace(0, np.nan)).fillna(0).clip(0, 100)


def empirical_percentile(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return (x.rank(method="average", pct=True) * 100).fillna(0).clip(0, 100)


def calculate_confidence(df: pd.DataFrame, model_features: list[str]) -> pd.Series:
    """
    Confidence reflects data support, not correctness probability.

    It decreases when key/model inputs are missing and when peer groups are too
    small for strong contextual comparison. It must be labelled as confidence
    in the analytical signal, not a probability of wrongdoing.
    """
    critical_fields = [
        "sanction_amount",
        "sanction_date",
        "work_category",
    ]
    available_critical = [c for c in critical_fields if c in df.columns]
    if available_critical:
        critical_missing = df[available_critical].isna().mean(axis=1)
        data_completeness = 1 - critical_missing
    else:
        data_completeness = pd.Series(0.5, index=df.index)

    model_available = [c for c in model_features if c in df.columns]
    if model_available:
        model_missing = df[model_available].isna().mean(axis=1)
        model_completeness = 1 - model_missing
    else:
        model_completeness = pd.Series(0.0, index=df.index)

    peer_counts = pd.to_numeric(
        df.get("cost_peer_group_n", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    peer_support = (peer_counts / 50.0).clip(0, 1)

    confidence = 100 * (
        0.50 * data_completeness
        + 0.30 * model_completeness
        + 0.20 * peer_support
    )
    return confidence.clip(0, 100).round(2)


def build_evidence_json(row: pd.Series) -> str:
    evidence = {
        "financial": {
            "score": float(row.get("financial_risk_score", 0) or 0),
            "overspend_pct": None if pd.isna(row.get("overspend_pct")) else float(row["overspend_pct"]),
            "utilization_pct": None if pd.isna(row.get("utilization_pct")) else float(row["utilization_pct"]),
            "cost_robust_z": None if pd.isna(row.get("cost_robust_z")) else float(row["cost_robust_z"]),
            "peer_group_n": None if pd.isna(row.get("cost_peer_group_n")) else int(row["cost_peer_group_n"]),
        },
        "execution": {
            "score": float(row.get("execution_risk_score", 0) or 0),
            "days_open_since_sanction": None if pd.isna(row.get("days_open_since_sanction")) else int(row["days_open_since_sanction"]),
            "days_over_general_one_year_benchmark": None if pd.isna(row.get("days_over_general_one_year_benchmark")) else int(row["days_over_general_one_year_benchmark"]),
            "days_since_last_expenditure": None if pd.isna(row.get("days_since_last_expenditure")) else int(row["days_since_last_expenditure"]),
        },
        "duplicate": {
            "score": float(row.get("duplicate_risk_score", 0) or 0),
            "is_candidate": bool(row.get("is_duplicate_candidate", False)),
            "group_count": None if pd.isna(row.get("duplicate_group_count")) else int(row["duplicate_group_count"]),
        },
        "data_integrity": {
            "score": float(row.get("data_integrity_risk_score", 0) or 0),
            "issue_count": int(row.get("data_quality_issue_count", 0) or 0),
        },
        "model": {
            "ml_anomaly_percentile": float(row.get("ml_anomaly_percentile", 0) or 0),
        },
    }
    return json.dumps(evidence, ensure_ascii=False)


# ============================================================
# MAIN RISK CALCULATION
# ============================================================


def calculate_risk(df: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing Isolation Forest model: {MODEL_PATH}\n"
            "Train/export the model first."
        )

    df = df.copy()

    package = joblib.load(MODEL_PATH)
    if not isinstance(package, dict):
        raise TypeError(
            "isolation_forest.pkl must contain a dict with at least "
            "model, imputer/preprocessor, and features."
        )

    model = package.get("model")
    imputer = package.get("imputer")
    preprocessor = package.get("preprocessor")
    features = package.get("features")

    if model is None:
        raise KeyError("Model package is missing key: 'model'")
    if features is None:
        raise KeyError("Model package is missing key: 'features'")

    features = list(features)

    missing_features = [column for column in features if column not in df.columns]
    if missing_features:
        raise ValueError(
            "The trained model expects features that are absent from master_works.csv: "
            + ", ".join(missing_features)
            + "\nRetrain the model against the v2 feature schema or restore those columns."
        )

    # Prevent circular leakage: final/priority/risk outputs must never be model inputs.
    forbidden = [
        column for column in features
        if (
            column.startswith("final_risk")
            or column.startswith("priority")
            or column == "risk_category"
            or column.startswith("ml_anomaly")
        )
    ]
    if forbidden:
        raise ValueError(
            "Model feature leakage detected. Retrain the model without: "
            + ", ".join(forbidden)
        )

    X = df[features].copy()

    # Existing project models use a numeric imputer. If a future model package
    # stores a complete preprocessor, use it instead.
    if preprocessor is not None:
        X_transformed = preprocessor.transform(X)
    elif imputer is not None:
        X_numeric = X.apply(pd.to_numeric, errors="coerce")
        X_transformed = imputer.transform(X_numeric)
    else:
        X_transformed = X.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()

    # IsolationForest decision_function is higher for more normal observations;
    # negative/inverted values therefore represent anomaly direction.
    decision = model.decision_function(X_transformed)
    df["ml_raw_decision_score"] = decision
    df["ml_raw_anomaly_score"] = -decision

    reference = _extract_reference_scores(package)
    df["ml_anomaly_percentile"], score_method = raw_to_percentile(
        df["ml_raw_anomaly_score"],
        reference,
    )
    df["ml_anomaly_score"] = df["ml_anomaly_percentile"].round(2)
    df["ml_score_method"] = score_method

    # --------------------------------------------------------
    # Transparent signals from one authoritative rule engine.
    # --------------------------------------------------------

    df, _ = add_risk_features(df)

    # --------------------------------------------------------
    # Final risk score.
    # This combines independent dimensions rather than a binary 3-flag count.
    # --------------------------------------------------------

    df["final_risk_score"] = weighted_mean(df, FINAL_WEIGHTS).round(2)

    df["risk_category"] = pd.cut(
        df["final_risk_score"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
    )

    # --------------------------------------------------------
    # Data support / confidence.
    # --------------------------------------------------------

    df["model_input_missing_pct"] = (
        df[features].isna().mean(axis=1) * 100
    ).round(2)
    df["confidence_score"] = calculate_confidence(df, features)

    # --------------------------------------------------------
    # Financial exposure percentile.
    # Use rank only for prioritization; do not call it risk.
    # --------------------------------------------------------

    df["financial_exposure_percentile"] = empirical_percentile(
        df["sanction_amount"]
    ).round(2)

    # --------------------------------------------------------
    # Investigation priority.
    # High risk + high financial exposure + stronger evidence support.
    # This is a triage score, not an official MPLADS score.
    # --------------------------------------------------------

    risk = df["final_risk_score"]
    exposure = df["financial_exposure_percentile"] / 100.0
    confidence = df["confidence_score"] / 100.0

    df["priority_score"] = (
        risk
        * (0.50 + 0.50 * exposure)
        * (0.70 + 0.30 * confidence)
    ).clip(0, 100).round(2)

    df["priority_rank"] = (
        df["priority_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    # --------------------------------------------------------
    # Explanation / evidence.
    # --------------------------------------------------------

    if "risk_explanation" not in df.columns:
        df["risk_explanation"] = "No strong rule-based signal"

    df["evidence_json"] = df.apply(build_evidence_json, axis=1)
    df["final_risk_version"] = FINAL_RISK_VERSION
    df["model_version"] = str(package.get("model_version", "unversioned"))

    # Deterministic output order: highest priority first.
    df = df.sort_values(
        ["priority_score", "final_risk_score", "sanction_amount"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return df


# ============================================================
# EXECUTION
# ============================================================


if __name__ == "__main__":
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing master dataset: {INPUT}\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(INPUT, low_memory=False)

    date_columns = [
        "recommended_date",
        "sanction_date",
        "completion_date",
        "first_expenditure_date",
        "last_expenditure_date",
        "monitoring_as_of_date",
    ]
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    df = calculate_risk(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")

    # Backward-compatibility mirror only. The application should treat
    # final_risk_data.csv as the single authoritative output.
    df.to_csv(LEGACY_OUTPUT, index=False, date_format="%Y-%m-%d")

    print("\n========================================")
    print("FINAL RISK SCORING COMPLETE")
    print("========================================")
    print("Final risk version:", FINAL_RISK_VERSION)
    print("Records:", len(df))
    print("Mean risk:", round(df["final_risk_score"].mean(), 2))
    print("Median risk:", round(df["final_risk_score"].median(), 2))
    print("High + Critical:", int(df["risk_category"].isin(["HIGH", "CRITICAL"]).sum()))
    print("Critical:", int(df["risk_category"].eq("CRITICAL").sum()))
    print("Multi-signal >=2:", int(df["independent_signal_count"].ge(2).sum()))
    print("\nRisk distribution:")
    print(df["risk_category"].value_counts().sort_index())
    print("\nSaved authoritative output:", OUTPUT)
    print("Saved compatibility mirror:", LEGACY_OUTPUT)
