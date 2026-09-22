from __future__ import annotations

"""
MPLADS AI MONITOR
-----------------
Isolation Forest anomaly-model training module.

Responsibilities
----------------
1. Load the engineered master_works dataset.
2. Validate the model feature contract.
3. Convert model inputs safely to numeric values.
4. Fit a median-imputation + Isolation Forest pipeline boundary.
5. Store a stable reference anomaly-score distribution for downstream scoring.
6. Persist a versioned, auditable model package.
7. Provide reusable load / predict helpers for batch or API pipelines.

Important:
- This module detects statistical anomalies; it does NOT establish fraud,
  corruption, illegality, or wrongdoing.
- Model features are deliberately limited to upstream analytical features.
- Final risk scoring remains the responsibility of final_risk.py.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_works.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"

MODEL_VERSION = "4.0"
MODEL_TYPE = "IsolationForest"

# These are the established upstream model features in the project.
# Keep "overrun_pct" for backward compatibility with the existing trained
# feature schema. feature_engineering.py currently creates this alias from
# the canonical overspend field.
MODEL_FEATURES: tuple[str, ...] = (
    "sanction_amount",
    "recommended_amount",
    "overrun_pct",
    "days_rec_to_sanction",
    "days_sanction_to_complete",
    "duplicate_group_count",
)

# Training parameters.
N_ESTIMATORS = 300
CONTAMINATION: str | float = "auto"
RANDOM_STATE = 42
N_JOBS = -1

# A reference distribution this large is enough for stable percentile
# conversion while preventing an unnecessarily huge pickle.
MAX_REFERENCE_SCORES = 100_000

# Minimum observations required for a meaningful unsupervised model.
MIN_ROWS = 50

# Minimum number of usable feature columns.
MIN_FEATURES = 2

# Require at least this many finite observations in a feature before using it.
MIN_VALID_VALUES_PER_FEATURE = 10


# ============================================================================
# DATA CONTRACT
# ============================================================================

@dataclass(frozen=True)
class ModelPackage:
    """
    Serializable model contract.

    The dict representation written by save_model() intentionally contains
    simple Python/scikit-learn objects so the existing final_risk.py can load
    it using joblib.load().
    """

    model: IsolationForest
    imputer: SimpleImputer
    features: list[str]
    reference_raw_anomaly_scores: np.ndarray
    model_version: str
    model_type: str
    trained_at_utc: str
    training_rows: int
    training_features: int
    random_state: int
    n_estimators: int
    contamination: str | float


# ============================================================================
# VALIDATION / NORMALIZATION
# ============================================================================

def _require_input_file(path: Path = INPUT_PATH) -> None:
    """Fail early with a precise message when the engineered dataset is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required training dataset was not found:\n{path}\n\n"
            "Run the feature-engineering pipeline first so that "
            "data/processed/master_works.csv exists."
        )

    if not path.is_file():
        raise ValueError(f"Training input path is not a file: {path}")


def _load_master(path: Path = INPUT_PATH) -> pl.DataFrame:
    """
    Load master_works.csv using Polars.

    infer_schema=False keeps source parsing conservative. Model columns are
    explicitly cast to Float64 afterward, which prevents a single malformed
    text value from changing the schema unpredictably.
    """
    _require_input_file(path)

    df = pl.read_csv(
        path,
        infer_schema=False,
        null_values=["", "NULL", "null", "NA", "N/A", "NaN"],
        ignore_errors=False,
    )

    if df.is_empty():
        raise ValueError(f"Training dataset is empty: {path}")

    return df


def _select_features(
    df: pl.DataFrame,
    candidates: Sequence[str] = MODEL_FEATURES,
) -> tuple[list[str], list[str]]:
    """
    Validate candidate features and return:
        (usable_features, excluded_features)

    A feature is usable when it exists and has enough finite numeric values.
    """
    usable: list[str] = []
    excluded: list[str] = []

    for column in candidates:
        if column not in df.columns:
            excluded.append(f"{column} [missing]")
            continue

        numeric = (
            df
            .select(
                pl.col(column)
                .cast(pl.Float64, strict=False)
                .alias(column)
            )
            .get_column(column)
        )

        finite_count = int(
            numeric
            .is_not_null()
            .sum()
        )

        # Polars Float64 nulls are distinct from +/-inf, so explicitly
        # inspect finite values through NumPy at this small validation boundary.
        values = numeric.to_numpy()
        finite_count = int(np.isfinite(values).sum())

        if finite_count < MIN_VALID_VALUES_PER_FEATURE:
            excluded.append(
                f"{column} [only {finite_count} finite values]"
            )
            continue

        usable.append(column)

    if len(usable) < MIN_FEATURES:
        details = "; ".join(excluded) if excluded else "none"
        raise ValueError(
            f"Only {len(usable)} usable anomaly features were found; "
            f"at least {MIN_FEATURES} are required.\n"
            f"Expected features: {list(candidates)}\n"
            f"Excluded: {details}"
        )

    return usable, excluded


def _build_feature_matrix(
    df: pl.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    """
    Convert selected Polars columns to a clean float64 NumPy matrix.

    This is the intentional ML boundary:
        Polars -> NumPy -> scikit-learn
    """
    matrix = (
        df
        .select(
            [
                pl.col(column)
                .cast(pl.Float64, strict=False)
                .alias(column)
                for column in features
            ]
        )
        .to_numpy()
    )

    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Unexpected feature matrix shape: {matrix.shape}"
        )

    if matrix.shape[0] < MIN_ROWS:
        raise ValueError(
            f"Only {matrix.shape[0]} rows are available for training. "
            f"At least {MIN_ROWS} rows are required."
        )

    if matrix.shape[1] < MIN_FEATURES:
        raise ValueError(
            f"Only {matrix.shape[1]} usable features are available."
        )

    # Convert +/-inf to missing values so SimpleImputer can handle them.
    matrix[~np.isfinite(matrix)] = np.nan

    if np.all(np.isnan(matrix)):
        raise ValueError(
            "All model feature values are missing/non-finite. "
            "The Isolation Forest cannot be trained."
        )

    return matrix


def _validate_feature_quality(
    matrix: np.ndarray,
    features: Sequence[str],
) -> None:
    """Produce a precise diagnostic for completely missing model columns."""
    for index, feature in enumerate(features):
        valid = int(np.isfinite(matrix[:, index]).sum())

        if valid == 0:
            raise ValueError(
                f"Feature '{feature}' contains no usable numeric values."
            )


# ============================================================================
# MODEL TRAINING
# ============================================================================

def _fit_model(
    X: np.ndarray,
) -> tuple[SimpleImputer, IsolationForest, np.ndarray]:
    """
    Fit imputer and Isolation Forest.

    Returns:
        imputer
        model
        raw anomaly-direction scores

    IsolationForest decision_function:
        larger  -> more normal
        smaller -> more abnormal

    Therefore:
        raw_anomaly = -decision_function

    Larger raw_anomaly means more statistically unusual.
    """
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=False,
        keep_empty_features=False,
    )

    X_imputed = imputer.fit_transform(X)

    if not np.isfinite(X_imputed).all():
        raise ValueError(
            "Non-finite values remain after imputation."
        )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    model.fit(X_imputed)

    decision = np.asarray(
        model.decision_function(X_imputed),
        dtype=np.float64,
    )

    raw_anomaly = -decision

    if not np.isfinite(raw_anomaly).all():
        raise ValueError(
            "Isolation Forest produced non-finite anomaly scores."
        )

    return imputer, model, raw_anomaly


def _build_reference_scores(
    raw_anomaly: np.ndarray,
    max_scores: int = MAX_REFERENCE_SCORES,
) -> np.ndarray:
    """
    Store a sorted empirical reference distribution.

    For the normal project scale this will usually contain every training
    observation. A deterministic evenly spaced sample is used only when the
    training set exceeds MAX_REFERENCE_SCORES.
    """
    values = np.asarray(raw_anomaly, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size < MIN_ROWS:
        raise ValueError(
            "Too few finite reference anomaly scores were produced."
        )

    values.sort()

    if values.size <= max_scores:
        return values

    # Deterministic down-sampling; no random state required.
    indices = np.linspace(
        0,
        values.size - 1,
        num=max_scores,
        dtype=np.int64,
    )

    return values[indices]


def _training_summary(
    X: np.ndarray,
    features: Sequence[str],
    raw_anomaly: np.ndarray,
) -> dict[str, Any]:
    """Create compact audit metadata without storing the training dataframe."""
    missing_pct = {
        feature: round(
            float(np.isnan(X[:, i]).mean() * 100.0),
            4,
        )
        for i, feature in enumerate(features)
    }

    return {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "feature_names": list(features),
        "missing_percent_by_feature": missing_pct,
        "anomaly_score_min": float(np.min(raw_anomaly)),
        "anomaly_score_median": float(np.median(raw_anomaly)),
        "anomaly_score_max": float(np.max(raw_anomaly)),
    }


# ============================================================================
# SERIALIZATION
# ============================================================================

def _package_to_dict(
    *,
    model: IsolationForest,
    imputer: SimpleImputer,
    features: Sequence[str],
    reference_raw_anomaly_scores: np.ndarray,
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the backward-compatible joblib package."""
    return {
        # Required by final_risk.py.
        "model": model,
        "imputer": imputer,
        "features": list(features),

        # Stable reference distribution for downstream percentile conversion.
        # IMPORTANT: these values are in anomaly direction:
        # larger = more abnormal.
        "reference_raw_anomaly_scores": np.asarray(
            reference_raw_anomaly_scores,
            dtype=np.float64,
        ),

        # Version / audit metadata.
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(training_summary["rows"]),
        "training_features": int(training_summary["features"]),
        "training_summary": dict(training_summary),

        # Reproducibility.
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,

        # Explicit contract documentation.
        "score_definition": (
            "raw_anomaly_score = -IsolationForest.decision_function(X)"
        ),
        "score_direction": (
            "higher values indicate greater statistical abnormality"
        ),
        "interpretation_notice": (
            "Anomaly score is a statistical monitoring signal and is not "
            "proof of wrongdoing, fraud, corruption, or illegality."
        ),
    }


def save_model(package: Mapping[str, Any], path: Path = MODEL_PATH) -> Path:
    """Atomically persist the model package."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    try:
        joblib.dump(
            dict(package),
            temporary_path,
            compress=3,
        )
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return path


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load and validate the persisted model package."""
    if not path.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found:\n{path}\n\n"
            "Train the model first with:\n"
            "python -m ml.anomaly_model"
        )

    package = joblib.load(path)

    if not isinstance(package, dict):
        raise TypeError(
            "Invalid isolation_forest.pkl: expected a dictionary package."
        )

    required = {"model", "imputer", "features"}

    missing = sorted(required - package.keys())

    if missing:
        raise KeyError(
            "Invalid Isolation Forest package. Missing keys: "
            + ", ".join(missing)
        )

    if not isinstance(package["features"], (list, tuple)):
        raise TypeError(
            "Model package 'features' must be a list or tuple."
        )

    if not isinstance(package["model"], IsolationForest):
        raise TypeError(
            "Model package 'model' is not an IsolationForest instance."
        )

    return package


# ============================================================================
# PREDICTION / INFERENCE
# ============================================================================

def _prepare_inference_matrix(
    df: pl.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    """Prepare an inference matrix using the exact trained feature contract."""
    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Inference data is missing trained model features: "
            + ", ".join(missing)
        )

    matrix = (
        df
        .select(
            [
                pl.col(feature)
                .cast(pl.Float64, strict=False)
                .alias(feature)
                for feature in features
            ]
        )
        .to_numpy()
    )

    matrix = np.asarray(matrix, dtype=np.float64)
    matrix[~np.isfinite(matrix)] = np.nan

    return matrix


def predict_anomalies(
    df: pl.DataFrame,
    package: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """
    Score a Polars DataFrame using a persisted model package.

    Returns the original dataframe plus:
        ml_raw_decision_score
        ml_raw_anomaly_score
        ml_anomaly_label

    No final risk score is calculated here.
    """
    if package is None:
        package = load_model()

    model = package["model"]
    imputer = package["imputer"]
    features = list(package["features"])

    X = _prepare_inference_matrix(df, features)
    X_imputed = imputer.transform(X)

    decision = np.asarray(
        model.decision_function(X_imputed),
        dtype=np.float64,
    )

    raw_anomaly = -decision

    labels = np.where(
        model.predict(X_imputed) == -1,
        "ANOMALY",
        "NORMAL",
    )

    return df.with_columns(
        pl.Series(
            "ml_raw_decision_score",
            decision,
            dtype=pl.Float64,
        ),
        pl.Series(
            "ml_raw_anomaly_score",
            raw_anomaly,
            dtype=pl.Float64,
        ),
        pl.Series(
            "ml_anomaly_label",
            labels,
            dtype=pl.String,
        ),
    )


# ============================================================================
# TRAINING ENTRY POINT
# ============================================================================

def train(
    input_path: Path = INPUT_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """
    Train the Isolation Forest and persist the complete model package.

    Returns the saved package metadata.
    """
    print("=" * 72)
    print("MPLADS AI MONITOR — ISOLATION FOREST TRAINING")
    print("=" * 72)
    print(f"Input : {input_path}")
    print(f"Model : {model_path}")
    print(f"Version: {MODEL_VERSION}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df = _load_master(input_path)

    print(f"Rows loaded: {df.height:,}")
    print(f"Columns    : {df.width}")

    # ------------------------------------------------------------------
    # Feature contract
    # ------------------------------------------------------------------
    features, excluded = _select_features(df)

    if excluded:
        print("\nExcluded model features:")
        for item in excluded:
            print(f"  - {item}")

    print("\nFeatures used:")
    for feature in features:
        print(f"  + {feature}")

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------
    X = _build_feature_matrix(df, features)
    _validate_feature_quality(X, features)

    print(f"\nFeature matrix: {X.shape[0]:,} × {X.shape[1]}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    imputer, model, raw_anomaly = _fit_model(X)

    print("\nIsolation Forest fitted.")
    print(f"Estimators : {N_ESTIMATORS}")
    print(f"Contamination: {CONTAMINATION}")
    print(f"Random state: {RANDOM_STATE}")

    # ------------------------------------------------------------------
    # Reference distribution
    # ------------------------------------------------------------------
    reference = _build_reference_scores(raw_anomaly)

    summary = _training_summary(
        X,
        features,
        raw_anomaly,
    )

    package = _package_to_dict(
        model=model,
        imputer=imputer,
        features=features,
        reference_raw_anomaly_scores=reference,
        training_summary=summary,
    )

    saved_path = save_model(
        package,
        model_path,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MODEL TRAINING COMPLETE")
    print("=" * 72)
    print(f"Training rows       : {summary['rows']:,}")
    print(f"Training features   : {summary['features']}")
    print(f"Reference scores    : {reference.size:,}")
    print(
        "Raw anomaly range   : "
        f"{summary['anomaly_score_min']:.6f} → "
        f"{summary['anomaly_score_max']:.6f}"
    )
    print(f"Saved model         : {saved_path}")
    print(f"Model size          : {saved_path.stat().st_size / 1024:.1f} KB")
    print("\nScore semantics:")
    print("  higher raw anomaly score = more statistically unusual")
    print("  anomaly score ≠ proof of wrongdoing")

    return package


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    train()
