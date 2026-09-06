from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_works.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "isolation_forest.pkl"
)


FEATURES = [
    "sanction_amount",
    "recommended_amount",
    "overrun_pct",
    "days_rec_to_sanction",
    "days_sanction_to_complete",
    "duplicate_group_count",
]


def train():

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    available = [
        column
        for column in FEATURES
        if column in df.columns
    ]

    if len(available) < 2:
        raise ValueError(
            "Not enough numerical features available for anomaly detection."
        )

    X = df[available].copy()

    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    model.fit(X)

    # Lower decision_function = more abnormal.
    raw = -model.decision_function(X)

    # Rank into 0–100 for easy dashboard interpretation.
    scores = (
        pd.Series(raw)
        .rank(pct=True)
        * 100
    )

    df["ml_anomaly_score"] = (
        scores.round(2)
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "features": available,
        },
        MODEL_PATH,
    )

    print("✅ Model trained.")
    print("Features:", available)
    print("Saved:", MODEL_PATH)

    return df


if __name__ == "__main__":
    train()