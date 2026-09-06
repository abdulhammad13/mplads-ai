from pathlib import Path

import pandas as pd
from sqlalchemy import text

from .database import engine


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_risk_data.csv"
)


# ============================================================
# REQUIRED ML / RISK COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "work_uid",
    "rule_risk_score",
    "ml_anomaly_score",
    "ml_anomaly_percentile",
    "final_risk_score",
    "risk_category",
    "priority_score",
    "priority_rank",
    "confidence_score",
]


# ============================================================
# LOAD FINAL DATA
# ============================================================

if __name__ == "__main__":

    print("\nReading final risk dataset...")
    print(INPUT)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"\nFinal risk dataset not found:\n{INPUT}"
        )

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(f"\nCSV rows: {len(df):,}")
    print(f"CSV columns: {len(df.columns):,}")


    # ========================================================
    # NORMALIZE POSSIBLE OLD COLUMN NAME
    # ========================================================

    if (
        "final_risk_category" in df.columns
        and "risk_category" not in df.columns
    ):
        df = df.rename(
            columns={
                "final_risk_category":
                    "risk_category"
            }
        )

        print(
            "Renamed: final_risk_category → risk_category"
        )


    # ========================================================
    # CHECK REQUIRED RISK COLUMNS
    # ========================================================

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        print(
            "\n❌ REQUIRED RISK COLUMNS ARE MISSING:"
        )

        for column in missing:
            print(
                f"   - {column}"
            )

        print(
            "\nCurrent risk/anomaly columns:"
        )

        print([
            column
            for column in df.columns
            if any(
                word in column.lower()
                for word in [
                    "risk",
                    "anomaly",
                    "score",
                    "flag"
                ]
            )
        ])

        raise ValueError(
            "\nThe final_risk_data.csv is not yet the "
            "complete ML/risk dataset."
        )


    # ========================================================
    # CLEAN RISK DATA TYPES
    # ========================================================

    numeric_columns = [
        "rule_risk_score",
        "ml_anomaly_score",
        "final_risk_score",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df["risk_category"] = (
        df["risk_category"]
        .astype("string")
        .str.strip()
        .str.upper()
    )


    # ========================================================
    # SHOW RISK SUMMARY BEFORE SQL LOAD
    # ========================================================

    print("\nRisk category distribution:")

    print(
        df["risk_category"]
        .value_counts(
            dropna=False
        )
    )


    print(
        "\nRisk score summary:"
    )

    print(
        df[
            [
                "rule_risk_score",
                "ml_anomaly_score",
                "final_risk_score",
            ]
        ].describe()
    )


    # ========================================================
    # WRITE TO SQL SERVER
    # ========================================================

    print(
        "\nReplacing SQL Server table: dbo.works ..."
    )

    df.to_sql(
        name="works",
        con=engine,
        schema="dbo",
        if_exists="replace",
        index=False,
        chunksize=1000,
    )


    # ========================================================
    # VERIFY SQL SERVER
    # ========================================================

    with engine.connect() as connection:

        row_count = connection.execute(
            text(
                "SELECT COUNT(*) "
                "FROM dbo.works"
            )
        ).scalar()


        columns = connection.execute(
            text(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'dbo'
                  AND TABLE_NAME = 'works'
                ORDER BY ORDINAL_POSITION
                """
            )
        ).fetchall()


    sql_columns = [
        row[0]
        for row in columns
    ]


    print(
        "\n========================================"
    )
    print(
        "✅ SQL SERVER LOAD COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        f"\nCSV rows : {len(df):,}"
    )

    print(
        f"SQL rows : {row_count:,}"
    )


    print(
        "\nML / Risk columns now in SQL:"
    )

    for column in REQUIRED_COLUMNS:

        print(
            f"   {'✅' if column in sql_columns else '❌'} "
            f"{column}"
        )


    # --------------------------------------------------------
    # Final consistency check
    # --------------------------------------------------------

    if row_count != len(df):

        raise RuntimeError(
            "Row-count mismatch between CSV and SQL Server."
        )

    if not all(
        column in sql_columns
        for column in REQUIRED_COLUMNS
    ):

        raise RuntimeError(
            "One or more required ML/risk columns "
            "were not created in SQL Server."
        )


    print(
        "\n✅ CSV and SQL Server are synchronized."
    )
