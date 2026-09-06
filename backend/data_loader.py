from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# OFFICIAL SOURCE FILES
# ============================================================

FILES = {
    "allocation": "Allocated Limit for Honble MPs.csv",
    "calamity": "Amount consented for Calamity.csv",
    "expenditure": "Expenditure on Completed and On-going Works as on Date.csv",
    "completed": "Works Completed.csv",
    "recommended": "Works Recommended.csv",
    "sanctioned": "Works Sanctioned.csv",
}


# ============================================================
# DATE COLUMNS
# ============================================================

DATE_COLUMNS = {
    "calamity": [
        "date_of_consent"
    ],

    "expenditure": [
        "expenditure_date"
    ],

    "completed": [
        "completion_date"
    ],

    "recommended": [
        "recommended_date",
        "sanction_date"
    ],

    "sanctioned": [
        "recommended_date",
        "sanction_date"
    ],
}


# ============================================================
# MONEY COLUMNS
# ============================================================

MONEY_COLUMNS = {
    "allocation": [
        "allocated_amount"
    ],

    "calamity": [
        "consent_amount"
    ],

    "expenditure": [
        "fund_disbursed_amount"
    ],

    "completed": [
        "amount_disbursed"
    ],

    "recommended": [
        "recommended_amount"
    ],

    "sanctioned": [
        "sanction_amount"
    ],
}


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_columns(df):
    """
    Convert government CSV column names into consistent
    Python-friendly names.

    Example:
        'Sanction Amount ( ₹ )'
        ->
        'sanction_amount'
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    return df


# ============================================================
# NORMALIZE MP COLUMN NAME
# ============================================================

def normalize_mp_column(df):
    """
    Different source files may use slightly different
    spellings for the MP column. Convert them all to 'mp'.
    """

    mp_variants = [
        "honble_members_of_parliament",
        "honble_members_of_parliaments",
        "hon_ble_members_of_parliament",
        "hon_ble_members_of_parliaments",
    ]

    for column in mp_variants:

        if column in df.columns:

            df = df.rename(
                columns={
                    column: "mp"
                }
            )

            break

    return df


# ============================================================
# REMOVE SUMMARY / TOTAL ROWS
# ============================================================

def remove_summary_rows(df):
    """
    Remove non-record summary rows such as:
        Grand Total
        Total
        Sub Total
        Subtotal

    This is done after reading the source but before
    cleaning/calculating features.

    Raw CSV files remain untouched.
    """

    summary_values = {
        "grand total",
        "total",
        "sub total",
        "subtotal",
    }

    summary_mask = (
        df
        .astype("string")
        .apply(
            lambda column: (
                column
                .str.strip()
                .str.lower()
                .isin(summary_values)
            )
        )
        .any(axis=1)
    )

    removed = int(summary_mask.sum())

    if removed > 0:
        print(
            f"  Removed summary rows: {removed}"
        )

    return df.loc[~summary_mask].copy()


# ============================================================
# CLEAN MONEY
# ============================================================

def clean_money(series):
    """
    Convert monetary strings into numeric values.

    Examples:
        '₹ 1,25,000' -> 125000
        '1,25,000'   -> 125000
        blank        -> NaN
    """

    cleaned = (
        series
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("Rs.", "", case=False, regex=False)
        .str.replace("Rs", "", case=False, regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(series):
    """
    Remove leading/trailing spaces and collapse repeated
    whitespace without destroying missing values.
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )


# ============================================================
# LOAD ALL OFFICIAL DATA
# ============================================================

def load_data():

    data = {}

    print("\nLoading official MPLADS datasets...\n")

    for name, filename in FILES.items():

        print(f"Loading: {filename}")

        # ----------------------------------------------------
        # File path
        # ----------------------------------------------------

        path = RAW_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"\nDataset not found:\n{path}"
            )

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            path,
            low_memory=False
        )

        original_rows = len(df)

        # ----------------------------------------------------
        # Standardize column names
        # ----------------------------------------------------

        df = clean_columns(df)

        # ----------------------------------------------------
        # Normalize MP field
        # ----------------------------------------------------

        df = normalize_mp_column(df)

        # ----------------------------------------------------
        # Remove summary rows
        # ----------------------------------------------------

        df = remove_summary_rows(df)

        # ----------------------------------------------------
        # Convert dates
        # ----------------------------------------------------

        for column in DATE_COLUMNS.get(name, []):

            if column in df.columns:

                df[column] = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

        # ----------------------------------------------------
        # Convert money columns
        # ----------------------------------------------------

        for column in MONEY_COLUMNS.get(name, []):

            if column in df.columns:

                df[column] = clean_money(
                    df[column]
                )

        # ----------------------------------------------------
        # Clean text columns
        # ----------------------------------------------------

        text_columns = [
            "state",
            "ida",
            "mp",
            "constituency",
            "work",
            "work_description",
            "vendor_name",
            "work_category",
            "work_status",
            "payment_status",
        ]

        for column in text_columns:

            if column in df.columns:

                df[column] = clean_text(
                    df[column]
                )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        data[name] = df

        print(
            f"  Rows: {original_rows:,} -> {len(df):,}"
        )

    print(
        "\n✅ All official datasets loaded successfully."
    )

    return data


# ============================================================
# SAVE CLEANED DATASETS
# ============================================================

def save_clean_data(data):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving cleaned datasets...\n")

    for name, df in data.items():

        output = (
            PROCESSED_DIR
            / f"{name}_clean.csv"
        )

        df.to_csv(
            output,
            index=False
        )

        print(
            f"Saved: {output.name} "
            f"({len(df):,} rows)"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    datasets = load_data()

    save_clean_data(
        datasets
    )

    print(
        "\n✅ DATA LOADER COMPLETE."
    )
