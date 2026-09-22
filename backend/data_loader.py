from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl


# ============================================================
# PROJECT PATHS
# ============================================================

# Actual project structure:
#
# mplads-ai/
# ├── backend/
# │   └── data_loader.py
# ├── data/
# │   ├── raw/
# │   └── processed/
# └── ...
#
# __file__ -> mplads-ai/backend/data_loader.py
# parents[0] -> backend/
# parents[1] -> mplads-ai/
#
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"


# ============================================================
# OFFICIAL SOURCE FILES
# ============================================================

FILES: Final[dict[str, str]] = {
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

DATE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "calamity": (
        "date_of_consent",
    ),

    "expenditure": (
        "expenditure_date",
    ),

    "completed": (
        "completion_date",
    ),

    "recommended": (
        "recommended_date",
        "sanction_date",
    ),

    "sanctioned": (
        "recommended_date",
        "sanction_date",
    ),
}


# ============================================================
# MONEY COLUMNS
# ============================================================

MONEY_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "allocation": (
        "allocated_amount",
    ),

    "calamity": (
        "consent_amount",
    ),

    "expenditure": (
        "fund_disbursed_amount",
    ),

    "completed": (
        "amount_disbursed",
    ),

    "recommended": (
        "recommended_amount",
    ),

    "sanctioned": (
        "sanction_amount",
    ),
}


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "sr_no",
    "s_no",
    "serial_no",
    "serial_number",
)


# ============================================================
# TEXT COLUMNS
# ============================================================

TEXT_COLUMNS: Final[tuple[str, ...]] = (
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
)


# ============================================================
# SUMMARY VALUES
# ============================================================

SUMMARY_VALUES: Final[tuple[str, ...]] = (
    "grand total",
    "total",
    "sub total",
    "subtotal",
)


# ============================================================
# NULL VALUES
# ============================================================

CSV_NULL_VALUES: Final[tuple[str, ...]] = (
    "",
    "NA",
    "N/A",
    "NULL",
    "null",
)


# ============================================================
# COLUMN-NAME NORMALIZATION
# ============================================================

def _normalize_column_name(column: str) -> str:
    """
    Convert a government CSV column name into a consistent
    Python-friendly snake_case name.

    Examples
    --------
    'Sanction Amount ( ₹ )'
        -> 'sanction_amount'

    'Honble Members of Parliament'
        -> 'honble_members_of_parliament'
    """

    name = str(column).strip().lower()

    characters: list[str] = []

    for character in name:
        if character.isalnum() or character == "_":
            characters.append(character)
        else:
            characters.append("_")

    name = "".join(characters)

    # Collapse repeated underscores.
    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize dataframe column names.

    Raises
    ------
    ValueError
        If multiple source columns become the same normalized
        column name.
    """

    original_columns = df.columns

    cleaned_columns = [
        _normalize_column_name(column)
        for column in original_columns
    ]

    seen: set[str] = set()
    duplicates: set[str] = set()

    for column in cleaned_columns:
        if column in seen:
            duplicates.add(column)
        seen.add(column)

    if duplicates:
        raise ValueError(
            "Column-name collision detected after normalization: "
            f"{sorted(duplicates)}. "
            f"Original columns: {original_columns}"
        )

    return df.rename(
        dict(zip(original_columns, cleaned_columns))
    )


# ============================================================
# NORMALIZE MP COLUMN
# ============================================================

def normalize_mp_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize known variations of the MP column to 'mp'.
    """

    mp_variants = (
        "honble_members_of_parliament",
        "honble_members_of_parliaments",
        "hon_ble_members_of_parliament",
        "hon_ble_members_of_parliaments",
    )

    for column in mp_variants:
        if column in df.columns:
            return df.rename({column: "mp"})

    return df


# ============================================================
# REMOVE SUMMARY / TOTAL ROWS
# ============================================================

def remove_summary_rows(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove non-record summary rows such as:

        Grand Total
        Total
        Sub Total
        Subtotal

    The serial-number column is preferred when available.

    If a serial-number column does not exist, a small set of
    identifying columns is used as a fallback.

    This avoids incorrectly deleting legitimate records simply
    because a descriptive field contains the word 'total'.
    """

    if df.is_empty():
        return df

    # --------------------------------------------------------
    # Prefer serial-number / identifier columns
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    if identifier_columns:

        summary_mask = pl.any_horizontal(
            [
                (
                    pl.col(column)
                    .cast(pl.String, strict=False)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .is_in(SUMMARY_VALUES)
                )
                for column in identifier_columns
            ]
        )

    else:

        # ----------------------------------------------------
        # Fallback identifying columns
        # ----------------------------------------------------

        candidate_columns = [
            column
            for column in (
                "mp",
                "state",
                "constituency",
                "work",
                "work_description",
            )
            if column in df.columns
        ]

        if not candidate_columns:
            return df

        summary_mask = pl.any_horizontal(
            [
                (
                    pl.col(column)
                    .cast(pl.String, strict=False)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .is_in(SUMMARY_VALUES)
                )
                for column in candidate_columns
            ]
        )

    removed = (
        df
        .select(
            summary_mask.cast(pl.UInt64).sum()
        )
        .item()
    )

    removed = int(removed or 0)

    if removed > 0:
        print(
            f"  Removed summary rows: {removed:,}"
        )

    return df.filter(~summary_mask)


# ============================================================
# CLEAN MONEY EXPRESSION
# ============================================================

def clean_money_expression(column: str) -> pl.Expr:
    """
    Convert monetary text into Float64.

    Supported examples:

        ₹ 1,25,000
        1,25,000
        Rs. 50,000
        Rs 50000
        50000.50

    Invalid or empty values become null.
    """

    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.replace_all("₹", "")
        .str.replace_all(
            r"(?i)\brs\.?\b",
            "",
        )
        .str.replace_all(",", "")
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(column)
    )


# ============================================================
# CLEAN NUMERIC EXPRESSION
# ============================================================

def clean_numeric_expression(column: str) -> pl.Expr:
    """
    Convert numeric-looking source fields to Float64.

    Invalid values become null.
    """

    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(column)
    )


# ============================================================
# CLEAN TEXT EXPRESSION
# ============================================================

def clean_text_expression(column: str) -> pl.Expr:
    """
    Normalize text values while preserving nulls.

    Operations:
        - cast to string safely
        - remove leading/trailing whitespace
        - collapse repeated whitespace
    """

    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .alias(column)
    )


# ============================================================
# CLEAN DATE EXPRESSION
# ============================================================

def clean_date_expression(column: str) -> pl.Expr:
    """
    Parse government-source date values using explicit formats.

    Supported formats include:

        DD/MM/YYYY
        DD-MM-YYYY
        DD.MM.YYYY
        YYYY-MM-DD
        MM/DD/YYYY

        DD/MM/YYYY HH:MM:SS
        DD-MM-YYYY HH:MM:SS
        DD.MM.YYYY HH:MM:SS
        YYYY-MM-DD HH:MM:SS
        MM/DD/YYYY HH:MM:SS

        DD/MM/YYYY HH:MM:SS.s
        DD-MM-YYYY HH:MM:SS.s
        YYYY-MM-DD HH:MM:SS.s

        DD-Mon-YYYY
        DD-Month-YYYY

        DD-Mon-YYYY HH:MM:SS
        DD-Month-YYYY HH:MM:SS

    Invalid values become null.
    """

    value = (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
    )

    formats = (
        # Date only
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",

        # Date + time
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",

        # Date + fractional seconds
        "%d/%m/%Y %H:%M:%S%.f",
        "%d-%m-%Y %H:%M:%S%.f",
        "%d.%m.%Y %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%m/%d/%Y %H:%M:%S%.f",

        # Month names
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d-%b-%Y %H:%M:%S",
        "%d-%B-%Y %H:%M:%S",
    )

    parsed = [
        value.str.strptime(
            pl.Datetime,
            format=date_format,
            strict=False,
        )
        for date_format in formats
    ]

    return (
        pl.coalesce(parsed)
        .alias(column)
    )


# ============================================================
# CLEAN ONE DATASET
# ============================================================

def clean_dataset(
    name: str,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Apply the complete cleaning pipeline to one dataset.

    Pipeline:

        raw dataframe
            ↓
        clean column names
            ↓
        normalize MP
            ↓
        remove summary rows
            ↓
        numeric conversion
            ↓
        date conversion
            ↓
        money conversion
            ↓
        text normalization
    """

    # --------------------------------------------------------
    # 1. Standardize column names
    # --------------------------------------------------------

    df = clean_columns(df)

    # --------------------------------------------------------
    # 2. Normalize MP column
    # --------------------------------------------------------

    df = normalize_mp_column(df)

    # --------------------------------------------------------
    # 3. Remove summary rows
    # --------------------------------------------------------

    df = remove_summary_rows(df)

    # --------------------------------------------------------
    # 4. Numeric identifier columns
    # --------------------------------------------------------

    numeric_expressions = [
        clean_numeric_expression(column)
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    if numeric_expressions:
        df = df.with_columns(
            numeric_expressions
        )

    # --------------------------------------------------------
    # 5. Date columns
    # --------------------------------------------------------

    date_expressions = [
        clean_date_expression(column)
        for column in DATE_COLUMNS.get(name, ())
        if column in df.columns
    ]

    if date_expressions:
        df = df.with_columns(
            date_expressions
        )

    # --------------------------------------------------------
    # 6. Money columns
    # --------------------------------------------------------

    money_expressions = [
        clean_money_expression(column)
        for column in MONEY_COLUMNS.get(name, ())
        if column in df.columns
    ]

    if money_expressions:
        df = df.with_columns(
            money_expressions
        )

    # --------------------------------------------------------
    # 7. Text columns
    # --------------------------------------------------------

    text_expressions = [
        clean_text_expression(column)
        for column in TEXT_COLUMNS
        if column in df.columns
    ]

    if text_expressions:
        df = df.with_columns(
            text_expressions
        )

    return df


# ============================================================
# LOAD SINGLE DATASET
# ============================================================

def load_dataset(
    name: str,
    filename: str,
) -> pl.DataFrame:
    """
    Load and clean one official MPLADS CSV dataset.

    All CSV fields are initially read as strings.

    This is intentional because government datasets can contain
    summary rows such as 'Grand Total' inside otherwise numeric
    columns.
    """

    path = RAW_DIR / filename

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"\nDataset path is not a file:\n{path}"
        )

    print(f"Loading: {filename}")

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------
    #
    # infer_schema=False is deliberate.
    #
    # Government CSVs can contain mixed values such as:
    #
    #     1
    #     2
    #     3
    #     Grand Total
    #
    # Reading everything initially as strings prevents Polars
    # from failing during schema inference.
    #
    # Correct data types are assigned later by our cleaning
    # pipeline.
    #

    df = pl.read_csv(
        path,
        infer_schema=False,
        null_values=CSV_NULL_VALUES,
        ignore_errors=False,
    )

    original_rows = df.height

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    df = clean_dataset(
        name=name,
        df=df,
    )

    print(
        f"  Rows: {original_rows:,} -> {df.height:,}"
    )

    return df


# ============================================================
# LOAD ALL OFFICIAL DATA
# ============================================================

def load_data() -> dict[str, pl.DataFrame]:
    """
    Load all official MPLADS datasets.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary containing one Polars dataframe for each
        official dataset.
    """

    data: dict[str, pl.DataFrame] = {}

    print(
        "\nLoading official MPLADS datasets...\n"
    )

    for name, filename in FILES.items():

        data[name] = load_dataset(
            name=name,
            filename=filename,
        )

    print(
        "\nAll official datasets loaded successfully."
    )

    return data


# ============================================================
# SAVE CLEANED DATASETS
# ============================================================

def save_clean_data(
    data: dict[str, pl.DataFrame],
) -> None:
    """
    Save cleaned datasets as CSV files under:

        data/processed/
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving cleaned datasets...\n"
    )

    for name, df in data.items():

        output = (
            PROCESSED_DIR
            / f"{name}_clean.csv"
        )

        df.write_csv(
            output,
            include_bom=False,
            null_value="",
        )

        print(
            f"Saved: {output.name} "
            f"({df.height:,} rows)"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete data-loading and cleaning pipeline.
    """

    datasets = load_data()

    save_clean_data(
        datasets
    )

    print(
        "\nDATA LOADER COMPLETE."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()