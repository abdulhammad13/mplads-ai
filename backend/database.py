import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# SETTINGS
# ============================================================

DB_SERVER = os.getenv(
    "DB_SERVER",
    r"LAPTOP-HF3PUVR1\SQLEXPRESS"
)

DB_NAME = os.getenv(
    "DB_NAME",
    "MPLADS_AI"
)

DB_DRIVER = os.getenv(
    "DB_DRIVER",
    "ODBC Driver 18 for SQL Server"
)

DB_TRUST_SERVER_CERTIFICATE = os.getenv(
    "DB_TRUST_SERVER_CERTIFICATE",
    "yes",
).strip().lower()

DB_CONNECT_TIMEOUT = int(
    os.getenv("DB_CONNECT_TIMEOUT", "30")
)


# ============================================================
# WINDOWS AUTHENTICATION
# ============================================================

DATABASE_URL = URL.create(
    "mssql+pyodbc",
    host=DB_SERVER,
    database=DB_NAME,
    query={
        "driver": DB_DRIVER,
        "trusted_connection": "yes",
        "TrustServerCertificate": DB_TRUST_SERVER_CERTIFICATE,
    },
)


# ============================================================
# ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args={"timeout": DB_CONNECT_TIMEOUT},
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# ============================================================
# TEST
# ============================================================

def test_connection():

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        print(
            "SQL Server connected successfully:",
            result.scalar()
        )


if __name__ == "__main__":
    test_connection()

