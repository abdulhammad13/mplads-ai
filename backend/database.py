from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, URL
from sqlalchemy.orm import Session, sessionmaker


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


# ============================================================
# DATABASE SETTINGS
# ============================================================

DB_SERVER = os.getenv(
    "DB_SERVER",
    r"LAPTOP-HF3PUVR1\SQLEXPRESS",
).strip()

DB_NAME = os.getenv(
    "DB_NAME",
    "MPLADS_AI",
).strip()

DB_DRIVER = os.getenv(
    "DB_DRIVER",
    "ODBC Driver 18 for SQL Server",
).strip()

DB_TRUST_SERVER_CERTIFICATE = os.getenv(
    "DB_TRUST_SERVER_CERTIFICATE",
    "yes",
).strip().lower()

DB_CONNECT_TIMEOUT = int(
    os.getenv(
        "DB_CONNECT_TIMEOUT",
        "30",
    )
)

DB_POOL_SIZE = int(
    os.getenv(
        "DB_POOL_SIZE",
        "5",
    )
)

DB_MAX_OVERFLOW = int(
    os.getenv(
        "DB_MAX_OVERFLOW",
        "10",
    )
)

DB_POOL_TIMEOUT = int(
    os.getenv(
        "DB_POOL_TIMEOUT",
        "30",
    )
)

DB_POOL_RECYCLE = int(
    os.getenv(
        "DB_POOL_RECYCLE",
        "1800",
    )
)


# ============================================================
# VALIDATION
# ============================================================

if not DB_SERVER:
    raise ValueError(
        "DB_SERVER cannot be empty."
    )

if not DB_NAME:
    raise ValueError(
        "DB_NAME cannot be empty."
    )

if not DB_DRIVER:
    raise ValueError(
        "DB_DRIVER cannot be empty."
    )

if DB_CONNECT_TIMEOUT <= 0:
    raise ValueError(
        "DB_CONNECT_TIMEOUT must be greater than 0."
    )

if DB_POOL_SIZE < 1:
    raise ValueError(
        "DB_POOL_SIZE must be at least 1."
    )

if DB_MAX_OVERFLOW < 0:
    raise ValueError(
        "DB_MAX_OVERFLOW cannot be negative."
    )

if DB_POOL_TIMEOUT <= 0:
    raise ValueError(
        "DB_POOL_TIMEOUT must be greater than 0."
    )

if DB_POOL_RECYCLE <= 0:
    raise ValueError(
        "DB_POOL_RECYCLE must be greater than 0."
    )


# ============================================================
# DATABASE URL
# ============================================================

DATABASE_URL = URL.create(
    drivername="mssql+pyodbc",
    host=DB_SERVER,
    database=DB_NAME,
    query={
        "driver": DB_DRIVER,
        "trusted_connection": "yes",
        "TrustServerCertificate": DB_TRUST_SERVER_CERTIFICATE,
    },
)


# ============================================================
# SQLALCHEMY ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=DB_POOL_RECYCLE,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    connect_args={
        "timeout": DB_CONNECT_TIMEOUT,
    },
)


# ============================================================
# SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ============================================================
# FASTAPI DATABASE DEPENDENCY
# ============================================================

def get_db() -> Generator[Session, None, None]:
    """
    Provide a SQLAlchemy session for FastAPI request handlers.

    The session is always closed after the request, including
    when the endpoint raises an exception.

    Usage:

        def endpoint(db: Session = Depends(get_db)):
            ...
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ============================================================
# CONNECTION HEALTH CHECK
# ============================================================

def test_connection() -> bool:
    """
    Test connectivity to the configured SQL Server database.

    Returns
    -------
    bool
        True when SELECT 1 succeeds.

    Raises
    ------
    ConnectionError
        If SQL Server cannot be reached.
    """

    try:
        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT 1 AS connection_test")
            )

            value = result.scalar_one()

            if value != 1:
                raise ConnectionError(
                    "SQL Server responded, but the connection "
                    "health check returned an unexpected value."
                )

        print(
            "SQL Server connected successfully."
        )

        return True

    except Exception as exc:

        raise ConnectionError(
            "Unable to connect to SQL Server.\n"
            f"Server: {DB_SERVER}\n"
            f"Database: {DB_NAME}\n"
            f"Driver: {DB_DRIVER}\n"
            f"Original error: {exc}"
        ) from exc


# ============================================================
# RAW CONNECTION HELPER
# ============================================================

def get_connection() -> Connection:
    """
    Open and return a SQLAlchemy database connection.

    The caller is responsible for closing the connection.

    Prefer:

        with get_connection() as connection:
            ...

    for normal usage.
    """

    return engine.connect()


# ============================================================
# ENGINE CLEANUP
# ============================================================

def dispose_engine() -> None:
    """
    Dispose of all connections currently held by the
    SQLAlchemy connection pool.

    Useful during application shutdown or controlled restarts.
    """

    engine.dispose()


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_connection()