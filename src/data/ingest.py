"""
Data ingestion module for BankGuard AI.

Handles loading the Kaggle Bank Customer Churn Prediction CSV,
column renaming, SQLite storage, and SQL-based feature extraction.
This module provides the single entry-point ``run_ingestion_pipeline()``
that is consumed by both the training script and the dashboard.
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "bank_churn.db"
RAW_CSV = RAW_DIR / "Bank Customer Churn Prediction.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. CSV loading ──────────────────────────────────────────────
def load_csv(path: Path | None = None) -> pd.DataFrame:
    """Load the Bank Customer Churn Prediction CSV.

    Performs essential cleaning on load:
    - Drops ``RowNumber`` and ``Surname`` columns.
    - Renames columns with spaces: ``Satisfaction Score`` →
      ``Satisfaction_Score``, ``Card Type`` → ``Card_Type``,
      ``Point Earned`` → ``Point_Earned``.

    Parameters
    ----------
    path : Path, optional
        CSV file path. Defaults to ``data/raw/Bank Customer Churn Prediction.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with 16 columns (18 original minus
        RowNumber and Surname).

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.  Download from
        https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn
    """
    path = path or RAW_CSV

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please download from:\n"
            "  https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn\n"
            "and place 'Bank Customer Churn Prediction.csv' in data/raw/"
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d records from %s", len(df), path)

    # Drop unnecessary columns
    drop_cols = [c for c in ["RowNumber", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info("Dropped columns: %s", drop_cols)

    # Rename columns with spaces
    rename_map = {
        "Satisfaction Score": "Satisfaction_Score",
        "Card Type": "Card_Type",
        "Point Earned": "Point_Earned",
    }
    df = df.rename(columns=rename_map)
    logger.info("Renamed columns: %s", list(rename_map.keys()))

    logger.info(
        "Cleaned dataset: %d rows × %d cols — churn rate %.1f%%",
        len(df),
        len(df.columns),
        100 * df["Exited"].mean(),
    )
    return df


# ── 2. SQLite persistence ──────────────────────────────────────
def load_to_sqlite(df: pd.DataFrame, db_path: Path | None = None) -> None:
    """Write the raw DataFrame into a SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer data.
    db_path : Path, optional
        SQLite database file path.
    """
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        df.to_sql("customers_raw", conn, if_exists="replace", index=False)
        logger.info("Loaded %d rows into SQLite (%s)", len(df), db_path)
    finally:
        conn.close()


# ── 3. SQL-based feature extraction ────────────────────────────
def query_features_from_sql(db_path: Path | None = None) -> pd.DataFrame:
    """Extract derived features via raw SQL.

    Derived columns:
      - **balance_salary_ratio**: ``Balance / MAX(EstimatedSalary, 1)``
      - **credit_age_ratio**: ``CreditScore / MAX(Age, 1)``
      - **is_zero_balance**: 1 if Balance = 0, else 0
      - **product_tenure_ratio**: ``NumOfProducts / MAX(Tenure, 1)``
      - **is_high_complain_risk**: 1 if Complain=1 AND Satisfaction_Score≤2

    Parameters
    ----------
    db_path : Path, optional
        SQLite database file path.

    Returns
    -------
    pd.DataFrame
        Customer data enriched with derived features.
    """
    db_path = db_path or DB_PATH

    query = """
    SELECT
        *,

        ROUND(
            CAST(Balance AS REAL) / MAX(EstimatedSalary, 1),
            4
        ) AS balance_salary_ratio,

        ROUND(
            CAST(CreditScore AS REAL) / MAX(Age, 1),
            4
        ) AS credit_age_ratio,

        CASE WHEN Balance = 0 THEN 1 ELSE 0 END AS is_zero_balance,

        ROUND(
            CAST(NumOfProducts AS REAL) / MAX(Tenure, 1),
            4
        ) AS product_tenure_ratio,

        CASE
            WHEN Complain = 1 AND Satisfaction_Score <= 2 THEN 1
            ELSE 0
        END AS is_high_complain_risk

    FROM customers_raw
    """

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(query, conn)
        logger.info("Queried %d feature rows from SQLite", len(df))
    finally:
        conn.close()
    return df


# ── 4. Orchestration ────────────────────────────────────────────
def run_ingestion_pipeline(
    csv_path: Path | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Run the full ingestion pipeline: load → SQLite → SQL features.

    This is the **single entry point** consumed by both
    ``src.models.train`` and ``dashboard.app``.

    Parameters
    ----------
    csv_path : Path, optional
        Path to the raw CSV.
    db_path : Path, optional
        Path to the SQLite DB.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame ready for preprocessing.
    """
    logger.info("▶ Starting ingestion pipeline")
    df_raw = load_csv(csv_path)
    load_to_sqlite(df_raw, db_path)
    df_features = query_features_from_sql(db_path)
    logger.info(
        "✓ Ingestion complete — %d rows, %d columns",
        len(df_features),
        len(df_features.columns),
    )
    return df_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = run_ingestion_pipeline()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Churn rate: {df['Exited'].mean():.2%}")
