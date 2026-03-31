"""
Feature engineering, encoding, scaling, and train/test splitting
for BankGuard AI.

All transformations mirror the training pipeline so that the API
inference path applies identical logic.  Artifacts (encoders,
scaler) are persisted with ``joblib`` under ``data/processed/``.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ENCODERS_PATH = PROCESSED_DIR / "encoders.pkl"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Exited"

# Columns that require label-encoding
CATEGORICAL_COLS = ["Geography", "Gender", "Card_Type"]

# Numeric columns to scale
NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Point_Earned", "balance_salary_ratio",
    "credit_age_ratio", "product_tenure_ratio",
]

# Binary / ordinal columns — passed through unchanged
PASSTHROUGH_COLS = [
    "HasCrCard", "IsActiveMember", "Complain",
    "Satisfaction_Score", "is_zero_balance", "is_high_complain_risk",
]


# ── Feature engineering ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived business features to the DataFrame.

    New columns
    -----------
    age_group : str
        'Young' (18-35), 'Mid' (36-55), 'Senior' (56+).
    wealth_segment : str
        Balance quartiles → Low / Mid-Low / Mid-High / High.
    engagement_score : float [0, 1]
        Weighted composite of IsActiveMember, Point_Earned,
        and Satisfaction_Score.
    churn_risk_flag : int (0 | 1)
        1 if Complain==1 AND IsActiveMember==0.
    products_per_year : float
        NumOfProducts / max(Tenure, 1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns from the ingestion pipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with the five new columns.
    """
    df = df.copy()

    # age_group
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 35, 55, 200],
        labels=["Young", "Mid", "Senior"],
    ).astype(str)

    # wealth_segment (based on Balance quartiles)
    df["wealth_segment"] = pd.cut(
        df["Balance"],
        bins=[-1, 50_000, 100_000, 150_000, float("inf")],
        labels=["Low", "Mid-Low", "Mid-High", "High"],
    ).astype(str)

    # engagement_score (0-1)
    point_norm = (df["Point_Earned"] / 10_000).clip(0, 1)
    sat_norm = df["Satisfaction_Score"] / 5.0
    df["engagement_score"] = (
        df["IsActiveMember"] * 0.4
        + point_norm * 0.3
        + sat_norm * 0.3
    ).clip(0, 1).round(4)

    # churn_risk_flag
    df["churn_risk_flag"] = (
        (df["Complain"] == 1) & (df["IsActiveMember"] == 0)
    ).astype(int)

    # products_per_year
    df["products_per_year"] = (
        df["NumOfProducts"] / df["Tenure"].clip(lower=1)
    ).round(4)

    logger.info("Engineered 5 new features (age_group, wealth_segment, "
                "engagement_score, churn_risk_flag, products_per_year)")
    return df


# ── Encoding & scaling ──────────────────────────────────────────
def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
) -> pd.DataFrame:
    """Label-encode categoricals and standard-scale numerics.

    When ``fit=True`` (training), encoders and scaler are fitted and
    saved to ``data/processed/``.  When ``fit=False`` (inference),
    previously saved artifacts are loaded.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both categorical and numeric columns.
    fit : bool
        Whether to fit new encoders/scaler or load existing ones.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with encoded categoricals and scaled
        numerics.  Passthrough columns are untouched.
    """
    df = df.copy()

    if fit:
        encoders: dict[str, LabelEncoder] = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, ENCODERS_PATH)
        logger.info("Fitted & saved %d label encoders → %s", len(encoders), ENCODERS_PATH)

        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Fitted & saved StandardScaler → %s", SCALER_PATH)
    else:
        encoders = joblib.load(ENCODERS_PATH)
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x, _k=known, _le=le: (
                    _le.transform([x])[0] if x in _k else -1
                )
            )
        scaler = joblib.load(SCALER_PATH)
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
        logger.info("Applied saved encoders & scaler from disk")

    return df


# ── Train / test split ──────────────────────────────────────────
def split_data(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed DataFrame (encoded + scaled).
    target : str
        Target column name.
    test_size : float
        Proportion held out for testing.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    drop_cols = [target, "CustomerId"]
    # Also drop engineered string columns that aren't for model input
    extra_drop = ["age_group", "wealth_segment"]
    all_drop = [c for c in drop_cols + extra_drop if c in df.columns]

    feature_cols = [c for c in df.columns if c not in all_drop]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    logger.info(
        "Split data — train=%d, test=%d, churn-rate train=%.2f%%, test=%.2f%%",
        len(X_train),
        len(X_test),
        100 * y_train.mean(),
        100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# ── Artifact loading helper ─────────────────────────────────────
def load_artifacts() -> dict[str, Any]:
    """Load saved encoders and scaler for inference.

    Returns
    -------
    dict
        ``{"encoders": dict[str, LabelEncoder], "scaler": StandardScaler}``
    """
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Loaded preprocessing artifacts from %s", PROCESSED_DIR)
    return {"encoders": encoders, "scaler": scaler}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from src.data.ingest import run_ingestion_pipeline

    df = run_ingestion_pipeline()
    df = engineer_features(df)
    df = encode_and_scale(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Features: {list(X_train.columns)}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
