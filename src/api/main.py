"""
FastAPI model-serving REST API for BankGuard AI.

Exposes endpoints for health checks, single/batch churn
predictions, and monitoring statistics.  Feature engineering
at inference exactly mirrors the training pipeline.
"""

import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.data.preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model.pkl"
ENCODERS_PATH = PROJECT_ROOT / "data" / "processed" / "encoders.pkl"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "data" / "processed" / "feature_cols.pkl"

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="BankGuard AI",
    description="Bank Customer Churn Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────────
_state: dict[str, Any] = {
    "model": None,
    "encoders": None,
    "scaler": None,
    "feature_cols": None,
    "model_loaded": False,
    "total_predictions": 0,
    "churn_probs": deque(maxlen=10_000),
    "latencies_ms": deque(maxlen=10_000),
}

VALID_GEOGRAPHIES = {"France", "Germany", "Spain"}
VALID_GENDERS = {"Male", "Female"}
VALID_CARD_TYPES = {"DIAMOND", "GOLD", "SILVER", "PLATINUM"}


# ── Pydantic v2 request model ───────────────────────────────────
class CustomerFeatures(BaseModel):
    """Input schema for a single bank customer prediction.

    All fields match the training feature set for the Bank Customer
    Churn Prediction dataset.
    """

    customer_id: str
    CreditScore: int = Field(..., ge=300, le=900)
    Geography: str
    Gender: str
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)
    Complain: int = Field(..., ge=0, le=1)
    Satisfaction_Score: int = Field(..., ge=1, le=5)
    Card_Type: str
    Point_Earned: int = Field(..., ge=0)

    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, v: str) -> str:
        """Ensure Geography is one of the allowed values."""
        if v not in VALID_GEOGRAPHIES:
            raise ValueError(
                f"Geography must be one of {VALID_GEOGRAPHIES}, got '{v}'"
            )
        return v

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Ensure Gender is one of the allowed values."""
        if v not in VALID_GENDERS:
            raise ValueError(f"Gender must be one of {VALID_GENDERS}, got '{v}'")
        return v

    @field_validator("Card_Type")
    @classmethod
    def validate_card_type(cls, v: str) -> str:
        """Ensure Card_Type is one of the allowed values."""
        if v not in VALID_CARD_TYPES:
            raise ValueError(
                f"Card_Type must be one of {VALID_CARD_TYPES}, got '{v}'"
            )
        return v


class PredictionResponse(BaseModel):
    """Output schema for a churn prediction."""

    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_segment: str
    model_version: str
    predicted_at: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    total_predictions: int
    timestamp: str


class MonitoringStats(BaseModel):
    """Monitoring statistics response."""

    total_predictions: int
    mean_churn_prob: float
    high_risk_pct: float
    avg_latency_ms: float


# ── Startup ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Load model and preprocessing artifacts on startup."""
    try:
        _state["model"] = joblib.load(MODEL_PATH)
        _state["encoders"] = joblib.load(ENCODERS_PATH)
        _state["scaler"] = joblib.load(SCALER_PATH)
        if FEATURE_COLS_PATH.exists():
            _state["feature_cols"] = joblib.load(FEATURE_COLS_PATH)
        _state["model_loaded"] = True
        logger.info("Model and artifacts loaded successfully")
    except FileNotFoundError as e:
        logger.warning("Could not load model artifacts: %s", e)
        _state["model_loaded"] = False


# ── Prediction helper ────────────────────────────────────────────
def _prepare_features(customer: CustomerFeatures) -> pd.DataFrame:
    """Mirror the training feature engineering exactly.

    Parameters
    ----------
    customer : CustomerFeatures
        Input customer features.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame ready for model prediction.
    """
    data = customer.model_dump()
    df = pd.DataFrame([data])

    # SQL-derived features (must match ingest.query_features_from_sql)
    df["balance_salary_ratio"] = (
        df["Balance"] / df["EstimatedSalary"].clip(lower=1)
    ).round(4)
    df["credit_age_ratio"] = (
        df["CreditScore"] / df["Age"].clip(lower=1)
    ).round(4)
    df["is_zero_balance"] = (df["Balance"] == 0).astype(int)
    df["product_tenure_ratio"] = (
        df["NumOfProducts"] / df["Tenure"].clip(lower=1)
    ).round(4)
    df["is_high_complain_risk"] = (
        (df["Complain"] == 1) & (df["Satisfaction_Score"] <= 2)
    ).astype(int)

    # Engineered features (must match preprocess.engineer_features)
    point_norm = (df["Point_Earned"] / 10_000).clip(0, 1)
    sat_norm = df["Satisfaction_Score"] / 5.0
    df["engagement_score"] = (
        df["IsActiveMember"] * 0.4 + point_norm * 0.3 + sat_norm * 0.3
    ).clip(0, 1).round(4)

    df["churn_risk_flag"] = (
        (df["Complain"] == 1) & (df["IsActiveMember"] == 0)
    ).astype(int)

    df["products_per_year"] = (
        df["NumOfProducts"] / df["Tenure"].clip(lower=1)
    ).round(4)

    # Encode categoricals
    encoders = _state["encoders"]
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(
            lambda x, _k=known, _le=le: (
                _le.transform([x])[0] if x in _k else -1
            )
        )

    # Scale numerics
    scaler = _state["scaler"]
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Select only feature columns used during training
    if _state["feature_cols"]:
        df = df[[c for c in _state["feature_cols"] if c in df.columns]]
    else:
        drop_cols = ["customer_id", "CustomerId", "age_group", "wealth_segment"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def _make_prediction(customer: CustomerFeatures) -> PredictionResponse:
    """Run prediction for a single customer.

    Parameters
    ----------
    customer : CustomerFeatures
        Input customer features.

    Returns
    -------
    PredictionResponse
    """
    start = time.perf_counter()

    X = _prepare_features(customer)
    model = _state["model"]
    proba = float(model.predict_proba(X)[:, 1][0])
    prediction = int(proba >= 0.5)

    if proba >= 0.60:
        risk_segment = "High"
    elif proba >= 0.30:
        risk_segment = "Medium"
    else:
        risk_segment = "Low"

    elapsed_ms = (time.perf_counter() - start) * 1000
    _state["total_predictions"] += 1
    _state["churn_probs"].append(proba)
    _state["latencies_ms"].append(elapsed_ms)

    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_segment=risk_segment,
        model_version="1.0.0",
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )


def _log_prediction(result: PredictionResponse) -> None:
    """Background task to log a prediction result.

    Parameters
    ----------
    result : PredictionResponse
    """
    logger.info(
        "Prediction logged: %s → %.4f (%s)",
        result.customer_id,
        result.churn_probability,
        result.risk_segment,
    )


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns
    -------
    HealthResponse
        Status (healthy/degraded), model_loaded flag,
        total predictions, and current timestamp.
    """
    return HealthResponse(
        status="healthy" if _state["model_loaded"] else "degraded",
        model_loaded=_state["model_loaded"],
        total_predictions=_state["total_predictions"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures, bg: BackgroundTasks):
    """Single customer churn prediction.

    Parameters
    ----------
    customer : CustomerFeatures
        Customer feature payload.

    Returns
    -------
    PredictionResponse
    """
    if not _state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = _make_prediction(customer)
    bg.add_task(_log_prediction, result)
    return result


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(customers: list[CustomerFeatures], bg: BackgroundTasks):
    """Batch churn predictions (max 1 000).

    Parameters
    ----------
    customers : list[CustomerFeatures]
        List of customer feature payloads.

    Returns
    -------
    list[PredictionResponse]
    """
    if not _state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Max batch size is 1000")

    results = []
    for cust in customers:
        result = _make_prediction(cust)
        bg.add_task(_log_prediction, result)
        results.append(result)
    return results


@app.get("/monitoring/stats", response_model=MonitoringStats)
async def monitoring_stats():
    """Monitoring statistics for operational awareness.

    Returns
    -------
    MonitoringStats
        Total predictions, mean churn probability,
        high-risk percentage (≥ 0.60), and average latency.
    """
    probs = list(_state["churn_probs"])
    lats = list(_state["latencies_ms"])

    return MonitoringStats(
        total_predictions=_state["total_predictions"],
        mean_churn_prob=round(float(np.mean(probs)) if probs else 0.0, 4),
        high_risk_pct=round(
            float(np.mean([p >= 0.60 for p in probs])) * 100 if probs else 0.0, 2
        ),
        avg_latency_ms=round(float(np.mean(lats)) if lats else 0.0, 2),
    )
