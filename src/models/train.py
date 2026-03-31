"""
Model training and evaluation module for BankGuard AI.

Trains three classifiers (XGBoost, RandomForest, Logistic Regression),
logs experiments to MLflow, selects the best model by ROC-AUC, and
saves the champion model and feature column list to ``data/processed/``.
"""

import logging
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data.ingest import run_ingestion_pipeline
from src.data.preprocess import TARGET, encode_and_scale, engineer_features, split_data

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "data" / "processed" / "feature_cols.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Model definitions ───────────────────────────────────────────
MODELS: dict[str, object] = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=4,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "LogisticRegression": LogisticRegression(
        C=0.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
}


def _evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like
        Predicted probabilities for positive class.

    Returns
    -------
    dict[str, float]
        Dictionary of metrics.
    """
    return {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_proba), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


def train_and_evaluate() -> dict[str, dict]:
    """Train all models, log to MLflow, select best by ROC-AUC.

    Returns
    -------
    dict[str, dict]
        ``{model_name: {params: ..., metrics: ...}}``
    """
    logger.info("═" * 60)
    logger.info("  BankGuard AI — Model Training Pipeline")
    logger.info("═" * 60)

    # ── Data ─────────────────────────────────────────────────────
    df = run_ingestion_pipeline()
    df = engineer_features(df)
    df = encode_and_scale(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)

    # Save feature columns for inference
    joblib.dump(list(X_train.columns), FEATURE_COLS_PATH)
    logger.info("Saved %d feature columns → %s", len(X_train.columns), FEATURE_COLS_PATH)

    # ── MLflow experiment ────────────────────────────────────────
    mlflow.set_experiment("BankGuard-AI")

    results: dict[str, dict] = {}
    best_auc = -1.0
    best_model = None
    best_name = ""

    for name, model in MODELS.items():
        logger.info("─ Training %s ─", name)
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(X_test)
            )

            metrics = _evaluate(y_test, y_pred, y_proba)
            params = model.get_params()

            # Log to MLflow
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=name)

            results[name] = {"params": params, "metrics": metrics}

            logger.info(
                "  %s — AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
                name,
                metrics["roc_auc"],
                metrics["f1"],
                metrics["precision"],
                metrics["recall"],
            )

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_name = name

    # ── Save champion ────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    logger.info("═" * 60)
    logger.info(
        "  ★ Champion: %s (AUC %.4f) → saved to %s",
        best_name,
        best_auc,
        MODEL_PATH,
    )
    logger.info("═" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    results = train_and_evaluate()

    print("\n" + "="*60)
    print("             MODEL COMPARISON RESULTS                 ")
    print("+" + "-"*58 + "+")
    print(f"| {'Model':<22} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} |")
    print("+" + "-"*58 + "+")
    for name, data in results.items():
        m = data["metrics"]
        print(
            f"| {name:<22} {m['roc_auc']:>7.4f} {m['f1']:>7.4f} "
            f"{m['precision']:>7.4f} {m['recall']:>7.4f} |"
        )
    print("+" + "-"*58 + "+")
