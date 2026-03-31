-- ============================================================
-- BankGuard AI — Database Schema
-- ============================================================
-- Creates the core tables for bank customer data, predictions,
-- and A/B experiment tracking.
-- ============================================================

-- Raw customer data (mirrors Kaggle CSV after cleaning)
CREATE TABLE IF NOT EXISTS customers_raw (
    CustomerId          INTEGER PRIMARY KEY,
    CreditScore         INTEGER NOT NULL CHECK (CreditScore >= 300 AND CreditScore <= 900),
    Geography           TEXT    NOT NULL CHECK (Geography IN ('France', 'Germany', 'Spain')),
    Gender              TEXT    NOT NULL CHECK (Gender IN ('Male', 'Female')),
    Age                 INTEGER NOT NULL CHECK (Age >= 18 AND Age <= 100),
    Tenure              INTEGER NOT NULL CHECK (Tenure >= 0 AND Tenure <= 10),
    Balance             REAL    NOT NULL CHECK (Balance >= 0),
    NumOfProducts       INTEGER NOT NULL CHECK (NumOfProducts >= 1 AND NumOfProducts <= 4),
    HasCrCard           INTEGER NOT NULL CHECK (HasCrCard IN (0, 1)),
    IsActiveMember      INTEGER NOT NULL CHECK (IsActiveMember IN (0, 1)),
    EstimatedSalary     REAL    NOT NULL CHECK (EstimatedSalary >= 0),
    Exited              INTEGER NOT NULL CHECK (Exited IN (0, 1)),
    Complain            INTEGER NOT NULL CHECK (Complain IN (0, 1)),
    Satisfaction_Score  INTEGER NOT NULL CHECK (Satisfaction_Score >= 1 AND Satisfaction_Score <= 5),
    Card_Type           TEXT    NOT NULL CHECK (Card_Type IN ('DIAMOND', 'GOLD', 'SILVER', 'PLATINUM')),
    Point_Earned        INTEGER NOT NULL CHECK (Point_Earned >= 0)
);

-- Churn prediction log
CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         INTEGER NOT NULL,
    churn_probability   REAL    NOT NULL,
    risk_segment        TEXT    NOT NULL CHECK (risk_segment IN ('High', 'Medium', 'Low')),
    model_version       TEXT    NOT NULL,
    predicted_at        TEXT    NOT NULL
);

-- A/B experiment enrollment and outcomes
CREATE TABLE IF NOT EXISTS ab_experiments (
    experiment_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name     TEXT    NOT NULL,
    variant             TEXT    NOT NULL CHECK (variant IN ('control', 'treatment')),
    customer_id         INTEGER NOT NULL,
    enrolled_at         TEXT    NOT NULL,
    churned             INTEGER          CHECK (churned IS NULL OR churned IN (0, 1))
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_predictions_risk
    ON churn_predictions (risk_segment);

CREATE INDEX IF NOT EXISTS idx_experiments_name_variant
    ON ab_experiments (experiment_name, variant);
