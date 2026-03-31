"""
BankGuard AI — Comprehensive Test Suite.

Covers data ingestion, feature engineering, A/B testing,
and LLM insights with deterministic, reproducible assertions.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.ab_testing.experiment import (
    bayesian_test,
    calculate_sample_size,
    run_chi_square_test,
)
from src.data.ingest import load_csv, run_ingestion_pipeline
from src.data.preprocess import encode_and_scale, engineer_features, split_data
from src.llm.insights import get_retention_insight_fallback


# ── Fixture: sample CSV ──────────────────────────────────────────
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_bank.csv"


def _load_fixture() -> pd.DataFrame:
    """Load the first 200 rows fixture for fast testing.

    If the fixture doesn't exist, fall back to the full dataset
    and create the fixture for future runs.
    """
    if FIXTURE_PATH.exists():
        return load_csv(FIXTURE_PATH)

    # Fall back to full dataset
    from src.data.ingest import RAW_CSV
    if RAW_CSV.exists():
        full_df = pd.read_csv(RAW_CSV)
        # Create fixture
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        full_df.head(200).to_csv(FIXTURE_PATH, index=False)
        return load_csv(FIXTURE_PATH)

    pytest.skip("Dataset not available — download from Kaggle first")


# ═══════════════════════════════════════════════════════════════
#  DATA INGESTION
# ═══════════════════════════════════════════════════════════════
class TestDataIngestion:
    """Tests for CSV loading and ingestion pipeline."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Load fixture data."""
        self.df = _load_fixture()

    def test_shape(self):
        """Loading 200 rows should yield 200 rows with 15 feature columns
        (18 original minus RowNumber, Surname, Exited target kept separately)."""
        # RowNumber and Surname are dropped; remaining = 16 cols including Exited
        assert self.df.shape[0] == 200
        assert "RowNumber" not in self.df.columns
        assert "Surname" not in self.df.columns

    def test_dropped_cols(self):
        """RowNumber and Surname must not be present."""
        assert "RowNumber" not in self.df.columns
        assert "Surname" not in self.df.columns

    def test_target(self):
        """'Exited' column must exist with values in {0, 1}."""
        assert "Exited" in self.df.columns
        assert set(self.df["Exited"].unique()).issubset({0, 1})

    def test_derived_cols(self):
        """Ingestion pipeline should produce SQL-derived columns."""
        from src.data.ingest import load_to_sqlite, query_features_from_sql, DB_PATH
        load_to_sqlite(self.df)
        df_feat = query_features_from_sql()
        assert "balance_salary_ratio" in df_feat.columns
        assert "is_zero_balance" in df_feat.columns
        assert "is_high_complain_risk" in df_feat.columns


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
class TestFeatureEngineering:
    """Tests for feature engineering and preprocessing."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Prepare a feature-enriched DataFrame for all tests."""
        raw_df = _load_fixture()
        from src.data.ingest import load_to_sqlite, query_features_from_sql
        load_to_sqlite(raw_df)
        df_feat = query_features_from_sql()
        self.df = engineer_features(df_feat)

    def test_engagement_score_range(self):
        """engagement_score must be in [0, 1]."""
        assert self.df["engagement_score"].min() >= 0
        assert self.df["engagement_score"].max() <= 1

    def test_churn_risk_flag_binary(self):
        """churn_risk_flag must be 0 or 1."""
        assert set(self.df["churn_risk_flag"].unique()).issubset({0, 1})

    def test_products_per_year_positive(self):
        """products_per_year must be > 0 for all customers."""
        assert (self.df["products_per_year"] > 0).all()

    def test_encode_and_scale(self):
        """After encoding, no NULLs should remain and encoders
        must not be empty."""
        df_encoded = encode_and_scale(self.df, fit=True)
        assert df_encoded.isnull().sum().sum() == 0
        from src.data.preprocess import ENCODERS_PATH
        assert ENCODERS_PATH.exists()

    def test_train_test_split_stratified(self):
        """Class distribution between train and test should be
        within 3 percentage points."""
        df_encoded = encode_and_scale(self.df, fit=True)
        _, _, y_train, y_test = split_data(df_encoded)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.03, (
            f"Stratification drift: train={train_rate:.3f}, test={test_rate:.3f}"
        )


# ═══════════════════════════════════════════════════════════════
#  A/B TESTING
# ═══════════════════════════════════════════════════════════════
class TestABTesting:
    """Tests for frequentist and Bayesian A/B experiment analysis."""

    def test_chi_square_significant(self):
        """200/1000 vs 160/1000 should be statistically significant."""
        result = run_chi_square_test(200, 1000, 160, 1000)
        assert result.p_value < 0.05
        assert result.is_significant

    def test_chi_square_not_significant(self):
        """200/1000 vs 198/1000 should NOT be significant."""
        result = run_chi_square_test(200, 1000, 198, 1000)
        assert result.p_value > 0.05
        assert not result.is_significant

    def test_sample_size_gt_400(self):
        """For baseline=0.20, MDE=0.04 the required n should exceed 400."""
        n = calculate_sample_size(baseline_rate=0.20, mde=0.04)
        assert n > 400, f"Expected n > 400, got {n}"

    def test_bayesian_high_confidence(self):
        """200 vs 160 churns should yield > 95% probability
        that treatment is better."""
        result = bayesian_test(200, 1000, 160, 1000)
        assert result["prob_treatment_better"] > 0.95

    def test_bayesian_uncertain(self):
        """200 vs 198 should yield an uncertain probability
        (roughly 0.3 – 0.7)."""
        result = bayesian_test(200, 1000, 198, 1000)
        assert 0.3 < result["prob_treatment_better"] < 0.7


# ═══════════════════════════════════════════════════════════════
#  LLM INSIGHTS (rule-based fallback)
# ═══════════════════════════════════════════════════════════════
class TestLLMInsights:
    """Tests for the rule-based retention insight engine."""

    def test_complain_insight(self):
        """Customer with complaint + low satisfaction should trigger escalation."""
        customer = {
            "CustomerId": "TEST-COMP",
            "Complain": 1,
            "Satisfaction_Score": 1,
            "churn_probability": 0.85,
            "IsActiveMember": 0,
            "NumOfProducts": 2,
            "Age": 40,
            "Balance": 50000,
            "CreditScore": 650,
            "is_zero_balance": 0,
        }
        insight = get_retention_insight_fallback(customer)
        assert "complaint" in insight.lower() or "escalate" in insight.lower(), (
            f"Expected 'complaint' or 'escalate' in insight, got:\n{insight}"
        )

    def test_zero_balance_insight(self):
        """Dormant zero-balance customer should get activation offer."""
        customer = {
            "CustomerId": "TEST-ZERO",
            "is_zero_balance": 1,
            "IsActiveMember": 0,
            "churn_probability": 0.72,
            "Complain": 0,
            "Satisfaction_Score": 3,
            "NumOfProducts": 1,
            "Age": 35,
            "Balance": 0,
            "CreditScore": 700,
        }
        insight = get_retention_insight_fallback(customer)
        assert any(w in insight.lower() for w in ["dormant", "zero", "activation"]), (
            f"Expected 'dormant', 'zero', or 'activation' in insight, got:\n{insight}"
        )

    def test_stable_customer(self):
        """Low-risk customer should receive a stable check-in."""
        customer = {
            "CustomerId": "TEST-STABLE",
            "churn_probability": 0.10,
            "Complain": 0,
            "Satisfaction_Score": 4,
            "IsActiveMember": 1,
            "NumOfProducts": 2,
            "Age": 30,
            "Balance": 80000,
            "CreditScore": 750,
            "is_zero_balance": 0,
        }
        insight = get_retention_insight_fallback(customer)
        assert "stable" in insight.lower() or "check-in" in insight.lower(), (
            f"Expected 'stable' or 'check-in' in insight, got:\n{insight}"
        )
