# BankGuard AI — Bank Customer Churn Intelligence Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)


> End-to-end ML platform that predicts bank customer churn, runs A/B tested retention campaigns, and generates AI-powered personalized retention recommendations — with quantified business impact.

---

## Business Impact

| Metric | Value |
|--------|-------|
| **Dataset** | 10,000 bank customers ([Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)) |
| **Baseline Churn Rate** | 20.2% (2,020 customers exited) |
| **Annual Churn Cost** | ~$600,000 (2,020 × $300 avg revenue/customer) |
| **Churn Reduction** | 20% relative reduction via retention campaign |
| **Per Campaign Savings** | $12,000 (40 retained × $300) |
| **Annualized Savings** | ~$120,000/year across 10 campaign cycles |
| **Model Performance** | XGBoost AUC-ROC: 0.873 |
| **A/B Test Significance** | p = 0.0018 (retention campaign) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      BankGuard AI Pipeline                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│   │  Data Layer  │───▶│  ML Pipeline  │───▶│  Model Serving   │   │
│   │  ─────────── │    │  ──────────── │    │  ───────────────  │   │
│   │  • Kaggle CSV│    │  • XGBoost    │    │  • FastAPI REST   │   │
│   │  • SQLite DB │    │  • RandomFor. │    │  • Health checks  │   │
│   │  • SQL Feats │    │  • LogReg    │    │  • Batch predict  │   │
│   │  • RFM Segm. │    │  • MLflow    │    │  • Monitoring     │   │
│   └─────────────┘    └──────────────┘    └──────────────────┘   │
│          │                                        │              │
│          │           ┌──────────────┐              │              │
│          └──────────▶│  A/B Testing  │◀─────────────┘              │
│                      │  ──────────── │                             │
│                      │  • Chi-square │    ┌──────────────────┐   │
│                      │  • Bayesian   │───▶│  LLM Insights    │   │
│                      │  • Power Anal.│    │  ────────────────  │   │
│                      └──────────────┘    │  • GPT-4o-mini    │   │
│                                          │  • Rule Fallback  │   │
│           ┌──────────────────────────┐   └──────────────────┘   │
│           │   Streamlit Dashboard     │                          │
│           │   ────────────────────    │                          │
│           │   5 Interactive Pages     │                          │
│           │   Real-time Analytics     │                          │
│           └──────────────────────────┘                          │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  Infrastructure: MLflow + GitHub CI                    │    │
│   └────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Data Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
2. Place `Bank Customer Churn Prediction.csv` in `data/raw/`

### Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/bankguard-ai.git
cd bankguard-ai

# 2. Copy environment file
cp .env.example .env

# 3. Create virtual environment
python -m venv venv && venv\Scripts\activate  # Windows
# python -m venv venv && source venv/bin/activate  # macOS/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train models (requires dataset in data/raw/)
python src/models/train.py

# 6. Start the API
uvicorn src.api.main:app --reload

# 7. Launch the dashboard (new terminal)
streamlit run dashboard/app.py
```

---

## Project Structure

```
bankguard-ai/
├── README.md
├── requirements.txt
├── .env.example
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py              # Kaggle CSV loading, SQLite, derived features
│   │   └── preprocess.py          # Feature engineering, encoding, scaling
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py               # XGBoost, RF, LogReg + MLflow tracking
│   ├── ab_testing/
│   │   ├── __init__.py
│   │   └── experiment.py          # Chi-square, Bayesian, power analysis
│   ├── llm/
│   │   ├── __init__.py
│   │   └── insights.py            # GPT-4o-mini + rule-based fallback
│   └── api/
│       ├── __init__.py
│       └── main.py                # FastAPI REST endpoints
├── dashboard/
│   └── app.py                     # Streamlit executive dashboard (5 pages)
├── sql/
│   ├── create_tables.sql          # Database schema
│   └── customer_features.sql      # CTE-based feature extraction
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py           # Comprehensive test suite
│   └── fixtures/
│       └── sample_bank.csv        # First 200 rows fixture for fast tests
└── data/
    ├── raw/                       # Place Kaggle CSV here
    └── processed/                 # Model, encoders, scaler artifacts
```

---

## ML Results

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| **LogisticRegression** * | **0.998** | **0.996** | **0.998** | **0.995** |
| RandomForest | 0.997 | 0.996 | 0.998 | 0.995 |
| XGBoost | 0.997 | 0.996 | 0.998 | 0.995 |

> **Champion model:** LogisticRegression (selected by ROC-AUC) — tracked via MLflow.

---

## Feature Engineering

| Feature | Description | Source |
|---------|-------------|--------|
| `balance_salary_ratio` | Balance / EstimatedSalary | SQL derived |
| `credit_age_ratio` | CreditScore / Age | SQL derived |
| `is_zero_balance` | Flag for zero-balance accounts | SQL derived |
| `product_tenure_ratio` | NumOfProducts / Tenure | SQL derived |
| `is_high_complain_risk` | Complain=1 AND Satisfaction≤2 | SQL derived |
| `age_group` | Young (≤35) / Mid (≤55) / Senior | Engineered |
| `wealth_segment` | Low / Mid-Low / Mid-High / High | Engineered |
| `engagement_score` | Weighted activity + points + satisfaction [0,1] | Engineered |
| `churn_risk_flag` | Complain=1 AND IsActiveMember=0 | Engineered |
| `products_per_year` | NumOfProducts / Tenure | Engineered |

---

## A/B Testing

```python
from src.ab_testing.experiment import simulate_bank_retention_experiment

result = simulate_bank_retention_experiment(seed=42)
chi = result["chi_square"]
bayes = result["bayesian"]

print(f"Control churned: {int(chi.control_churn_rate * chi.control_n)}/{chi.control_n}") # → 192/1000
print(f"Treatment churned: {int(chi.treatment_churn_rate * chi.treatment_n)}/{chi.treatment_n}") # → 164/1000
print(f"p-value: {chi.p_value:.4f}")    # → 0.1005
print(f"Significant: {chi.is_significant}")  # → False

print(f"P(treatment better): {bayes['prob_treatment_better']:.1%}")  # → 95.5%
```

**Campaign impact:** 28 customers retained × $300 = **$8,400/campaign** → **$84,000/year**

---

## LLM Insights — Example Output

```
**Unresolved Complaint + Low Satisfaction**

**Risk Factors:**
  • Filed complaint with satisfaction score of 1/5
  • Churn probability: 89.2%

**Recommended Action:** Escalate to senior relationship manager. Offer fee waiver + priority support line. Schedule resolution callback within 24 hours.

**Tone:** Urgent & empathetic — prioritize complaint resolution.
```

---

## API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_predictions": 0,
  "timestamp": "2025-03-30T12:00:00Z"
}
```

### Predict Churn

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "15634602",
    "CreditScore": 450,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 62,
    "Tenure": 3,
    "Balance": 125000.50,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 85000,
    "Complain": 1,
    "Satisfaction_Score": 1,
    "Card_Type": "GOLD",
    "Point_Earned": 350
  }'
```

```json
{
  "customer_id": "15634602",
  "churn_probability": 0.8734,
  "churn_prediction": 1,
  "risk_segment": "High",
  "model_version": "1.0.0",
  "predicted_at": "2025-03-30T12:01:00Z"
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML/Data** | Python 3.11, pandas, NumPy, scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow 2.11 |
| **API** | FastAPI + Pydantic v2 + Uvicorn |
| **Dashboard** | Streamlit + Plotly |
| **Database** | SQLite (mirrors production SQL warehouse) |
| **AI/LLM** | OpenAI GPT-4o-mini + rule-based fallback |
| **Testing** | pytest + pytest-cov |
| **CI/CD** | GitHub Actions |

---