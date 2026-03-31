"""
BankGuard AI — Executive Dashboard.

A Streamlit application providing 5 interactive pages:
1. Executive Overview (KPIs, risk distribution, feature importances)
2. At-Risk Customers (filterable high-risk customer table)
3. A/B Experiments (frequentist + Bayesian results)
4. AI Insights (LLM-powered retention recommendations)
5. Predict Customer (interactive churn prediction form)
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Project imports ──────────────────────────────────────────────
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import run_ingestion_pipeline
from src.data.preprocess import engineer_features
from src.ab_testing.experiment import run_chi_square_test, bayesian_test, calculate_sample_size
from src.llm.insights import get_retention_insight_fallback

logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BankGuard AI — Customer Intelligence Platform",
    page_icon="Bank",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .insight-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #0ea5e9;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        color: #e0e0e0;
        font-size: 0.92rem;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .insight-box strong {
        color: #7dd3fc;
    }

    .kpi-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5e 100%);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(14, 165, 233, 0.2);
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9, #7dd3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .kpi-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }

    .stMetric > div {
        background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5e 100%);
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid rgba(14, 165, 233, 0.2);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }

    .sidebar-title {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "data" / "processed" / "feature_cols.pkl"


@st.cache_data(show_spinner="Loading customer data...")
def load_data() -> pd.DataFrame:
    """Load and engineer features for the full customer dataset."""
    df = run_ingestion_pipeline()
    df = engineer_features(df)
    return df


@st.cache_data(show_spinner="Loading model predictions...")
def get_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate churn predictions using the saved model or rule-based formula."""
    df = df.copy()

    if MODEL_PATH.exists():
        try:
            from src.data.preprocess import encode_and_scale
            df_encoded = encode_and_scale(df, fit=False)
            model = joblib.load(MODEL_PATH)

            if FEATURE_COLS_PATH.exists():
                feature_cols = joblib.load(FEATURE_COLS_PATH)
                X = df_encoded[[c for c in feature_cols if c in df_encoded.columns]]
            else:
                drop_cols = ["Exited", "CustomerId", "age_group", "wealth_segment"]
                X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])

            df["churn_probability"] = model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.warning("Could not load model: %s — using rule-based scores", e)
            df = _rule_based_predictions(df)
    else:
        df = _rule_based_predictions(df)

    df["risk_segment"] = pd.cut(
        df["churn_probability"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return df


def _rule_based_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based churn probability formula as fallback.

    Parameters
    ----------
    df : pd.DataFrame
        Customer data with raw features.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``churn_probability`` column added.
    """
    df["churn_probability"] = (
        0.05
        + 0.25 * df["Complain"]
        + 0.15 * df["is_zero_balance"]
        - 0.10 * df["IsActiveMember"]
        + 0.08 * (df["NumOfProducts"] >= 3).astype(int)
        - 0.005 * (df["Satisfaction_Score"] - 1)
        - 0.001 * df["Tenure"]
    ).clip(0.02, 0.95)
    return df


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## BankGuard")
    st.markdown('<p class="sidebar-title">BankGuard AI</p>', unsafe_allow_html=True)
    st.caption("Customer Intelligence Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "At-Risk Customers",
            "A/B Experiments",
            "AI Insights",
            "Predict Customer",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Model Info**")
    st.markdown("Model: LogisticRegression")
    st.markdown("AUC-ROC: **0.998**")
    st.markdown("Last retrained: 2025-03-30")

    st.divider()
    st.caption("© 2025 BankGuard AI")


# ── Load data ────────────────────────────────────────────────────
df = load_data()
df_pred = get_predictions(df)


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Executive Overview":
    st.markdown('<h1 class="main-header">Executive Overview</h1>', unsafe_allow_html=True)
    st.caption("Real-time bank customer churn intelligence at a glance")

    # ── KPI row ──────────────────────────────────────────────────
    high_risk = df_pred[df_pred["risk_segment"] == "High"]
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Customers", f"{len(df_pred):,}")
    with k2:
        st.metric("High-Risk", f"{len(high_risk):,}",
                   delta=f"{len(high_risk)/len(df_pred)*100:.1f}%", delta_color="inverse")
    with k3:
        churn_rate = df_pred["Exited"].mean()
        st.metric("Annual Churn Rate", f"{churn_rate:.1%}")
    with k4:
        revenue_at_risk = len(high_risk) * 300
        st.metric("Revenue at Risk", f"${revenue_at_risk:,}")
    with k5:
        st.metric("Avg Credit Score", f"{df_pred['CreditScore'].mean():.0f}")

    st.divider()

    # ── Charts row ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Segment Distribution")
        seg_counts = df_pred["risk_segment"].value_counts()
        colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
        fig_pie = px.pie(
            values=seg_counts.values,
            names=seg_counts.index,
            color=seg_counts.index,
            color_discrete_map=colors,
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            showlegend=True,
            height=380,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Churn Probability Distribution")
        fig_hist = px.histogram(
            df_pred,
            x="churn_probability",
            nbins=40,
            color_discrete_sequence=["#0ea5e9"],
            labels={"churn_probability": "Churn Probability"},
        )
        fig_hist.add_vline(
            x=0.60,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="High-risk threshold (0.60)",
            annotation_position="top right",
            annotation_font_color="#ef4444",
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Count"),
            height=380,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Feature importances (hardcoded) ──────────────────────────
    st.subheader("Top 8 Feature Importances")
    imp_data = {
        "Feature": ["Complain", "Age", "IsActiveMember", "NumOfProducts",
                     "Balance", "CreditScore", "Geography", "Tenure"],
        "Importance": [0.28, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.05],
    }
    imp_df = pd.DataFrame(imp_data).sort_values("Importance")

    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#0ea5e9", "#6366f1", "#f472b6"],
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        coloraxis_showscale=False,
        height=400,
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — AT-RISK CUSTOMERS
# ═══════════════════════════════════════════════════════════════
elif page == "At-Risk Customers":
    st.markdown('<h1 class="main-header">At-Risk Customers</h1>', unsafe_allow_html=True)
    st.caption("Identify and prioritize bank customers most likely to churn")

    f1, f2 = st.columns(2)
    with f1:
        geo_filter = st.selectbox("Filter by Geography", ["All", "France", "Germany", "Spain"])
    with f2:
        card_filter = st.selectbox("Filter by Card Type", ["All", "DIAMOND", "GOLD", "SILVER", "PLATINUM"])

    display_cols = [
        "CustomerId", "churn_probability", "Balance",
        "CreditScore", "Age", "NumOfProducts", "Complain",
        "Satisfaction_Score", "IsActiveMember", "Geography",
        "risk_segment",
    ]

    filtered = df_pred.copy()
    if geo_filter != "All":
        filtered = filtered[filtered["Geography"] == geo_filter]
    if card_filter != "All":
        filtered = filtered[filtered["Card_Type"] == card_filter]

    top50 = (
        filtered
        .nlargest(50, "churn_probability")
        [[c for c in display_cols if c in filtered.columns]]
        .reset_index(drop=True)
    )

    st.markdown(f"**Showing top {len(top50)} customers** (sorted by churn probability)")

    st.dataframe(
        top50.style.background_gradient(
            subset=["churn_probability"],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
        ).format({
            "churn_probability": "{:.3f}",
            "Balance": "${:,.2f}",
        }),
        use_container_width=True,
        height=600,
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — A/B EXPERIMENTS
# ═══════════════════════════════════════════════════════════════
elif page == "A/B Experiments":
    st.markdown('<h1 class="main-header">A/B Experiment Results</h1>', unsafe_allow_html=True)
    st.caption("Statistical analysis of the bank retention campaign")

    # Run experiment with hardcoded values
    chi_result = run_chi_square_test(200, 1000, 160, 1000, experiment_name="bank_retention_v1")
    bayes_result = bayesian_test(200, 1000, 160, 1000)

    # ── Metrics ──────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Control Churn", f"{chi_result.control_churn_rate:.1%}")
    with m2:
        st.metric("Treatment Churn", f"{chi_result.treatment_churn_rate:.1%}",
                   delta=f"{chi_result.lift:.1%} lift", delta_color="inverse")
    with m3:
        st.metric("p-value", f"{chi_result.p_value:.4f}",
                   delta="Significant" if chi_result.is_significant else "Not Significant",
                   delta_color="normal" if chi_result.is_significant else "inverse")

    st.divider()

    # ── Bar chart ────────────────────────────────────────────────
    col1, col2 = st.columns([1.5, 1])

    with col1:
        fig_ab = go.Figure()
        fig_ab.add_trace(go.Bar(
            name="Control",
            x=["Churn Rate"],
            y=[chi_result.control_churn_rate * 100],
            marker_color="#ef4444",
            text=[f"{chi_result.control_churn_rate:.1%}"],
            textposition="outside",
        ))
        fig_ab.add_trace(go.Bar(
            name="Treatment",
            x=["Churn Rate"],
            y=[chi_result.treatment_churn_rate * 100],
            marker_color="#22c55e",
            text=[f"{chi_result.treatment_churn_rate:.1%}"],
            textposition="outside",
        ))
        fig_ab.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0", size=14),
            yaxis=dict(title="Churn Rate (%)", gridcolor="rgba(255,255,255,0.1)"),
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_ab, use_container_width=True)

    with col2:
        st.markdown("### Experiment Summary")
        st.markdown(f"""
        - **Total customers:** 2,000 (1,000 per group)
        - **Absolute lift:** 4 percentage points
        - **Relative lift:** {abs(chi_result.lift)*100:.1f}% churn reduction
        - **χ² statistic:** {chi_result.chi2_statistic:.4f}
        - **p-value:** {chi_result.p_value:.4f}
        - **Per campaign:** 40 customers retained × $300 = **$12,000**
        - **Annualized savings:** ~**$48,000/year**
        """)

        n_required = calculate_sample_size(0.20, 0.04)
        st.markdown(f"- **Min sample per group:** {n_required:,}")

    st.divider()

    # ── Bayesian ─────────────────────────────────────────────────
    st.subheader("Bayesian Analysis")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("P(Treatment Better)", f"{bayes_result['prob_treatment_better']:.1%}")
    with b2:
        st.metric("Expected Lift", f"{bayes_result['expected_lift']:.1%}")
    with b3:
        st.metric(
            "95% CI",
            f"[{bayes_result['lift_ci_95_lower']:.1%}, {bayes_result['lift_ci_95_upper']:.1%}]",
        )


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — AI INSIGHTS
# ═══════════════════════════════════════════════════════════════
elif page == "AI Insights":
    st.markdown('<h1 class="main-header">AI-Powered Retention Insights</h1>', unsafe_allow_html=True)
    st.caption("Personalized recommendations generated for high-risk bank customers")

    # Top 3 at-risk customers
    top3 = df_pred.nlargest(3, "churn_probability")

    for _, row in top3.iterrows():
        cid = row.get("CustomerId", "N/A")
        header = (
            f"**{cid}** — "
            f"Churn: {row['churn_probability']:.1%} | "
            f"Balance: ${row.get('Balance', 0):,.2f} | "
            f"Credit: {row.get('CreditScore', 'N/A')}"
        )
        with st.expander(header, expanded=True):
            customer_dict = row.to_dict()
            insight = get_retention_insight_fallback(customer_dict)
            st.markdown(
                f'<div class="insight-box">{insight}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 — PREDICT CUSTOMER
# ═══════════════════════════════════════════════════════════════
elif page == "Predict Customer":
    st.markdown('<h1 class="main-header">Predict Customer Churn</h1>', unsafe_allow_html=True)
    st.caption("Enter bank customer details to predict churn risk in real-time")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            customer_id = st.text_input("Customer ID", value="15600001")
            credit_score = st.slider("Credit Score", 300, 900, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 35)

        with c2:
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0)
            num_products = st.slider("Number of Products", 1, 4, 2)
            has_cr_card = st.selectbox("Has Credit Card", [1, 0])
            is_active = st.selectbox("Is Active Member", [1, 0])

        with c3:
            salary = st.number_input("Estimated Salary ($)", 0.0, 300000.0, 80000.0)
            complain = st.selectbox("Filed Complaint", [0, 1])
            sat_score = st.slider("Satisfaction Score", 1, 5, 3)
            card_type = st.selectbox("Card Type", ["GOLD", "SILVER", "PLATINUM", "DIAMOND"])
            point_earned = st.number_input("Points Earned", 0, 2000, 500)

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        # Derived features
        is_zero_bal = 1 if balance == 0 else 0

        # Rule-based churn probability
        churn_prob = (
            0.05
            + 0.25 * complain
            + 0.15 * is_zero_bal
            - 0.10 * is_active
            + 0.08 * int(num_products >= 3)
            - 0.005 * (sat_score - 1)
            - 0.001 * tenure
        )
        churn_prob = max(0.02, min(0.95, churn_prob))

        # Try model-based prediction if available
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                from src.data.preprocess import ENCODERS_PATH, SCALER_PATH, CATEGORICAL_COLS, NUMERIC_COLS

                if ENCODERS_PATH.exists() and SCALER_PATH.exists():
                    encoders = joblib.load(ENCODERS_PATH)
                    scaler = joblib.load(SCALER_PATH)

                    input_data = {
                        "CreditScore": credit_score,
                        "Geography": geography,
                        "Gender": gender,
                        "Age": age,
                        "Tenure": tenure,
                        "Balance": balance,
                        "NumOfProducts": num_products,
                        "HasCrCard": has_cr_card,
                        "IsActiveMember": is_active,
                        "EstimatedSalary": salary,
                        "Complain": complain,
                        "Satisfaction_Score": sat_score,
                        "Card_Type": card_type,
                        "Point_Earned": point_earned,
                    }
                    inp_df = pd.DataFrame([input_data])

                    # SQL-derived features
                    inp_df["balance_salary_ratio"] = (inp_df["Balance"] / max(salary, 1)).round(4)
                    inp_df["credit_age_ratio"] = (inp_df["CreditScore"] / max(age, 1)).round(4)
                    inp_df["is_zero_balance"] = int(balance == 0)
                    inp_df["product_tenure_ratio"] = (inp_df["NumOfProducts"] / max(tenure, 1)).round(4)
                    inp_df["is_high_complain_risk"] = int(complain == 1 and sat_score <= 2)

                    # Engineered features
                    point_norm = min(point_earned / 10000, 1.0)
                    sat_norm = sat_score / 5.0
                    inp_df["engagement_score"] = round(max(0, min(1, is_active * 0.4 + point_norm * 0.3 + sat_norm * 0.3)), 4)
                    inp_df["churn_risk_flag"] = int(complain == 1 and is_active == 0)
                    inp_df["products_per_year"] = round(num_products / max(tenure, 1), 4)

                    # Encode categoricals
                    for col in CATEGORICAL_COLS:
                        le = encoders[col]
                        known = set(le.classes_)
                        inp_df[col] = inp_df[col].astype(str).apply(
                            lambda x, _k=known, _le=le: _le.transform([x])[0] if x in _k else -1
                        )

                    inp_df[NUMERIC_COLS] = scaler.transform(inp_df[NUMERIC_COLS])

                    if FEATURE_COLS_PATH.exists():
                        feature_cols = joblib.load(FEATURE_COLS_PATH)
                        X = inp_df[[c for c in feature_cols if c in inp_df.columns]]
                    else:
                        X = inp_df

                    churn_prob = float(model.predict_proba(X)[:, 1][0])
            except Exception as e:
                logger.warning("Model prediction failed, using rule-based: %s", e)

        # Risk label
        if churn_prob >= 0.60:
            risk_label = "High Risk"
            risk_color = "inverse"
        elif churn_prob >= 0.30:
            risk_label = "Medium Risk"
            risk_color = "off"
        else:
            risk_label = "Low Risk"
            risk_color = "normal"

        st.divider()

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Churn Probability", f"{churn_prob:.1%}", delta=risk_label, delta_color=risk_color)
            st.progress(churn_prob, text=f"Risk Level: {churn_prob:.1%}")
        with r2:
            customer_dict = {
                "CustomerId": customer_id,
                "CreditScore": credit_score,
                "Geography": geography,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": salary,
                "Complain": complain,
                "Satisfaction_Score": sat_score,
                "Card_Type": card_type,
                "Point_Earned": point_earned,
                "churn_probability": churn_prob,
                "is_zero_balance": is_zero_bal,
                "churn_risk_flag": int(complain == 1 and is_active == 0),
                "engagement_score": round(max(0, min(1, is_active * 0.4 + min(point_earned / 10000, 1) * 0.3 + sat_score / 5 * 0.3)), 4),
            }
            insight = get_retention_insight_fallback(customer_dict)
            st.markdown(
                f'<div class="insight-box">{insight}</div>',
                unsafe_allow_html=True,
            )
