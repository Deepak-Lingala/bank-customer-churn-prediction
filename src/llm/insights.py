"""
LLM-powered retention insights module for BankGuard AI.

Provides both OpenAI GPT-4o-mini integration and a deterministic
rule-based fallback engine for generating personalized bank
customer retention recommendations.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ── Context builder ──────────────────────────────────────────────
def build_customer_context(customer_dict: dict[str, Any]) -> str:
    """Format a structured text summary of a bank customer profile.

    Parameters
    ----------
    customer_dict : dict
        Customer record with bank-churn-related features.

    Returns
    -------
    str
        Human-readable text block suitable as LLM context.
    """
    return (
        f"Customer ID: {customer_dict.get('CustomerId', customer_dict.get('customer_id', 'N/A'))}\n"
        f"Credit Score: {customer_dict.get('CreditScore', 'N/A')}\n"
        f"Geography: {customer_dict.get('Geography', 'N/A')}\n"
        f"Gender: {customer_dict.get('Gender', 'N/A')}\n"
        f"Age: {customer_dict.get('Age', 'N/A')}\n"
        f"Tenure: {customer_dict.get('Tenure', 'N/A')} years\n"
        f"Balance: ${customer_dict.get('Balance', 0):,.2f}\n"
        f"Products: {customer_dict.get('NumOfProducts', 'N/A')}\n"
        f"Has Credit Card: {'Yes' if customer_dict.get('HasCrCard', 0) else 'No'}\n"
        f"Active Member: {'Yes' if customer_dict.get('IsActiveMember', 0) else 'No'}\n"
        f"Estimated Salary: ${customer_dict.get('EstimatedSalary', 0):,.2f}\n"
        f"Complaint Filed: {'Yes' if customer_dict.get('Complain', 0) else 'No'}\n"
        f"Satisfaction Score: {customer_dict.get('Satisfaction_Score', 'N/A')}/5\n"
        f"Card Type: {customer_dict.get('Card_Type', 'N/A')}\n"
        f"Points Earned: {customer_dict.get('Point_Earned', 'N/A')}\n"
        f"Churn Probability: {customer_dict.get('churn_probability', 'N/A')}\n"
        f"Engagement Score: {customer_dict.get('engagement_score', 'N/A')}\n"
        f"Zero Balance: {'Yes' if customer_dict.get('is_zero_balance', 0) else 'No'}\n"
        f"Churn Risk Flag: {'Yes' if customer_dict.get('churn_risk_flag', 0) else 'No'}"
    )


# ── OpenAI integration ──────────────────────────────────────────
def get_retention_insight_openai(
    customer_dict: dict[str, Any],
    api_key: str | None = None,
) -> str:
    """Generate a retention insight using GPT-4o-mini.

    Automatically falls back to the rule-based engine if no API key
    is available or if the API call fails.

    Parameters
    ----------
    customer_dict : dict
        Customer record with bank-churn-related features.
    api_key : str, optional
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.

    Returns
    -------
    str
        Formatted retention recommendation.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key or api_key.startswith("sk-your"):
        logger.info("No valid OpenAI key — using rule-based fallback")
        return get_retention_insight_fallback(customer_dict)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        context = build_customer_context(customer_dict)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a bank customer success analyst. "
                        "Given a customer profile, provide:\n"
                        "1. Top 2 risk factors driving churn probability\n"
                        "2. One specific, actionable retention action\n"
                        "3. Recommended message tone (urgent/empathetic/celebratory)\n"
                        "Keep the response concise (under 150 words)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze this bank customer and provide retention recommendations:\n\n{context}",
                },
            ],
            temperature=0.7,
            max_tokens=300,
        )
        insight = response.choices[0].message.content
        logger.info("Generated LLM insight for %s", customer_dict.get("CustomerId"))
        return insight

    except ImportError:
        logger.warning("openai package not installed — falling back to rules")
        return get_retention_insight_fallback(customer_dict)
    except Exception as e:
        logger.warning("OpenAI API error: %s — falling back to rules", e)
        return get_retention_insight_fallback(customer_dict)


# ── Rule-based fallback ──────────────────────────────────────────
def get_retention_insight_fallback(customer_dict: dict[str, Any]) -> str:
    """Generate a retention insight using deterministic banking rules.

    Rules (evaluated in priority order)
    ------------------------------------
    1. Complain==1 AND Satisfaction_Score≤2 → CSM escalation + fee waiver
    2. is_zero_balance==1 AND IsActiveMember==0 → Dormant account promo
    3. NumOfProducts==1 AND churn_prob>0.6 → Product bundling
    4. Age>55 AND Balance>50k AND churn_prob>0.5 → Private banking
    5. CreditScore<550 → Financial wellness
    6. Default → Quarterly check-in

    Parameters
    ----------
    customer_dict : dict
        Customer record.

    Returns
    -------
    str
        Formatted retention recommendation with emoji.
    """
    complain = customer_dict.get("Complain", 0)
    sat_score = customer_dict.get("Satisfaction_Score", 3)
    is_zero_bal = customer_dict.get("is_zero_balance", 0)
    is_active = customer_dict.get("IsActiveMember", 1)
    num_products = customer_dict.get("NumOfProducts", 1)
    churn_prob = customer_dict.get("churn_probability", 0.5)
    age = customer_dict.get("Age", 35)
    balance = customer_dict.get("Balance", 0)
    credit_score = customer_dict.get("CreditScore", 650)

    # Rule 1: Unresolved complaint + low satisfaction
    if complain == 1 and sat_score <= 2:
        return (
            "**Unresolved Complaint + Low Satisfaction**\n\n"
            "**Risk Factors:**\n"
            f"  • Filed complaint with satisfaction score of {sat_score}/5\n"
            f"  • Churn probability: {churn_prob:.1%}\n\n"
            "**Recommended Action:** Escalate to senior relationship manager. "
            "Offer fee waiver + priority support line. Schedule resolution "
            "callback within 24 hours.\n\n"
            "**Tone:** Urgent & empathetic — prioritize complaint resolution."
        )

    # Rule 2: Dormant zero-balance account
    if is_zero_bal == 1 and is_active == 0:
        return (
            "**Dormant Zero-Balance Account**\n\n"
            "**Risk Factors:**\n"
            "  • Zero account balance — no financial engagement\n"
            f"  • Inactive member status, churn probability: {churn_prob:.1%}\n\n"
            "**Recommended Action:** Offer cashback activation promo within "
            "14 days. Send personalized re-engagement email with deposit bonus.\n\n"
            "**Tone:** Warm & inviting — emphasize value of returning."
        )

    # Rule 3: Single-product customer at risk
    if num_products == 1 and churn_prob > 0.6:
        return (
            "**Single-Product Customer — Low Switching Cost**\n\n"
            "**Risk Factors:**\n"
            f"  • Only {num_products} product — minimal lock-in\n"
            f"  • Churn probability: {churn_prob:.1%}\n\n"
            "**Recommended Action:** Bundle savings + credit card with "
            "bonus points. Offer cross-sell discount for adding a second product.\n\n"
            "**Tone:** Consultative — position bundling as financial convenience."
        )

    # Rule 4: High-value senior
    if age > 55 and balance > 50_000 and churn_prob > 0.5:
        return (
            f"**High-Value Senior Customer** (Balance: ${balance:,.0f})\n\n"
            "**Risk Factors:**\n"
            f"  • Age {age}, high balance — attractive to competitors\n"
            f"  • Churn probability: {churn_prob:.1%}\n\n"
            "**Recommended Action:** Assign dedicated private banking advisor. "
            "Offer premium card upgrade with enhanced rewards and priority access.\n\n"
            "**Tone:** Respectful & exclusive — emphasize white-glove service."
        )

    # Rule 5: Financial stress signal
    if credit_score < 550:
        return (
            f"**Financial Stress Signal** (Credit Score: {credit_score})\n\n"
            "**Risk Factors:**\n"
            f"  • Low credit score ({credit_score}) — possible financial difficulty\n"
            f"  • Churn probability: {churn_prob:.1%}\n\n"
            "**Recommended Action:** Offer financial wellness consultation. "
            "Consider restructured loan terms or temporary fee relief.\n\n"
            "**Tone:** Supportive & non-judgmental — build trust through guidance."
        )

    # Rule 6: Default — stable customer
    return (
        "**Customer Stable — Quarterly Check-in**\n\n"
        "**Risk Factors:**\n"
        "  • No major risk factors identified\n"
        f"  • Churn probability: {churn_prob:.1%}\n\n"
        "**Recommended Action:** Schedule quarterly check-in. "
        "Offer loyalty tier upgrade and highlight new product features.\n\n"
        "**Tone:** Positive & celebratory — reinforce value and build advocacy."
    )


# ── Batch generation ─────────────────────────────────────────────
def batch_generate_insights(
    customers_list: list[dict[str, Any]],
    use_llm: bool = False,
) -> list[dict[str, str]]:
    """Generate retention insights for a list of customers.

    Parameters
    ----------
    customers_list : list[dict]
        List of customer dictionaries.
    use_llm : bool
        Whether to use OpenAI (``True``) or rule-based fallback.

    Returns
    -------
    list[dict]
        Each item has ``customer_id`` and ``insight`` keys.
    """
    insights = []
    gen_fn = get_retention_insight_openai if use_llm else get_retention_insight_fallback

    for cust in customers_list:
        insight = gen_fn(cust)
        cid = cust.get("CustomerId", cust.get("customer_id", "unknown"))
        insights.append({
            "customer_id": cid,
            "insight": insight,
        })
    logger.info("Generated %d insights (LLM=%s)", len(insights), use_llm)
    return insights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sample = {
        "CustomerId": 15634602,
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
        "Point_Earned": 350,
        "churn_probability": 0.91,
        "engagement_score": 0.12,
        "churn_risk_flag": 1,
        "is_zero_balance": 0,
    }
    print(build_customer_context(sample))
    print("\n" + "═" * 55)
    print(get_retention_insight_fallback(sample))
