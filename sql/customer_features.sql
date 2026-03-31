-- ============================================================
-- BankGuard AI — Customer Feature Extraction Query
-- ============================================================
-- Full CTE-based SQL that derives risk segments, wealth segments,
-- and churn risk indicators from raw bank customer data.
-- ============================================================

WITH base_features AS (
    -- CTE 1: Derived financial and risk indicators
    SELECT
        *,

        -- Balance to salary ratio
        ROUND(
            CAST(Balance AS REAL) / MAX(EstimatedSalary, 1),
            4
        ) AS balance_salary_ratio,

        -- Credit score to age ratio
        ROUND(
            CAST(CreditScore AS REAL) / MAX(Age, 1),
            4
        ) AS credit_age_ratio,

        -- Zero balance flag
        CASE WHEN Balance = 0 THEN 1 ELSE 0 END AS is_zero_balance,

        -- Product to tenure ratio
        ROUND(
            CAST(NumOfProducts AS REAL) / MAX(Tenure, 1),
            4
        ) AS product_tenure_ratio,

        -- High complaint risk: filed complaint + very low satisfaction
        CASE
            WHEN Complain = 1 AND Satisfaction_Score <= 2 THEN 1
            ELSE 0
        END AS is_high_complain_risk

    FROM customers_raw
),

risk_segments AS (
    -- CTE 2: Add segments and flags
    SELECT
        *,

        -- Wealth segment based on balance quartiles
        CASE
            WHEN Balance < 50000  THEN 'Low'
            WHEN Balance < 100000 THEN 'Mid-Low'
            WHEN Balance < 150000 THEN 'Mid-High'
            ELSE 'High'
        END AS wealth_segment,

        -- Age group classification
        CASE
            WHEN Age <= 35 THEN 'Young'
            WHEN Age <= 55 THEN 'Mid'
            ELSE 'Senior'
        END AS age_group,

        -- Churn risk flag: complained AND not active
        CASE
            WHEN Complain = 1 AND IsActiveMember = 0 THEN 1
            ELSE 0
        END AS churn_risk_flag

    FROM base_features
)

-- Final SELECT ordered by risk
SELECT *
FROM risk_segments
ORDER BY is_high_complain_risk DESC, Balance DESC;
