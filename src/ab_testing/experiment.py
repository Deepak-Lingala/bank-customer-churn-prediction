"""
A/B testing and experiment analysis module for BankGuard AI.

Provides frequentist (chi-square) and Bayesian (Beta-Binomial)
statistical testing for retention experiments, along with power
analysis for sample-size calculation.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ── Sample-size calculation ──────────────────────────────────────
def calculate_sample_size(
    baseline_rate: float = 0.20,
    mde: float = 0.04,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Two-proportion z-test power analysis.

    Parameters
    ----------
    baseline_rate : float
        Expected baseline churn rate (e.g. 0.20 for banking).
    mde : float
        Minimum detectable effect (absolute, e.g. 0.04).
    alpha : float
        Significance level.
    power : float
        Statistical power.

    Returns
    -------
    int
        Required sample size per group.
    """
    p1 = baseline_rate
    p2 = baseline_rate - mde
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = (
        (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
         + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (p1 - p2) ** 2
    )
    n = math.ceil(n)
    logger.info(
        "Sample-size calculation: baseline=%.3f, MDE=%.3f → n=%d per group",
        baseline_rate,
        mde,
        n,
    )
    return n


# ── Result dataclass ─────────────────────────────────────────────
@dataclass
class ExperimentResult:
    """Container for A/B experiment results.

    Attributes
    ----------
    experiment_name : str
    control_n : int
    treatment_n : int
    control_churn_rate : float
    treatment_churn_rate : float
    lift : float
        Relative lift (negative means treatment reduced churn).
    p_value : float
    chi2_statistic : float
    is_significant : bool
    """

    experiment_name: str
    control_n: int
    treatment_n: int
    control_churn_rate: float
    treatment_churn_rate: float
    lift: float
    p_value: float
    chi2_statistic: float
    is_significant: bool


# ── Frequentist chi-square test ──────────────────────────────────
def run_chi_square_test(
    ctrl_churned: int,
    ctrl_total: int,
    trt_churned: int,
    trt_total: int,
    experiment_name: str = "bank_retention_v1",
    alpha: float = 0.05,
) -> ExperimentResult:
    """Run a chi-square test of independence for an A/B experiment.

    Parameters
    ----------
    ctrl_churned : int
        Number of churned customers in the control group.
    ctrl_total : int
        Total customers in the control group.
    trt_churned : int
        Number of churned customers in the treatment group.
    trt_total : int
        Total customers in the treatment group.
    experiment_name : str
        Name for this experiment.
    alpha : float
        Significance threshold.

    Returns
    -------
    ExperimentResult
    """
    table = np.array([
        [ctrl_churned, ctrl_total - ctrl_churned],
        [trt_churned, trt_total - trt_churned],
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(table)

    ctrl_rate = ctrl_churned / ctrl_total
    trt_rate = trt_churned / trt_total
    relative_lift = (trt_rate - ctrl_rate) / ctrl_rate if ctrl_rate > 0 else 0.0

    result = ExperimentResult(
        experiment_name=experiment_name,
        control_n=ctrl_total,
        treatment_n=trt_total,
        control_churn_rate=round(ctrl_rate, 4),
        treatment_churn_rate=round(trt_rate, 4),
        lift=round(relative_lift, 4),
        p_value=round(p_value, 6),
        chi2_statistic=round(chi2, 4),
        is_significant=p_value < alpha,
    )
    logger.info(
        "Chi-square test '%s': χ²=%.4f, p=%.6f, lift=%.2f%% → %s",
        experiment_name,
        chi2,
        p_value,
        relative_lift * 100,
        "SIGNIFICANT" if result.is_significant else "not significant",
    )
    return result


# ── Bayesian test ────────────────────────────────────────────────
def bayesian_test(
    ctrl_churned: int,
    ctrl_total: int,
    trt_churned: int,
    trt_total: int,
    n_samples: int = 100_000,
    seed: int = 42,
) -> dict[str, float]:
    """Beta-Binomial Bayesian A/B test.

    Uses a non-informative Beta(1, 1) prior with 100 k Monte Carlo
    samples to estimate the probability that the treatment is better.

    Parameters
    ----------
    ctrl_churned : int
        Churned customers in control group.
    ctrl_total : int
        Total control customers.
    trt_churned : int
        Churned customers in treatment group.
    trt_total : int
        Total treatment customers.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``prob_treatment_better``, ``expected_lift``,
        ``lift_ci_95_lower``, ``lift_ci_95_upper``.
    """
    rng = np.random.default_rng(seed)

    ctrl_samples = rng.beta(ctrl_churned + 1, ctrl_total - ctrl_churned + 1, n_samples)
    trt_samples = rng.beta(trt_churned + 1, trt_total - trt_churned + 1, n_samples)

    # "better" means lower churn
    prob_better = float(np.mean(trt_samples < ctrl_samples))
    lift_samples = (trt_samples - ctrl_samples) / ctrl_samples
    expected_lift = float(np.mean(lift_samples))
    ci_lower = float(np.percentile(lift_samples, 2.5))
    ci_upper = float(np.percentile(lift_samples, 97.5))

    logger.info(
        "Bayesian test: P(treatment better)=%.3f, E[lift]=%.3f, 95%% CI=[%.3f, %.3f]",
        prob_better,
        expected_lift,
        ci_lower,
        ci_upper,
    )
    return {
        "prob_treatment_better": round(prob_better, 4),
        "expected_lift": round(expected_lift, 4),
        "lift_ci_95_lower": round(ci_lower, 4),
        "lift_ci_95_upper": round(ci_upper, 4),
    }


# ── Demo simulation ─────────────────────────────────────────────
def simulate_bank_retention_experiment(seed: int = 42) -> dict:
    """Simulate a bank retention A/B experiment for demonstration.

    Creates a scenario with 1 000 customers per group, a baseline
    churn rate of 20 %, and a true lift of 20 % relative
    (treatment churn ≈ 16 %).

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    dict
        Contains ``chi_square`` (ExperimentResult) and
        ``bayesian`` (dict) keys.
    """
    rng = np.random.RandomState(seed)
    n_per_group = 1000
    baseline_churn = 0.20
    true_lift = 0.20  # 20% relative reduction

    ctrl_churned = int(rng.binomial(n_per_group, baseline_churn))
    trt_churned = int(rng.binomial(n_per_group, baseline_churn * (1 - true_lift)))

    chi_result = run_chi_square_test(
        ctrl_churned, n_per_group, trt_churned, n_per_group,
        experiment_name="bank_retention_discount_v1",
    )
    bayes_result = bayesian_test(
        ctrl_churned, n_per_group, trt_churned, n_per_group,
    )

    logger.info("Simulation complete — control=%d, treatment=%d", ctrl_churned, trt_churned)
    return {"chi_square": chi_result, "bayesian": bayes_result}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    result = simulate_bank_retention_experiment()
    chi = result["chi_square"]
    bayes = result["bayesian"]

    print(f"\n{'═' * 55}")
    print(f"  A/B Experiment: {chi.experiment_name}")
    print(f"{'═' * 55}")
    print(f"  Control churn:   {chi.control_churn_rate:.1%} ({chi.control_n} customers)")
    print(f"  Treatment churn: {chi.treatment_churn_rate:.1%} ({chi.treatment_n} customers)")
    print(f"  Relative lift:   {chi.lift:.1%}")
    print(f"  χ² statistic:    {chi.chi2_statistic:.4f}")
    print(f"  p-value:         {chi.p_value:.6f}")
    print(f"  Significant:     {'✓ YES' if chi.is_significant else '✗ NO'}")
    print(f"\n  Bayesian: P(trt better) = {bayes['prob_treatment_better']:.1%}")
    print(f"  Expected lift: {bayes['expected_lift']:.1%}")
    print(f"  95% CI: [{bayes['lift_ci_95_lower']:.1%}, {bayes['lift_ci_95_upper']:.1%}]")
