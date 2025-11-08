from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .data import CHANNELS
from .transforms import adstock, hill


@dataclass
class Allocation:
    horizon_weeks: int = 13
    baseline_weeks: int = 8
    lower_multiplier: float = .5
    upper_multiplier: float = 1.5
    posterior_draws: int = 256
    restarts: int = 5


def project_shares(candidate, lower, upper):
    candidate, lower, upper = map(lambda value: np.asarray(value, float), (candidate, lower, upper))
    if not all(np.isfinite(value).all() for value in (candidate, lower, upper)) or candidate.shape != lower.shape or lower.shape != upper.shape or np.any(lower < 0) or np.any(lower > upper) or lower.sum() > 1 + 1e-10 or upper.sum() < 1 - 1e-10:
        raise ValueError("Infeasible fixed-budget bounds")
    lo, hi = float(np.min(candidate - upper) - 1), float(np.max(candidate - lower) + 1)
    for _ in range(100):
        midpoint = (lo + hi) / 2
        if np.clip(candidate - midpoint, lower, upper).sum() > 1:
            lo = midpoint
        else:
            hi = midpoint
    return np.clip(candidate - (lo + hi) / 2, lower, upper)


def scenario_draws(weekly_spend, previous_state, future_features, processor, posterior):
    spend = np.asarray(weekly_spend) / np.asarray(processor.spend_scale)
    state = previous_state.copy()
    total = np.zeros(len(posterior["intercept"]))
    for feature_row in future_features:
        state = spend[None, :] + posterior["decay"] * state
        response = hill(state, posterior["half_saturation"], posterior["slope"])
        total += posterior["intercept"] + posterior["control_beta"] @ feature_row + np.sum(response * posterior["media_beta"], axis=1)
    return total * processor.revenue_scale


def optimize_budget(training, processor, posterior, config, seed=42):
    if config.horizon_weeks < 1 or config.baseline_weeks < 1 or config.baseline_weeks > len(training) or config.restarts < 1 or config.posterior_draws < 1:
        raise ValueError("Invalid allocation horizon, baseline window, or restart count")
    if not 0 <= config.lower_multiplier <= 1 <= config.upper_multiplier:
        raise ValueError("Bounds must contain the historical allocation")
    baseline = training[CHANNELS].tail(config.baseline_weeks).mean().to_numpy(float)
    budget = float(baseline.sum())
    if budget <= 0:
        raise ValueError("Allocation requires a positive budget")
    shares = baseline / budget
    lower, upper = shares * config.lower_multiplier, shares * config.upper_multiplier
    historic, _ = processor.transform(training)
    previous = np.stack([adstock(historic, decay)[-1] for decay in posterior["decay"]])
    # Both counterfactuals use the same future controls, dates, and starting carryover.
    import pandas as pd
    future = training.iloc[[-1]].copy()
    future = pd.concat([future] * config.horizon_weeks, ignore_index=True)
    future["DATE"] = pd.date_range(training.DATE.iloc[-1] + pd.Timedelta(days=7), periods=config.horizon_weeks, freq="7D")
    _, features = processor.transform(future)
    def expected(weekly):
        return scenario_draws(weekly, previous, features, processor, posterior)
    def objective(candidate):
        return -float(expected(candidate * budget).mean() / processor.revenue_scale)
    rng = np.random.default_rng(seed)
    solutions = []
    for restart in range(config.restarts):
        start = shares if restart == 0 else project_shares(shares * rng.uniform(.5, 1.5, len(shares)), lower, upper)
        result = minimize(objective, start, method="SLSQP", bounds=list(zip(lower, upper)),
                          constraints=[{"type": "eq", "fun": lambda candidate: candidate.sum() - 1}],
                          options={"ftol": 1e-9, "maxiter": 1000})
        valid = (result.success and abs(result.x.sum() - 1) < 1e-6 and
                 np.all(result.x >= lower - 1e-7) and np.all(result.x <= upper + 1e-7))
        if valid:
            solutions.append(result)
    if not solutions:
        raise RuntimeError("No feasible optimizer solution converged")
    best = min(solutions, key=lambda value: value.fun)
    proposed = project_shares(best.x, lower, upper) * budget
    baseline_revenue, proposed_revenue = expected(baseline), expected(proposed)
    if np.any(baseline_revenue <= 0):
        raise ValueError("Scenario includes nonpositive baseline revenue; inspect model specification")
    uplift = (proposed_revenue / baseline_revenue - 1) * 100
    return {"horizon_weeks": config.horizon_weeks, "weekly_budget": budget,
            "channels": [{"channel": name, "baseline_weekly_spend": float(old), "proposed_weekly_spend": float(new)}
                         for name, old, new in zip(CHANNELS, baseline, proposed)],
            "baseline_expected_revenue": float(baseline_revenue.mean()),
            "proposed_expected_revenue": float(proposed_revenue.mean()),
            "uplift_percent_mean": float(uplift.mean()),
            "uplift_percent_p05": float(np.quantile(uplift, .05)), "uplift_percent_p95": float(np.quantile(uplift, .95)),
            "probability_positive_uplift": float(np.mean(uplift > 0)),
            "converged_starts": len(solutions), "posterior_draws": len(uplift),
            "assumption": "Constant weekly channel spend; last training controls held fixed; conditional model projection"}
