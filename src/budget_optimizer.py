"""
Constrained budget reallocation via scipy.optimize.  # === CORE CONTRIBUTION ===

Given the fitted Hill saturation curves (alpha, K per channel) and the fitted Ridge
coefficients (the linear weight each channel's saturated feature carries in the revenue
equation), find the weekly spend split across the 5 channels that maximizes predicted
revenue, holding TOTAL weekly budget fixed at its current (historical average) level —
"reallocate the same money better," the brief's ask, not "spend more."

Steady-state simplification: optimizing spend week-by-week through the full adstock
recursion is a much harder (path-dependent) control problem. Standard MMM practice
(Robyn, Meta's own writeups) instead optimizes the STEADY-STATE weekly spend level: if a
channel receives a constant weekly spend s indefinitely, its adstock converges to
s / (1 - lambda) (geometric series sum). Plugging that into the channel's Hill curve gives
a smooth, differentiable map from "weekly spend level" to "saturated response," letting
this be a plain constrained nonlinear program:

    maximize   sum_c  beta_c * Hill( s_c / (1 - lambda_c) ; alpha_c, K_c )
    subject to sum_c s_c = TOTAL_BUDGET,   0 <= s_c <= spend_cap_c

beta_c is the channel's fitted Ridge coefficient converted back out of the pipeline's
StandardScaler into original (0-1 Hill-output) units — i.e. "how many rupees of revenue
one unit of fully-saturated Hill response is worth." spend_cap_c defaults to 3x the
channel's historical max weekly spend (prevents the optimizer from proposing spend levels
far outside anything the fitted saturation curve was ever calibrated on).

Solved with SLSQP (`scipy.optimize.minimize`), which handles the equality budget
constraint and box bounds directly — no external LP/QP solver needed.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from config import MEDIA_CHANNELS
from src.saturation import hill_saturation


def _extract_channel_betas(model, feature_names: list[str]) -> dict[str, float]:
    """Undo the pipeline's StandardScaler to get each channel's Hill-feature coefficient
    in original (unscaled, 0-1 Hill-output) units."""
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]
    coefs_orig = ridge.coef_ / scaler.scale_
    betas = {}
    for ch in MEDIA_CHANNELS:
        col = f"{ch}_hill"
        idx = feature_names.index(col)
        betas[ch] = float(coefs_orig[idx])
    return betas


def optimize_budget(
    model, feature_names: list[str], hill_params: dict[str, dict],
    lambdas: dict[str, float], current_spend: dict[str, float],
    spend_cap_multiplier: float = 3.0,
) -> dict:
    betas = _extract_channel_betas(model, feature_names)
    total_budget = float(sum(current_spend.values()))
    channels = MEDIA_CHANNELS
    n = len(channels)

    caps = np.array([max(current_spend[c], 1.0) * spend_cap_multiplier for c in channels])
    x0 = np.array([current_spend[c] for c in channels])
    # rescale x0 onto the budget simplex in case of float drift
    x0 = x0 / x0.sum() * total_budget
    x0 = np.minimum(x0, caps * 0.999)

    def steady_state_response(s: np.ndarray) -> float:
        total = 0.0
        for i, ch in enumerate(channels):
            lam = lambdas[ch]
            adstock_ss = s[i] / (1 - lam)
            p = hill_params[ch]
            sat = hill_saturation(np.array([adstock_ss]), p["alpha"], p["K"])[0]
            total += betas[ch] * sat
        return total

    def neg_objective(s: np.ndarray) -> float:
        return -steady_state_response(s)

    constraints = [{"type": "eq", "fun": lambda s: np.sum(s) - total_budget}]
    bounds = [(0.0, float(caps[i])) for i in range(n)]

    result = minimize(neg_objective, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})

    optimal_spend = {ch: max(0.0, float(result.x[i])) for i, ch in enumerate(channels)}
    # renormalize tiny SLSQP equality-constraint drift back onto the exact budget
    drift = total_budget - sum(optimal_spend.values())
    if abs(drift) > 1e-6:
        largest = max(optimal_spend, key=optimal_spend.get)
        optimal_spend[largest] += drift

    current_response = steady_state_response(np.array([current_spend[c] for c in channels]))
    optimal_response = steady_state_response(np.array([optimal_spend[c] for c in channels]))

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "total_budget": total_budget,
        "current_spend": current_spend,
        "optimal_spend": optimal_spend,
        "current_steady_state_response": float(current_response),
        "optimal_steady_state_response": float(optimal_response),
        "predicted_lift_pct": float(
            (optimal_response - current_response) / abs(current_response) * 100
        ) if current_response != 0 else float("nan"),
        "betas": betas,
    }
