"""
Stage 2 — Hill saturation curve.  # === CORE CONTRIBUTION ===

Saturated(x) = x^alpha / (x^alpha + K^alpha),  x >= 0, alpha > 0, K > 0

alpha controls the curve's shape (alpha>1: S-shaped, slow ramp-up then diminishing
returns; alpha<=1: concave from the origin, immediate diminishing returns — typical for
already-saturated channels). K is the half-saturation point: the spend level at which the
channel delivers 50% of its ceiling response. Fit per channel on the ADSTOCKED spend
series (saturation acts on carryover-adjusted exposure, not raw weekly spend — this is
the standard Stage1->Stage2 ordering in MMM: memory first, then diminishing returns on the
resulting effective exposure).

Fitting method: alpha, K are not identifiable from a pure curve-fit against noisy revenue
in closed form, so this is solved as a constrained nonlinear least-squares problem
(scipy.optimize.curve_fit) minimizing the residual between a *scaled* Hill output and a
proxy target (the channel's own adstocked spend rank-correlated against revenue via a
single-channel OLS pass) — see `fit_hill_channel` docstring for the exact target
construction. Falls back to a coarse alpha/K grid + revenue-correlation objective if
curve_fit fails to converge (rare, guarded explicitly, not silently swallowed).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def hill_saturation(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
    """Saturated(x) = x^alpha / (x^alpha + K^alpha). Vectorized, x >= 0."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    xa = np.power(x, alpha)
    return xa / (xa + K**alpha + 1e-12)


def fit_hill_channel(adstocked_spend: np.ndarray, revenue: np.ndarray,
                      seed: int = 42) -> dict:
    """
    Fit alpha, K for one channel so that hill_saturation(adstocked_spend; alpha, K),
    linearly rescaled, best matches revenue in a least-squares sense. This gives Hill
    parameters that are directly useful as a Ridge feature (Stage 3 fits the linear
    scale/intercept jointly with every other predictor) — Stage 2 only needs the CURVE
    SHAPE (alpha, K), not the linear scale, since Ridge re-scales every feature anyway.

    We fit a 4-parameter model y = a * Hill(x; alpha, K) + b via curve_fit so alpha/K are
    identified against revenue's actual scale, then discard a, b (Ridge will refit its own
    coefficient on top of the Hill-transformed feature).
    """
    x = np.asarray(adstocked_spend, dtype=float)
    y = np.asarray(revenue, dtype=float)
    # A strict x > 0 mask is wrong here: low-lambda adstock still leaves an
    # exponentially tiny nonzero carryover on true zero-spend weeks (e.g. 3.9e-08 on a
    # series whose real spend tops out at ~32,000), which pollutes the low quantile used
    # for the K lower bound below and can drag the whole fit toward a near-degenerate
    # curve. Treat anything below 1e-6 of the series max as "no meaningful spend."
    thresh = 1e-6 * x.max() if x.max() > 0 else 0.0
    x_pos = x[x > thresh]
    if len(x_pos) < 5 or x_pos.std() == 0:
        # channel barely used in this slice — return a near-linear default so the
        # feature degenerates gracefully to raw spend rather than crashing the pipeline
        return {"alpha": 1.0, "K": float(np.median(x[x > 0])) if len(x_pos) else 1.0,
                "converged": False, "reason": "insufficient nonzero spend to fit Hill"}

    K0 = float(np.median(x_pos))
    a0 = float((y.max() - y.min()) or 1.0)
    b0 = float(y.min())

    def model(x_, alpha, K, a, b):
        return a * hill_saturation(x_, alpha, K) + b

    p0 = [1.0, K0, a0, b0]
    # Bounds are anchored to the data's own spend distribution (quantiles, not raw
    # min/max — a single outlier week must not set the K range) and alpha is kept in
    # [0.3, 3.0]: wide enough to span "already-diminishing" (alpha<1) through a
    # pronounced S-curve (alpha~3), but excludes near-step-function corner solutions
    # (alpha->5, K->~0) that curve_fit can land on with only ~100-150 training weeks
    # and produce a channel feature that is effectively a binary "spent vs not" flag —
    # found empirically on print_S (alpha hit the old 5.0 bound, K collapsed near 0);
    # see TECHNICAL_JOURNEY.md Iteration 2.
    K_lo = float(np.quantile(x_pos, 0.10))
    K_hi = float(np.quantile(x_pos, 0.90)) * 2.0
    if K_hi <= K_lo:
        K_hi = K_lo * 2.0 + 1.0
    bounds = ([0.3, K_lo, -np.inf, -np.inf],
              [3.0, K_hi, np.inf, np.inf])
    p0[1] = float(np.clip(K0, K_lo, K_hi))
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        alpha, K, a, b = popt
        return {"alpha": float(alpha), "K": float(K), "converged": True}
    except Exception as e:  # pragma: no cover - guarded fallback, not silently swallowed
        # coarse grid fallback: maximize correlation between Hill(x) and revenue
        alpha_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        K_grid = np.quantile(x_pos, [0.25, 0.5, 0.75])
        best, best_corr = (1.0, K0), -np.inf
        for a_ in alpha_grid:
            for K_ in K_grid:
                sat = hill_saturation(x, a_, K_)
                if sat.std() == 0:
                    continue
                corr = np.corrcoef(sat, y)[0, 1]
                if corr > best_corr:
                    best_corr, best = corr, (float(a_), float(K_))
        return {"alpha": best[0], "K": best[1], "converged": False,
                "reason": f"curve_fit failed ({e}); used correlation grid fallback"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(1000, 400, 150))
    true_alpha, true_K = 1.8, 900.0
    y = 5000 * hill_saturation(x, true_alpha, true_K) + rng.normal(0, 50, 150) + 1000
    fit = fit_hill_channel(x, y)
    print("fit:", fit, " true alpha/K:", true_alpha, true_K)
