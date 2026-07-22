"""
Stage 1 — geometric adstock transformation.  # === CORE CONTRIBUTION ===

Adstock(t) = spend(t) + lambda * Adstock(t-1),   Adstock(0) = spend(0)

lambda in [0,1) is the week-over-week carryover (decay) rate: a channel with lambda near
1 has long "memory" (a rupee spent this week still moves revenue many weeks out — TV,
brand-building), a channel with lambda near 0 has almost no memory (Search, bottom-funnel,
converts in the week it's clicked). This module implements the recursive transform itself
(no external MMM library) and the per-channel lambda selection: a grid search that picks,
for each channel independently, the lambda whose resulting adstocked feature — fed through
the rest of the fixed pipeline (Hill saturation with a quick default, then Ridge) — gives
the lowest MAPE on the VALIDATION slice (weeks 151-170), never on train error and never on
the holdout. This mirrors ridge_pipeline's alpha search protocol; adstock lambda and Ridge
alpha are two nested hyperparameters of the same estimator and are selected the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def geometric_adstock(spend: np.ndarray, lam: float) -> np.ndarray:
    """Apply Adstock(t) = spend(t) + lam*Adstock(t-1) along axis 0. lam in [0, 1)."""
    if not (0.0 <= lam < 1.0):
        raise ValueError(f"lam must be in [0, 1), got {lam}")
    spend = np.asarray(spend, dtype=float)
    out = np.empty_like(spend)
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = spend[t] + lam * out[t - 1]
    return out


def adstock_all_channels(df: pd.DataFrame, channels: list[str],
                          lambdas: dict[str, float]) -> pd.DataFrame:
    """Return a DataFrame of adstocked series, one column per channel, same index as df."""
    out = pd.DataFrame(index=df.index)
    for ch in channels:
        out[ch] = geometric_adstock(df[ch].to_numpy(), lambdas[ch])
    return out


def grid_search_lambda(
    train_df: pd.DataFrame, val_df: pd.DataFrame, channel: str,
    lambda_grid: list[float], score_fn,
) -> tuple[float, dict[float, float]]:
    """
    For one channel, try every lambda in lambda_grid; `score_fn(lam) -> validation MAPE`
    is supplied by the caller (ridge_pipeline), which knows how to hold every other
    channel's lambda fixed, refit Hill+Ridge, and score on the validation slice. This
    function is intentionally channel-agnostic and stateless — it just runs the grid and
    returns the best lambda plus the full score trace (written to outputs for the book).
    """
    scores = {lam: score_fn(lam) for lam in lambda_grid}
    best_lam = min(scores, key=scores.get)
    return best_lam, scores


if __name__ == "__main__":
    # smoke test: pure impulse response check
    spend = np.array([100.0, 0, 0, 0, 0])
    for lam in (0.1, 0.7):
        a = geometric_adstock(spend, lam)
        print(f"lam={lam}: {np.round(a, 2)}")
        assert a[0] == 100.0
        assert np.all(np.diff(a) <= 0)  # pure decay after the single impulse
