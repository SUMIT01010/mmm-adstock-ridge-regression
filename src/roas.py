"""
ROAS / revenue decomposition.  # === CORE CONTRIBUTION ===

Given a fitted Ridge pipeline (StandardScaler + Ridge) over the Hill(Adstock(spend))
features + controls, decompose predicted revenue into:
    base       = prediction with every media channel's Hill feature set to 0
    contrib_c  = model.predict(X) - model.predict(X with ONLY channel c zeroed)
This "zero-out" decomposition (rather than reading raw coefficients off the model) is
deliberately robust to the StandardScaler inside the pipeline and to the fact the model
is additive-but-not-linear-in-spend (Hill is nonlinear) — it directly measures "how much
predicted revenue disappears if this channel's saturated, carryover-adjusted exposure is
removed," which is the quantity a CFO actually wants ("incremental revenue from channel
c"), holding every other channel's spend and the intercept/controls fixed.

Channel ROAS = incremental (attributed) revenue / raw spend, per channel, over the window
being decomposed (train+val fit period or holdout, caller's choice).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MEDIA_CHANNELS


def _hill_cols(X: pd.DataFrame) -> list[str]:
    return [f"{c}_hill" for c in MEDIA_CHANNELS if f"{c}_hill" in X.columns]


def decompose_revenue(model, X: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed like X with columns: base, <channel>_contrib (one per
    media channel), pred_total (= model.predict(X), sanity check against base+sum(contrib))."""
    hill_cols = _hill_cols(X)
    pred_total = model.predict(X)

    X_zero_all = X.copy()
    X_zero_all[hill_cols] = 0.0
    base = model.predict(X_zero_all)

    contrib = {}
    for ch in MEDIA_CHANNELS:
        col = f"{ch}_hill"
        X_zero_c = X.copy()
        X_zero_c[col] = 0.0
        pred_without_c = model.predict(X_zero_c)
        contrib[f"{ch}_contrib"] = pred_total - pred_without_c

    out = pd.DataFrame({"base": base, **contrib, "pred_total": pred_total}, index=X.index)
    # Additive decomposition is only exact for a purely linear model; Ridge here is
    # linear in the (already-nonlinear) Hill features, so base + sum(contrib) == pred_total
    # exactly (no cross-terms) — asserted, not assumed.
    reconstructed = out["base"] + out[[f"{c}_contrib" for c in MEDIA_CHANNELS]].sum(axis=1)
    max_err = float(np.max(np.abs(reconstructed - out["pred_total"])))
    assert max_err < 1e-6, f"decomposition does not reconstruct prediction, max err={max_err}"
    return out


def channel_roas(decomposition: pd.DataFrame, raw_spend: pd.DataFrame) -> dict[str, dict]:
    """incremental revenue / raw spend, per channel, summed over the decomposed window."""
    result = {}
    for ch in MEDIA_CHANNELS:
        incr = float(decomposition[f"{ch}_contrib"].clip(lower=0).sum())
        spend = float(raw_spend[ch].sum())
        result[ch] = {
            "incremental_revenue": incr,
            "total_spend": spend,
            "roas": (incr / spend) if spend > 0 else float("nan"),
        }
    return result


def revenue_waterfall(decomposition: pd.DataFrame) -> dict:
    """base vs incremental-per-channel share of total predicted revenue."""
    total = float(decomposition["pred_total"].sum())
    base = float(decomposition["base"].sum())
    shares = {"base": base / total}
    for ch in MEDIA_CHANNELS:
        shares[ch] = float(decomposition[f"{ch}_contrib"].sum()) / total
    return {"total_revenue": total, "shares": shares}
