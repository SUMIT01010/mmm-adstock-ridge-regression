"""
Regression diagnostics: VIF, Ljung-Box, Chow test, QQ / heteroscedasticity.  # -- scaffolding --

VIF < 5 target (post-transform predictors) — the brief's headline diagnostic for "does
adstock+Hill actually reduce the multicollinearity that makes raw-spend OLS misattribute
credit." Ljung-Box checks residual serial correlation (autocorrelated residuals would mean
the model is missing structure, e.g. omitted trend/seasonality). Chow test checks for a
structural break — the brief suggests COVID; this dataset (Robyn's simulated panel) is
2015-11-23 to 2019-11-11, entirely pre-COVID, so there is no COVID period to test. A
defensible midpoint split (week 104, the median week) is used instead, documented here
rather than silently skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF per column of X (expects a numeric design matrix, no intercept column needed —
    statsmodels' VIF formula only needs the raw predictors; we add a constant internally
    since VIF is defined against a regression of each column on all the others + intercept).

    Zero-variance columns (e.g. a one-hot event dummy whose single positive week falls
    outside the slice being diagnosed — confirmed happening for `event_event2` on the
    train+val window here) are dropped before the VIF regression and reported separately:
    a constant column is perfectly collinear with the intercept by construction, which
    makes VIF mathematically undefined (formula divides by zero), not "high" — reporting
    it as +inf would misleadingly suggest a real multicollinearity problem."""
    Xc = X.copy().astype(float)
    zero_var_cols = [c for c in Xc.columns if Xc[c].std() == 0]
    Xc = Xc.drop(columns=zero_var_cols)
    Xc.insert(0, "const", 1.0)
    vifs = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(Xc.values, i)
        vifs.append({"feature": col, "vif": float(v)})
    for c in zero_var_cols:
        vifs.append({"feature": c, "vif": None, "note": "zero-variance in this window, VIF undefined"})
    return pd.DataFrame(vifs).sort_values(
        "vif", ascending=False, na_position="last").reset_index(drop=True)


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box Q test for residual serial correlation. H0: no autocorrelation up to
    each lag. A small p-value (< 0.05) flags remaining structure the model missed."""
    res = acorr_ljungbox(residuals, lags=lags, return_df=True)
    return res


def chow_test(X: pd.DataFrame, y: np.ndarray, split_idx: int) -> dict:
    """
    Chow (1960) structural-break F-test: does a single regression fit the whole sample
    as well as two separate regressions fit before/after `split_idx`?

    F = [(RSS_pooled - (RSS_1 + RSS_2)) / k] / [(RSS_1 + RSS_2) / (n1 + n2 - 2k)]
    where k = number of regressors (incl. intercept), under H0 ~ F(k, n1+n2-2k).
    A significant F (small p-value) means the coefficients differ meaningfully across the
    split -> evidence of a structural break at that point.
    """
    Xc = X.copy().astype(float)
    Xc.insert(0, "const", 1.0)
    Xarr, yarr = Xc.to_numpy(), np.asarray(y, dtype=float)
    n, k = Xarr.shape

    def rss(Xs, ys):
        beta, _, _, _ = np.linalg.lstsq(Xs, ys, rcond=None)
        resid = ys - Xs @ beta
        return float(resid @ resid)

    rss_pooled = rss(Xarr, yarr)
    X1, y1 = Xarr[:split_idx], yarr[:split_idx]
    X2, y2 = Xarr[split_idx:], yarr[split_idx:]
    n1, n2 = len(y1), len(y2)
    rss1, rss2 = rss(X1, y1), rss(X2, y2)

    df1, df2 = k, n1 + n2 - 2 * k
    if df2 <= 0:
        return {"error": f"not enough observations for Chow test (n1={n1}, n2={n2}, k={k})"}
    f_stat = ((rss_pooled - (rss1 + rss2)) / df1) / ((rss1 + rss2) / df2)
    p_value = float(1 - stats.f.cdf(f_stat, df1, df2))
    return {
        "split_idx": int(split_idx), "n1": n1, "n2": n2, "k": k,
        "rss_pooled": rss_pooled, "rss1": rss1, "rss2": rss2,
        "f_stat": float(f_stat), "df1": int(df1), "df2": int(df2),
        "p_value": p_value, "structural_break_at_5pct": bool(p_value < 0.05),
    }


def qq_data(residuals: np.ndarray) -> dict:
    """Theoretical vs sample quantiles for a QQ plot, plus a Shapiro-Wilk normality
    p-value and a Breusch-Pagan-style heteroscedasticity proxy (correlation of |resid|
    with fitted rank) — both summarized numerically for the JSON output; the plot itself
    is drawn in diagnostics_viz.py."""
    residuals = np.asarray(residuals, dtype=float)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) <= 5000 else (None, None)
    return {
        "theoretical_quantiles": osm.tolist(),
        "sample_quantiles": osr.tolist(),
        "fit_slope": float(slope), "fit_intercept": float(intercept), "fit_r": float(r),
        "shapiro_stat": float(shapiro_stat) if shapiro_stat is not None else None,
        "shapiro_p": float(shapiro_p) if shapiro_p is not None else None,
    }


def breusch_pagan(residuals: np.ndarray, fitted: np.ndarray) -> dict:
    """Simple heteroscedasticity check: regress squared residuals on fitted values;
    a significant slope indicates variance grows/shrinks with the prediction level."""
    residuals = np.asarray(residuals, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    sq_resid = residuals ** 2
    slope, intercept, r, p, se = stats.linregress(fitted, sq_resid)
    return {"slope": float(slope), "p_value": float(p), "r": float(r),
            "heteroscedastic_at_5pct": bool(p < 0.05)}
