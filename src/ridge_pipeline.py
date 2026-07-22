"""
Stage 3 — feature assembly + Ridge regression + joint hyperparameter search.
# === CORE CONTRIBUTION ===

Orchestrates Stages 1-3 end to end:
  1. per-channel geometric adstock (adstock.py) with lambda selected by coordinate-descent
     grid search against VALIDATION MAPE (never train error, never the holdout);
  2. per-channel Hill saturation (saturation.py) fit on the resulting adstocked series;
  3. Ridge regression on [Hill(Adstock(spend_c)) for c in channels] + linear controls
     (event flags, month dummies, trend, non-media numeric controls), with the Ridge
     penalty alpha ALSO selected on validation MAPE via a grid search.

Hyperparameter search protocol (why coordinate descent, not a full grid):
5 channels x 10 lambda values x 11 alpha values = 5,500 combinations for a brute-force
joint grid — wasteful and, worse, each channel's optimal lambda is near-independent of
the others' lambdas conditional on Ridge's regularization (Ridge already shrinks
correlated/redundant transformed features, so channel-level carryover shape doesn't
interact much with other channels' carryover shape). We instead do 2 sweeps of
coordinate descent: hold every channel's lambda fixed except one, grid-search that one
channel's lambda against validation MAPE (Hill + a fixed default Ridge alpha=1.0 during
this inner loop, to isolate carryover shape from regularization strength), then move to
the next channel. After both lambda sweeps converge, Hill parameters are refit at the
final lambdas and a SEPARATE grid search picks the Ridge alpha. This is 5*10*2 + 11 = 111
fits instead of 5,500, and is stated up front rather than silently done — see
TECHNICAL_JOURNEY.md for the sweep-by-sweep lambda trace and whether it converged.

Baseline: `fit_ols_baseline` fits plain OLS (Ridge alpha=0, LinearRegression) on RAW
untransformed spend (no adstock, no Hill) + the same controls, exactly the brief's "why
adstock/saturation matters" comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    EVENT_COL, EXTRA_NUMERIC_CONTROLS, MEDIA_CHANNELS, RIDGE_ALPHA_GRID,
)
from src.adstock import geometric_adstock
from src.saturation import fit_hill_channel, hill_saturation

# NOTE: `is_event` (boolean "any event this week") is deliberately EXCLUDED from the
# model's controls even though data_audit.py keeps it in the panel for readability/EDA.
# With only 2 event weeks total in 208 (event1, event2 — one each), and the train+val
# window (weeks 1-170) containing only event1, `is_event` is then IDENTICAL to
# `event_event1` in-window (both are 1 on exactly that one row, 0 elsewhere) -> perfectly
# collinear -> infinite VIF, confirmed empirically (see TECHNICAL_JOURNEY.md Iteration 3).
# The one-hot event dummies (`event_event1`, `event_event2`) already carry strictly finer
# information than the boolean OR of them, so only the dummies are used as controls.
CONTROL_COLS_BASE = ["trend"] + EXTRA_NUMERIC_CONTROLS


def _month_event_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("month_") or c.startswith("event_")]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    denom = n - n_features - 1
    if denom <= 0:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / denom)


# ------------------------------------------------------------------ transform pipeline
def adstock_transform(df: pd.DataFrame, lambdas: dict[str, float]) -> pd.DataFrame:
    """Adstock every media channel over the FULL series (index-respecting recursion),
    then the caller slices train/val/holdout — adstock at week t depends on weeks < t,
    so it must be computed on the full chronological series, not independently per split
    (re-computing per split would truncate carryover from the prior period)."""
    out = pd.DataFrame(index=df.index)
    for ch in MEDIA_CHANNELS:
        out[ch] = geometric_adstock(df[ch].to_numpy(), lambdas[ch])
    return out


def hill_transform(adstocked: pd.DataFrame, hill_params: dict[str, dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=adstocked.index)
    for ch in MEDIA_CHANNELS:
        p = hill_params[ch]
        out[ch] = hill_saturation(adstocked[ch].to_numpy(), p["alpha"], p["K"])
    return out


def build_feature_matrix(df: pd.DataFrame, lambdas: dict[str, float],
                          hill_params: dict[str, dict]) -> tuple[pd.DataFrame, list[str]]:
    adstocked = adstock_transform(df, lambdas)
    saturated = hill_transform(adstocked, hill_params)
    controls = df[CONTROL_COLS_BASE + _month_event_cols(df)].astype(float)
    X = pd.concat([saturated.add_suffix("_hill"), controls], axis=1)
    return X, list(X.columns)


def build_baseline_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Raw untransformed spend + same controls — no adstock, no Hill."""
    raw = df[MEDIA_CHANNELS].astype(float)
    controls = df[CONTROL_COLS_BASE + _month_event_cols(df)].astype(float)
    X = pd.concat([raw, controls], axis=1)
    return X, list(X.columns)


# ------------------------------------------------------------------- hyperparam search
def _fit_hill_for_lambdas(df: pd.DataFrame, train_idx, lambdas: dict[str, float],
                          y_train: np.ndarray) -> dict[str, dict]:
    """
    Fit each channel's Hill(alpha, K) against a PARTIAL-RESIDUAL target, not raw revenue.

    Dead end found first (documented in TECHNICAL_JOURNEY.md Iteration 2): curve-fitting
    Hill(adstocked_c) directly against raw revenue is a 4-parameter fit (alpha, K, scale,
    intercept) trying to explain a target that is actually driven by 4 other channels +
    seasonality + trend + events. With ~100-150 training weeks per channel this is
    underdetermined for the weaker channels (print_S in particular) and curve_fit lands
    on degenerate near-step-function solutions (alpha/K pinned at their bounds) that
    overfit noise and generalize badly. Fix: fit a quick linear reference model (Ridge,
    alpha=1.0, on adstocked-but-unsaturated spend + controls) first, take its residual,
    and add back channel c's own linear contribution (`resid + beta_c * adstocked_c`) —
    standard partial-regression / backfitting logic. Hill is then fit against THAT
    (channel c's isolated marginal relationship with revenue), which is far better
    conditioned. This is still fit on TRAIN only.
    """
    adstocked = adstock_transform(df, lambdas)
    controls = df[CONTROL_COLS_BASE + _month_event_cols(df)].astype(float)
    channel_cols = [f"{ch}_raw" for ch in MEDIA_CHANNELS]
    X_lin = pd.concat([adstocked.set_axis(channel_cols, axis=1), controls], axis=1)

    lin_model = make_ridge_pipeline(1.0)
    lin_model.fit(X_lin.iloc[train_idx], y_train)
    scaler, ridge = lin_model.named_steps["scale"], lin_model.named_steps["ridge"]
    coefs_orig = ridge.coef_ / scaler.scale_   # undo standardization -> original-feature-scale betas
    pred_train = lin_model.predict(X_lin.iloc[train_idx])
    resid_train = y_train - pred_train

    hill_params = {}
    for j, ch in enumerate(MEDIA_CHANNELS):
        beta_c = coefs_orig[j]
        x_train = adstocked[ch].to_numpy()[train_idx]
        partial_target = resid_train + beta_c * x_train
        hill_params[ch] = fit_hill_channel(x_train, partial_target)
    return hill_params


def coordinate_descent_lambdas(
    df: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
    lambda_grid: list[float], n_sweeps: int = 2, default_alpha: float = 1.0,
) -> tuple[dict[str, float], list[dict]]:
    """Coordinate-descent lambda search; returns final lambdas + a full trace (for the
    book/TECHNICAL_JOURNEY): one entry per (sweep, channel, lambda) with val MAPE."""
    lambdas = {ch: 0.5 for ch in MEDIA_CHANNELS}
    y_train, y_val = y[train_idx], y[val_idx]
    trace = []

    def score(candidate_lambdas: dict[str, float]) -> float:
        hill_params = _fit_hill_for_lambdas(df, train_idx, candidate_lambdas, y_train)
        X, _ = build_feature_matrix(df, candidate_lambdas, hill_params)
        model = make_ridge_pipeline(default_alpha)
        model.fit(X.iloc[train_idx], y_train)
        pred_val = model.predict(X.iloc[val_idx])
        pred_val = np.clip(pred_val, 1e-6, None)
        return mape(y_val, pred_val)

    for sweep in range(n_sweeps):
        for ch in MEDIA_CHANNELS:
            scores = {}
            for lam in lambda_grid:
                cand = dict(lambdas)
                cand[ch] = lam
                s = score(cand)
                scores[lam] = s
                trace.append({"sweep": sweep, "channel": ch, "lambda": lam, "val_mape": s})
            best_lam = min(scores, key=scores.get)
            lambdas[ch] = best_lam
    return lambdas, trace


def select_ridge_alpha(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray, alpha_grid: list[float],
) -> tuple[float, dict[float, float]]:
    scores = {}
    for a in alpha_grid:
        model = make_ridge_pipeline(a)
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_val), 1e-6, None)
        scores[a] = mape(y_val, pred)
    best_alpha = min(scores, key=scores.get)
    return best_alpha, scores


def full_hyperparam_search(
    df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
    lambda_grid: list[float], alpha_grid: list[float],
) -> dict:
    """Runs the full Stage1-3 hyperparameter search protocol described in the module
    docstring and returns everything downstream code / the book needs."""
    y = df["revenue"].to_numpy()
    y_train, y_val = y[train_idx], y[val_idx]

    lambdas, lambda_trace = coordinate_descent_lambdas(
        df, y, train_idx, val_idx, lambda_grid)
    hill_params = _fit_hill_for_lambdas(df, train_idx, lambdas, y_train)
    X, feature_names = build_feature_matrix(df, lambdas, hill_params)
    best_alpha, alpha_scores = select_ridge_alpha(
        X.iloc[train_idx], y_train, X.iloc[val_idx], y_val, alpha_grid)

    return {
        "lambdas": lambdas,
        "lambda_trace": lambda_trace,
        "hill_params": hill_params,
        "ridge_alpha": best_alpha,
        "ridge_alpha_scores": alpha_scores,
        "feature_names": feature_names,
        "X": X,
        "y": y,
    }


# ------------------------------------------------------------------------------ fitting
def make_ridge_pipeline(alpha: float) -> Pipeline:
    """StandardScaler + Ridge. Scaling matters here because feature scales span orders
    of magnitude (Hill outputs are in [0,1]; competitor_sales_B is O(1e6); trend is
    O(1e2); dummies are 0/1) — an unscaled Ridge solve is ill-conditioned (confirmed:
    sklearn raised LinAlgWarning on the raw-scale fit) and, more importantly, an
    unscaled L2 penalty shrinks large-raw-scale features less than small-scale ones
    for reasons that have nothing to do with their actual explanatory power."""
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=0))])


def fit_ridge(X_train: pd.DataFrame, y_train: np.ndarray, alpha: float) -> Pipeline:
    model = make_ridge_pipeline(alpha)
    model.fit(X_train, y_train)
    return model


def fit_ols_baseline(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[Pipeline, list[str]]:
    """Raw OLS baseline (alpha=0 Ridge == LinearRegression, wrapped in the same scaler
    pipeline for a fair, numerically stable apples-to-apples comparison)."""
    X, feature_names = build_baseline_matrix(df)
    y = df["revenue"].to_numpy()
    model = Pipeline([("scale", StandardScaler()), ("ols", LinearRegression())])
    model.fit(X.iloc[train_idx], y[train_idx])
    return model, feature_names
