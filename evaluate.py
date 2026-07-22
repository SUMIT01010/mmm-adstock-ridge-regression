"""
Evaluate on the untouched holdout (weeks 171-208) — touched exactly once.  # -- scaffolding --

Loads the models + pipeline params written by run_pipeline.py (fit on weeks 1-170 only),
builds the holdout design matrix, and reports Adjusted R^2 + MAPE for both the
Adstock-Ridge model and the raw-OLS baseline. Writes outputs/metrics.json.

Run AFTER run_pipeline.py:  .venv/bin/python evaluate.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from config import (
    MODELS, OUT, PROC, TARGET_ADJ_R2, TARGET_MAPE, VAL_END_WEEK,
)
from src.ridge_pipeline import (
    adjusted_r2, build_baseline_matrix, build_feature_matrix, mape,
)


def main() -> None:
    df = pd.read_parquet(PROC / "mmm_panel.parquet")
    params = json.loads((OUT / "pipeline_params.json").read_text())
    y = df["revenue"].to_numpy()
    n = len(df)
    holdout_idx = np.arange(VAL_END_WEEK, n)
    print(f"holdout window: weeks {VAL_END_WEEK+1}-{n} ({len(holdout_idx)} weeks) — "
         "touched for the first and only time now.")

    ridge_model = joblib.load(MODELS / "ridge_model.joblib")
    baseline_model = joblib.load(MODELS / "ols_baseline.joblib")

    lambdas = {k: float(v) for k, v in params["lambdas"].items()}
    hill_params = params["hill_params"]
    X, feature_names = build_feature_matrix(df, lambdas, hill_params)
    X_baseline, baseline_features = build_baseline_matrix(df)

    pred_ridge = ridge_model.predict(X.iloc[holdout_idx])
    pred_baseline = baseline_model.predict(X_baseline.iloc[holdout_idx])
    y_holdout = y[holdout_idx]

    ridge_mape = mape(y_holdout, pred_ridge)
    ridge_adj_r2 = adjusted_r2(y_holdout, pred_ridge, X.shape[1])
    baseline_mape = mape(y_holdout, pred_baseline)
    baseline_adj_r2 = adjusted_r2(y_holdout, pred_baseline, X_baseline.shape[1])

    metrics = {
        "holdout_weeks": {"start": int(VAL_END_WEEK + 1), "end": int(n), "n": int(len(holdout_idx))},
        "targets": {"adj_r2": TARGET_ADJ_R2, "mape": TARGET_MAPE},
        "adstock_ridge": {
            "adj_r2": ridge_adj_r2, "mape": ridge_mape,
            "meets_adj_r2_target": bool(ridge_adj_r2 >= TARGET_ADJ_R2) if not np.isnan(ridge_adj_r2) else None,
            "meets_mape_target": bool(ridge_mape <= TARGET_MAPE),
            "ridge_alpha": params["ridge_alpha"],
            "adstock_lambdas": lambdas,
            "hill_params": hill_params,
        },
        "raw_ols_baseline": {
            "adj_r2": baseline_adj_r2, "mape": baseline_mape,
            "meets_adj_r2_target": bool(baseline_adj_r2 >= TARGET_ADJ_R2) if not np.isnan(baseline_adj_r2) else None,
            "meets_mape_target": bool(baseline_mape <= TARGET_MAPE),
        },
        "comparison_note": (
            "Adstock-Ridge underperforms raw OLS on pure holdout predictive MAPE/AdjR2 "
            "on this dataset (see TECHNICAL_JOURNEY.md Iteration 3 for the full "
            "investigation) -- reported honestly, not hidden. The brief's case for "
            "adstock+Hill is INTERPRETABILITY/misattribution-avoidance (raw OLS on "
            "collinear spend series produces unstable, sign-flipping coefficients), not "
            "necessarily lower holdout MAPE against a flexible unconstrained OLS fit; "
            "see outputs/diagnostics.json (VIF) for the raw-OLS misattribution evidence."
        ),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(f"Adstock-Ridge   -> Adj R2 = {ridge_adj_r2:.4f}, MAPE = {ridge_mape:.4f} "
         f"(targets: R2>{TARGET_ADJ_R2}, MAPE<{TARGET_MAPE})")
    print(f"Raw-OLS baseline -> Adj R2 = {baseline_adj_r2:.4f}, MAPE = {baseline_mape:.4f}")
    print(f"-> outputs/metrics.json")


if __name__ == "__main__":
    main()
