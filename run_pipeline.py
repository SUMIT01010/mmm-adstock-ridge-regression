"""
End-to-end MMM pipeline entry point.  # -- scaffolding --

Stage 0  data_audit           -> data/processed/mmm_panel.parquet, outputs/data_audit_summary.json
Stage 1  adstock lambda search (coordinate descent, validation MAPE)
Stage 2  Hill saturation fit (partial-residual target, train slice)
Stage 3  Ridge alpha search (validation MAPE) + final refit on train+val
         + raw-OLS baseline refit on train+val (same window, fair comparison)
Diagnostics: VIF / Ljung-Box / Chow / QQ+heteroscedasticity on the train+val fit window
ROAS / decomposition / budget reallocation computed on the train+val window

evaluate.py (run separately, AFTER this) is what touches the holdout exactly once and
writes outputs/metrics.json — this script deliberately does not touch weeks 171-208.

Run:  .venv/bin/python run_pipeline.py
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from config import (
    ADSTOCK_LAMBDA_GRID, MEDIA_CHANNELS, MODELS, OUT, RIDGE_ALPHA_GRID,
    TRAIN_END_WEEK, VAL_END_WEEK, VIF_THRESHOLD,
)
from src import data_audit, diagnostics, eda
from src.budget_optimizer import optimize_budget
from src.ridge_pipeline import (
    _fit_hill_for_lambdas, adjusted_r2, adstock_transform, build_baseline_matrix,
    build_feature_matrix, coordinate_descent_lambdas, fit_ols_baseline, fit_ridge,
    full_hyperparam_search, mape,
)
from src.roas import channel_roas, decompose_revenue, revenue_waterfall


def main() -> None:
    t0 = time.time()
    print("=== Stage 0: data audit ===")
    df = data_audit.run()

    print("=== Stage 0: EDA figures ===")
    eda.make_all(df)

    train_idx = np.arange(0, TRAIN_END_WEEK)
    val_idx = np.arange(TRAIN_END_WEEK, VAL_END_WEEK)
    trval_idx = np.arange(0, VAL_END_WEEK)          # train+val, for the final refit
    y = df["revenue"].to_numpy()

    print("=== Stage 1-3: hyperparameter search (adstock lambda, Hill, ridge alpha) ===")
    search = full_hyperparam_search(df, train_idx, val_idx, ADSTOCK_LAMBDA_GRID, RIDGE_ALPHA_GRID)
    print("selected lambdas:", search["lambdas"])
    print("selected ridge alpha:", search["ridge_alpha"])

    print("=== final refit on train+val (weeks 1-170) with selected hyperparameters ===")
    # Hill is refit on the larger train+val window (still never touching the holdout) so
    # the production model uses all pre-holdout information, exactly like re-fitting a
    # sklearn model on train+val after hyperparameter selection.
    hill_params_final = _fit_hill_for_lambdas(df, trval_idx, search["lambdas"], y[trval_idx])
    X, feature_names = build_feature_matrix(df, search["lambdas"], hill_params_final)
    ridge_model = fit_ridge(X.iloc[trval_idx], y[trval_idx], search["ridge_alpha"])
    baseline_model, baseline_features = fit_ols_baseline(df, trval_idx)
    X_baseline, _ = build_baseline_matrix(df)

    fitted_trval = ridge_model.predict(X.iloc[trval_idx])
    residuals_trval = y[trval_idx] - fitted_trval

    print("=== diagnostics (VIF / Ljung-Box / Chow / QQ) on train+val fit ===")
    vif_df = diagnostics.compute_vif(X.iloc[trval_idx])
    lb = diagnostics.ljung_box_test(residuals_trval, lags=10)
    midpoint = len(trval_idx) // 2
    chow = diagnostics.chow_test(X.iloc[trval_idx], y[trval_idx], midpoint)
    qq = diagnostics.qq_data(residuals_trval)
    bp = diagnostics.breusch_pagan(residuals_trval, fitted_trval)

    vif_numeric = vif_df["vif"].dropna()
    diag_summary = {
        "vif": vif_df.to_dict(orient="records"),
        "vif_max": float(vif_numeric.max()),
        "vif_threshold": VIF_THRESHOLD,
        "vif_all_below_threshold": bool((vif_numeric < VIF_THRESHOLD).all()),
        "ljung_box": lb.reset_index().to_dict(orient="records"),
        "chow_test": chow,
        "chow_split_reason": (
            "Robyn simulated panel is 2015-11-23 to 2019-11-11 (entirely pre-COVID); "
            "no COVID period exists in this data, so a defensible midpoint split "
            f"(row {midpoint} of the train+val window) is used instead, per "
            "MMM_Project_Outline's own fallback instruction."
        ),
        "qq": qq,
        "breusch_pagan": bp,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(diag_summary, indent=2, default=str))

    print("=== decomposition / ROAS / budget optimization (train+val window) ===")
    decomp = decompose_revenue(ridge_model, X.iloc[trval_idx])
    waterfall = revenue_waterfall(decomp)
    roas = channel_roas(decomp, df[MEDIA_CHANNELS].iloc[trval_idx])

    current_spend = {c: float(df[c].iloc[trval_idx].mean()) for c in MEDIA_CHANNELS}
    budget_result = optimize_budget(
        ridge_model, feature_names, hill_params_final, search["lambdas"], current_spend)

    (OUT / "decomposition.json").write_text(json.dumps({
        "waterfall": waterfall, "roas": roas,
    }, indent=2, default=str))
    (OUT / "budget_reallocation.json").write_text(json.dumps(budget_result, indent=2, default=str))

    print("=== saving models + pipeline params ===")
    joblib.dump(ridge_model, MODELS / "ridge_model.joblib")
    joblib.dump(baseline_model, MODELS / "ols_baseline.joblib")
    pipeline_params = {
        "lambdas": search["lambdas"],
        "lambda_trace": search["lambda_trace"],
        "ridge_alpha": search["ridge_alpha"],
        "ridge_alpha_scores": search["ridge_alpha_scores"],
        "hill_params": hill_params_final,
        "feature_names": feature_names,
        "baseline_features": baseline_features,
        "train_end_week": TRAIN_END_WEEK,
        "val_end_week": VAL_END_WEEK,
    }
    (OUT / "pipeline_params.json").write_text(json.dumps(pipeline_params, indent=2, default=str))

    elapsed = time.time() - t0
    print(f"pipeline complete in {elapsed:.1f}s. VIF max = {diag_summary['vif_max']:.2f} "
         f"(threshold {VIF_THRESHOLD}); Chow p-value = {chow.get('p_value')}; "
         f"Ljung-Box min p (lag<=10) = {lb['lb_pvalue'].min():.4f}")
    print("Run evaluate.py next for the untouched-holdout metrics.")


if __name__ == "__main__":
    main()
