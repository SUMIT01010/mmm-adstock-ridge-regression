"""
Single entry point: regenerate every figure (EDA + post-model diagnostics/results).
# -- scaffolding --

Run AFTER run_pipeline.py + evaluate.py:  .venv/bin/python make_figures.py
Writes every figure to outputs/figures/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import joblib

from config import FIGS, MODELS, OUT, PROC, VAL_END_WEEK
from src import diagnostics, eda
from src.diagnostics_viz import (
    plot_adstock_vs_raw, plot_budget_reallocation, plot_channel_roas, plot_chow_break,
    plot_hill_curves, plot_residual_diagnostics, plot_revenue_waterfall, plot_vif_bar,
)
from src.ridge_pipeline import adstock_transform, build_feature_matrix


def main() -> None:
    df = pd.read_parquet(PROC / "mmm_panel.parquet")
    params = json.loads((OUT / "pipeline_params.json").read_text())
    diag = json.loads((OUT / "diagnostics.json").read_text())
    decomp = json.loads((OUT / "decomposition.json").read_text())
    budget = json.loads((OUT / "budget_reallocation.json").read_text())

    lambdas = {k: float(v) for k, v in params["lambdas"].items()}
    hill_params = params["hill_params"]

    print("=== Stage 0 EDA figures ===")
    eda.make_all(df)

    print("=== adstock vs raw ===")
    adstocked = adstock_transform(df, lambdas)
    plot_adstock_vs_raw(df, adstocked, lambdas)

    print("=== Hill saturation curves ===")
    trval = np.arange(0, VAL_END_WEEK)
    plot_hill_curves(adstocked.iloc[trval], hill_params)

    print("=== VIF bar chart ===")
    vif_df = pd.DataFrame(diag["vif"])
    vif_df["vif"] = vif_df["vif"].astype(float)
    plot_vif_bar(vif_df.dropna(subset=["vif"]))

    print("=== residual diagnostics panel ===")
    ridge_model = joblib.load(MODELS / "ridge_model.joblib")
    X, _ = build_feature_matrix(df, lambdas, hill_params)
    y = df["revenue"].to_numpy()
    fitted = ridge_model.predict(X.iloc[trval])
    residuals = y[trval] - fitted
    plot_residual_diagnostics(residuals, fitted)

    print("=== Chow structural-break visualization ===")
    midpoint = len(trval) // 2
    plot_chow_break(df["DATE"].iloc[trval], y[trval], midpoint, diag["chow_test"])

    print("=== revenue decomposition waterfall ===")
    plot_revenue_waterfall(decomp["waterfall"])

    print("=== channel ROAS ===")
    plot_channel_roas(decomp["roas"])

    print("=== budget reallocation before/after ===")
    plot_budget_reallocation(budget)

    figs = sorted(p.name for p in FIGS.glob("*.png"))
    print(f"\n{len(figs)} figures written to {FIGS}:")
    for f in figs:
        print(" -", f)


if __name__ == "__main__":
    main()
