"""
Streamlit serving app — MMM results dashboard.  # -- scaffolding --

Shows revenue decomposition, channel ROAS, and budget reallocation (before/after) from
the artifacts written by run_pipeline.py / evaluate.py / make_figures.py. Read-only
dashboard (no live refit — refitting is run_pipeline.py's job); loads outputs/*.json.

Run:  .venv/bin/streamlit run app.py --server.headless true
Artifacts must exist first:  .venv/bin/python run_pipeline.py && .venv/bin/python evaluate.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import MEDIA_CHANNELS, OUT, PROC
from src.plotstyle import BASE_COLOR, CHANNEL_COLORS, CHANNEL_LABELS, INK_2, MUTED, apply_style

apply_style()
st.set_page_config(page_title="MMM · Adstock-Ridge", layout="wide")


@st.cache_data
def load_json(name: str) -> dict | None:
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_panel() -> pd.DataFrame:
    p = PROC / "mmm_panel.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


metrics = load_json("metrics.json")
decomp = load_json("decomposition.json")
budget = load_json("budget_reallocation.json")
diag = load_json("diagnostics.json")
panel = load_panel()

st.title("Marketing Mix Model — Adstock-Ridge Regression")
st.caption("Robyn Sample Data (Meta OSS), 208 weeks, 5 media channels. "
          "Artifacts from run_pipeline.py / evaluate.py.")

if metrics is None or decomp is None or budget is None:
    st.error("No pipeline outputs found — run `.venv/bin/python run_pipeline.py` then "
            "`.venv/bin/python evaluate.py` first.")
    st.stop()

# ------------------------------------------------------------------------ headline metrics
c1, c2, c3, c4 = st.columns(4)
r = metrics["adstock_ridge"]
b = metrics["raw_ols_baseline"]
c1.metric("Holdout Adj R2 (Adstock-Ridge)", f"{r['adj_r2']:.3f}",
         help=f"target > {metrics['targets']['adj_r2']}")
c2.metric("Holdout MAPE (Adstock-Ridge)", f"{r['mape']*100:.1f}%",
         help=f"target < {metrics['targets']['mape']*100:.0f}%")
c3.metric("Holdout Adj R2 (raw OLS baseline)", f"{b['adj_r2']:.3f}")
c4.metric("Holdout MAPE (raw OLS baseline)", f"{b['mape']*100:.1f}%")
st.caption(metrics["comparison_note"])

st.divider()

# --------------------------------------------------------------------- revenue decomposition
st.subheader("Revenue decomposition")
left, right = st.columns([1, 1])
with left:
    waterfall = decomp["waterfall"]
    labels = ["Base"] + [CHANNEL_LABELS[c] for c in MEDIA_CHANNELS]
    shares = [waterfall["shares"]["base"]] + [waterfall["shares"][c] for c in MEDIA_CHANNELS]
    colors = [BASE_COLOR] + [CHANNEL_COLORS[c] for c in MEDIA_CHANNELS]
    fig, ax = plt.subplots(figsize=(6, 4))
    cum = 0.0
    for i, (lab, s, col) in enumerate(zip(labels, shares, colors)):
        ax.bar(lab, s, bottom=cum if i > 0 else 0, color=col)
        ax.annotate(f"{s*100:.1f}%", xy=(i, cum + s / 2), ha="center", va="center",
                   fontsize=8, color="white" if s > 0.05 else INK_2)
        cum += s
    ax.set_ylabel("share of predicted revenue")
    st.pyplot(fig, use_container_width=True)
with right:
    st.dataframe(pd.DataFrame({
        "component": labels,
        "share_of_revenue": [f"{s*100:.2f}%" for s in shares],
    }), hide_index=True, use_container_width=True)
    st.caption(f"Total predicted revenue over the fit window: "
              f"₹{waterfall['total_revenue']:,.0f}")

st.divider()

# ------------------------------------------------------------------------------- channel ROAS
st.subheader("Channel ROAS")
roas = decomp["roas"]
left, right = st.columns([1, 1])
with left:
    vals = [roas[c]["roas"] for c in MEDIA_CHANNELS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([CHANNEL_LABELS[c] for c in MEDIA_CHANNELS], vals,
          color=[CHANNEL_COLORS[c] for c in MEDIA_CHANNELS])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}x", xy=(i, v), ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color=INK_2, linewidth=1.0, linestyle="--")
    ax.set_ylabel("ROAS")
    st.pyplot(fig, use_container_width=True)
with right:
    roas_df = pd.DataFrame([
        {"channel": CHANNEL_LABELS[c], "incremental_revenue": roas[c]["incremental_revenue"],
         "total_spend": roas[c]["total_spend"], "roas": roas[c]["roas"]}
        for c in MEDIA_CHANNELS
    ])
    st.dataframe(roas_df.round(2), hide_index=True, use_container_width=True)

st.divider()

# ----------------------------------------------------------------- budget reallocation
st.subheader("Budget reallocation (same total budget)")
left, right = st.columns([1, 1])
with left:
    current = [budget["current_spend"][c] for c in MEDIA_CHANNELS]
    optimal = [budget["optimal_spend"][c] for c in MEDIA_CHANNELS]
    x = np.arange(len(MEDIA_CHANNELS)); width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width / 2, current, width, label="current", color=MUTED)
    ax.bar(x + width / 2, optimal, width, label="optimal",
          color=[CHANNEL_COLORS[c] for c in MEDIA_CHANNELS])
    ax.set_xticks(x); ax.set_xticklabels([CHANNEL_LABELS[c] for c in MEDIA_CHANNELS])
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)
with right:
    st.metric("Predicted lift (steady-state)", f"{budget['predicted_lift_pct']:+.1f}%")
    realloc_df = pd.DataFrame({
        "channel": [CHANNEL_LABELS[c] for c in MEDIA_CHANNELS],
        "current_weekly_spend": current,
        "optimal_weekly_spend": optimal,
    })
    st.dataframe(realloc_df.round(0), hide_index=True, use_container_width=True)
    st.caption("Steady-state optimization (constant weekly spend -> converged adstock -> "
              "Hill saturation); see src/budget_optimizer.py for the full method note. "
              "Caps at 3x historical max per channel.")

st.divider()

# ---------------------------------------------------------------------------- diagnostics
if diag is not None:
    st.subheader("Model diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max VIF (post-transform)", f"{diag['vif_max']:.1f}",
             help=f"target < {diag['vif_threshold']}")
    c2.metric("Chow test p-value", f"{diag['chow_test'].get('p_value', float('nan')):.3f}")
    lb_pvals = [row["lb_pvalue"] for row in diag["ljung_box"]]
    c3.metric("Ljung-Box min p-value (lag<=10)", f"{min(lb_pvals):.3f}")
    st.caption(diag["chow_split_reason"])
