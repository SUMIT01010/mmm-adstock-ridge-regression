"""
Stage 0 — exploratory / audit-time visuals.  # -- scaffolding --

Three figures, mirroring the shape of Clustering's Section-4 visualization table adapted
for MMM/time-series data:
  1. per-channel raw spend-vs-revenue time series (208 weeks) with event/holiday markers
  2. spend / zero-spend heatmap across channels x weeks (makes the pulsed-media pattern
     found in data_audit.py visible, motivates why zero-spend != missing)
  3. raw-channel correlation matrix (motivates why plain OLS on raw spend misattributes —
     correlated media flights get their coefficients confused)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGS, MEDIA_CHANNELS, REVENUE_COL
from src.plotstyle import CHANNEL_COLORS, CHANNEL_LABELS, GRID, INK, INK_2, MUTED, SURFACE, apply_style

apply_style()


def plot_channel_timeseries(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(len(MEDIA_CHANNELS) + 1, 1, figsize=(11, 2.0 * (len(MEDIA_CHANNELS) + 1)),
                             sharex=True)
    dates = df["DATE"]

    ax = axes[0]
    ax.plot(dates, df[REVENUE_COL], color=INK, linewidth=1.4)
    ax.set_ylabel("revenue", fontsize=8)
    ax.set_title("Weekly revenue and media spend, 208 weeks (Robyn simulated panel)",
                 fontsize=11, color=INK, loc="left")
    event_weeks = df.loc[df["is_event"], "DATE"]
    for w in event_weeks:
        ax.axvline(w, color="#e34948", linewidth=1.0, linestyle="--", alpha=0.8)

    for i, ch in enumerate(MEDIA_CHANNELS):
        ax = axes[i + 1]
        ax.plot(dates, df[ch], color=CHANNEL_COLORS[ch], linewidth=1.2)
        ax.set_ylabel(CHANNEL_LABELS[ch], fontsize=8)
        for w in event_weeks:
            ax.axvline(w, color="#e34948", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("week")
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "eda_channel_timeseries.png", bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_zero_spend_heatmap(df: pd.DataFrame, save: bool = True):
    mat = (df[MEDIA_CHANNELS] > 0).astype(int).to_numpy().T  # 1 = spent, 0 = zero-spend
    fig, ax = plt.subplots(figsize=(11, 2.6))
    im = ax.imshow(mat, aspect="auto", cmap="Greens", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(MEDIA_CHANNELS)))
    ax.set_yticklabels([CHANNEL_LABELS[c] for c in MEDIA_CHANNELS], fontsize=9)
    ax.set_xlabel("week index")
    ax.set_title("Spend (green) vs zero-spend (white) by channel x week — pulsed media, "
                 "not missing data", fontsize=10, loc="left")
    for i, ch in enumerate(MEDIA_CHANNELS):
        pct_zero = 100 * (1 - mat[i].mean())
        ax.annotate(f"{pct_zero:.0f}% zero", xy=(1.005, i), xycoords=("axes fraction", "data"),
                    fontsize=7, color=INK_2, va="center")
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "eda_zero_spend_heatmap.png", bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_raw_correlation_matrix(df: pd.DataFrame, save: bool = True):
    cols = MEDIA_CHANNELS + [REVENUE_COL]
    corr = df[cols].corr()
    labels = [CHANNEL_LABELS.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.values[i, j]) > 0.5 else INK)
    ax.set_title("Raw-channel correlation matrix (motivates Ridge over plain OLS)",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "eda_raw_correlation_matrix.png", bbox_inches="tight")
        plt.close(fig)
    return fig


def make_all(df: pd.DataFrame) -> list[str]:
    plot_channel_timeseries(df)
    plot_zero_spend_heatmap(df)
    plot_raw_correlation_matrix(df)
    return ["eda_channel_timeseries.png", "eda_zero_spend_heatmap.png",
            "eda_raw_correlation_matrix.png"]


if __name__ == "__main__":
    panel = pd.read_parquet("data/processed/mmm_panel.parquet")
    files = make_all(panel)
    print("wrote:", files)
