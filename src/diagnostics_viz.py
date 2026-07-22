"""
Post-modelling diagnostic + result visuals.  # -- scaffolding --

Every function takes already-computed arrays/dicts (produced by run_pipeline.py /
evaluate.py) rather than recomputing anything — this module is purely presentation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from config import FIGS, MEDIA_CHANNELS, VIF_THRESHOLD
from src.plotstyle import (
    BASE_COLOR, CHANNEL_COLORS, CHANNEL_LABELS, GRID, INK, INK_2, MUTED, apply_style,
)
from src.saturation import hill_saturation

apply_style()


def plot_adstock_vs_raw(df: pd.DataFrame, adstocked: pd.DataFrame, lambdas: dict, save=True):
    fig, axes = plt.subplots(len(MEDIA_CHANNELS), 1, figsize=(11, 2.0 * len(MEDIA_CHANNELS)),
                             sharex=True)
    dates = df["DATE"]
    for i, ch in enumerate(MEDIA_CHANNELS):
        ax = axes[i]
        ax.plot(dates, df[ch], color=MUTED, linewidth=1.0, label="raw spend")
        ax.plot(dates, adstocked[ch], color=CHANNEL_COLORS[ch], linewidth=1.4,
               label=f"adstocked (lambda={lambdas[ch]:.2f})")
        ax.set_ylabel(CHANNEL_LABELS[ch], fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_title("Adstock-transformed vs raw spend, per channel", fontsize=10, loc="left")
    axes[-1].set_xlabel("week")
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "adstock_vs_raw.png", bbox_inches="tight"); plt.close(fig)
    return fig


def plot_hill_curves(adstocked: pd.DataFrame, hill_params: dict, save=True):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for i, ch in enumerate(MEDIA_CHANNELS):
        ax = axes[i]
        x = adstocked[ch].to_numpy()
        p = hill_params[ch]
        xs = np.linspace(0, max(x.max(), 1) * 1.05, 200)
        ys = hill_saturation(xs, p["alpha"], p["K"])
        ax.plot(xs, ys, color=CHANNEL_COLORS[ch], linewidth=2)
        y_obs = hill_saturation(x, p["alpha"], p["K"])
        ax.scatter(x, y_obs, color=CHANNEL_COLORS[ch], s=10, alpha=0.35)
        ax.axvline(p["K"], color=INK_2, linewidth=0.8, linestyle="--")
        ax.annotate("K", xy=(p["K"], 0.02), fontsize=8, color=INK_2)
        ax.set_title(f"{CHANNEL_LABELS[ch]}  (alpha={p['alpha']:.2f}, K={p['K']:.0f})",
                    fontsize=9)
        ax.set_xlabel("adstocked spend"); ax.set_ylabel("saturated response")
    for j in range(len(MEDIA_CHANNELS), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Hill saturation curves per channel (fitted alpha, K)", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "hill_saturation_curves.png", bbox_inches="tight"); plt.close(fig)
    return fig


def plot_vif_bar(vif_df: pd.DataFrame, save=True):
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(vif_df))))
    colors = ["#e34948" if v > VIF_THRESHOLD else "#1baf7a" for v in vif_df["vif"]]
    ax.barh(vif_df["feature"], vif_df["vif"], color=colors)
    ax.axvline(VIF_THRESHOLD, color=INK_2, linewidth=1.2, linestyle="--")
    ax.annotate(f"VIF={VIF_THRESHOLD}", xy=(VIF_THRESHOLD, len(vif_df) - 1),
               fontsize=8, color=INK_2, xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("VIF"); ax.invert_yaxis()
    ax.set_title("VIF per post-transform predictor", fontsize=10, loc="left")
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "vif_bar.png", bbox_inches="tight"); plt.close(fig)
    return fig


def plot_residual_diagnostics(residuals: np.ndarray, fitted: np.ndarray, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # QQ plot
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax = axes[0]
    ax.scatter(osm, osr, s=14, color="#2a78d6", alpha=0.7)
    ax.plot(osm, slope * np.array(osm) + intercept, color=INK_2, linewidth=1.2)
    ax.set_title(f"QQ plot (r={r:.3f})", fontsize=10)
    ax.set_xlabel("theoretical quantiles"); ax.set_ylabel("sample quantiles")

    # residuals vs fitted
    ax = axes[1]
    ax.scatter(fitted, residuals, s=14, color="#eda100", alpha=0.7)
    ax.axhline(0, color=INK_2, linewidth=1.0, linestyle="--")
    ax.set_title("Residuals vs fitted", fontsize=10)
    ax.set_xlabel("fitted"); ax.set_ylabel("residual")

    # ACF of residuals (Ljung-Box visual)
    ax = axes[2]
    plot_acf(residuals, ax=ax, lags=min(20, len(residuals) // 2 - 1), color="#1baf7a")
    ax.set_title("Residual autocorrelation (Ljung-Box)", fontsize=10)

    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "residual_diagnostics.png", bbox_inches="tight"); plt.close(fig)
    return fig


def plot_chow_break(dates: pd.Series, revenue: np.ndarray, split_idx: int, chow: dict, save=True):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(dates, revenue, color=INK, linewidth=1.2)
    ax.axvline(dates.iloc[split_idx], color="#e34948", linewidth=1.5, linestyle="--")
    ax.annotate(f"Chow split (week {split_idx})\nF={chow.get('f_stat', float('nan')):.2f}, "
               f"p={chow.get('p_value', float('nan')):.3f}",
               xy=(dates.iloc[split_idx], ax.get_ylim()[1]), fontsize=8, color=INK_2,
               ha="left", va="top", xytext=(6, -6), textcoords="offset points")
    ax.set_title("Structural break test (Chow) split point", fontsize=10, loc="left")
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "chow_break.png", bbox_inches="tight"); plt.close(fig)
    return fig


def plot_revenue_waterfall(waterfall: dict, save=True):
    labels = ["Base"] + [CHANNEL_LABELS[c] for c in MEDIA_CHANNELS]
    shares = [waterfall["shares"]["base"]] + [waterfall["shares"][c] for c in MEDIA_CHANNELS]
    colors = [BASE_COLOR] + [CHANNEL_COLORS[c] for c in MEDIA_CHANNELS]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cum = 0.0
    for i, (lab, s, c) in enumerate(zip(labels, shares, colors)):
        ax.bar(lab, s, bottom=cum if i > 0 else 0, color=c)
        ax.annotate(f"{s*100:.1f}%", xy=(i, cum + s / 2), ha="center", va="center",
                   fontsize=8, color="white" if s > 0.05 else INK_2)
        cum += s
    ax.set_ylabel("share of predicted revenue")
    ax.set_title("Revenue decomposition: base vs incremental per channel", fontsize=10, loc="left")
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "revenue_waterfall.png", bbox_inches="tight"); plt.close(fig)
    return fig


def plot_channel_roas(roas: dict, save=True):
    chs = MEDIA_CHANNELS
    vals = [roas[c]["roas"] for c in chs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([CHANNEL_LABELS[c] for c in chs], vals, color=[CHANNEL_COLORS[c] for c in chs])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}x", xy=(i, v), ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.axhline(1.0, color=INK_2, linewidth=1.0, linestyle="--")
    ax.set_ylabel("ROAS (incremental revenue / spend)")
    ax.set_title("Channel ROAS", fontsize=10, loc="left")
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "channel_roas.png", bbox_inches="tight"); plt.close(fig)
    return fig


def plot_budget_reallocation(opt_result: dict, save=True):
    chs = MEDIA_CHANNELS
    current = [opt_result["current_spend"][c] for c in chs]
    optimal = [opt_result["optimal_spend"][c] for c in chs]
    x = np.arange(len(chs)); width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, current, width, label="current (avg weekly)", color=MUTED)
    ax.bar(x + width / 2, optimal, width, label="optimal (reallocated)",
          color=[CHANNEL_COLORS[c] for c in chs])
    ax.set_xticks(x); ax.set_xticklabels([CHANNEL_LABELS[c] for c in chs])
    ax.set_ylabel("weekly spend")
    lift = opt_result.get("predicted_lift_pct", float("nan"))
    ax.set_title(f"Budget reallocation, same total budget (predicted lift {lift:+.1f}%)",
                fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIGS / "budget_reallocation.png", bbox_inches="tight"); plt.close(fig)
    return fig
