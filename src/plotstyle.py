"""
Shared matplotlib style for figures / app / book.  # -- scaffolding --
Same palette-formula approach as project-2: fixed categorical slot per channel (color
follows the same channel across every chart), light surface, thin marks, muted axis ink.
"""
from __future__ import annotations

import matplotlib as mpl

# fixed categorical slot per media channel (never re-assigned when a chart drops one)
CHANNEL_COLORS = {
    "tv_S": "#2a78d6",         # blue
    "ooh_S": "#1baf7a",        # aqua
    "print_S": "#eda100",      # yellow
    "facebook_S": "#4a3aa7",   # violet
    "search_S": "#e34948",     # red
}
CHANNEL_LABELS = {
    "tv_S": "TV", "ooh_S": "OOH", "print_S": "Print",
    "facebook_S": "Facebook", "search_S": "Search",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BASE_COLOR = "#898781"     # base/organic revenue in waterfall charts
SEQ_BLUE = "#2a78d6"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "figure.dpi": 150,
    })
