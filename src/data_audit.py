"""
Stage 0 — data audit + validated panel construction.  # -- scaffolding --

Runs BEFORE any modelling stage (per run_pipeline.py). Fills the gap the brief left open:
the brief jumps straight to "Stage 1 — Adstock" and assumes a clean panel exists. This
module is what makes that assumption true, and it documents every coercion/decision it
makes to `outputs/data_audit_summary.json` so nothing is silently imputed.

Schema mapping (Robyn `dt_simulated_weekly.csv` -> brief's illustrative feature list):
    DATE                -> weekly date index (continuity-checked, no gaps allowed)
    revenue              -> target
    tv_S, ooh_S, print_S, facebook_S, search_S
                          -> the brief's "5 media channels", literal spend (config.MEDIA_CHANNELS)
    facebook_I            -> Facebook impressions (exposure, not spend -> linear control)
    search_clicks_P        -> paid-search clicks (brief's "Paid Search clicks" -> linear control)
    competitor_sales_B    -> competitor sales (brief has no analogue; kept as a linear control
                             — omitting it would push its explanatory power onto media coefficients)
    newsletter             -> owned-channel send volume (not paid media -> linear control)
    events                -> categorical, "na" for 206/208 weeks, {"event1","event2"} once each
                             -> becomes `is_event` boolean (the brief's "Holiday flags"); the two
                             raw event weeks are also kept as one-hot dummies since regression can
                             use finer information than a single binary flag
The brief's "Promo discount depth" and "Price index" have no counterpart in this dataset —
Robyn's simulated panel does not include a promo/price series. Documented as a scope gap,
not silently invented: the model has no promo/price control, so absorption of any pricing
effect will land on `competitor_sales_B`/seasonality/trend, noted in TECHNICAL_JOURNEY.md.

Zero-spend weeks (spend columns are genuinely 0 for many weeks — pulsed media, confirmed
via a raw scan: tv_S 116/208 zero weeks, ooh_S 123/208, print_S 121/208, facebook_S
107/208, search_S 32/208) are a real business signal ("this channel bought nothing this
week") and MUST NOT be imputed/filled — a zero adstock/spend feeds correctly into the Hill
saturation curve (Saturated(0) = 0) and the Ridge fit. This module explicitly checks that
zero != missing by requiring zero-spend cells to be non-null; a NULL in a spend column
(never observed in the raw file, but checked defensively) is treated as true missing data
and would raise, not get silently filled with 0 or a mean.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import (
    DATE_COL, EVENT_COL, EXTRA_NUMERIC_CONTROLS, MEDIA_CHANNELS,
    N_WEEKS_EXPECTED, OUT, PROC, RAW, REVENUE_COL,
)

NUMERIC_COLS = [REVENUE_COL] + MEDIA_CHANNELS + EXTRA_NUMERIC_CONTROLS


def _profile(df: pd.DataFrame) -> dict:
    prof = {}
    for c in df.columns:
        col = df[c]
        entry = {
            "dtype_raw": str(col.dtype),
            "n_missing": int(col.isna().sum()),
            "pct_missing": float(col.isna().mean()),
        }
        if c in NUMERIC_COLS:
            numeric = pd.to_numeric(col, errors="coerce")
            entry["n_zero"] = int((numeric == 0).sum())
            entry["pct_zero"] = float((numeric == 0).mean())
            entry["min"] = float(numeric.min())
            entry["max"] = float(numeric.max())
            entry["mean"] = float(numeric.mean())
        prof[c] = entry
    return prof


def _validate_temporal_continuity(dates: pd.Series) -> dict:
    """Assert the 208-week index has no gaps: every consecutive pair is exactly 7 days."""
    diffs = dates.diff().dropna()
    gap_days = diffs.dt.days
    bad = gap_days[gap_days != 7]
    is_continuous = bool((bad.empty))
    assert is_continuous, (
        f"non-weekly gaps found at positions {bad.index.tolist()}: "
        f"day deltas {bad.tolist()}")
    assert len(dates) == N_WEEKS_EXPECTED, (
        f"expected {N_WEEKS_EXPECTED} weeks, got {len(dates)}")
    return {
        "n_weeks": int(len(dates)),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "all_gaps_exactly_7_days": is_continuous,
        "min_gap_days": int(gap_days.min()) if len(gap_days) else None,
        "max_gap_days": int(gap_days.max()) if len(gap_days) else None,
    }


def run() -> pd.DataFrame:
    raw_path = RAW / "dt_simulated_weekly.csv"
    df = pd.read_csv(raw_path)

    decisions: list[str] = []
    raw_profile = _profile(df)

    # --- dtype coercion ---------------------------------------------------------
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d")
    decisions.append(f"coerced {DATE_COL} to datetime64[ns] (format %Y-%m-%d)")

    for c in NUMERIC_COLS:
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        after_na = df[c].isna().sum()
        if after_na > before_na:
            raise ValueError(
                f"column {c}: {after_na - before_na} values failed numeric "
                "coercion (would have been silently NaN-filled) — aborting per "
                "the 'zero != missing, never silently impute' rule")
        decisions.append(f"coerced {c} to numeric (float); {int(after_na)} true NaN found")

    # --- zero-vs-missing guard for spend columns --------------------------------
    for c in MEDIA_CHANNELS:
        n_missing = int(df[c].isna().sum())
        n_zero = int((df[c] == 0).sum())
        if n_missing > 0:
            raise ValueError(
                f"media channel {c} has {n_missing} true-missing weeks — this "
                "pipeline refuses to auto-impute media spend (would corrupt "
                "adstock/saturation); investigate the raw source instead")
        decisions.append(
            f"{c}: {n_zero}/{len(df)} weeks are genuine zero-spend (pulsed media) — "
            "kept as 0, NOT imputed; these are meaningful baseline observations")

    # --- temporal continuity -----------------------------------------------------
    continuity = _validate_temporal_continuity(df[DATE_COL])
    decisions.append(
        f"validated weekly continuity: {continuity['n_weeks']} weeks, "
        f"{continuity['date_min']} -> {continuity['date_max']}, "
        "every consecutive gap == 7 days (no missing/duplicated weeks)")

    # --- holiday / event flags (brief's "Holiday flags") -------------------------
    df[EVENT_COL] = df[EVENT_COL].fillna("na").astype(str)
    df["is_event"] = (df[EVENT_COL] != "na").astype(bool)
    event_dummies = pd.get_dummies(df[EVENT_COL], prefix="event")
    event_dummies = event_dummies.drop(columns=[c for c in event_dummies.columns
                                                  if c.endswith("_na")], errors="ignore")
    n_events = int(df["is_event"].sum())
    decisions.append(
        f"events: {EVENT_COL}=='na' -> is_event=False ({len(df) - n_events}/{len(df)} weeks); "
        f"{n_events} weeks flagged as promo/holiday events "
        f"({sorted(df.loc[df['is_event'], EVENT_COL].unique().tolist())}); "
        "one-hot event dummies kept alongside the boolean flag for finer signal")

    # --- seasonality controls (Fable's data-pipeline responsibility per the brief's
    # division-of-work table: "Data pipeline + seasonality dummies") -------------
    df["month"] = df[DATE_COL].dt.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df["trend"] = np.arange(len(df), dtype=float)  # linear time trend control
    decisions.append(
        "added month-of-year dummies (11, month=1 dropped as reference) and a linear "
        "trend index as seasonality/base-demand controls, appended to the panel")

    panel = pd.concat(
        [df[[DATE_COL, REVENUE_COL] + MEDIA_CHANNELS + EXTRA_NUMERIC_CONTROLS +
            ["is_event", "trend"]],
         event_dummies, month_dummies],
        axis=1,
    )

    # --- promo/price scope gap, documented not invented --------------------------
    decisions.append(
        "brief's 'Promo discount depth' and 'Price index' have no counterpart in the "
        "Robyn simulated panel; NOT fabricated — omitted, with its absorption risk "
        "(onto competitor_sales_B / seasonality / trend) noted in TECHNICAL_JOURNEY.md")

    out_path = PROC / "mmm_panel.parquet"
    panel.to_parquet(out_path, index=False)

    summary = {
        "raw_shape": list(df.shape),
        "panel_shape": list(panel.shape),
        "raw_column_profile": raw_profile,
        "temporal_continuity": continuity,
        "media_channels": MEDIA_CHANNELS,
        "extra_numeric_controls": EXTRA_NUMERIC_CONTROLS,
        "decisions": decisions,
        "output_panel": str(out_path),
    }
    (OUT / "data_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"data_audit: {panel.shape[0]} weeks x {panel.shape[1]} cols -> {out_path}")
    print(f"summary -> {OUT / 'data_audit_summary.json'}")
    return panel


if __name__ == "__main__":
    run()
