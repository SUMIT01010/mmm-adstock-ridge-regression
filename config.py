"""
Regression project — central configuration.  # --- scaffolding ---

Marketing Mix Modelling (Adstock-Ridge Regression) on the Robyn Sample Data (Meta OSS,
208 weeks). All paths/knobs live here so every other module imports from one place, same
pattern as project-1/project-2 (see ../skills/uv-python311-ml-env-on-mac.md).

No torch, no xgboost/lightgbm anywhere in this project — pure numpy/pandas/scikit-learn/
scipy. The Mac libomm import-order issue documented in the skill file does not apply here.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
MODELS = OUT / "models"
BOOK = ROOT / "book"
for _d in (RAW, PROC, OUT, FIGS, MODELS, BOOK / "figs"):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = min(2, os.cpu_count() or 2)

# ------------------------------------------------------------------------ data source
# Robyn's R package only ships dt_simulated_weekly as an .RData binary (R/data/); the
# Python port of the tutorial ships the identical table as a plain CSV — same 208-week
# simulated panel, verified live (see TECHNICAL_JOURNEY.md Iteration 1). Open, no auth.
ROBYN_CSV_URL = (
    "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/python/src/"
    "robyn/tutorials/resources/dt_simulated_weekly.csv"
)

DATE_COL = "DATE"
REVENUE_COL = "revenue"
# The brief's feature list ("TV GRP", "promo discount depth", "price index") is an
# illustrative sketch, not the literal Robyn schema — see data_audit.py header for the
# 1:1 mapping actually used. Five paid-media spend channels, confirmed non-negotiable
# ("5 media channels" is the brief's hard number):
MEDIA_CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
# Non-spend media signals kept as additional controls (impressions/clicks are exposure
# metrics, not spend — they do not get adstocked/saturated as media, they ride along as
# linear controls exactly like competitor sales / newsletter).
EXTRA_NUMERIC_CONTROLS = ["facebook_I", "search_clicks_P", "competitor_sales_B", "newsletter"]
EVENT_COL = "events"          # categorical; "na" = no event -> becomes the holiday flag
N_WEEKS_EXPECTED = 208

# ------------------------------------------------------------------------------ splits
# Brief (MMM_Project_Outline.md) specifies a 2-way split (train 1-170 / holdout 171-208).
# The task brief given for THIS build refines that into a 3-way temporal split so that
# adstock lambda / Hill alpha,K / Ridge alpha are all selected on a validation slice
# instead of on train error or (worse) on the holdout itself. Holdout is touched exactly
# once, at the very end, in evaluate.py. Documented as a deliberate protocol upgrade in
# TECHNICAL_JOURNEY.md, not a silent deviation.
TRAIN_END_WEEK = 150   # weeks 1-150   (0-indexed rows 0:150)  -> fit adstock/saturation/Ridge
VAL_END_WEEK = 170      # weeks 151-170 (rows 150:170)          -> select lambda/alpha/K/ridge-alpha
# weeks 171-208 (rows 170:208) -> untouched holdout, reported once

# ------------------------------------------------------------------------- adstock grid
ADSTOCK_LAMBDA_GRID = [round(x, 2) for x in
                       [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

# ------------------------------------------------------------------------ Hill / ridge
RIDGE_ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

# ---------------------------------------------------------------------- primary metric
TARGET_ADJ_R2 = 0.85
TARGET_MAPE = 0.08

# ----------------------------------------------------------------- diagnostics config
VIF_THRESHOLD = 5.0
