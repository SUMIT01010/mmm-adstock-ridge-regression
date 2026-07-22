# Marketing Mix Modelling — Adstock-Ridge Regression

## Objective

A brand spends across TV, OOH, Print, Facebook, and Search. The CFO wants to know which
rupee of spend drove how much incremental revenue, so budget can be reallocated to the
channels that actually move revenue instead of spread evenly or by habit. Plain OLS on
raw weekly spend fails at this: media effects **carry over** across weeks (a TV flight
this week still moves revenue two months out) and spend has **diminishing returns**
(doubling budget does not double the lift) — both distort raw-spend coefficients, and
correlated media flights (multiple channels running simultaneously) make coefficients
unstable on top of that. Interpretability is the product here, not a black-box score.

## Approach

Three-stage econometric pipeline, each stage implemented from its definition (no MMM
library):

1. **Geometric adstock** (`src/adstock.py`) — `Adstock(t) = spend(t) + λ·Adstock(t-1)`,
   λ per channel selected by coordinate-descent grid search against **validation MAPE**.
2. **Hill saturation** (`src/saturation.py`) — `Saturated(x) = x^α/(x^α+K^α)`, α/K fit per
   channel via `scipy.optimize.curve_fit` against a **partial-residual target** (isolates
   each channel's own marginal relationship with revenue before curve-fitting its shape —
   see TECHNICAL_JOURNEY.md Iteration 2 for why a direct fit against raw revenue failed).
3. **Ridge regression** (`src/ridge_pipeline.py`) — `StandardScaler + Ridge` on
   `[Hill(Adstock(spend_c))]` + seasonality/event/trend/other-numeric controls, α selected
   by grid search on **validation MAPE**. A raw-OLS baseline (untransformed spend, same
   controls) is fit alongside for the "why adstock/saturation matters" comparison.

Downstream: `src/roas.py` (zero-out revenue decomposition -> channel ROAS),
`src/budget_optimizer.py` (steady-state constrained LP reallocation, `scipy.optimize`),
`src/diagnostics.py` (VIF / Ljung-Box / Chow / QQ+heteroscedasticity).

**Metric protocol** — three-way temporal split (train weeks 1-150 / validation 151-170 /
holdout 171-208, holdout touched exactly once): see `config.py` for the exact indices.

## Results

Holdout (weeks 171-208, 38 weeks), reported once, from `outputs/metrics.json`:

| Model | Adj R² | MAPE | Meets R²>0.85 target | Meets MAPE<8% target |
|---|---|---|---|---|
| **Adstock-Ridge** (this project) | 0.610 | 12.7% | No | No |
| Raw-OLS baseline | 0.695 | 8.6% | No | No (close) |

**Neither model hits the brief's targets on this dataset, reported honestly.** The
raw-OLS baseline predicts the holdout slightly better than the adstock+Hill+Ridge model —
the opposite of the brief's framing. This is a real, investigated finding (see
TECHNICAL_JOURNEY.md Iterations 2-3), not a bug: the brief's case for adstock/Hill is
**avoiding misattribution among correlated channels**, not necessarily beating a flexible
unconstrained OLS fit on pure holdout MAPE on a 38-week window. Evidence for the
misattribution story: `outputs/diagnostics.json` — post-transform VIF is still above the
5.0 target for `search_S_hill` (17.5), `competitor_sales_B` (14.3) and `trend` (10.1) on
this panel (11 month dummies + a linear trend over only ~4 years of weekly data is itself
collinear — documented, not swept under the rug); Ljung-Box does not reject residual
autocorrelation at lag ≤10 (min p = 0.075, borderline); Chow test at the dataset's
pre-COVID midpoint finds **no** structural break (p = 0.91 — expected, this panel predates
COVID entirely, see below).

Selected hyperparameters (train 1-150 -> validate 151-170, refit on 1-170 for holdout):

| Channel | λ (adstock decay) | Hill α | Hill K |
|---|---|---|---|
| TV | 0.05 | 0.61 | 111,842 |
| OOH | 0.10 | 1.50 | 1,080 |
| Print | 0.20 | 3.00 | 11,612 |
| Facebook | 0.80 | 3.00 | 21,306 |
| Search | 0.80 | 2.42 | 30,713 |

Ridge α = 0.3 (selected on validation MAPE, grid `[0.01 .. 1000]`).

Revenue decomposition + channel ROAS (`outputs/decomposition.json`, fit window):

| Channel | Revenue share | ROAS |
|---|---|---|
| Base (organic) | 72.7% | — |
| TV | 8.1% | 9.25x |
| Search | 14.5% | 51.2x |
| OOH | 2.2% | 0.82x |
| Print | 2.0% | 9.45x |
| Facebook | 0.4% | 3.48x |

Search's 51x ROAS is a genuine model output, flagged as an outlier in
TECHNICAL_JOURNEY.md — Search has low absolute spend (₹0.9M of the fit window's total)
and a Hill curve fit that attributes it disproportionate marginal lift; a real MMM
deployment would sanity-check this against a marketing-science prior before acting on it.

## Environment (zero to running)

```bash
cd Regression
uv venv --python 3.11
uv pip install -r requirements.txt
```

System Python is untouched; everything lives in `./.venv` (pinned to 3.11, see
`../skills/uv-python311-ml-env-on-mac.md`). No torch, no xgboost/lightgbm anywhere in this
project — pure numpy/pandas/scikit-learn/scipy/statsmodels.

## Pipeline (run in order)

```bash
.venv/bin/python download_data.py    # Robyn simulated weekly panel, 208 weeks     (~2 s)
.venv/bin/python run_pipeline.py     # Stage 0-3 + diagnostics + ROAS + budget opt  (~10 s)
.venv/bin/python evaluate.py         # PRIMARY: untouched-holdout Adj R2/MAPE       (~2 s)
.venv/bin/python make_figures.py     # every figure in outputs/figures/             (~15 s)
```

All numbers written to `outputs/*.json`. `run_pipeline.py` never touches weeks 171-208;
`evaluate.py` is the only script that does, and only once.

## Serving

```bash
.venv/bin/streamlit run app.py --server.headless true
# revenue decomposition waterfall, channel ROAS, budget reallocation before/after,
# diagnostics summary — reads outputs/*.json + data/processed/mmm_panel.parquet
```

Verified: launches headless, `/_stcore/health` -> HTTP 200, main page -> HTTP 200, no
exceptions in the server log.

Docker: `docker build -t regression-mmm . && docker run -p 8501:8501 regression-mmm`
(authored, read-through verified for path correctness; **not built** — no Docker Desktop
on this machine, per program convention).

## Layout

| Path | What | Ownership |
|---|---|---|
| `src/adstock.py` | geometric adstock transform + per-channel λ grid search | **CORE** |
| `src/saturation.py` | Hill saturation curve fit (partial-residual target) | **CORE** |
| `src/ridge_pipeline.py` | feature assembly, coordinate-descent λ search, Ridge α search, final fit | **CORE** |
| `src/roas.py` | zero-out revenue decomposition + channel ROAS | **CORE** |
| `src/budget_optimizer.py` | steady-state constrained LP budget reallocation (`scipy.optimize`) | **CORE** |
| `src/data_audit.py` | Stage 0 profiling, temporal-continuity check, zero-vs-missing guard | scaffolding |
| `src/diagnostics.py` | VIF, Ljung-Box, Chow test, QQ/heteroscedasticity | scaffolding |
| `src/eda.py` | Stage 0 visuals (time series, zero-spend heatmap, correlation matrix) | scaffolding |
| `src/diagnostics_viz.py` | post-model visuals (adstock/Hill/VIF/residuals/waterfall/ROAS/budget) | scaffolding |
| `src/plotstyle.py` | shared matplotlib style | scaffolding |
| `config.py`, `download_data.py`, `run_pipeline.py`, `evaluate.py`, `make_figures.py`, `app.py`, `Dockerfile` | wiring / serving / figures | scaffolding |
| `outputs/`, `data/` | metrics JSONs, fitted models, figures, processed panel (generated by the pipeline) | — |
| `book/` | technical-book PDF (math + diagnostics deep-dive) | — |

## Data source (open, no registration)

Robyn Sample Data (`dt_simulated_weekly`), Meta's open-source MMM package. The R package
only ships this table as a binary `.RData` file; the identical panel is available as a
plain CSV in Robyn's own Python-port tutorial resources — verified live, 208 rows, public
`raw.githubusercontent.com` URL, no auth wall (see `download_data.py` for the exact URL
and `data/processed/data_manifest.json` for the SHA-256 + shape record). Schema: `DATE`,
`revenue`, 5 media spend channels (`tv_S`, `ooh_S`, `print_S`, `facebook_S`, `search_S`),
plus `facebook_I` (impressions), `search_clicks_P` (clicks), `competitor_sales_B`,
`newsletter`, and `events` (2 promo/holiday weeks out of 208). The brief's illustrative
"Promo discount depth" / "Price index" features have no counterpart in this dataset and
are not fabricated — see `src/data_audit.py` header for the full schema mapping and scope
note.
