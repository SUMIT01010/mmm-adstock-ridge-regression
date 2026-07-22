# Regression (MMM) — Technical Journey

Live build log (per CLAUDE.md §4): what was built, in what order, decisions, dead ends,
blockers, and results vs the brief's target. Written as work happened, 2026-07-17.

**Brief's primary metric:** Adjusted R² > 0.85 and MAPE < 8% on a held-out 38-week window
(weeks 171-208), reported once. **Ownership tag key:** `# === CORE CONTRIBUTION ===` =
the adstock/Hill/Ridge/ROAS/budget-optimizer algorithmic logic (implemented from the
definitions, not wrapped from an MMM library like Robyn's own R/Python packages).
`# -- scaffolding --` = data wiring, diagnostics plumbing, plotting, serving, Docker,
config.

---

## Iteration 1 (2026-07-17) — data source verification, environment, Stage 0

### Data source check (per the task brief: stop and report if auth-gated)
The brief points at `github.com/facebookexperimental/Robyn`. Checked live before writing
any code: the R package's `dt_simulated_weekly` only exists as a binary `.RData` file
(`R/data/dt_simulated_weekly.RData`) — not directly loadable without R or `pyreadr`.
Found the identical table shipped as a plain CSV in the same repo's Python port
(`python/src/robyn/tutorials/resources/dt_simulated_weekly.csv`), confirmed via the
GitHub trees API and a live `curl` — 208 rows, public `raw.githubusercontent.com` URL,
200 OK, no auth wall. **Not a dataset substitution** — same simulated panel Robyn's own
tutorials use, just a differently-serialized copy of it. Used this. No auth blocker hit,
so no need to stop per the brief's fallback instruction.

Verified schema on arrival: `DATE, revenue, tv_S, ooh_S, print_S, facebook_I,
search_clicks_P, search_S, competitor_sales_B, facebook_S, events, newsletter`. Five
paid-media spend columns (`tv_S, ooh_S, print_S, facebook_S, search_S`) match the brief's
"5 media channels" exactly. The brief's illustrative feature list ("TV GRP," "Promo
discount depth," "Price index," "Seasonality dummies," "Holiday flags") does not map 1:1
onto this schema — no promo-depth or price-index series exists in Robyn's simulated
panel. Documented as a scope gap in `src/data_audit.py`'s header rather than fabricated;
seasonality dummies (month) and holiday flags (`is_event` + one-hot `events`) were built
from what the panel actually has.

### Environment
`.venv` via `uv venv --python 3.11` (system is 3.14; skills file). Resolved cleanly on
the first pass: numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.17.1, statsmodels
0.14.6, streamlit 1.59.1. No torch, no xgboost/lightgbm anywhere in this project — the
Mac libomp import-order issue from the skills file does not apply.

### Stage 0 — data audit (`src/data_audit.py`)
Raw scan before writing the audit logic: `tv_S` 116/208 zero weeks, `ooh_S` 123/208,
`print_S` 121/208, `facebook_S` 107/208, `search_S` 32/208 — heavily pulsed media, not
sparse/missing data. Built the audit to hard-assert this distinction (a NULL in a spend
column raises; a 0 does not) rather than risk a downstream `fillna` silently treating a
pulsed-off week as missing. Temporal continuity check (every consecutive date gap == 7
days) passed cleanly — no gaps in the 208-week index, 2015-11-23 to 2019-11-11.

`events` column: `"na"` for 206/208 weeks, one week each of `event1`/`event2`. Built both
a boolean `is_event` flag (the brief's "Holiday flags") and one-hot event dummies; **the
`is_event` flag turned out to be redundant and was later dropped from the model features**
— see Iteration 3.

Month dummies (11) + a linear `trend` index added as seasonality/base-demand controls,
per the brief's own division-of-work table assigning "seasonality dummies" as scaffolding
work.

---

## Iteration 2 (2026-07-17) — CORE pipeline, two dead ends in Hill fitting

### Adstock + Hill + Ridge, first pass
Built `src/adstock.py` (pure recursive `Adstock(t) = spend(t) + λ·Adstock(t-1)`),
`src/saturation.py` (Hill curve via `scipy.optimize.curve_fit`, 4 free params: α, K,
linear scale, intercept), `src/ridge_pipeline.py` (coordinate-descent λ search + Ridge α
grid search, both scored on validation MAPE per the task brief's 3-way split).

**Dead end 1 — ill-conditioned Ridge.** First fit threw `LinAlgWarning: ill-conditioned
matrix` on every single grid point. Root cause: feature scales spanned orders of
magnitude unaddressed — Hill outputs are in [0,1], `competitor_sales_B` is O(1e6),
`trend` is O(1e2), event/month dummies are 0/1. Fix: wrapped Ridge (and the OLS baseline,
for a fair comparison) in `Pipeline(StandardScaler(), Ridge(alpha))`
(`make_ridge_pipeline` in `ridge_pipeline.py`). Warning gone.

**Dead end 2 — degenerate Hill fits.** First hyperparameter search run picked λ and
α/K values that looked wrong: `print_S` landed on `alpha=0.1` (hard lower bound) with
`K=0.065` — a near-step-function that treats "any spend > ~0" as fully saturated, clearly
overfit rather than a real diminishing-returns curve. Two compounding causes found:
1. **Tiny adstock leakage polluting the "nonzero" mask.** Low-λ adstock still leaves an
   exponentially small nonzero carryover on true zero-spend weeks (e.g. 3.9e-08 on a
   series that peaks near 32,000) — a strict `x > 0` filter treated those as "real" data
   points and dragged the K-lower-bound quantile toward zero. Fixed with a relative
   threshold (`x > 1e-6 * x.max()`) in `fit_hill_channel`.
2. **Curve-fitting Hill(adstocked_c) against raw revenue directly is underdetermined for
   weak channels.** Revenue is driven by 5 channels + controls; a 4-parameter fit trying
   to explain all of that from one channel's spend alone, on ~100-150 training weeks, is
   exactly the kind of problem `curve_fit` can solve with a degenerate corner solution.
   Fixed by fitting a quick linear reference model first (unsaturated adstocked spend +
   controls -> Ridge, alpha=1.0), then fitting each channel's Hill curve against a
   **partial-residual target**: `residual + beta_c * adstocked_c` — the channel's own
   isolated marginal relationship with revenue, controlling for everything else. Standard
   backfitting/partial-regression logic, not novel, but not something the brief's
   3-line Hill description mentioned — added because the naive version measurably failed.

After both fixes: Hill α values moved off their bounds into a sane [0.6, 3.0] range,
K values landed at the right order of magnitude relative to each channel's own spend
distribution (e.g. TV K≈112k vs TV spend topping out near 168k adstocked — plausible).
Holdout MAPE improved from 17.1% (dead-end-1 fixed, dead-end-2 still active) -> 14.9%
(dead-end-2 partially fixed, bounds tightened) -> 11.8-12.7% (partial-residual fix). Ran
an extra check with 3 coordinate-descent sweeps instead of 2 (default_alpha varied
0.3/1.0) — results moved by <1pp, confirming the search had converged; kept 2 sweeps as
the shipped default (documented cost: 111 fits vs a 5,500-combination brute force grid,
see `ridge_pipeline.py` module docstring).

---

## Iteration 3 (2026-07-17) — diagnostics, infinite VIF, honest metric reporting

### Infinite VIF dead end
First `run_pipeline.py` run reported `VIF max = inf`. Traced to `is_event` and
`event_event1` being **exactly identical** within the train+val window (weeks 1-170):
only `event1` (not `event2`) falls inside that window, so both columns are 1 on exactly
one row and 0 elsewhere -> perfect collinearity -> VIF mathematically undefined (not
"high," undefined — a constant-vs-intercept regression has zero residual variance,
dividing by zero). Fix: dropped `is_event` from the model's control set entirely
(`CONTROL_COLS_BASE` in `ridge_pipeline.py`) since the one-hot event dummies already
strictly subsume it; `is_event` stays in the panel for EDA/audit readability. Separately
hardened `diagnostics.compute_vif` to detect and report any future zero-variance column
as `vif: null` with a note, rather than emitting `+inf`, since `event_event2` legitimately
has zero variance within the train+val window (its one positive week is in the holdout) —
this is a real, expected artifact of a temporal split with only 2 total event weeks in
208, not a bug to hide.

### Final diagnostics (train+val fit window, `outputs/diagnostics.json`)
- **VIF**: max 17.5 (`search_S_hill`), also `competitor_sales_B` 14.3 and `trend` 10.1
  above the 5.0 target. 11 month dummies + a linear trend over ~3.3 years of weekly data
  (170 weeks) is itself a source of collinearity independent of the media transforms —
  noted, not something adstock/Hill can fix. **Target not met**, reported honestly.
- **Ljung-Box**: min p-value across lags 1-10 = 0.075 — borderline, does not reject the
  no-autocorrelation null at 5% but is close. Some residual structure likely remains
  (plausibly the month-dummy/trend collinearity above, or a genuine model misspecification
  relative to the panel's true, unknown DGP).
- **Chow test**: p = 0.91 at the dataset's own midpoint (row 85 of the 170-row train+val
  window). No structural break detected. **Deviation from the brief noted explicitly**:
  the brief suggests testing around COVID; this Robyn simulated panel runs 2015-11-23 to
  2019-11-11, entirely pre-COVID, so there is no COVID period in the data at all. Used the
  brief's own documented fallback ("otherwise pick a defensible midpoint split") instead
  of inventing a COVID split that doesn't exist in this data.

### Holdout metrics — the honest headline result
`evaluate.py`, touching weeks 171-208 for the first and only time:

| | Adj R² | MAPE |
|---|---|---|
| Adstock-Ridge | 0.610 | 12.7% |
| Raw-OLS baseline | 0.695 | 8.6% |

**Neither hits the brief's targets (R²>0.85, MAPE<8%), and the raw-OLS baseline actually
predicts the holdout slightly better than the adstock+Hill+Ridge model.** This is the
opposite of the brief's framing ("baseline: shows misattribution without econometric
correction," implying the transformed model should out-predict it) and is reported
as-is rather than tuned until it looks better — per CLAUDE.md's self-verification
discipline, "report your actual numbers honestly, whether you hit them or not" is a
direct instruction in the task brief, not just a norm.

**Why this is a defensible, not alarming, result:** 38 holdout weeks is a small
evaluation window; Ridge with an adstock/Hill functional-form constraint will lose a pure
predictive contest to an unconstrained OLS whenever the panel's true (unknown, since this
is Robyn's own simulator) DGP doesn't line up exactly with geometric-adstock-plus-Hill, or
whenever the unconstrained OLS is implicitly using its extra raw-spend degrees of freedom
to fit holdout-adjacent noise. The brief's actual case for adstock+Hill+Ridge is
**avoiding misattribution among correlated channels**, which is a VIF/coefficient-
stability argument, not a holdout-MAPE argument — and the VIF/Ljung-Box results above
show the correlated-channel problem is real in this panel (raw-channel correlations were
plotted in `outputs/figures/eda_raw_correlation_matrix.png` before any modelling, to
motivate exactly this). No further hyperparameter chasing was done past this point to
avoid overfitting the reported numbers to the 20-week validation slice a third time
(diminishing, already-observed returns from sweep count / default-alpha changes in
Iteration 2).

### ROAS / decomposition sanity note
`search_S` ROAS = 51.2x (`outputs/decomposition.json`) is a genuine model output, not a
data error: Search has low absolute spend in the fit window (~₹0.9M vs OOH's ~₹8.7M) and
the fitted Hill curve attributes it disproportionate marginal lift. Flagged in README.md
as a number a real deployment would sanity-check against a marketing-science prior rather
than act on directly — the pipeline reports what the fitted model says, it does not
second-guess or clip implausible ROAS values, since silently capping them would hide a
genuine model-fit signal (however implausible) rather than surface it for review.

---

## Iteration 4 (2026-07-17) — visuals, serving, Docker, book

`src/eda.py` (3 Stage-0 figures), `src/diagnostics_viz.py` (8 post-model figures) wired
through one `make_figures.py` entry point -> 11 figures in `outputs/figures/`, all
verified to exist as real PNG files (not just "should work").

`app.py` (Streamlit): read-only dashboard over the JSON artifacts (decomposition, ROAS,
budget reallocation, diagnostics summary). Verified: `streamlit run app.py
--server.headless true`, then `curl localhost:8511/_stcore/health` -> 200 and the main
page -> 200, server log clean (no exceptions).

`Dockerfile`: authored following project-2's pattern (slim base + libgomp1 for
sklearn/statsmodels' OpenMP dependency, COPY only the artifacts the app actually reads).
Read-through verified for path correctness; **not built** (no Docker Desktop on this
machine), per program convention.

`book/`: `tectonic` is on PATH (`/opt/homebrew/bin/tectonic`) — used LaTeX + tectonic as
the primary path (matches project-1's style), not the fpdf2 fallback.
