# Marketing Mix Modelling with Adstock Transformation
**Method tag:** Adstock-Ridge Regression | **Domain:** Marketing Science / Econometrics

---

## Core Problem

A brand spends across TV, Digital, Search, and Promotions. The CFO wants to know which rupee of spend drove how much incremental revenue. Standard MLR fails because:
- Media effects **carry over** across weeks (adstock / memory effect)
- Spend exhibits **diminishing returns** (doubling budget ≠ double sales lift)

MMM solves this with linear regression on econometrically transformed predictors — where coefficients are the deliverable, not predictions.

---

## Data

| Item | Detail |
|---|---|
| **Primary** | Robyn Sample Data (Meta OSS) — 208 weeks, 5 media channels, weekly revenue |
| **Download** | `github.com/facebookexperimental/Robyn` |
| **Features** | TV GRP, Digital impressions, Paid Search clicks, Promo discount depth, Price index, Seasonality dummies, Holiday flags |
| **Target** | Weekly brand revenue (units or ₹) |
| **Split** | Temporal — Weeks 1–170 train / Weeks 171–208 hold-out |

---

## Modelling Pipeline

### Stage 1 — Adstock Transformation
Model carryover (memory) of each media channel:

```
Adstock(t) = spend(t) + λ × Adstock(t-1)
```

- `λ` = decay rate per channel, estimated via grid search on hold-out MAPE
- TV: λ ≈ 0.7 (long memory) | Search: λ ≈ 0.1 (short memory)

### Stage 2 — Hill Saturation Curve
Model diminishing returns per channel:

```
Saturated(x) = x^α / (x^α + K^α)
```

- `α` = shape parameter, `K` = half-saturation point
- Fitted per channel; ensures non-linear spend response is captured before regression

### Stage 3 — Ridge Regression
Regress weekly sales on transformed variables + base controls:

```
Sales = β₀ + β₁·Adstock_TV + β₂·Adstock_Digital + ... + β_k·Seasonality + ε
```

- **Ridge (L2)** handles multicollinearity across correlated media channels
- Fitted **β coefficients** = media contribution per unit spend (the business output)
- **Baseline:** Raw OLS on untransformed spends — shows misattribution without econometric correction

### Diagnostics
- VIF < 5 on all predictors post-transformation
- Ljung-Box test for residual serial correlation
- Chow test for structural breaks (e.g. COVID period)
- Residual QQ plot + heteroscedasticity check

---

## Outputs

| Deliverable | Description |
|---|---|
| Revenue decomposition | Base sales + incremental contribution per channel (%) |
| Channel ROAS | Revenue attributed ÷ spend, per channel |
| Saturation curves | Sales lift vs. spend level (visualises diminishing returns) |
| Budget reallocation | Optimal spend split via constrained LP (`scipy.optimize`) |
| Adjusted R² | Target > 0.85 on hold-out period |
| Hold-out MAPE | Target < 8% on 38-week out-of-sample window |

---

## Why Linear Regression (Not XGBoost)

OLS is the right tool here — not a baseline. A black-box model cannot tell the CFO *"₹1 spent on TV generates ₹3.2 in incremental revenue."* Ridge regression coefficients, after the adstock + saturation transforms, carry direct business meaning. Interpretability is the product.

---

## Division of Work

| You code | Claude Code handles |
|---|---|
| Adstock transformation function | Data pipeline + seasonality dummies |
| Hill saturation curve | VIF checker + Ljung-Box wrapper |
| Ridge regression pipeline | Decomposition waterfall chart |
| ROAS computation | Budget reallocation LP setup |
| Budget reallocation optimizer | Robyn data loader |

---

## Recruiter Relevance

| Firm | Connection |
|---|---|
| BCG | MMM is a core FMCG/pharma client deliverable |
| Mastercard | Data & Services team sells MMM as a product to advertisers |
| JPMC | Card spend attribution across marketing channels |
| D.E. Shaw / QRT | Factor decomposition methodology analogue in finance |
| Sun Pharma / Piramal | Promotional spend attribution for pharma detailing |

---

## Differentiator from Batch

"Used Cars Price Prediction (MLR)" appears on **6+ CVs verbatim** in your PGDBA batch. This project uses the same statistical engine (OLS/Ridge) in a scenario where regression is not a commodity exercise — it is the industry-standard methodology for a ₹multi-crore business decision.
