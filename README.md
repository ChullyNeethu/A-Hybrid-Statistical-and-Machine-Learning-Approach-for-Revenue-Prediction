# A-Hybrid-Statistical-and-Machine-Learning-Approach-for-Revenue-Prediction
# Hybrid Statistical + ML Revenue Forecasting

A three-layer forecasting framework for industrial B2B revenue prediction, combining classical time-series modeling, gradient-boosted residual correction, and live CRM pipeline signals. Built and validated on 25 years of monthly revenue data from a precision engineering manufacturer.

## Why This Exists

Industrial B2B revenue is hard to forecast: customer bases are small, transactions are large and infrequent, and procurement cycles introduce lumpy, event-driven demand on top of genuine seasonal structure. Classical models like SARIMA capture the seasonal backbone well but miss non-linear demand shifts; pure machine learning models overfit on the ~300 data points a 25-year monthly series provides. This project resolves that tension with a layered architecture where each method does only what it's best suited for.

## Architecture

```
Data sources
     │
     ▼
Layer 1 — SARIMA (statistical baseline)
     │  residuals
     ▼
Layer 2 — XGBoost / LightGBM (non-linear residual correction)
     │
     ▼            ┌─────────────────────────────┐
     │            │ Layer 3 — CRM Backlog Module │
     │            │ (quotation pipeline signal)  │
     │            └─────────────────────────────┘
     ▼                          │
        Weighted combination ◄──┘
                 │
                 ▼
        3-month revenue forecast
```

1. **SARIMA (statistical baseline)** — trained on 25 years of log-transformed monthly revenue to capture long-term trend, annual seasonality, and autocorrelation.
2. **XGBoost / LightGBM (residual correction)** — trained on SARIMA's residuals to capture non-linear effects, feature interactions, and macroeconomic regime shifts, using a 46-feature matrix (temporal, lag, rolling-stat, category-mix, PMI, and quotation-pipeline features).
3. **CRM backlog module** — converts live, probability-weighted quotation pipeline data into a directional demand signal using empirically derived lead-time conversions, applied as a capped nudge (±10%) rather than a blended value.

The three layers are merged through a **weighted combination** (statistical rigor + ML flexibility + forward-looking pipeline intelligence) to produce the final 3-month forecast.

## Data Pipeline

| Source | Description | Time Range |
|---|---|---|
| Monthly Revenue CSVs | Customer-level transactions, aggregated monthly | 2023–2026 (~35 files) |
| Rolling 12-Month Sales | Long-horizon SARIMA training backbone | 2001–2026 |
| Quotation Files (×2) | CRM pipeline — probability-weighted backlog | Current + historical open quotes |
| PMI File | Purchasing Managers' Index (macro feature) | 1994–present |
| Industrial Production Index & Shipment Proxy | Coincident demand indicators (national statistics) | Variable |
| Customer Analysis | Customer concentration/dynamics features | — |

### Data Quality Pipeline

Before modeling, a validation stage runs:
- **Geographic filtering** — isolates the relevant country's transactions from a multi-country dataset
- **Number-format cleaning** — corrects European thousand-separator/decimal conventions
- **Completeness check** — reindexes to a gap-free monthly range and forward-fills gaps
- **Zero/negative handling** — interpolates or falls back to median for invalid revenue values
- **Outlier winsorization** — clips values outside ±3σ
- **Stationarity check (ADF test)** — informs SARIMA's differencing order

## Feature Engineering

Eight feature groups feed the residual models, all respecting a strict no-leakage rule (features at time *t* use only information available before *t*):

| Group | Examples | Purpose |
|---|---|---|
| Temporal | month, quarter, sin/cos encoding, Q4 flag | Seasonality |
| Lag | 1–12 month lags of revenue | Autocorrelation |
| Rolling stats | 3/6/12-month mean & std | Trend, volatility |
| YoY growth | 12-month % change | Momentum |
| Category mix | % of product-line revenue | Product shift |
| O/S ratio | orders-to-sales ratio | Demand–supply balance |
| Macro | PMI level, 3-month MA, diff, regime | Economic context |
| Quotation pipeline | weighted pipeline, quote count, avg age | Near-future demand |

## Modeling Details

**XGBoost residual corrector**
- Residuals clipped at ±4σ to avoid corrupting the training target after inverse log-transform
- Feature matrix inner-joined on time index; columns >50% missing dropped, remainder median-imputed
- Separate Month+1 specialist model (shallow trees, learning rate 0.02, recency-weighted samples) vs. a general model for Months 2–3 (Optuna-tuned)
- SHAP values computed for feature-importance interpretability

**LightGBM residual corrector**
- Bayesian hyperparameter search (Optuna, 80 trials, `TimeSeriesSplit`) with a custom overcorrection penalty discouraging corrections >2× the actual residual
- Hard regularization floors (max 8 leaves, max depth 3, strong L1/L2) given the small sample size
- 3-seed ensemble (seeds 42/7/123) averaged to reduce variance
- Same Month+1 specialist / general-model split as XGBoost

**Post-processing corrections**
- **January correction** — blends the raw forecast 75% toward the historical January midpoint to offset December-spike carryover
- **Backlog adjustment** — pipeline value compared to its historical average, applied as a capped directional nudge (not a blended absolute value)
- **Bias calibration** — per-month multiplicative correction blending the 25-year seasonal index with walk-forward validation ratios
- **Month-specialist corrections** — targeted scalar adjustments for historically volatile months, clipped to [0.6, 1.6]

## Experimental Design

- **Walk-forward validation**: SARIMA refit at each step on all data up to time *t*; residual models retrained to produce 1-, 2-, and 3-month-ahead forecasts
- **Evaluation window**: 3-month out-of-sample holdout (Oct–Dec) following ~297 months of training data
- **Metrics**: MAPE, hit rate (±5% / ±10%), and bias, computed per horizon

## Results

| Configuration | Mean MAPE (3-month holdout) |
|---|---|
| SARIMA baseline | 10.9% |
| SARIMA + XGBoost | 6.2% |
| **SARIMA + LightGBM** | **4.4%** |

- The LightGBM hybrid met the primary ±5% MAPE target and outperformed XGBoost in every forecast month.
- CRM backlog integration was the single biggest driver of Month+1 accuracy, cutting first-month error from an 18.1% SARIMA baseline to 1.6% in the best configuration by detecting a below-average weighted pipeline and applying a downward correction.
- External macroeconomic signals (national Industrial Production Index, customer shipment proxy) were tested as additional features but showed **no net benefit** at the quarterly level — the IPI in particular degraded Q4 accuracy substantially, while the shipment proxy was largely redundant with existing internal features.

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.13 |
| RAM | 8 GB min, 16 GB recommended |
| Storage | ~500 MB free |
| Notebook | Jupyter / IPython (.ipynb) |
| OS | Windows / macOS / Linux |
| Environment | conda or venv |

Key libraries: `statsmodels` (SARIMA), `xgboost`, `lightgbm`, `optuna`, `shap`, `pandas`, `numpy`.

## Limitations & Future Work

- Currently scoped to a single country's operations; multi-country forecasting (with cross-border correlation structure) is a planned extension.
- Correction caps (±15%/±35%/±45% per horizon) are currently fixed; adaptive, regime-aware caps could improve December-type spike forecasting.
- Additional forward-looking signals (customer inquiry volumes, sales-rep activity) are identified as promising directions beyond quotation backlog alone.



