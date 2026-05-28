# QOIN — Quantitative Optimized Indian Nifty

Phase 1 local dashboard: systematic factor analysis for NIFTY 50, BANK NIFTY, and ICICI Bank.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run src/app.py
```

Set `QOIN_DATA_DIR` to override the default data path:

```bash
QOIN_DATA_DIR="/path/to/cleaned/data" streamlit run src/app.py
```

## Three-window design

| Window | Default | Purpose |
|--------|---------|---------|
| PCA / Regression | 126 days | Factor covariance, eigenvalues, OLS coefficients, geometric projection |
| Mean reversion | 15 days | Rolling std of regression residual; ±2σ threshold |
| Fingerprint | 5 days | Z-scored factor vector for cosine-similarity historical matching |

All three windows are adjustable from the sidebar. All rolling computations use only data on or before day t (no lookahead). A single `as_of(df, date)` helper enforces this everywhere.

## Per-factor levels-vs-returns decisions

| Factor | Transform | Rationale |
|--------|-----------|-----------|
| `gold_return` | log-return | Level non-stationary |
| `usd_inr_return` | log-return | Level non-stationary |
| `us_10y_diff`, `us_5y_diff` | 1st diff | Rate level → change in rate |
| `india_10y_diff`, `india_5y_diff` | 1st diff | Same |
| `cpi_diff`, `wpi_diff` | 1st diff (monthly → daily) | Price index level → inflation change |
| `gdp_yoy` | YoY growth % (quarterly → daily) | Already a growth rate; approx stationary |
| `trade_balance` | Level (monthly → daily) | Spread series; approx stationary |
| `repo_rate_diff` | 1st diff (monthly → daily) | Rate level → policy change |
| `fii_net`, `dii_net` | Level (daily, from 2018) | Flow series; approx stationary |
| `icici_nii_qoq`, `icici_pat_qoq` | QoQ change (quarterly → daily) | Already a change; approx stationary |

Daily series from foreign markets (US bonds, gold) are reindexed to the Indian trading-day calendar and **forward-filled** at the price level before differencing. On Indian trading days when a foreign market was closed, the return is 0 (no new information). This prevents NaN gaps from breaking rolling z-scores.

## Quarterly fundamentals release-lag assumption

ICICI Bank quarterly results have no explicit release date in the source file. **Release date is approximated as fiscal quarter end + 45 calendar days.** Example: Q1 FY2010 (Apr–Jun 2009) → period end = 30 Jun 2009 → release = 14 Aug 2009. This assumption is conservative (actual results often release in ~45–60 days). Shown in the diagnostics panel.

## What is NOT in this app

- **Trading strategy** — no position sizing, no entry/exit rules
- **Backtesting** — no PnL simulation, no drawdown analysis
- **Machine learning** — no model training, no prediction outputs
- **Sentiment analysis** — no news, social, or options flow data
- **Momentum signals** — no trend-following or cross-sectional momentum
- **Execution** — no order routing, no broker integration
- **Live data** — offline only; all data loaded from local CSVs

## Project structure

```
src/
  data/loader.py          — CSV loading, stationarity transforms, daily alignment
  math_engine/pipeline.py — as_of(), rolling_zscore(), PCA+OLS, fingerprint search
  views/biplot.py         — View 1: factor loading arrows + stock positions
  views/parallel_coords.py— View 2: full z-state + fingerprint match overlays
  views/heatmap.py        — View 3: factor correlation matrix + stock rows
  views/mean_reversion.py — View 4: price chart, factor-implied mean, ±2σ bands
  app.py                  — Streamlit entrypoint
```
