# FX & Commodities Backtesting Project
## May 2026 | Master Documentation

**Project Root:** `<local-path>`
**Code Folder:**  `...\New Selection tech\Code for Portfolio`
**Documented:**   May 2026

---

## 1. Project Overview

This project backtests FX and Commodities strategies using Broker Vol 0.5 position sizing. It covers:

- **9 portfolios** across three strategy variants (Standard, 3-Day hold rule, 3-Day + ML filter)
- **3 comparison reports** across all portfolio pairs
- **ML model pipeline** — per-instrument directional model used as a trade filter
- **Slippage sensitivity analysis** — 1-pip and 2-pip per leg versions of all outputs
- **Trade-level model data builder** — monthly CSVs for ML model training/testing

The April 2026 backtest is the baseline. The May 2026 backtest was rebuilt with updated selection/sizing parameters — it is **not** an extension of April; it is a separate configuration.

---

## 2. Software Dependencies

```
Python 3.12+
pandas
numpy
matplotlib
quantstats
xlsxwriter
openpyxl
```

Install:
```
pip install pandas numpy matplotlib quantstats xlsxwriter openpyxl
```

---

## 3. Strategy Variants

| Variant | Description |
|---------|-------------|
| **Standard (Normal)** | No holding period restriction |
| **3 Days** | Minimum 3-day hold rule before a trade can close |
| **3 Days ML** | Same 3-day rule + ML directional model filter applied |

All strategies use **Broker Vol 0.5 sizing**. All PL uses **`PL With cost`** column — never `Returns_final_per_trade` (see Section 8).

---

## 4. Portfolio Definitions

| Tag | Instruments |
|-----|-------------|
| **FX** | FX pairs only |
| **COMM** | XAG/USD (Silver), XAU/USD (Gold), XBR/USD (Brent Crude) |
| **BOTH** | FX + Commodities combined |

---

## 5. Complete Master Path Directory

### 5.1 Project Roots

| Label | Path |
|-------|------|
| May 2026 root | `<local-path>` |
| May 2026 New Selection | `<local-path>` |
| April 2026 root | `<local-path>` |
| April 2026 New Selection | `<local-path>` |
| Code folder | `...\May 2026\New Selection tech\Code for Portfolio` |

### 5.2 Input Data — Trade-Level CSVs and Daily Excel (May 2026)

Each folder below lives under `<local-path>`

| Strategy Variant | Portfolio | Trade Unit Folder |
|------------------|-----------|-------------------|
| Standard | FX | `Trade level data mking FX\Trade Unit\` |
| Standard | COMM | `Trade level data mking Comm\Trade Unit\` |
| Standard | BOTH | `Trade level data mking both\Trade Unit\` |
| 3 Days | FX | `Trade level data mking FX 3 days\Trade Unit\` |
| 3 Days | COMM | `Trade level data mking Comm 3 days\Trade Unit\` |
| 3 Days | BOTH | `Trade level data mking both 3 days\Trade Unit\` |
| 3 Days ML | FX | `Trade level data mking FX 3 days ML\Trade Unit\` |
| 3 Days ML | COMM | `Trade level data mking Comm 3 days ML\Trade Unit\` |
| 3 Days ML | BOTH | `Trade level data mking both 3 days ML\Trade Unit\` |

**Files inside each Trade Unit folder:**

| File | Description |
|------|-------------|
| `Fx_daily_data_for_portfolio_05_vol.xlsx` | Daily portfolio returns (PL With cost per day) |
| `Fx - TradeDaily_With_Broker_vol_05_v1_transactions_with_equity with cost flt col.csv` | Full trade-level data |
| `portfolio\` | Original (no slippage) report outputs |
| `portfolio 1pip\` | 1-pip-per-leg slippage report outputs |
| `portfolio adj\` | 2-pip-per-leg slippage report outputs |

*(Replace `Fx` prefix with `Comm` or `both` for COMM and BOTH portfolios)*

### 5.3 Input Data — April 2026 (used in comparisons only)

Path root: `<local-path>`

| Portfolio | Trade Unit Folder |
|-----------|-------------------|
| FX | `Trade level data mking FX\Trade Unit\` |
| COMM | `Trade level data mking Comm\Trade Unit\` |
| BOTH | `Trade level data mking both\Trade Unit\` |

### 5.4 BT Results (parquet files for model data builder)

| Variant | Path |
|---------|------|
| Standard | `<local-path>` |
| 3 Days | `<local-path>` |

Parquet filename pattern: `{CCY_TF_P1_P2_P3_P4_P5_P6_Condition}.parquet`

### 5.5 Monthly Combination Lists

| Variant | Path |
|---------|------|
| Standard | `...\New Selection tech\Results\Combinations list with Sharpe\` |
| 3 Days | `...\New Selection tech\Results\Combinations list with Sharpe 3 days\` |

Filename pattern: `Comb list with param for live MM-YYYY.csv`

### 5.6 Output — Portfolio Report Suites

Each `portfolio\` subfolder contains 14 files (see Section 7.1).

| Strategy | Portfolio | No Slippage | 1-pip | 2-pip |
|----------|-----------|-------------|-------|-------|
| Standard | FX | `Trade level data mking FX\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| Standard | COMM | `Trade level data mking Comm\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| Standard | BOTH | `Trade level data mking both\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| 3 Days | FX | `Trade level data mking FX 3 days\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| 3 Days | COMM | `Trade level data mking Comm 3 days\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| 3 Days | BOTH | `Trade level data mking both 3 days\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| 3 Days ML | FX | `Trade level data mking FX 3 days ML\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| 3 Days ML | COMM | `Trade level data mking Comm 3 days ML\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |
| 3 Days ML | BOTH | `Trade level data mking both 3 days ML\Trade Unit\portfolio\` | `\portfolio 1pip\` | `\portfolio adj\` |

All paths above are relative to `<local-path>`

### 5.7 Output — Comparison Reports

All comparison folders sit directly under `<local-path>`

| Comparison | No Slippage | 1-pip | 2-pip |
|-----------|-------------|-------|-------|
| April vs May (Standard) | `Apr vs May Comparison\` | `Apr vs May Comparison 1pip\` | `Apr vs May Comparison adj\` |
| April vs May (3 Days) | `Apr vs May 3 Days Comparison\` | `Apr vs May 3 Days Comparison 1pip\` | `Apr vs May 3 Days Comparison adj\` |
| 3 Days vs 3 Days ML | `3 Days vs ML Comparison\` | `3 Days vs ML Comparison 1pip\` | `3 Days vs ML Comparison adj\` |

Each comparison folder contains:
- `*.xlsx` — side-by-side metrics + monthly returns (all 3 strategies in one file)
- `*.html` — interactive HTML with charts
- `FX_*_Chart.png`, `COMM_*_Chart.png`, `BOTH_*_Chart.png` — 3-panel equity/bar/diff charts

### 5.8 Output — ML Model Pipeline

| Item | Path |
|------|------|
| Trade-level CSVs (Standard) | `Results\Trade level data for model\Trade level data\` |
| Trade-level CSVs (3 Days) | `Results\Trade level data for model\Trade level data 3 days\` |
| ML model scripts | `Results\Model Results 3 days\` |
| ML predictions | `Results\Model Results 3 days\Live_Predictions\` |
| MTM data (Standard) | `MTM data\` |
| MTM data (3 Days) | `MTM data 3 days\` |
| MTM output (Standard) | `MTM output\` |
| MTM output (3 Days) | `MTM output 3 days\` |
| MTM output (3 Days ML) | `MTM output ML 3 days\` |

### 5.9 Summary Files

| File | Path |
|------|------|
| Slippage summary (2-pip) | `Code for Portfolio\slippage_reports_summary.xlsx` |
| Slippage summary (1-pip) | `Code for Portfolio\slippage_reports_summary_1pip.xlsx` |

---

## 6. Scripts — What Each One Does

All scripts live in `Code for Portfolio\` and are copied to `Master Documentation\Code\`

### 6.1 `quant_analysis_report.py` — Standard Portfolio Reports
**Runs for:** FX, COMM, BOTH (standard, no holding period rule)
**Reads:** Daily Excel + Trade CSV from `Trade level data mking FX\`, `..Comm\`, `..both\`
**Outputs to:** `portfolio\` subfolder in each strategy's Trade Unit folder
**Run:**
```
cd "...\Code for Portfolio"
python quant_analysis_report.py
```

### 6.2 `quant_analysis_report_3days.py` — 3-Day Portfolio Reports
**Runs for:** FX 3 days, COMM 3 days, BOTH 3 days
**Reads:** `Trade level data mking FX 3 days\`, etc.
**Outputs to:** `portfolio\` in each 3-day Trade Unit folder
**Run:**
```
python quant_analysis_report_3days.py
```

### 6.3 `quant_analysis_report_3days_ml.py` — 3-Day ML Portfolio Reports
**Runs for:** FX 3 days ML, COMM 3 days ML, BOTH 3 days ML
**Reads:** `Trade level data mking FX 3 days ML\`, etc.
**Outputs to:** `portfolio\` in each 3-day ML Trade Unit folder
**Run:**
```
python quant_analysis_report_3days_ml.py
```

### 6.4 `apr_may_comparison.py` — April vs May Standard Comparison
**Compares:** April 2026 standard vs May 2026 standard (all 3 portfolios)
**Reads:** April + May Trade Unit folders (standard)
**Outputs to:** `Apr vs May Comparison\`
**Run:**
```
python apr_may_comparison.py
```

### 6.5 `apr_may_3days_comparison.py` — April vs May 3-Days Comparison
**Compares:** April 2026 standard vs May 2026 3-day (all 3 portfolios)
**Note:** April never had a 3-day variant — April standard is used as the baseline
**Outputs to:** `Apr vs May 3 Days Comparison\`
**Run:**
```
python apr_may_3days_comparison.py
```

### 6.6 `days3_vs_ml_comparison.py` — 3 Days Base vs 3 Days ML Comparison
**Compares:** May 2026 3-day (base) vs May 2026 3-day ML (all 3 portfolios)
**Both sides from May 2026**
**Outputs to:** `3 Days vs ML Comparison\`
**Run:**
```
python days3_vs_ml_comparison.py
```

### 6.7 `build_model_trade_data.py` — ML Model Training Data Builder
**Purpose:** For each monthly combination CSV, loads the matching BT parquet files and labels rows as `train` (historical months) or `test` (the target month). Outputs one CSV per month.
**Runs two modes:** `normal` and `3days`
**Key logic:**
- Target month rows → `split = "test"`
- All prior months rows → `split = "train"`
- Parquet filename: `{CCY}_{TF}_{P1}_{P2}_{P3}_{P4}_{P5}_{P6}_{Condition}.parquet`
**Outputs:**
- Normal: `Results\Trade level data for model\Trade level data\YYYY_MM_trade_data.csv`
- 3 Days: `Results\Trade level data for model\Trade level data 3 days\YYYY_MM_trade_data.csv`
**Run:**
```
python build_model_trade_data.py
```

### 6.8 `slippage_all_reports.py` — 2-Pip Slippage Adjusted Reports (ALL)
**Purpose:** Applies 2-pip entry + 2-pip exit slippage to all 9 portfolios and all 3 comparison pairs. Generates identical report suites (no slippage labels in output).
**Slippage:** `SLIP = 0.0002` (0.02% per leg, 4 pips round-trip)
**Outputs to:** `portfolio adj\` and `*Comparison adj\` folders
**Run:**
```
python slippage_all_reports.py
```

### 6.9 `slippage_1pip_reports.py` — 1-Pip Slippage Adjusted Reports (ALL)
**Purpose:** Same as above but with 1-pip per leg (2 pips round-trip total).
**Slippage:** `SLIP = 0.0001` (0.01% per leg, 2 pips round-trip)
**Outputs to:** `portfolio 1pip\` and `*Comparison 1pip\` folders
**Run:**
```
python slippage_1pip_reports.py
```

### 6.10 `market_crash_analysis.py`
**Purpose:** Commodity crash analysis — how COMM strategy performed during historical drawdown periods in Silver, Gold, Brent. Plots entry/exit points on price charts.
**Note:** Originally built for April 2026 data. Update `COMM_TRADE_FILE` and `COMM_DAILY_FILE` at top of script before running on May data.

---

## 7. Output Files — Portfolio Report Suite

Each `portfolio\`, `portfolio 1pip\`, or `portfolio adj\` folder contains the same 14 files:

| File | Description |
|------|-------------|
| `01_{TAG}_Overall_Summary.xlsx` | Total Return, CAGR, Sharpe, Sortino, Max DD, Calmar, Win Rates, Trade Stats |
| `02_{TAG}_Yearly_Performance.xlsx` | Return, Vol, Sharpe, Max DD, Win Rate by calendar year |
| `03_{TAG}_Monthly_Performance.xlsx` | Monthly return pivot table (Year × Month heatmap) |
| `04_{TAG}_Drawdown_Analysis.xlsx` | Top 15 drawdown periods: start, trough, end, magnitude, duration |
| `05_{TAG}_Trade_Statistics.xlsx` | Trades by Instrument, Direction, Condition, Year |
| `05b_{TAG}_Instrument_Performance.xlsx` | Per-instrument metrics — Overall and 2026 Q1 tabs |
| `06_{TAG}_Rolling_Metrics.xlsx` | Rolling 63-day return, annualised vol, Sharpe |
| `07_{TAG}_2026_Q1_Performance.xlsx` | Jan/Feb/Mar 2026 performance with trade breakdown |
| `{TAG}_Overall_Charts.png` | 5-panel: equity curve, drawdown, monthly heatmap, rolling Sharpe, rolling 3M return |
| `{TAG}_2026_Q1_Report_charts.png` | 3-panel 2026 Q1: monthly equity curves, daily returns bar, Q1 drawdown |
| `{TAG}_Instrument_Charts_Overall.png` | 4-panel bar chart by instrument (PL, Win Rate, Sharpe, Trade Count) — full history |
| `{TAG}_Instrument_Charts_2026_Q1.png` | Same instrument chart — 2026 Q1 only |
| `{TAG}_QuantStats_Overall.html` | Full QuantStats tearsheet — entire backtest history |
| `{TAG}_QuantStats_2026_Q1.html` | QuantStats tearsheet — 2026 Q1 only |
| `{TAG}_2026_Q1_Report.html` | Custom HTML: monthly table, trade stats, instrument breakdown, charts |

`{TAG}` = `FX`, `COMM`, or `BOTH`

---

## 8. Key Design Decisions

### Always use `PL With cost`, never `Returns_final_per_trade`
`Returns_final_per_trade` double-counts MTM PL and produces inflated returns. All scripts use `PL With cost` exclusively.

### Daily returns come from the daily Excel file
All return-based metrics (CAGR, Sharpe, Max DD, equity curve) use the `PL With cost` column from the daily Excel — not aggregated from the trade CSV — because the daily Excel already accounts for portfolio-level equity.

Exception: slippage scripts derive adjusted daily returns by aggregating `new_PL_With_cost` from the trade CSV, since the Excel file is based on original (pre-slippage) prices.

### Slippage formula
```
slippage_impact = SLIP * amount_invested_per_trade / equity_start_of_day
new_Returns = Returns_final_per_trade - slippage_impact
new_PL_With_cost = new_Returns - Total_Cost / equity_start_of_day
```
- OPEN rows: SLIP applied once (entry leg)
- CLOSE rows: SLIP applied once (exit leg)
- OPEN_CLOSE rows: 2 × SLIP (same-day open and close)

### Train/test labeling (ML model data)
- Target month = `test`
- All prior months = `train`
- Rows after the target month are dropped (future data, not known at decision time)

### Why BOTH does not exactly equal FX + COMM
BOTH is constructed with a combined equity base, giving different `equity_start_of_day` and different `flt col` filtering. PL will not sum to FX + COMM exactly — this is by design.

### Why May 2026 has ~3x more trades than April 2026
The May backtest was rebuilt with updated selection parameters — it is not April + one month. Historical performance for the overlapping period differs between April and May configurations.

---

## 9. Slippage Sensitivity Results (Key Metrics)

### Standard Portfolios

| Portfolio | No Slippage | 1 pip/leg (2 total) | 2 pips/leg (4 total) |
|-----------|-------------|---------------------|----------------------|
| FX | 68.63% / Sharpe 2.529 | 53.06% / 2.082 | 38.93% / 1.627 |
| COMM | 61.67% / 2.063 | 57.16% / 1.948 | 52.77% / 1.833 |
| BOTH | 79.87% / 2.753 | 63.30% / 2.323 | 48.26% / 1.884 |

### 3-Day Portfolios

| Portfolio | 1 pip/leg (2 total) | 2 pips/leg (4 total) |
|-----------|---------------------|----------------------|
| FX 3 Days | 57.01% / Sharpe 2.436 | 34.09% / 1.598 |
| COMM 3 Days | 61.27% / 2.515 | 55.72% / 2.338 |
| BOTH 3 Days | 65.54% / 2.697 | 41.39% / 1.868 |

### 3-Day ML Portfolios

| Portfolio | 1 pip/leg (2 total) | 2 pips/leg (4 total) |
|-----------|---------------------|----------------------|
| FX 3 Days ML | 60.23% / Sharpe 2.432 | 36.85% / 1.633 |
| COMM 3 Days ML | 62.91% / 2.755 | 57.72% / 2.580 |
| BOTH 3 Days ML | 71.45% / 2.741 | 46.52% / 1.958 |

---

## 10. Step-by-Step Reproduction Guide

To regenerate everything from scratch (assuming trade-level data files exist):

### Step 1 — Standard Portfolio Reports
```
cd "<local-path>"
python quant_analysis_report.py
```

### Step 2 — 3-Day Portfolio Reports
```
python quant_analysis_report_3days.py
```

### Step 3 — 3-Day ML Portfolio Reports
```
python quant_analysis_report_3days_ml.py
```

### Step 4 — Comparison Reports (original, no slippage)
```
python apr_may_comparison.py
python apr_may_3days_comparison.py
python days3_vs_ml_comparison.py
```

### Step 5 — ML Model Training Data
```
python build_model_trade_data.py
```
*(Outputs monthly train/test CSVs to Results\Trade level data for model\)*

### Step 6 — Slippage Analysis (all portfolios + comparisons)
```
python slippage_all_reports.py      # 2-pip per leg (4-pip round-trip)
python slippage_1pip_reports.py     # 1-pip per leg (2-pip round-trip)
```

---

## 11. Trade-Level CSV Column Reference

| Column | Description |
|--------|-------------|
| `date` | Row date (datetime) |
| `price` | Execution price |
| `TradeID` | Trade identifier |
| `Strategy` | Strategy name |
| `Currency` | Instrument (e.g. EUR/USD, XAU/USD) |
| `Timeframe` | Timeframe (e.g. D1) |
| `Condition` | Long or Short |
| `Direction` | 1 = Long, -1 = Short |
| `Opening Date` | Trade entry date |
| `Closing Date` | Trade exit date |
| `DayStatus` | OPEN / CLOSE / OPEN_CLOSE / ACTIVE |
| `Returns_final_per_trade` | Return before costs (do not use for final PL) |
| `equity_start_of_day` | Portfolio equity at start of day — in `Pipeline/run_pipeline.py` (2026-06-26 on), this is net of prior days' cost+slippage, computed recursively day-by-day; see `Pipeline/README.md` |
| `amount_invested_per_trade` | Position size in base currency |
| `Delta (Close flag)` | 1 = trade closed on this row |
| `Total Cost` | Transaction + financing costs |
| `PL With cost` | Final PL including all costs — USE THIS |

---

## 12. Related Projects / External Dependencies

| Project | Path |
|---------|------|
| April 2026 backtest | `<local-path>` |
| May 2026 BT results (parquet) | `<local-path>` |
| May 2026 BT results 3 days | `<local-path>` |
| Combination lists | `...\New Selection tech\Results\Combinations list with Sharpe\` |

---

*This documentation covers all work performed in the FX & Commodities backtesting project, May 2026. All Python scripts are copied to the `Code\` subfolder of this documentation folder for archival.*
