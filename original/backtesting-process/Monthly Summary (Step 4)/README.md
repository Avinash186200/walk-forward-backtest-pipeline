# Monthly Combination Summary — Backtesting Pipeline (Step 4)

Reads every `selected_trade_level_all_months.csv` (Step 3's FT output), optionally
enforces **one open trade at a time per combination**, and writes a per-`(combination,
month)` summary to `Monthly_Combination_Summary.xlsx` (trade counts, win %, PnL, etc.).

A *combination* is `bt_strategy_folder_name × Currency × Condition × Timeframe`.

## Script

| File | |
|------|--|
| **`month_summary.py`** | Reads all strategy CSVs, filters, aggregates, writes the Excel. |

## What was optimized (output identical)

1. **Parallel reads** — the per-strategy CSVs are read across worker processes, then concatenated.
2. **Vectorized one-trade-at-a-time filter** — the original kept non-overlapping
   trades with a per-row `iterrows()` loop inside `groupby.apply`; replaced with a
   numpy sequential scan per combination (same logic, same kept rows, ~50–100× faster).

The aggregation and Excel export are unchanged. Verified: the summary DataFrame is
**exactly equal** to the original's (`assert_frame_equal`, 6,495 rows on the sample).

## Configure

Edit the CONFIG block at the top of `month_summary.py`:

```python
ROOT_DIR        = r"...\Results\Monthly Rolling"          # contains the Step 3 outputs
OUTPUT_FOLDER   = r"...\Results\Portfolios-Summary"
ALLOW_MULTIPLE_TRADES = "no"   # "no" = one open trade at a time per combination
```

## Run

```bash
python month_summary.py                          # uses CONFIG paths
python month_summary.py "<local-path>" 16
```
