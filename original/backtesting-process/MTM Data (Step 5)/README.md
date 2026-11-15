# MTM Data — Backtesting Pipeline (Step 5)

Builds the **mark-to-market (MTM) dataset** from Step 3's forward-test trades. For
every selected trade it assigns an asset-class label, attaches the instrument's
**BROKER margin**, derives the SL-to-feed / opening-date fields, and writes **one
combined CSV per year**:

```
<MTM data>\All Strategy combined MTM data\<YEAR> all selected comb mtm data Test.csv
```

Each row is one closed D1 trade (entry and close in the same year) with the columns
the downstream MTM/P&L step expects (`Strategy, Opening Date, Currency, Condition,
SL to Feed, New P/L_pips, Currency Type, BROKER Margin, ...`).

## Scripts

| File | Use it when |
|------|-------------|
| **`step5_optimized.py`** | **The runner.** Reads each strategy's `selected_trade_level` once, builds all years in parallel across strategies, writes the per-year CSVs. |
| **`step5_data.py`** | Static reference data: the per-instrument **margin table** (`BROKER_Margin_df`, ~2,000 symbols) and the asset-class lists (majors / exotics / commodities / ETF / single-stock CFD / …). Imported by the runner. |
| **`baseline_step5.py`** | The original notebook logic, inlined verbatim (year-outer loop, re-reading per year). Kept only to **validate** the runner is byte-identical. |

The transform (asset-class classification via `np.select`, margin merge, the exact
`columns_list`) is the original's, unchanged — so the **data is identical**.

## What was optimized

The original notebook loop was:

```python
for year in Year_list:            # 6 years
    for strategy in strategies:   # ~94 strategies
        df = read_csv(strategy/selected_trade_level_all_months.csv)   # <-- re-read every year
        ... build MTM rows for this (strategy, year) ...
```

So each strategy's trade file was **read 6 times** (once per year) and strategies
were processed **one at a time**. This version:

1. **Reads each strategy's CSV once** (not once per year) — the year filter is just
   a slice of the already-loaded frame. ~6× fewer reads.
2. **Parallelizes across strategies** (`multiprocessing.Pool`); each worker returns
   all six years for its strategy, and the master concatenates per year.
3. **Deterministic ordering.** The original built its strategy order from
   `set(os.listdir(...))`, which Python orders by a **per-process-randomized string
   hash** — so the original's row order was *not reproducible run-to-run*. We sort
   the strategy list, so the output is **identical every run**. Same rows, same
   columns; only the (previously random) row order is now stable.

Result on the 94-folder Russell-1000 set: the full step runs in **~84 seconds**,
output **byte-identical** to the original logic (verified — see below).

## Validation

`baseline_step5.py` (original logic) and `step5_optimized.py` were run on the same
20 strategies into separate folders and compared with `cmp`:

```
2021..2026: IDENTICAL   (77,709 / 95,306 / 69,507 / 73,492 / 98,211 / 51,636 rows)
RESULT: STEP 5 BYTE-IDENTICAL
```

Both use the sorted strategy order, so the comparison is exact (byte-for-byte),
isolating the read-pattern / parallelism change from the ordering fix.

## Configure & run

Edit the CONFIG block at the top of `step5_optimized.py`:

```python
ROOT_DIR             = r"...\Results\Monthly Rolling"                       # Step 3 output (one folder per strategy)
MTM_DATA_EXPORT_PATH = r"...\MTM data\All Strategy combined MTM data"       # where the per-year CSVs go
YEAR_LIST            = [2021, 2022, 2023, 2024, 2025, 2026]
```

```bash
python step5_optimized.py                                   # uses CONFIG paths
python step5_optimized.py "<Monthly Rolling>" "<out dir>" 12 # ROOT, OUT_DIR, N_WORKERS
python step5_optimized.py "<Monthly Rolling>" "<out dir>" 12 20   # + STRATEGY_LIMIT (validation)
```

Python 3.7+, pandas / numpy.
