# FT Trade-Level Extraction — Backtesting Pipeline (Step 3)

Consumes Step 2's deduped selection (`walk_forward_selection_sharpe_ge_1_unique.csv`)
and, for every selected `(strategy, iteration, trade_month)`, pulls the matching
trades from that iteration's backtest trade-level parquet. Per strategy folder it
writes three things:

| Output | Contents |
|--------|----------|
| `selected_trade_level_all_months.csv` | Trades whose **entry month == the selected trade_month** — the **forward-test (FT)** set. **This is Step 4's input.** |
| `Train data/train_trades_<YYYYMMDD>.csv` | Per trade-month: trades whose entry is **before** that month (the train set). |
| `Train_Test data/train_test_trades_<YYYYMMDD>.csv` | Per trade-month: train + test (entry **before** + entry **in** the month). |

## Scripts

| File | Use it when |
|------|-------------|
| **`ft_trade_level.py`** | Core logic + single-folder function. Import it or call `process_strategy_selection_folder`. |
| **`ft_trade_level_parallel.py`** | **Recommended for full runs.** Memory-aware scheduler across strategy folders. Same per-folder output, much faster. |

The parallel runner imports `ft_trade_level.py`, so **per-strategy output is
byte-identical** regardless of which you run (verified with `cmp` across the
`selected_trade_level` file and every Train/Train_Test month file).

## What was optimized

The original notebook ran **three separate passes** (one per output type), each
re-reading the same trade parquets, and processed one strategy after another. This
version:

1. **One pass, shared parquet cache** — the FT, train, and test sets for a folder
   are computed together, so each iteration's parquet is read **once** instead of
   ~3×.
2. **Parallel across strategy folders**, under a **memory budget** (estimated from
   each folder's selection size) so reads don't pile up. Trade parquets are small
   relative to Step 2's inputs, so many folders run at once.

Output is identical to the original — only the read pattern and scheduling change.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.7+, pandas / numpy / pyarrow.

## Configure

Edit the CONFIG block at the top of `ft_trade_level.py`:

```python
SELECTION_ROOT  = r"...\Results\Monthly Rolling"   # Step 2 output: one folder per strategy
BT_RESULTS_ROOT = r"...\BT results"                # backtest trade-level parquets
SELECTION_FILE_NAME = "walk_forward_selection_sharpe_ge_1_unique.csv"
SKIP_DONE = True   # resume: skip folders that already have selected_trade_level_all_months.csv
```

## Run

```bash
python ft_trade_level_parallel.py                                  # uses CONFIG paths
python ft_trade_level_parallel.py "<local-path>" 60 16   # ROOT, mem-budget GB, workers
```

Resumable: folders that already have `selected_trade_level_all_months.csv` are skipped.
