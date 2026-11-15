# Monthly Rollup — Backtesting Pipeline (Step 1)

Rolls **trade-level parquet files** up to **monthly returns**, one row per
`(strategy, iteration, year, month)`. No filtering, no stats, no selection — just
`read trades → group by exit month → write CSV`.

This is Step 1 of the backtesting pipeline: it consumes the per-iteration trade files
produced by the backtest stage and emits compact monthly summaries used by the later
selection / portfolio steps.

## Scripts

| File | Use it when |
|------|-------------|
| **`monthly_rollup.py`** | Single-process. The reference implementation — correct and reasonably fast. Run this if you want the simplest thing. |
| **`monthly_rollup_parallel.py`** | **Recommended for full runs.** Processes many strategies at once (one worker process per strategy). Same per-strategy output as the single-process version, much faster wall-clock. |
| **`benchmark_workers.py`** | One-off: finds the best worker count for *your* machine (runs one shared pool of many strategies at each `NW`, with a cache warm-up, so the comparison is fair). |

All three share one config and one set of core functions — `monthly_rollup_parallel.py`
and `benchmark_workers.py` import `monthly_rollup.py`, so **per-strategy output is identical**
regardless of which you run.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.7+ (works on 3.7 through 3.12). Depends on pandas, numpy, pyarrow.

## Configure

Edit the **CONFIG block at the top of `monthly_rollup.py`** — this is the single source of
truth for paths (the other two scripts read from it):

```python
STRATEGY_ROOT_DIR = r"...\BT results"            # input: one folder per strategy
TRADE_SUBDIR_NAME = "Trade Level Data"           # subfolder holding the *.parquet trade files
OUTPUT_ROOT_DIR   = r"...\Results\Monthly Rolling"  # output: one folder per strategy

TRADE_EXIT_COL  = "Closing Date"                 # column rolled up by month
TRADE_PNL_COL   = "Profit_Loss"
CAPITAL_BASE    = 100_000.0                       # monthly_return = sum(Profit_Loss) / CAPITAL_BASE
```

Expected input layout:

```
STRATEGY_ROOT_DIR/
  <Strategy A>/Trade Level Data/*.parquet   <- each parquet = one backtest iteration's trades
  <Strategy B>/Trade Level Data/*.parquet
```

## Run

```bash
# Full run, parallel (recommended). NW = worker processes, TPW = pyarrow threads per worker.
# Defaults are NW=8 TPW=4 (production-proven); override as needed:
NW=8 TPW=4 python monthly_rollup_parallel.py

# Or single-process:
python monthly_rollup.py
```

On Windows PowerShell, set the env vars first:

```powershell
$env:NW="8"; $env:TPW="4"; python monthly_rollup_parallel.py
```

**Resume behavior:** a strategy is skipped if its output folder already exists in
`OUTPUT_ROOT_DIR`. So a re-run only processes new/missing strategies. To force a clean
re-run of everything, point `OUTPUT_ROOT_DIR` at an empty folder (or delete the folders
you want redone). Note: an **empty** leftover folder also counts as "done" and gets
skipped — delete any partial/empty folders before resuming.

## Choosing the worker count

The per-strategy group-by step is **single-threaded**, so the number of *workers* (`NW`)
is what scales — up to your physical core count. Reads use `TPW` pyarrow threads each.

- **Default: `NW=8, TPW=4`** — the production-proven config (see below).
- Keep `NW ≤ physical cores` and `NW × TPW ≤ logical cores` (avoids thread oversubscription,
  which can drastically *slow* things down).
- Don't exceed your physical core count for `NW`; on a single SATA SSD, returns flatten out.
- Run `benchmark_workers.py` to find the best value for your hardware — it writes
  `worker_benchmark_results.txt` into `OUTPUT_ROOT_DIR`.

**Ceiling to be aware of:** one strategy is processed by one worker, so total wall-clock is
floored by your single *largest* strategy — more workers clear the many small strategies
faster but can't speed up the biggest one.

### Measured performance (24-core / SATA-SSD / 128 GB box)

Real production run, `NW=8, TPW=4`: **46 strategies, ~3.4M trade files in 141 min**.

- ~**690 files/s** steady-state while all 8 workers were busy
- ~**400 files/s** overall average, dragged down by the long single-strategy tail
  (the largest strategy, 772k files, runs on one worker)
- vs. the single-process version at ~95 files/s → roughly **7x** faster wall-clock

Use `benchmark_workers.py` to check whether more workers help on your machine before
changing the defaults.

## Output

Two CSVs per strategy in `OUTPUT_ROOT_DIR/<strategy>/`:

- **`monthly_stats_all.csv`** — `year, month, month_end, monthly_return, monthly_trades, is_profitable, iteration_id, strategy_name`
- **`iteration_summary_all.csv`** — `iteration_id, strategy_name, symbol, leg, file_path`

## Performance notes / history

- The optimization replaces a per-file `pd.read_parquet` Python loop with batched
  `pyarrow.dataset` reads + one vectorized group-by per `BATCH_SIZE` files.
- Compatibility fixes baked in for older stacks (Python 3.7 / pandas 1.0.1):
  `from __future__ import annotations` (so `list[Path]` hints work), and `groupby().reset_index()`
  instead of `groupby(as_index=False).agg(named=…)` (which silently drops keys on pandas 1.0.x).
- The flush logic always writes a final flush, so no strategy ever loses its tail rows.
