# Optimized backtest strategies - ready to run

191 strategy notebooks, each confirmed to produce byte-identical output to
the original (slow) version on real multi-symbol, multi-parameter test
runs. The O(n^2) per-row indicator loop has been replaced with vectorized
math (`live_indicators.py`); the O(n*m) exit-matching loop has been replaced
with a single `searchsorted` call. Typical per-task speedup measured: 50x-220x
(some up to 2000x+, a few lower for strategies that didn't have the bottleneck
in the first place).

## 1. One-time setup

Edit **`config.py`** - four values, used by every notebook:

```python
DATA_DIR = "..."          # folder containing your raw OHLC parquet file
DATA_FILE = "..."          # filename of that parquet file
RESULTS_BASE_DIR = "..."   # where each strategy's BT/FT/Trade Level Data/etc. get written
CPU_COUNT = 40              # number of parallel worker processes
```

That's it - nothing else in this folder needs editing. Every notebook reads
these four values instead of having its own hardcoded path.

Keep `config.py`, `live_indicators.py`, and every `*.ipynb` in this same
folder - they're imported relative to the current working directory.

## 2. Running everything in one go

```bash
cd READY_FOR_SERVER
python run_all_strategies.py
```

This runs all 191 strategies **sequentially** (one after another), and each
strategy internally uses a multiprocessing Pool of `config.CPU_COUNT`
workers (same parallelism model the original notebooks used by hand in
Jupyter - just without having to open and run each one manually). The raw
data file is loaded **once** and shared across all 191 strategies instead of
being re-loaded per strategy.

Progress and a final summary are printed to the console and also written to
`run_all_strategies.log` (so you can `tail -f` it or check back after a long
run).

### Running a subset

```bash
python run_all_strategies.py "Bulls_strategy - Long - OPTIMIZED.ipynb" "Bulls_strategy - Short - OPTIMIZED.ipynb"
```

Useful for resuming after an interrupted run - each notebook's own
"already completed" check (based on which output files already exist) means
re-running a strategy that's partially done picks up where it left off
rather than redoing finished combinations.

### Estimated runtime

Computed directly from each notebook's own parameter grid x your symbol
universe, combined with measured per-task time (~31ms/task including full
parquet export I/O): roughly **2-5 hours at 40 workers for a 700-symbol
universe**, scaling linearly with your real symbol count. Real-world I/O
contention (many workers writing parquet files concurrently) may push this
higher - treat it as a planning estimate, not a guarantee, and watch the
first hour of the log to sanity-check the actual pace.

## 3. What's NOT in this folder

See `../TO_BE_CHECKED/` for:
- 2 strategies (`Swing strategy 3`) where the original has a pre-existing
  bug and the optimized version intentionally does NOT replicate it -
  needs your decision before use.
- 2 strategies (`TEMA Strategy 4`) that use a different code structure our
  test harness doesn't cover - needs a manual smoke test.
- 5 strategies left completely untouched (2 use ADX and were judged too
  risky to vectorize safely, 2 have a pre-existing bug unrelated to this
  work, 1 needs volume data that wasn't available for testing).

## 4. If something goes wrong

- **`FileNotFoundError` on first run**: make sure `RESULTS_BASE_DIR`'s
  *parent* directory exists - the script creates `RESULTS_BASE_DIR` itself
  but the original notebooks' own per-strategy subfolder creation
  (`os.mkdir`, not `os.makedirs`) needs the immediate parent to already
  exist.
- **`ModuleNotFoundError: live_indicators`**: you're not running from inside
  this folder, or `live_indicators.py` got separated from the notebooks.
- **Numpy errors mentioning `DTypePromotionError` or dtype promotion**: this
  is already handled (see the top of `live_indicators.py`) regardless of
  your numpy version - if you still see this, double check you copied the
  current `live_indicators.py`, not an older one.
