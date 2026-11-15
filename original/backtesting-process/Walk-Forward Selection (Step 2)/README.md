# Walk-Forward Monthly Selection — Backtesting Pipeline (Step 2)

Consumes the per-strategy **`monthly_stats_all.csv`** produced by Step 1 and, for
every parameter combination, computes a **recency-weighted walk-forward Sharpe**
over a rolling **60-month** window. It then picks the best parameter set per
`(symbol, strategy, side)` for each trade-month (keeping `Sharpe ≥ MIN_SHARPE`)
and deduplicates to one row per `(strategy, side, month)`.

This is Step 2 of the backtesting pipeline: it turns raw monthly performance into
a clean monthly **selection** of which parameter set to trade next month — the
input consumed by Step 3.

## What it writes (per strategy folder)

| File | Contents |
|------|----------|
| `walk_forward_scores_all.csv` | Every rolling-window score for every param combo |
| `walk_forward_selection_sharpe_ge_1.csv` | Best param per (base-combo, eval-month), filtered `Sharpe ≥ 1` |
| `walk_forward_selection_sharpe_ge_1_unique.csv` | Deduped to one best row per (strategy, side, month) — **Step 3's input** |

## Scripts

| File | Use it when |
|------|-------------|
| **`walk_forward_selection.py`** | Single-process. The reference implementation — correct and already optimized (vectorized rolling math). Run this for the simplest thing. |
| **`walk_forward_selection_parallel.py`** | Cross-strategy parallel — one worker per folder, memory-budget scheduled. Good when folders are similar in size. |
| **`walk_forward_selection_within.py`** | **Recommended for full runs.** Within-strategy parallel: one folder at a time, but that folder's combos are split across all cores. Best when folder sizes are skewed (a few giant folders). |

Both parallel runners import `walk_forward_selection.py`, so the **per-strategy
output is byte-identical** regardless of which you run.

## Why two parallel runners — and why within-strategy wins here

The strategy folders are **extremely skewed in size**: most are a few MB, but a
handful are **4 GB / ~7 million rows** (≈100k param-combos for one strategy).
That skew breaks the obvious "one worker per folder" design:

- **Cross-strategy** (`_parallel`): N folders run at once. Two failure modes on
  this data — (1) several multi-GB folders load simultaneously and **exhaust RAM**
  (a 4 GB CSV peaks at ~15–30 GB in pandas); and (2) once you throttle the budget
  so a giant runs alone, **one worker grinds its 100k combos sequentially while
  the other N-1 cores sit idle**. We measured a single giant taking ~57 min this
  way, with the rest of the machine doing nothing.

- **Within-strategy** (`_within`): folders run **one at a time** (so only one
  folder's data is ever in RAM → the giants can't OOM), and that folder's combos
  are **partitioned across all cores**. The bottleneck — the per-combo
  walk-forward loop — is exactly what gets parallelized, so the giants get the
  full machine and finish ~10× faster. Small folders finish in seconds; the
  giants benefit most.

Rule of thumb: **uniform folder sizes → `_parallel`; skewed sizes / giant folders
→ `_within`.** Output is identical either way (combos are partitioned in sorted
key order and reassembled in order).

## What was optimized

The original notebook computed the rolling weighted Sharpe with a **per-window
Python `for` loop** (`row.to_dict()`每 iteration) — O(combos × months × 60) — and
ran a redundant duplicate pass. This version:

1. **Vectorizes** the rolling weighted mean / std with a sliding-window view.
   The weighted mean and std use the *same element-wise operations* as the
   original (not a BLAS dot-product), so results are **byte-for-byte identical**
   (verified with `cmp` against the original, including a 2.1M-row scores file).
2. **Parallelizes** across strategy folders (one worker per strategy).
3. **Drops** the redundant duplicate pass.

On a 94-folder dataset this turns a ~37-hour sequential run into roughly
**~1–2 hours**, with identical output.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.7+ (the sliding window uses `as_strided`, so it works on
older numpy too). Depends on pandas, numpy.

## Configure

Edit the **CONFIG block at the top of `walk_forward_selection.py`** — the single
source of truth (the parallel script imports it):

```python
ROOT_DIR = r"...\Results\Monthly Rolling"   # one subfolder per strategy, each with monthly_stats_all.csv
WINDOW_MONTHS = 60                          # rolling window (5 years)
MIN_SHARPE = 1.0                            # selection filter
WEIGHT_CONFIG = { ... }                     # recency weights (must sum to 1.0)
SKIP_DONE = True                            # resume: skip folders that already have the _unique.csv
```

## Run

```bash
# single-process reference
python walk_forward_selection.py

# parallel (recommended) — optional ROOT_DIR and worker-count overrides
python walk_forward_selection_parallel.py
python walk_forward_selection_parallel.py "<local-path>" 30
```

**Resumable:** folders that already have `walk_forward_selection_sharpe_ge_1_unique.csv`
are skipped, so a re-run after an interruption picks up where it left off.
