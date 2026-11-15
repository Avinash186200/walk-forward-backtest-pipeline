"""
Walk-Forward Monthly Selection — Backtesting Pipeline (Step 2)

Consumes the per-strategy `monthly_stats_all.csv` produced by Step 1 and, for
every parameter combination, computes a recency-weighted walk-forward Sharpe
over a rolling 60-month window, then selects the best parameter set per
(symbol, strategy, side) for each trade-month (Sharpe >= MIN_SHARPE), and
deduplicates to one row per (strategy, side, month).

Per strategy folder it writes:
  - walk_forward_scores_all.csv                       (all rolling scores)
  - walk_forward_selection_sharpe_ge_1.csv            (best per base-combo/month, Sharpe>=1)
  - walk_forward_selection_sharpe_ge_1_unique.csv     (deduped -> Step 3's input)

This is the single-process REFERENCE implementation. It is the optimized core:
the original per-window Python loop is replaced with a vectorized sliding-window
weighted mean/std (same element-wise math -> byte-identical results), and the
redundant duplicate pass is removed. For full runs use
`walk_forward_selection_parallel.py`, which imports this module and processes
many strategies at once for the same per-strategy output, much faster wall-clock.
"""
import os
import glob
import sys
import time
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from numpy.lib.stride_tricks import as_strided

# ============================================================
# CONFIG  — single source of truth (the parallel script imports this)
# ============================================================

# Root folder containing one subfolder per strategy, each with monthly_stats_all.csv
ROOT_DIR = os.environ.get("PIPE_MONTHLY_ROLLING", "<local-path>")

MONTHLY_CSV_PATTERN = "monthly_stats_all.csv"   # input file name inside each strategy folder
WINDOW_MONTHS = 60                              # rolling window (5 years)
MIN_SHARPE = 1.0                                # selection filter
DAYFIRST_MONTH_END_PARSE = True                 # month_end strings are day-first (e.g. 14-01-2026)
SKIP_DONE = True                                # resume: skip folders that already have the _unique.csv

# Recency weights (must sum to 1.0)
WEIGHT_CONFIG = {
    "recent_3m": 0.20,   # last 3 months
    "rest_year1": 0.20,  # months 4-12
    "year2": 0.25,       # months 13-24
    "year3": 0.12,       # months 25-36
    "year4": 0.12,       # months 37-48
    "year5": 0.12,       # months 49-60
}

# Output file names
SCORES_NAME = "walk_forward_scores_all.csv"
SELECTION_NAME = "walk_forward_selection_sharpe_ge_1.csv"
UNIQUE_NAME = "walk_forward_selection_sharpe_ge_1_unique.csv"

# ============================================================
# WEIGHTS
# ============================================================

def build_weights():
    n = WINDOW_MONTHS
    w = np.zeros(n)
    w_r3, w_y1, w_y2, w_y3, w_y4, w_y5 = (
        WEIGHT_CONFIG[k] for k in ["recent_3m", "rest_year1", "year2", "year3", "year4", "year5"]
    )
    for i in range(n):
        m = (n - 1) - i  # 0 = newest
        if m <= 2:    w[i] = w_r3 / 3
        elif m <= 11: w[i] = w_y1 / 9
        elif m <= 23: w[i] = w_y2 / 12
        elif m <= 35: w[i] = w_y3 / 12
        elif m <= 47: w[i] = w_y4 / 12
        else:         w[i] = w_y5 / 12
    w /= w.sum()
    return w

WEIGHTS_60 = build_weights()

# ============================================================
# CORE
# ============================================================

def _sliding_window(a, w):
    """(len(a)-w+1, w) read-only sliding view; as_strided so it works on numpy<1.20."""
    a = np.ascontiguousarray(a)
    n = a.shape[0] - w + 1
    s = a.strides[0]
    return as_strided(a, shape=(n, w), strides=(s, s), writeable=False)

def _find_col(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError("column not found, tried %s" % candidates)
    return None

def expand_to_calendar_months(group):
    """Calendar month-end continuity for one param combo: collapse duplicate
    months, reindex to a full month-end range, zero-fill missing performance,
    ffill/bfill identifiers."""
    g = group.sort_values("month_end").copy()
    g["month_end"] = pd.to_datetime(g["month_end"], errors="coerce", dayfirst=DAYFIRST_MONTH_END_PARSE)
    g = g.dropna(subset=["month_end"])
    agg_map = {}
    if "monthly_return" in g.columns: agg_map["monthly_return"] = "sum"
    if "monthly_trades" in g.columns: agg_map["monthly_trades"] = "sum"
    for c in [c for c in g.columns if c not in ["month_end", "monthly_return", "monthly_trades"]]:
        agg_map[c] = "last"
    g = (g.groupby("month_end", as_index=False).agg(agg_map)
           .sort_values("month_end").reset_index(drop=True))
    if "monthly_return" in g.columns:
        g["is_profitable"] = g["monthly_return"] > 0
    try:
        full_idx = pd.date_range(g["month_end"].min(), g["month_end"].max(), freq="ME")
    except ValueError:
        full_idx = pd.date_range(g["month_end"].min(), g["month_end"].max(), freq="M")
    g = g.set_index("month_end").reindex(full_idx)
    g.index.name = "month_end"
    for c in ["monthly_return", "monthly_trades"]:
        if c in g.columns: g[c] = g[c].fillna(0)
    for c in ["year", "month", "iteration_id", "strategy_name", "symbol", "side",
              "strategy", "base_combo_key", "param_combo_key"]:
        if c in g.columns: g[c] = g[c].ffill().bfill()
    g["year"] = g.index.year
    g["month"] = g.index.month
    return g.reset_index()

def compute_wf_metrics(group):
    """Vectorized rolling weighted-Sharpe. Same element-wise ops as the original
    per-window loop, so outputs match bit-for-bit."""
    group = expand_to_calendar_months(group)
    group = group.sort_values("month_end").reset_index(drop=True)
    n = len(group)
    if n < WINDOW_MONTHS:
        return pd.DataFrame()

    pnl = group["monthly_return"].to_numpy(dtype=float)
    trd = group["monthly_trades"].to_numpy(dtype=float)
    swp = _sliding_window(pnl, WINDOW_MONTHS)   # (n-59, 60)
    swt = _sliding_window(trd, WINDOW_MONTHS)

    wmean = (swp * WEIGHTS_60).sum(axis=1)
    wstd = np.sqrt(((swp - wmean[:, None]) ** 2 * WEIGHTS_60).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(wstd > 0, (wmean / wstd) * np.sqrt(12), np.nan)
    trades = swt.sum(axis=1)

    out = group.iloc[WINDOW_MONTHS - 1:].reset_index(drop=True).copy()
    eval_month = out["month_end"]
    out["eval_month"] = eval_month
    out["trade_month"] = (pd.to_datetime(eval_month) + MonthBegin(1)).dt.normalize()
    out["wf_pnl"] = wmean
    out["wf_sharpe"] = sharpe
    out["wf_trades"] = trades
    return out

def write_unique(strategy_folder):
    """Dedup the selection file to one best row per (strategy, side, month)."""
    sel_path = os.path.join(strategy_folder, SELECTION_NAME)
    if not os.path.exists(sel_path):
        return
    df = pd.read_csv(sel_path)
    strategy_col = _find_col(df, ["strategy_name", "strategy", "strat_name"])
    side_col = _find_col(df, ["side", "long_short", "direction"])
    month_col = "trade_month" if "trade_month" in df.columns else _find_col(df, ["eval_month", "month", "month_end"])
    df[month_col] = pd.to_datetime(df[month_col])
    sharpe_col = _find_col(df, ["wf_sharpe", "sharpe", "sr", "sharp"])
    pnl_col = _find_col(df, ["wf_pnl", "pnl", "monthly_return", "return"])
    df_sorted = df.sort_values([strategy_col, side_col, month_col, sharpe_col, pnl_col],
                               ascending=[True, True, True, False, False])
    unique_df = df_sorted.groupby([strategy_col, side_col, month_col], as_index=False).first()
    unique_df.to_csv(os.path.join(strategy_folder, UNIQUE_NAME), index=False)

def process_strategy(strategy_folder):
    """Full Step-2 for one strategy folder. Returns (folder, status)."""
    if SKIP_DONE and os.path.exists(os.path.join(strategy_folder, UNIQUE_NAME)):
        return strategy_folder, "skip-done"
    files = glob.glob(os.path.join(strategy_folder, MONTHLY_CSV_PATTERN))
    if not files:
        return strategy_folder, "no-file"
    df = pd.read_csv(files[0])
    req = ["year", "month", "month_end", "monthly_return", "monthly_trades", "iteration_id", "strategy_name"]
    if any(c not in df.columns for c in req):
        return strategy_folder, "missing-cols"
    df["month_end"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)) + pd.offsets.MonthEnd(0)
    # memory-efficient first/last token: n=1 avoids building the full list-of-tokens
    # for multi-million-row iteration_id columns. Identical to .str.split("_").str[0]/[-1].
    _it = df["iteration_id"].astype(str)
    df["symbol"] = _it.str.split("_", n=1).str[0]
    df["side"] = _it.str.rsplit("_", n=1).str[-1]
    df["strategy"] = df["strategy_name"].astype(str)
    df["base_combo_key"] = df["symbol"].astype(str) + "|" + df["strategy"].astype(str) + "|" + df["side"].astype(str)
    df["param_combo_key"] = df["iteration_id"].astype(str)

    all_res = []
    for key, g in df.groupby("param_combo_key"):
        g = g.copy()
        g["month_end"] = pd.to_datetime(g["month_end"], errors="coerce")
        g = g.dropna(subset=["month_end"])
        res = compute_wf_metrics(g)
        if not res.empty:
            res["strategy_folder"] = os.path.basename(strategy_folder)
            all_res.append(res)
    if not all_res:
        return strategy_folder, "no-combos"

    scores_df = pd.concat(all_res, ignore_index=True)
    scores_df.to_csv(os.path.join(strategy_folder, SCORES_NAME), index=False)

    scores_sorted = scores_df.sort_values(["base_combo_key", "eval_month", "wf_sharpe", "wf_pnl"],
                                          ascending=[True, True, False, False])
    best = scores_sorted.groupby(["base_combo_key", "eval_month"], as_index=False).first()
    final_sel = best[best["wf_sharpe"] >= MIN_SHARPE].copy()
    try:
        parts = final_sel["base_combo_key"].str.split("|", expand=True)
        final_sel["symbol"] = parts[0]
        final_sel["strategy"] = parts[1]
        final_sel["side"] = parts[2]
        final_sel.to_csv(os.path.join(strategy_folder, SELECTION_NAME), index=False)
    except Exception:
        pass
    write_unique(strategy_folder)
    return strategy_folder, "ok"

def iter_strategy_folders(root_dir=None):
    root_dir = root_dir or ROOT_DIR
    return [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, f))]

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT_DIR
    folders = iter_strategy_folders(root)
    print("Step 2: %d strategy folders under %s" % (len(folders), root), flush=True)
    t0 = time.time()
    for fld in folders:
        _, status = process_strategy(fld)
        print("  %-11s %s" % (status, os.path.basename(fld)), flush=True)
    print("DONE in %.1fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
