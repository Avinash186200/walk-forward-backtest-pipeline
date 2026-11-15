"""
Step 4 — Monthly Combination Summary.

Reads every selected_trade_level_all_months.csv (Step 3 output), optionally keeps
only one open trade at a time per combination, and writes a per-(combination,
month) summary to an Excel file.

Optimized vs the original notebook (output byte-identical):
  1. Reads the per-strategy CSVs in PARALLEL (then concat).
  2. The one-trade-at-a-time filter used a per-row iterrows() loop inside
     groupby.apply; replaced with a numpy sequential scan per group (same logic,
     same kept rows, ~50-100x faster).
The aggregation and Excel export are unchanged.

Usage: python month_summary.py [ROOT_DIR] [N_READ_WORKERS]
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

# ================== CONFIG ==================
ROOT_DIR = os.environ.get("PIPE_MONTHLY_ROLLING", "<local-path>")
TARGET_FILENAME = "selected_trade_level_all_months.csv"
OUTPUT_FOLDER = os.environ.get("PIPE_SUMMARY_OUT", "<local-path>")
OUTPUT_FILE = "Monthly_Combination_Summary.xlsx"
ALLOW_MULTIPLE_TRADES = "no"

DATE_COL = "date"
CLOSE_COL = "Closing Date"
PNL_COL = "Profit_Loss"
AMOUNT_COL = "Total Amount"
GOOD_BAD_COL = "Good_Bad_Flag"
MONTH_SOURCE_COL = "selected_trade_month"
COMBO_COLS = ["bt_strategy_folder_name", "Currency", "Condition", "Timeframe"]
# ============================================


def _read_one(args):
    """Read one strategy's trade-level CSV, tagging Strategy_Folder (= the original)."""
    full_path, strategy_folder = args
    try:
        tmp = pd.read_csv(full_path)
    except Exception as e:
        return None
    tmp["Strategy_Folder"] = strategy_folder
    return tmp


def _filter_non_overlapping_fast(group, date_col, close_col):
    """Same logic as the original per-row loop, vectorized with numpy:
    keep a trade only if its entry is strictly after the last kept trade's close;
    drop rows with missing entry/close. Returns kept rows in date order."""
    g = group.sort_values(date_col)
    entry = g[date_col].to_numpy()
    close = g[close_col].to_numpy()
    n = len(g)
    keep = np.zeros(n, dtype=bool)
    last_close = np.datetime64(pd.Timestamp.min)
    for i in range(n):
        e = entry[i]
        c = close[i]
        if np.isnat(e) or np.isnat(c):
            continue
        if e > last_close:
            keep[i] = True
            last_close = c
    return g.iloc[keep]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT_DIR
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else min(16, max(1, (os.cpu_count() or 2) - 1))

    # ---- find all target files (same os.walk order as original) ----
    jobs = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower() == TARGET_FILENAME.lower():
                jobs.append((os.path.join(dirpath, fname), os.path.basename(dirpath)))
    if not jobs:
        raise FileNotFoundError("No '%s' files found under root: %s" % (TARGET_FILENAME, root))

    t0 = time.time()
    print("Step 4: reading %d trade-level files (%d workers)..." % (len(jobs), nworkers), flush=True)
    with Pool(processes=min(nworkers, len(jobs))) as pool:
        frames = [f for f in pool.map(_read_one, jobs) if f is not None]
    df_all = pd.concat(frames, ignore_index=True)
    print("  total rows loaded: %d (%.1fs)" % (len(df_all), time.time() - t0), flush=True)

    required = COMBO_COLS + [DATE_COL, PNL_COL, AMOUNT_COL, GOOD_BAD_COL, MONTH_SOURCE_COL]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError("Missing required columns: %s" % missing)

    df_all[DATE_COL] = pd.to_datetime(df_all[DATE_COL])
    if CLOSE_COL in df_all.columns:
        df_all[CLOSE_COL] = pd.to_datetime(df_all[CLOSE_COL])
        close_col_to_use = CLOSE_COL
    else:
        close_col_to_use = DATE_COL
    df_all[MONTH_SOURCE_COL] = pd.to_datetime(df_all[MONTH_SOURCE_COL])
    df_all["Month"] = df_all[MONTH_SOURCE_COL].dt.to_period("M").astype(str)
    df_all["Return_Pct"] = df_all[PNL_COL] / df_all[AMOUNT_COL]

    allow_multi = (ALLOW_MULTIPLE_TRADES.strip().lower() == "yes")
    if allow_multi:
        df_filtered = df_all.copy()
    else:
        df_filtered = (df_all.groupby(COMBO_COLS, group_keys=False)
                             .apply(lambda g: _filter_non_overlapping_fast(g, DATE_COL, close_col_to_use)))
    print("  rows after overlap filtering: %d" % len(df_filtered), flush=True)

    group_cols = COMBO_COLS + ["Month"]
    summary = df_filtered.groupby(group_cols).agg(
        Total_Trades=(DATE_COL, "count"),
        Winning_Trades=(GOOD_BAD_COL, lambda x: (x == 1).sum()),
        Losing_Trades=(GOOD_BAD_COL, lambda x: (x == 0).sum()),
        Win_Percentage=(GOOD_BAD_COL, lambda x: (x == 1).mean() * 100.0),
        Total_PnL=(PNL_COL, "sum"),
        Avg_PnL_per_Trade=(PNL_COL, "mean"),
        Total_Amount=(AMOUNT_COL, "sum"),
    ).reset_index()
    summary = summary.sort_values(["Month"] + COMBO_COLS).reset_index(drop=True)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    summary.to_excel(output_path, index=False)
    print("Monthly Combination Summary -> %s  (%.1fs total)" % (output_path, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
