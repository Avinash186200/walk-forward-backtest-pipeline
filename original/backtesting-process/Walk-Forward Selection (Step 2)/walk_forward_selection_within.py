"""
Step 2 runner — WITHIN-STRATEGY parallelism.

Processes strategy folders ONE AT A TIME (so only one folder's data is ever in
RAM -> no cross-strategy OOM, even for the 4 GB / 7M-row giants), but splits that
folder's ~100k param-combos across all worker cores. This fixes the real
bottleneck: the per-combo walk-forward loop, which a single worker would grind
sequentially.

Output is byte-identical to walk_forward_selection.py: same prep, the combos are
partitioned in SORTED key order and re-assembled in that order, so the scores
CSV row order is unchanged.

Usage: python walk_forward_selection_within.py [ROOT_DIR] [N_WORKERS]
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

import walk_forward_selection as wfs

PARTS_PER_WORKER = 3   # more partitions than workers -> better load balance via dynamic map


def _compute_partition(sub_df):
    """Run the per-combo walk-forward for every combo in this partition,
    in sorted key order, returning the concatenated scores (or None)."""
    out = []
    for _, g in sub_df.groupby("param_combo_key", sort=True):
        g = g.copy()
        g["month_end"] = pd.to_datetime(g["month_end"], errors="coerce")
        g = g.dropna(subset=["month_end"])
        res = wfs.compute_wf_metrics(g)
        if not res.empty:
            out.append(res)
    if not out:
        return None
    return pd.concat(out, ignore_index=True)


def process_folder_within(folder, pool, nparts):
    if wfs.SKIP_DONE and os.path.exists(os.path.join(folder, wfs.UNIQUE_NAME)):
        return "skip-done"
    files = glob.glob(os.path.join(folder, wfs.MONTHLY_CSV_PATTERN))
    if not files:
        return "no-file"
    df = pd.read_csv(files[0])
    req = ["year", "month", "month_end", "monthly_return", "monthly_trades", "iteration_id", "strategy_name"]
    if any(c not in df.columns for c in req):
        return "missing-cols"
    # --- same prep as walk_forward_selection.process_strategy ---
    df["month_end"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)) + pd.offsets.MonthEnd(0)
    _it = df["iteration_id"].astype(str)
    df["symbol"] = _it.str.split("_", n=1).str[0]
    df["side"] = _it.str.rsplit("_", n=1).str[-1]
    df["strategy"] = df["strategy_name"].astype(str)
    df["base_combo_key"] = df["symbol"].astype(str) + "|" + df["strategy"].astype(str) + "|" + df["side"].astype(str)
    df["param_combo_key"] = df["iteration_id"].astype(str)

    # partition combos into contiguous SORTED-key blocks (preserves global order on concat)
    keys = np.sort(df["param_combo_key"].unique())
    if len(keys) == 0:
        return "no-combos"
    nparts = min(nparts, len(keys))
    blocks = np.array_split(keys, nparts)
    key_to_part = {}
    for pi, blk in enumerate(blocks):
        for k in blk:
            key_to_part[k] = pi
    part_idx = df["param_combo_key"].map(key_to_part).to_numpy()
    parts = [df.iloc[np.where(part_idx == pi)[0]] for pi in range(nparts)]

    results = [r for r in pool.map(_compute_partition, parts) if r is not None]
    if not results:
        return "no-combos"
    scores_df = pd.concat(results, ignore_index=True)
    scores_df["strategy_folder"] = os.path.basename(folder)
    scores_df.to_csv(os.path.join(folder, wfs.SCORES_NAME), index=False)

    scores_sorted = scores_df.sort_values(["base_combo_key", "eval_month", "wf_sharpe", "wf_pnl"],
                                          ascending=[True, True, False, False])
    best = scores_sorted.groupby(["base_combo_key", "eval_month"], as_index=False).first()
    final_sel = best[best["wf_sharpe"] >= wfs.MIN_SHARPE].copy()
    try:
        p = final_sel["base_combo_key"].str.split("|", expand=True)
        final_sel["symbol"] = p[0]; final_sel["strategy"] = p[1]; final_sel["side"] = p[2]
        final_sel.to_csv(os.path.join(folder, wfs.SELECTION_NAME), index=False)
    except Exception:
        pass
    wfs.write_unique(folder)
    return "ok"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else wfs.ROOT_DIR
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    folders = wfs.iter_strategy_folders(root)
    # smallest-first: quick wins early, the giants (which benefit most from the
    # within-folder parallelism) last
    folders.sort(key=lambda f: os.path.getsize(os.path.join(f, wfs.MONTHLY_CSV_PATTERN))
                 if os.path.exists(os.path.join(f, wfs.MONTHLY_CSV_PATTERN)) else 0)
    n = len(folders)
    nparts = nworkers * PARTS_PER_WORKER
    print("Step 2 (within-strategy): %d folders, one at a time, %d workers x %d parts\n  root: %s"
          % (n, nworkers, PARTS_PER_WORKER, root), flush=True)
    t0 = time.time()
    with Pool(processes=nworkers) as pool:
        for i, folder in enumerate(folders, 1):
            ft = time.time()
            try:
                status = process_folder_within(folder, pool, nparts)
            except Exception as e:
                status = "ERR:%s" % e
            print("  [%d/%d] %-11s %-40s %.1fs" % (i, n, status, os.path.basename(folder), time.time() - ft), flush=True)
    print("DONE in %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
