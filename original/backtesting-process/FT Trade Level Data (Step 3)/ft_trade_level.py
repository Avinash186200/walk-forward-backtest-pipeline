"""
Step 3 core — extract FT trade-level data for selected combos.

Faithful to the original notebook's cell 2 (selected_trade_level_all_months.csv)
and cell 7 (Train data + Train_Test data per month). The per-folder logic is
byte-for-byte the original; the only changes are (a) a folder-local parquet cache
shared across the two outputs so each parquet is read once, and (b) the driver
runs folders in parallel under a memory budget (see step3_parallel.py).
"""
import os
import glob
import pandas as pd

# ---- CONFIG (single source of truth; parallel runner imports this) ----
SELECTION_ROOT = os.environ.get("PIPE_MONTHLY_ROLLING", "<local-path>")
BT_RESULTS_ROOT = os.environ.get("PIPE_BT_RESULTS", "<local-path>")
SELECTION_FILE_NAME = "walk_forward_selection_sharpe_ge_1_unique.csv"
OUTPUT_TRADELEVEL_NAME = "selected_trade_level_all_months.csv"
TRAIN_OUTPUT_SUBFOLDER_NAME = "Train data"
TRAIN_TEST_OUTPUT_SUBFOLDER_NAME = "Train_Test data"
TRADE_SUBFOLDER_CANDIDATES = [
    "trade level data", "Trade level data", "Trade Level Data",
    "Trade_Level_Data", "TradeLevelData",
]
SKIP_DONE = True  # resume: skip folders that already have the selected_trade_level output


def find_column(df, candidates, what_for, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError("Could not find column for %s. Tried: %s" % (what_for, candidates))
    return None


def find_trade_subfolder(bt_strategy_folder):
    for cand in TRADE_SUBFOLDER_CANDIDATES:
        path = os.path.join(bt_strategy_folder, cand)
        if os.path.isdir(path):
            return path
    return None


def load_iteration_trades(bt_strategy_folder_name, iteration_id, cache):
    key = (bt_strategy_folder_name, iteration_id)
    if key in cache:
        return cache[key]
    bt_strategy_folder = os.path.join(BT_RESULTS_ROOT, str(bt_strategy_folder_name))
    if not os.path.isdir(bt_strategy_folder):
        cache[key] = pd.DataFrame(); return cache[key]
    trade_subfolder = find_trade_subfolder(bt_strategy_folder)
    if trade_subfolder is None:
        cache[key] = pd.DataFrame(); return cache[key]
    parquet_path = os.path.join(trade_subfolder, "%s.parquet" % iteration_id)
    if not os.path.exists(parquet_path):
        matches = glob.glob(os.path.join(trade_subfolder, "*%s*.parquet" % iteration_id))
        if matches:
            parquet_path = matches[0]
        else:
            cache[key] = pd.DataFrame(); return cache[key]
    try:
        trades = pd.read_parquet(parquet_path)
    except Exception:
        cache[key] = pd.DataFrame()
        return cache[key]
    cache[key] = trades
    return trades


def add_entry_year_month(trades_all):
    if trades_all.empty:
        return trades_all
    trades = trades_all.copy()
    if "EntryDate Year" in trades.columns and "EntryDate Month" in trades.columns:
        trades["_entry_year_tmp"] = trades["EntryDate Year"]
        trades["_entry_month_tmp"] = trades["EntryDate Month"]
    else:
        if "date" not in trades.columns:
            trades["_entry_year_tmp"] = pd.NA
            trades["_entry_month_tmp"] = pd.NA
            return trades
        dates = pd.to_datetime(trades["date"], dayfirst=True, errors="coerce")
        trades["_entry_year_tmp"] = dates.dt.year
        trades["_entry_month_tmp"] = dates.dt.month
    return trades


# --- cell 2 filter: entry month == trade_month (no tmp cols if EntryDate cols exist) ---
def filter_trade_month_cell2(trades_all, trade_month):
    if trades_all.empty:
        return trades_all
    if "EntryDate Year" in trades_all.columns and "EntryDate Month" in trades_all.columns:
        year_col, month_col = "EntryDate Year", "EntryDate Month"
    else:
        if "date" not in trades_all.columns:
            return pd.DataFrame()
        trades_all = trades_all.copy()
        dates = pd.to_datetime(trades_all["date"], dayfirst=True, errors="coerce")
        trades_all["_entry_year_tmp"] = dates.dt.year
        trades_all["_entry_month_tmp"] = dates.dt.month
        year_col, month_col = "_entry_year_tmp", "_entry_month_tmp"
    mask = (trades_all[year_col] == trade_month.year) & (trades_all[month_col] == trade_month.month)
    return trades_all.loc[mask].copy()


# --- cell 7 filters: use add_entry_year_month (always adds tmp cols) ---
def filter_train_before_month(trades_all, trade_month):
    if trades_all.empty:
        return trades_all
    trades = add_entry_year_month(trades_all)
    if trades["_entry_year_tmp"].isna().all():
        return pd.DataFrame()
    ty, tm = trade_month.year, trade_month.month
    mask = (trades["_entry_year_tmp"] < ty) | ((trades["_entry_year_tmp"] == ty) & (trades["_entry_month_tmp"] < tm))
    return trades.loc[mask].copy()


def filter_trade_month_cell7(trades_all, trade_month):
    if trades_all.empty:
        return trades_all
    trades = add_entry_year_month(trades_all)
    if trades["_entry_year_tmp"].isna().all():
        return pd.DataFrame()
    ty, tm = trade_month.year, trade_month.month
    mask = (trades["_entry_year_tmp"] == ty) & (trades["_entry_month_tmp"] == tm)
    return trades.loc[mask].copy()


def _resolve_cols(sel):
    return {
        "bt": find_column(sel, ["strategy_folder", "strategy_folder_name", "bt_strategy_folder", "bt_folder"], "BT folder"),
        "iter": find_column(sel, ["iteration_id", "iteration", "combo_key", "iteration_name"], "iteration id"),
        "sym": find_column(sel, ["symbol", "org_symbol", "ticker"], "symbol"),
        "side": find_column(sel, ["side", "long_short", "direction"], "side"),
        "tm": find_column(sel, ["trade_month"], "trade month"),
    }


def _bt_name(raw):
    raw = str(raw)
    return os.path.basename(raw) if ("\\" in raw or "/" in raw) else raw


def process_strategy_selection_folder(selection_folder):
    """Produce selected_trade_level_all_months.csv (cell 2) + Train data / Train_Test
    data per month (cell 7), sharing one parquet cache. Byte-identical per output."""
    sel_path = os.path.join(selection_folder, SELECTION_FILE_NAME)
    if not os.path.exists(sel_path):
        return selection_folder, "no-selection"
    if SKIP_DONE and os.path.exists(os.path.join(selection_folder, OUTPUT_TRADELEVEL_NAME)):
        return selection_folder, "skip-done"
    sel = pd.read_csv(sel_path)
    if sel.empty:
        return selection_folder, "empty-sel"
    cols = _resolve_cols(sel)
    sel["trade_month"] = pd.to_datetime(sel[cols["tm"]])

    cache = {}
    trade_frames = []                # cell 2
    month_to_train = {}              # cell 7 train
    month_to_combined = {}           # cell 7 train+test

    for _, row in sel.iterrows():
        bt = _bt_name(row[cols["bt"]])
        iteration_id = str(row[cols["iter"]])
        symbol = str(row[cols["sym"]])
        side = row[cols["side"]]
        trade_month = row["trade_month"]

        trades_all = load_iteration_trades(bt, iteration_id, cache)
        if trades_all.empty:
            continue

        # cell 2: selected_trade_level (entry month == trade_month)
        tm2 = filter_trade_month_cell2(trades_all, trade_month)
        if not tm2.empty:
            tm2 = tm2.copy()
            tm2["selected_trade_month"] = trade_month
            tm2["bt_strategy_folder_name"] = bt
            tm2["iteration_id"] = iteration_id
            tm2["side"] = side
            tm2["symbol_from_selection"] = symbol
            trade_frames.append(tm2)

        key = trade_month.normalize()

        # cell 7 train: entry month < trade_month
        tr = filter_train_before_month(trades_all, trade_month)
        if not tr.empty:
            tr = tr.copy()
            tr["selected_trade_month"] = trade_month
            tr["bt_strategy_folder_name"] = bt
            tr["iteration_id"] = iteration_id
            tr["side"] = side
            tr["symbol_from_selection"] = symbol
            tr["dataset_type"] = "train"
        # cell 7 test: entry month == trade_month
        te = filter_trade_month_cell7(trades_all, trade_month)
        if not te.empty:
            te = te.copy()
            te["selected_trade_month"] = trade_month
            te["bt_strategy_folder_name"] = bt
            te["iteration_id"] = iteration_id
            te["side"] = side
            te["symbol_from_selection"] = symbol
            te["dataset_type"] = "test"

        if not tr.empty:
            month_to_train.setdefault(key, []).append(tr)
        combined_parts = []
        if not tr.empty:
            combined_parts.append(tr)
        if not te.empty:
            combined_parts.append(te)
        if combined_parts:
            month_to_combined.setdefault(key, []).append(pd.concat(combined_parts, ignore_index=True))

    # write cell-2 output
    if trade_frames:
        all_trades = pd.concat(trade_frames, ignore_index=True)
        all_trades.to_csv(os.path.join(selection_folder, OUTPUT_TRADELEVEL_NAME), index=False)

    # write cell-7 train + train_test
    if month_to_train:
        train_out = os.path.join(selection_folder, TRAIN_OUTPUT_SUBFOLDER_NAME)
        os.makedirs(train_out, exist_ok=True)
        for month_key, frames in month_to_train.items():
            mdf = pd.concat(frames, ignore_index=True)
            mdf.to_csv(os.path.join(train_out, "train_trades_%s.csv" % month_key.strftime("%Y%m%d")), index=False)
    if month_to_combined:
        tt_out = os.path.join(selection_folder, TRAIN_TEST_OUTPUT_SUBFOLDER_NAME)
        os.makedirs(tt_out, exist_ok=True)
        for month_key, frames in month_to_combined.items():
            mdf = pd.concat(frames, ignore_index=True)
            mdf["dataset_type_order"] = mdf["dataset_type"].map({"train": 0, "test": 1})
            mdf = mdf.sort_values(["dataset_type_order"]).drop(columns=["dataset_type_order"])
            mdf.to_csv(os.path.join(tt_out, "train_test_trades_%s.csv" % month_key.strftime("%Y%m%d")), index=False)

    return selection_folder, ("ok" if trade_frames or month_to_train else "no-trades")


def iter_selection_folders(root=None):
    root = root or SELECTION_ROOT
    return [os.path.join(root, n) for n in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, n))]
