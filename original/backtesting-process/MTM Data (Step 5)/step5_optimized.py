"""
Step 5 — Create MTM data (optimized).

For each strategy, read selected_trade_level_all_months.csv ONCE (the original
re-read it once per year), build the per-(strategy, year) MTM rows, and write one
combined CSV per year. Parallel across strategies.

Output data is identical to the original; strategy order is made deterministic
(sorted) instead of the original's non-deterministic set() ordering, so runs are
reproducible. Transform logic (asset-class classification, margin merge, column
set) is byte-for-byte the original (see step5_data.py for the lists/margin table).

Usage: python step5_optimized.py [ROOT] [OUT_DIR] [N_WORKERS] [STRATEGY_LIMIT]
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import step5_data as sd

ROOT_DIR = os.environ.get("PIPE_MONTHLY_ROLLING", "<local-path>")
MTM_DATA_EXPORT_PATH = os.environ.get("PIPE_MTM_OUT", "<local-path>")
TRADE_FILE = "selected_trade_level_all_months.csv"
YEAR_LIST = [2021, 2022, 2023, 2024, 2025, 2026]

COLUMNS_LIST = ['Strategy', 'Opening Date', 'Currency', 'Timeframe', 'Condition',
                'Trading Price Point', 'Stop_Loss', 'TakeProfit_by_pips', 'Closing Date',
                'Capping Date', 'SL to Feed', 'Data_Type', 'New P/L_pips', 'Year', 'Month',
                'Day', 'Hour', 'Day Level Date', 'Currency Type', 'BROKER Margin', 'Open close Flag']
_FILTER_VALUES = ["Non-Exotic", "Exotic", "Commodities", "ETF", "bond",
                  "Single Stock CFD", "Index CFD", "NDF", "Single Stock CFD"]


def _transform_year(raw, year):
    """Verbatim transform from the original cell, for one (strategy-data, year)."""
    d = raw.copy()
    d['Open close Flag'] = 'Closed'
    d = d[(d['EntryDate Year'] == year) & (d['ClosingDate Year'] == year)]
    d = d[d['Timeframe'] == 'D1'].reset_index(drop=True)
    d = d.rename(columns={'Profit_Loss': 'New P/L_pips'})
    d['Strategy'] = d['Strategy'].str.replace('77 strategy', '77').str.replace('RSI-Larry 25', 'RSI 25')
    d['date'] = pd.to_datetime(d['date'])
    d['Closing Date'] = pd.to_datetime(d['Closing Date'])
    d['Year'] = d['date'].dt.year
    d['Month'] = d['date'].dt.month
    d['Day'] = d['date'].dt.day
    filters = [d['Currency'].isin(sd.Major_pairs_list),
               d['Currency'].isin(sd.Exotic_pairs_list),
               d['Currency'].isin(sd.Commodities_pairs_list),
               d['Currency'].isin(sd.ETF_pairs_list),
               d['Currency'].isin(sd.bond),
               d['Currency'].isin(sd.futures),
               d['Currency'].isin(sd.Index_CFDs),
               d['Currency'].isin(sd.NDFs),
               d['Currency'].isin(sd.Single_Stock_CFDs)]
    d['Currency Type'] = np.select(filters, _FILTER_VALUES)
    d = d.rename(columns={'date': 'Opening Date', 'Train Cut Off': 'Cut Off', 'Model Passing': 'Passed Trades'})
    d['Capping Date'] = d['Closing Date']
    d['Data_Type'] = 'SL_Updated'
    d['Stop_Loss'] = 0
    d['TakeProfit_by_pips'] = 0
    d['SL to Feed'] = np.where(d['Condition'] == 'Long', d['Exit_bidclose'], d['Exit_askclose'])
    d['Day Level Date'] = d['Opening Date']
    d['Hour'] = d['Opening Date'].dt.hour
    d1 = d.merge(sd.BROKER_Margin_df, on=['Currency'], how='left')
    final = d1[COLUMNS_LIST].copy()
    final['Closing Type'] = 'No_SL_TP'
    return final


def _process_strategy(args):
    root, strategy = args
    path = os.path.join(root, strategy, TRADE_FILE)
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None
    return strategy, {year: _transform_year(raw, year) for year in YEAR_LIST}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else MTM_DATA_EXPORT_PATH
    nworkers = int(sys.argv[3]) if len(sys.argv) > 3 else min(12, max(1, (os.cpu_count() or 2) - 1))
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None

    strategies = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if limit:
        strategies = strategies[:limit]
    os.makedirs(out_dir, exist_ok=True)

    print("Step 5: %d strategies, %d workers -> %s" % (len(strategies), nworkers, out_dir), flush=True)
    t0 = time.time()
    with Pool(processes=min(nworkers, len(strategies))) as pool:
        results = [r for r in pool.map(_process_strategy, [(root, s) for s in strategies]) if r is not None]
    # reassemble in sorted-strategy order (deterministic)
    by_strategy = dict(results)
    for year in YEAR_LIST:
        parts = [by_strategy[s][year] for s in strategies if s in by_strategy]
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS_LIST + ['Closing Type'])
        out = os.path.join(out_dir, "%s all selected comb mtm data Test.csv" % year)
        combined.to_csv(out, index=False)
        print("  %s -> %d rows" % (year, len(combined)), flush=True)
    print("DONE in %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
