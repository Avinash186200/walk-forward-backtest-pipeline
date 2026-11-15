"""
Step 5 (companion) — per-year "combinations list" file.

This is S-5 cell 7: for each year it reads that year's MTM `... Test.csv` (produced by
step5_optimized.py), takes the unique (Strategy, Currency, Condition, Timeframe) combos,
builds the Combination key, and writes:

    <year> All selected comb percent_invested and Risk.xlsx

with Risk blank and percent_invested = 0.01 (a flat 1% per combination — the seed sizing
the downstream MTM/portfolio engine then scales). This is the second per-year input the
Step-6+ pipeline needs alongside the MTM Test csv.

Logic is byte-for-byte the original cell. Usage:
    python build_combinations_list.py [MTM_DIR] [YEARS]
        MTM_DIR  folder holding the "<year> all selected comb mtm data Test.csv" files
                 (default: PIPE_MTM_OUT env or the standard MTM output folder)
        YEARS    comma list, default 2021,2022,2023,2024,2025,2026
"""
import os
import sys
import pandas as pd

MTM_DATA_EXPORT_PATH = os.environ.get(
    "PIPE_MTM_OUT",
    "<local-path>",
)
YEAR_LIST = [2021, 2022, 2023, 2024, 2025, 2026]


def build_one(mtm_dir, year):
    df = pd.read_csv(os.path.join(mtm_dir, "%s all selected comb mtm data Test.csv" % year))
    comb_list = df[['Strategy', 'Currency', 'Condition', 'Timeframe']].drop_duplicates().reset_index(drop=True)
    comb_list['Combination'] = (comb_list['Strategy'].map(str) + "_" +
                                comb_list['Currency'].str[0:3].map(str) + "_" +
                                comb_list['Currency'].str[4:].map(str) + "_" +
                                comb_list['Condition'].map(str) + "_" +
                                comb_list['Timeframe'].map(str))
    comb_list['Risk'] = ""
    comb_list['percent_invested'] = 0.01
    out = os.path.join(mtm_dir, "%s All selected comb percent_invested and Risk.xlsx" % year)
    comb_list.to_excel(out, index=False)
    return out, len(comb_list)


def main():
    mtm_dir = sys.argv[1] if len(sys.argv) > 1 else MTM_DATA_EXPORT_PATH
    years = ([int(y) for y in sys.argv[2].split(",")] if len(sys.argv) > 2 else YEAR_LIST)
    for y in years:
        try:
            out, n = build_one(mtm_dir, y)
            print("  %s -> %d combinations  (%s)" % (y, n, os.path.basename(out)), flush=True)
        except FileNotFoundError:
            print("  %s -> SKIP (no MTM Test csv)" % y, flush=True)


if __name__ == "__main__":
    main()
