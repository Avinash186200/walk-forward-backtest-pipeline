"""
Builds a new combined MTM dataset pairing FX T+3 with Commodities T+15
(cross-horizon combination, requested because those two individually had the
best Sharpe ratios). Mirrors the exact rule already used for the existing
same-horizon Combined folders (verified against T+3 and T+15, all years):
  - Commodities side: kept in full, unfiltered.
  - FX side: drop any combination whose Currency contains XAU, XAG, or XBR
    (metals/oil quoted as FX-style pairs, already booked under Commodities -
    dropped here to avoid double-counting the same underlying instrument).
"""
import pandas as pd
from pathlib import Path

BASE = Path("<local-path>")
FX_DIR = BASE / "T+3" / "FX MTM Data"
COMM_DIR = BASE / "T+15" / "Comm MTM Data"
OUT_DIR = BASE / "Combined (FX T+3 + Comm T+15)" / "MTM Data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
METAL_TAGS = ("XAU", "XAG", "XBR")


def drop_metal_fx(df, currency_col="Currency"):
    mask = df[currency_col].astype(str).apply(lambda c: any(tag in c for tag in METAL_TAGS))
    return df[~mask].copy()


all_years_csv = []
all_years_xlsx = []

for year in YEARS:
    fx_csv_path = FX_DIR / f"{year} all selected comb mtm data Test(FX).csv"
    comm_csv_path = COMM_DIR / f"{year} all selected comb mtm data Test(comm).csv"
    fx_xlsx_path = FX_DIR / f"{year} All selected comb percent_invested and Risk(FX).xlsx"
    comm_xlsx_path = COMM_DIR / f"{year} All selected comb percent_invested and Risk(comm).xlsx"

    if not (fx_csv_path.exists() and comm_csv_path.exists()):
        print(f"  skip {year}: missing source file(s)")
        continue

    fx_csv = pd.read_csv(fx_csv_path)
    comm_csv = pd.read_csv(comm_csv_path)
    fx_csv_clean = drop_metal_fx(fx_csv)
    fx_csv_clean.insert(0, "Asset_Class", "FX")
    comm_csv_tagged = comm_csv.copy()
    comm_csv_tagged.insert(0, "Asset_Class", "Commodities")
    combined_csv = pd.concat([fx_csv_clean, comm_csv_tagged], ignore_index=True)

    fx_xlsx = pd.read_excel(fx_xlsx_path)
    comm_xlsx = pd.read_excel(comm_xlsx_path)
    fx_xlsx_clean = drop_metal_fx(fx_xlsx)
    fx_xlsx_clean.insert(0, "Asset_Class", "FX")
    fx_xlsx_clean.insert(1, "Year", year)
    comm_xlsx_tagged = comm_xlsx.copy()
    comm_xlsx_tagged.insert(0, "Asset_Class", "Commodities")
    comm_xlsx_tagged.insert(1, "Year", year)
    combined_xlsx = pd.concat([fx_xlsx_clean, comm_xlsx_tagged], ignore_index=True)

    csv_out = OUT_DIR / f"{year} all selected comb mtm data Test(FXT3CommT15).csv"
    xlsx_out = OUT_DIR / f"{year} All selected comb percent_invested and Risk(FXT3CommT15).xlsx"
    combined_csv.to_csv(csv_out, index=False)
    combined_xlsx.to_excel(xlsx_out, index=False)
    print(f"  {year}: FX {len(fx_csv)}->{len(fx_csv_clean)} rows (dropped {len(fx_csv)-len(fx_csv_clean)} metal/oil), "
          f"+ Comm {len(comm_csv)} rows = {len(combined_csv)} combined rows")

    all_years_csv.append(combined_csv)
    all_years_xlsx.append(combined_xlsx)

master_csv = pd.concat(all_years_csv, ignore_index=True)
master_xlsx = pd.concat(all_years_xlsx, ignore_index=True)
master_csv.to_csv(OUT_DIR / "MASTER all years combined mtm data (FXT3CommT15).csv", index=False)
master_xlsx.to_excel(OUT_DIR / "MASTER all years combined risk and percent_invested (FXT3CommT15).xlsx", index=False)
print(f"\nMASTER: {len(master_csv)} mtm rows, {len(master_xlsx)} risk rows -> {OUT_DIR}")
