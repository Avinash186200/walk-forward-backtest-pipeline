"""
One-shot backtesting pipeline: MTM Data + Raw Data  ->  Trade Unit files + full quant-stat report.

Replaces the manual notebook chain (vol script -> S6 -> S6.1 -> S7 -> S8 -> S9 -> S10 -> S11 -> S12
-> ad hoc report). Edit config.ini, then run:

    python run_pipeline.py

Everything else - intermediate folders, year discovery, the two Trade Unit deliverables, and the
final 16-file report suite - is handled automatically. See config.ini for every setting that can
be changed; nothing in this file needs editing for normal use.
"""

from __future__ import annotations

import argparse
import configparser
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ======================================================================
# CONFIG
# ======================================================================

def _tag_suffix(tag: str) -> str:
    """File-name tag suffix. '(comm)' for tag='comm'; '' for a blank tag, so files
    produced without a tag (e.g. '<year> all selected comb mtm data Test.csv') match."""
    return "(%s)" % tag if tag else ""


@dataclass
class Config:
    raw_data_dir: Path
    mtm_data_dir: Path
    output_dir: Path

    tag: str = "comm"
    label: str = "COMM"
    years: list = field(default_factory=list)

    total_portfolio_capital: float = 1_000_000
    max_gross_exposure: float = 0.90

    vol_lookback_days: int = 90
    normalize_gross_after_invvol: bool = True
    max_leverage_multiple: Optional[float] = None

    production_scenario: str = "tv05_cap100"
    run_all_scenarios: bool = False

    initial_equity: float = 1_000_000
    txn_cost_per_unit: float = 0.00002
    financing_cost_rate: float = 0.00005
    slippage_pips: float = 1.0

    force_vol_wall: bool = False

    n_trials: int = 1

    @property
    def mtm_output_dir(self) -> Path:
        return self.output_dir / "_intermediate" / "mtm_output"

    @property
    def combined_test_dir(self) -> Path:
        return self.mtm_output_dir / "Combined_Test"

    @property
    def broker_grid_dir(self) -> Path:
        return self.combined_test_dir / "Broker_Grid"

    @property
    def trade_level_dir(self) -> Path:
        return self.output_dir / "_intermediate" / "Trade Level"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "_intermediate" / "Summary"

    @property
    def final_dir(self) -> Path:
        return self.output_dir / "Final"

    @property
    def report_dir(self) -> Path:
        return self.output_dir / "Final" / "Report"

    def price_file(self) -> Path:
        return self.raw_data_dir / "All_Instruments_Master_D1.csv"

    def vol_wall_file(self) -> Path:
        return self.raw_data_dir / "Master_Data_with_Vol_Wall_30_60_90.parquet"

    def iteration_file(self, year: int) -> Path:
        return self.mtm_data_dir / f"{year} All selected comb percent_invested and Risk{_tag_suffix(self.tag)}.xlsx"

    def trade_file(self, year: int) -> Path:
        return self.mtm_data_dir / f"{year} all selected comb mtm data Test{_tag_suffix(self.tag)}.csv"


def load_config(config_path: Path) -> Config:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    base = config_path.resolve().parent

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (base / path).resolve()

    years_raw = parser.get("data", "years", fallback="").strip()
    years = [int(y.strip()) for y in years_raw.split(",") if y.strip()] if years_raw else []

    return Config(
        raw_data_dir=resolve(parser.get("paths", "raw_data_dir")),
        mtm_data_dir=resolve(parser.get("paths", "mtm_data_dir")),
        output_dir=resolve(parser.get("paths", "output_dir")),
        tag=parser.get("data", "tag", fallback="comm"),
        label=parser.get("data", "label", fallback="COMM"),
        years=years,
        total_portfolio_capital=parser.getfloat("portfolio", "total_portfolio_capital", fallback=1_000_000),
        max_gross_exposure=parser.getfloat("portfolio", "max_gross_exposure", fallback=0.90),
        initial_equity=parser.getfloat("portfolio", "initial_equity", fallback=1_000_000),
        vol_lookback_days=parser.getint("vol_targeting", "vol_lookback_days", fallback=90),
        production_scenario=parser.get("vol_targeting", "production_scenario", fallback="tv05_cap100"),
        run_all_scenarios=parser.getboolean("vol_targeting", "run_all_scenarios", fallback=False),
        txn_cost_per_unit=parser.getfloat("costs", "txn_cost_per_unit", fallback=0.00002),
        financing_cost_rate=parser.getfloat("costs", "financing_cost_rate", fallback=0.00005),
        slippage_pips=parser.getfloat("costs", "slippage_pips", fallback=1.0),
        force_vol_wall=parser.getboolean("run", "force_vol_wall", fallback=False),
        n_trials=parser.getint("statistics", "n_trials", fallback=1),
    )


# Full scenario grid from the original "S - 7" notebook (name, target_annual_vol, gross_cap).
VOL_SCENARIOS = [
    ("tv05_cap100", 0.05, 1.0), ("tv05_nocap", 0.05, None),
    ("tv10_cap100", 0.10, 1.0), ("tv10_nocap", 0.10, None),
    ("tv14_cap100", 0.14, 1.0), ("tv14_nocap", 0.14, None),
    ("tv16_cap100", 0.16, 1.0), ("tv16_nocap", 0.16, None),
    ("tv18_cap100", 0.18, 1.0), ("tv18_nocap", 0.18, None),
    ("tv20_cap100", 0.20, 1.0), ("tv20_nocap", 0.20, None),
]


def discover_years(cfg: Config) -> list[int]:
    pattern = re.compile(rf"^(\d{{4}}) all selected comb mtm data Test{re.escape(_tag_suffix(cfg.tag))}\.csv$", re.IGNORECASE)
    years = []
    for f in cfg.mtm_data_dir.glob("*.csv"):
        m = pattern.match(f.name)
        if m:
            years.append(int(m.group(1)))
    if not years:
        raise FileNotFoundError(
            f"No '<year> all selected comb mtm data Test({cfg.tag}).csv' files found in {cfg.mtm_data_dir}"
        )
    return sorted(years)


# ======================================================================
# STEP A — Instrument volatility wall  ("Python script to calculate Instrument vol")
# ======================================================================

def build_vol_wall(cfg: Config) -> Path:
    out_path = cfg.vol_wall_file()
    src_path = cfg.price_file()

    if out_path.exists() and not cfg.force_vol_wall and out_path.stat().st_mtime >= src_path.stat().st_mtime:
        print(f"[Step A] Vol wall up to date, reusing: {out_path}")
        return out_path

    print("[Step A] Building instrument volatility wall (30/60/90D)...")
    df = pd.read_csv(src_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["Symbol", "date"]).reset_index(drop=True)

    df["midclose"] = (df["askclose"] + df["bidclose"]) / 2.0
    df["ret"] = df.groupby("Symbol")["midclose"].pct_change()

    grp_ret = df.groupby("Symbol")["ret"]
    for window in (30, 60, 90):
        df[f"Vol_{window}D"] = grp_ret.transform(lambda x: x.rolling(window).std() * np.sqrt(252))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[Step A] Saved: {out_path} ({len(df):,} rows)")
    return out_path


# ======================================================================
# STEP B — Per-year MTM engine + walk-forward Test combination  ("S - 6")
# ======================================================================

def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _make_combination_key(df: pd.DataFrame) -> pd.Series:
    cur = df["Currency"].str.replace("/", "_", regex=False)
    return df["Strategy"] + "_" + cur + "_" + df["Condition"] + "_" + df["Timeframe"]


def run_mtm_for_year(cfg: Config, test_year: int, prices: pd.DataFrame) -> dict:
    output_dir = cfg.mtm_output_dir / str(test_year)
    output_dir.mkdir(parents=True, exist_ok=True)

    iter_df = pd.read_excel(cfg.iteration_file(test_year), sheet_name="Sheet1")
    trades = pd.read_csv(cfg.trade_file(test_year))
    for col in ("Opening Date", "Closing Date", "Day Level Date"):
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col])

    for d in (trades, iter_df):
        for col in ["Strategy", "Currency", "Condition", "Timeframe"]:
            if col not in d.columns:
                d[col] = ""
            d[col] = _clean_str(d[col])

    trades["Combination"] = _make_combination_key(trades)
    if "Combination" not in iter_df.columns:
        iter_df["Combination"] = _make_combination_key(iter_df)
    else:
        iter_df["Combination"] = _clean_str(iter_df["Combination"])

    iter_meta = iter_df[[c for c in [
        "Strategy", "Currency", "Condition", "Timeframe", "Combination",
        "Risk", "percent_invested", "Rollover", "Currency Type",
    ] if c in iter_df.columns]].drop_duplicates(subset=["Combination"]).copy()

    trades = trades.merge(iter_meta[["Combination", "percent_invested"]], on="Combination", how="left")
    trades = trades.rename(columns={"New P/L_pips": "Final_PnL", "SL to Feed": "ClosingPrice_SLtoFeed"})
    trades = trades[~trades["percent_invested"].isna()].copy()

    # Scale percent_invested once so the sum respects max_gross_exposure
    sum_pi = iter_meta["percent_invested"].sum()
    scale_factor = (cfg.max_gross_exposure / sum_pi) if sum_pi > cfg.max_gross_exposure else 1.0
    iter_meta["percent_invested_scaled"] = iter_meta["percent_invested"] * scale_factor

    trades = trades.merge(iter_meta[["Combination", "percent_invested_scaled"]], on="Combination", how="left")
    trades["percent_invested_scaled"] = trades["percent_invested_scaled"].fillna(0.0)
    trades = trades[trades["percent_invested_scaled"] > 0].copy()

    trades["trade_capital"] = cfg.total_portfolio_capital * trades["percent_invested_scaled"]
    trades["Qty"] = trades["trade_capital"] / trades["Trading Price Point"]
    trades["Direction"] = np.where(trades["Condition"].str.lower().str.contains("short"), -1, 1)

    if trades.empty:
        raise RuntimeError(f"[{test_year}] No trades left after percent_invested filtering.")

    trade_daily = _expand_trades_to_daily(trades, prices, cfg.total_portfolio_capital, test_year)
    if trade_daily.empty:
        raise RuntimeError(f"[{test_year}] TradeDaily is empty - check symbol/date alignment.")

    trade_daily.to_csv(output_dir / "TradeDaily_All.csv", index=False)

    comb_daily = (
        trade_daily.groupby(["date", "Combination"], as_index=False)
        .agg(position_value=("position_value", "sum"),
             daily_pnl=("daily_pnl", "sum"),
             daily_return_vs_trade_capital=("daily_return_vs_trade_capital", "sum"),
             exposure_vs_portfolio=("exposure_vs_portfolio", "sum"))
        .merge(iter_meta[["Combination", "percent_invested", "percent_invested_scaled"]].drop_duplicates("Combination"),
               on="Combination", how="left")
    )
    comb_daily["exposure_pct_portfolio_raw"] = comb_daily["position_value"] / cfg.total_portfolio_capital
    comb_daily["return_pct_portfolio_raw"] = comb_daily["daily_pnl"] / cfg.total_portfolio_capital
    comb_daily = comb_daily.sort_values(["date", "Combination"])

    portfolio_daily = (
        comb_daily.groupby("date", as_index=False)
        .agg(position_value=("position_value", "sum"),
             daily_pnl=("daily_pnl", "sum"),
             return_pct_portfolio_raw=("return_pct_portfolio_raw", "sum"))
        .sort_values("date")
    )
    portfolio_daily["exposure_pct_portfolio_raw"] = portfolio_daily["position_value"] / cfg.total_portfolio_capital
    portfolio_daily["portfolio_return_raw"] = portfolio_daily["daily_pnl"] / cfg.total_portfolio_capital
    portfolio_daily["exposure_scale_factor"] = np.where(
        portfolio_daily["exposure_pct_portfolio_raw"] > cfg.max_gross_exposure,
        cfg.max_gross_exposure / portfolio_daily["exposure_pct_portfolio_raw"],
        1.0,
    )
    portfolio_daily["position_value_capped"] = portfolio_daily["position_value"] * portfolio_daily["exposure_scale_factor"]
    portfolio_daily["daily_pnl_capped"] = portfolio_daily["daily_pnl"] * portfolio_daily["exposure_scale_factor"]
    portfolio_daily["exposure_pct_portfolio"] = portfolio_daily["position_value_capped"] / cfg.total_portfolio_capital
    portfolio_daily["portfolio_return"] = portfolio_daily["daily_pnl_capped"] / cfg.total_portfolio_capital

    comb_daily = comb_daily.merge(portfolio_daily[["date", "exposure_scale_factor"]], on="date", how="left")
    comb_daily["position_value_capped"] = comb_daily["position_value"] * comb_daily["exposure_scale_factor"]
    comb_daily["daily_pnl_capped"] = comb_daily["daily_pnl"] * comb_daily["exposure_scale_factor"]
    comb_daily["exposure_pct_portfolio"] = comb_daily["position_value_capped"] / cfg.total_portfolio_capital
    comb_daily["return_pct_portfolio"] = comb_daily["daily_pnl_capped"] / cfg.total_portfolio_capital

    for d in (portfolio_daily, comb_daily):
        d["Year"] = d["date"].dt.year
        d["DataSplit"] = np.where(d["Year"] < test_year, "Train", np.where(d["Year"] == test_year, "Test", "Ignore"))

    comb_daily.to_csv(output_dir / "CombinationDaily_All.csv", index=False)
    portfolio_daily.to_csv(output_dir / "PortfolioDaily_All.csv", index=False)

    test_slice_comb = comb_daily[comb_daily["DataSplit"] == "Test"].copy()
    if not test_slice_comb.empty:
        test_slice_comb.to_csv(output_dir / "CombinationDaily_Test.csv", index=False)

    test_slice_port = portfolio_daily[portfolio_daily["DataSplit"] == "Test"].copy()
    if not test_slice_port.empty:
        test_slice_port.to_csv(output_dir / "PortfolioDaily_Test.csv", index=False)

    return {"trade_daily": trade_daily, "comb_daily": comb_daily, "portfolio_daily": portfolio_daily}


def _expand_trades_to_daily_orig(trades_df: pd.DataFrame, prices_df: pd.DataFrame, total_portfolio_capital: float,
                                 test_year: int) -> pd.DataFrame:
    # Original per-trade implementation. Kept for byte-identical validation of the fast path.
    prices_by_symbol = {
        sym: g.sort_values("date").set_index("date")["price"]
        for sym, g in prices_df.groupby("Symbol")
    }

    all_rows = []
    for idx, row in trades_df.iterrows():
        series = prices_by_symbol.get(row["Currency"])
        if series is None:
            continue
        pr = series.loc[row["Opening Date"]:row["Closing Date"]]
        if pr.empty:
            continue

        pr = pr.rename("price").reset_index()
        pr["TradeID"] = f"{test_year}_{idx}"
        pr["Strategy"] = row["Strategy"]
        pr["Currency"] = row["Currency"]
        pr["Timeframe"] = row["Timeframe"]
        pr["Condition"] = row["Condition"]
        pr["Combination"] = row["Combination"]
        pr["Data_Type"] = row.get("Data_Type", np.nan)
        pr["Qty"] = row["Qty"]
        pr["Direction"] = row["Direction"]
        pr["trade_capital"] = row["trade_capital"]
        pr["Opening Date"] = row["Opening Date"]
        pr["Closing Date"] = row["Closing Date"]
        pr["Final_PnL"] = row.get("Final_PnL", 0.0)
        all_rows.append(pr)

    if not all_rows:
        return pd.DataFrame()

    daily = pd.concat(all_rows, ignore_index=True).sort_values(["TradeID", "date"])
    return _finalize_trade_daily(daily, total_portfolio_capital)


def _finalize_trade_daily(daily: pd.DataFrame, total_portfolio_capital: float) -> pd.DataFrame:
    daily["price_prev"] = daily.groupby("TradeID")["price"].shift(1)
    daily["position_value"] = daily["Qty"] * daily["price"]
    daily["exposure_vs_trade_capital"] = daily["position_value"] / daily["trade_capital"]
    daily["exposure_vs_portfolio"] = daily["position_value"] / total_portfolio_capital
    daily["daily_pnl"] = daily["Direction"] * daily["Qty"] * (daily["price"] - daily["price_prev"])
    daily.loc[daily["price_prev"].isna(), "daily_pnl"] = 0.0
    daily["daily_return_vs_trade_capital"] = daily["daily_pnl"] / daily["trade_capital"]
    return daily


def _expand_trades_to_daily(trades_df: pd.DataFrame, prices_df: pd.DataFrame, total_portfolio_capital: float,
                            test_year: int) -> pd.DataFrame:
    # Vectorized expansion. The original built one small DataFrame per trade and concatenated
    # ~N_trades of them - O(trades) Python/pandas overhead that scales badly once there are
    # hundreds of instruments / hundreds of thousands of trades (fine for ~25 commodities, slow
    # for ~500 equities). Here each trade's date window is found with two searchsorted() calls
    # into its symbol's sorted price array, then every column is materialized in one pass with
    # np.repeat. Same rows, same column order, same sort -> byte-identical output.
    prices_by_symbol = {}
    for sym, g in prices_df.groupby("Symbol"):
        g = g.sort_values("date")
        prices_by_symbol[sym] = (g["date"].to_numpy(), g["price"].to_numpy())

    idx_vals = trades_df.index.to_numpy()
    currency = trades_df["Currency"].to_numpy()
    opens = trades_df["Opening Date"].to_numpy()
    closes = trades_df["Closing Date"].to_numpy()
    n = len(trades_df)

    date_chunks, price_chunks = [], []
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pg = prices_by_symbol.get(currency[i])
        if pg is None:
            continue
        darr, parr = pg
        lo = np.searchsorted(darr, opens[i], side="left")
        hi = np.searchsorted(darr, closes[i], side="right")   # series.loc[open:close] is inclusive of both ends
        if hi <= lo:
            continue
        date_chunks.append(darr[lo:hi])
        price_chunks.append(parr[lo:hi])
        counts[i] = hi - lo

    if not date_chunks:
        return pd.DataFrame()

    def col(name, default):
        arr = trades_df[name].to_numpy() if name in trades_df.columns else np.full(n, default)
        return np.repeat(arr, counts)

    tradeid = np.repeat(np.array(["%s_%s" % (test_year, i) for i in idx_vals], dtype=object), counts)
    daily = pd.DataFrame({
        "date": np.concatenate(date_chunks),
        "price": np.concatenate(price_chunks),
        "TradeID": tradeid,
        "Strategy": col("Strategy", np.nan),
        "Currency": np.repeat(currency, counts),
        "Timeframe": col("Timeframe", np.nan),
        "Condition": col("Condition", np.nan),
        "Combination": col("Combination", np.nan),
        "Data_Type": col("Data_Type", np.nan),
        "Qty": col("Qty", np.nan),
        # original assigns Direction via an iterrows object-Series, which yields int64; match it
        "Direction": np.repeat(trades_df["Direction"].to_numpy().astype(np.int64), counts),
        "trade_capital": col("trade_capital", np.nan),
        "Opening Date": np.repeat(opens, counts),
        "Closing Date": np.repeat(closes, counts),
        "Final_PnL": col("Final_PnL", 0.0),
    }).sort_values(["TradeID", "date"])
    return _finalize_trade_daily(daily, total_portfolio_capital)


def run_mtm_all_years(cfg: Config) -> pd.DataFrame:
    print(f"[Step B] Running per-year MTM engine for {cfg.years}...")
    prices = pd.read_csv(cfg.price_file(), low_memory=False)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.rename(columns={"askclose": "price"})
    keep_cols = ["Symbol", "date", "price"] + (["Timeframe"] if "Timeframe" in prices.columns else [])
    prices = prices[keep_cols]

    all_test = []
    for year in cfg.years:
        print(f"  - {year} ...")
        result = run_mtm_for_year(cfg, year, prices)
        p = result["portfolio_daily"]
        p = p[p["DataSplit"] == "Test"].copy()
        if not p.empty:
            p["TestYear"] = year
            all_test.append(p)

    if not all_test:
        raise RuntimeError("No test-year portfolio data produced across the requested years.")

    combined = pd.concat(all_test, ignore_index=True)
    cfg.combined_test_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cfg.combined_test_dir / "PortfolioDaily_Test_AllYears.csv", index=False)
    print(f"[Step B] Combined out-of-sample portfolio: {len(combined):,} rows")
    return combined


# ======================================================================
# STEP C — Instrument-level daily data across years  ("S - 6.1")
# ======================================================================

def build_instrument_level_data(cfg: Config) -> pd.DataFrame:
    print("[Step C] Building instrument-level daily data across years...")
    value_cols = ["exposure_pct_portfolio", "return_pct_portfolio"]
    frames = []

    for year in cfg.years:
        file_path = cfg.mtm_output_dir / str(year) / "CombinationDaily_Test.csv"
        if not file_path.exists():
            print(f"  [skip] missing {file_path}")
            continue

        df = pd.read_csv(file_path)
        parts = df["Combination"].astype(str).str.rsplit("_", n=4, expand=True)
        while parts.shape[1] < 5:
            parts[parts.shape[1]] = None
        parts = parts.iloc[:, :5]
        df["Base"], df["Quote"] = parts[1], parts[2]
        df["Instruments"] = df["Base"].astype(str) + "/" + df["Quote"].astype(str)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].copy()
        df["YearMonthDay"] = df["date"].dt.strftime("%Y-%m-%d")

        df = df[["YearMonthDay", "Instruments"] + value_cols]
        agg = df.groupby(["YearMonthDay", "Instruments"], as_index=False)[value_cols].sum()
        agg["Date"] = pd.to_datetime(agg["YearMonthDay"])
        agg["Year"], agg["Month"], agg["Day"] = agg["Date"].dt.year, agg["Date"].dt.month, agg["Date"].dt.day
        agg = agg.rename(columns={"exposure_pct_portfolio": "Exposer", "return_pct_portfolio": "Returns"})
        frames.append(agg.assign(SourceYear=year))

    if not frames:
        raise ValueError("No CombinationDaily_Test.csv files found - check Step B output.")

    master = pd.concat(frames, ignore_index=True).sort_values(["YearMonthDay", "Instruments"]).reset_index(drop=True)
    master = master.drop(columns=["YearMonthDay"])

    cfg.combined_test_dir.mkdir(parents=True, exist_ok=True)
    out_file = cfg.combined_test_dir / "Instrument_level_daily_Test_AllYears.xlsx"
    master.to_excel(out_file, index=False, sheet_name="Master")
    print(f"[Step C] Saved: {out_file} ({len(master):,} rows)")
    return master


# ======================================================================
# STEP D — Ex-ante (1/vol) targeted portfolio  ("S - 7", live cell 9)
# ======================================================================

def _choose_vol(merged: pd.DataFrame) -> pd.Series:
    vol = merged["Vol_90D"].copy()
    bad_90 = vol.isna() | (vol <= 0)
    vol.loc[bad_90] = merged.loc[bad_90, "Vol_60D"]
    bad_60 = vol.isna() | (vol <= 0)
    vol.loc[bad_60] = merged.loc[bad_60, "Vol_30D"]
    return vol


def _run_broker_scenario(df: pd.DataFrame, ret_matrix: pd.DataFrame, target_annual_vol: float,
                          gross_cap: Optional[float], cfg: Config) -> pd.DataFrame:
    scen_df = df.copy()
    target_daily = target_annual_vol / math.sqrt(252.0)
    ret_dates = ret_matrix.index

    for col in ["exposure_invvol", "exposure_final", "Returns_invvol", "Returns_final",
                "scale_to_target_vol_raw", "scale_to_target_vol", "portfolio_90d_vol"]:
        scen_df[col] = np.nan

    for d, day in scen_df.groupby("date"):
        mask = scen_df["date"] == d
        instruments = day["Instruments"].tolist()
        vols = day["vol_used"].values
        base_exp = day["Exposer"].values
        base_ret = day["Returns"].values

        inv = base_exp / np.maximum(vols, 1e-8)
        if cfg.normalize_gross_after_invvol:
            gross_orig, gross_inv = np.sum(np.abs(base_exp)), np.sum(np.abs(inv))
            if gross_inv > 0:
                inv = inv / gross_inv * gross_orig
        day = day.copy()
        day["exposure_invvol"] = inv

        if d not in ret_dates:
            scen_df.loc[mask, ["exposure_invvol", "exposure_final", "scale_to_target_vol_raw", "scale_to_target_vol"]] = \
                np.column_stack([inv, inv, np.ones_like(inv), np.ones_like(inv)])
            continue

        pos = ret_dates.get_loc(d)
        window = ret_matrix.iloc[max(0, pos - cfg.vol_lookback_days + 1): pos + 1]
        cols_avail = [c for c in instruments if c in window.columns]
        if not cols_avail:
            scen_df.loc[mask, ["exposure_invvol", "exposure_final", "scale_to_target_vol_raw", "scale_to_target_vol"]] = \
                np.column_stack([inv, inv, np.ones_like(inv), np.ones_like(inv)])
            continue

        day_idx = day.set_index("Instruments")
        weights_inv = day_idx.loc[cols_avail, "exposure_invvol"].values
        window_sub = window[cols_avail].fillna(0.0)
        port_hist = np.dot(window_sub.values, weights_inv)
        hist_vol = np.nanstd(port_hist, ddof=0)

        if not np.isfinite(hist_vol) or hist_vol <= 0:
            scale_raw = 1.0
        else:
            scale_raw = target_daily / hist_vol
            if cfg.max_leverage_multiple is not None:
                scale_raw = min(scale_raw, cfg.max_leverage_multiple)

        exposure_candidate = inv * scale_raw
        gross_candidate = np.sum(np.abs(exposure_candidate))
        gross_factor = (gross_cap / gross_candidate) if (gross_cap is not None and gross_candidate > gross_cap > 0) else 1.0
        scale_final = scale_raw * gross_factor
        exposure_final = inv * scale_final

        ratio_inv = np.divide(inv, np.where(base_exp == 0, np.nan, base_exp))
        ratio_final = np.divide(exposure_final, np.where(base_exp == 0, np.nan, base_exp))

        scen_df.loc[mask, "exposure_invvol"] = inv
        scen_df.loc[mask, "exposure_final"] = exposure_final
        scen_df.loc[mask, "Returns_invvol"] = base_ret * ratio_inv
        scen_df.loc[mask, "Returns_final"] = base_ret * ratio_final
        scen_df.loc[mask, "scale_to_target_vol_raw"] = scale_raw
        scen_df.loc[mask, "scale_to_target_vol"] = scale_final
        scen_df.loc[mask, "portfolio_90d_vol"] = hist_vol

    return scen_df


def build_broker_vol_portfolio(cfg: Config, instrument_master: pd.DataFrame) -> pd.DataFrame:
    print("[Step D] Building ex-ante (1/vol) targeted portfolio...")

    inst = instrument_master.copy()
    master = pd.read_parquet(cfg.vol_wall_file())
    master["date"] = pd.to_datetime(master["date"]).dt.normalize()

    vol_df = master[["date", "Symbol", "Vol_30D", "Vol_60D", "Vol_90D"]].copy()
    vol_df["Year"], vol_df["Month"], vol_df["Day"] = vol_df["date"].dt.year, vol_df["date"].dt.month, vol_df["date"].dt.day
    vol_df = vol_df.rename(columns={"Symbol": "symbol_mst"}).drop(columns=["date"])

    merged = inst.merge(
        vol_df, left_on=["Year", "Month", "Day", "Instruments"],
        right_on=["Year", "Month", "Day", "symbol_mst"], how="left",
    ).drop(columns=["symbol_mst"])

    ret_matrix = master.pivot(index="date", columns="Symbol", values="ret").sort_index()

    merged["vol_used"] = _choose_vol(merged)
    df = merged[merged["vol_used"] > 0].copy()
    df["date"] = pd.to_datetime(df["Date"])

    scenarios = VOL_SCENARIOS if cfg.run_all_scenarios else [
        s for s in VOL_SCENARIOS if s[0] == cfg.production_scenario
    ]
    if not scenarios:
        raise ValueError(f"Unknown scenario '{cfg.production_scenario}'. Choose from: {[s[0] for s in VOL_SCENARIOS]}")

    cfg.broker_grid_dir.mkdir(parents=True, exist_ok=True)
    production_df = None

    for name, target_annual_vol, gross_cap in scenarios:
        print(f"  - scenario {name} (target_vol={target_annual_vol}, gross_cap={gross_cap})")
        scen_df = _run_broker_scenario(df, ret_matrix, target_annual_vol, gross_cap, cfg)

        daily = scen_df.groupby("date", as_index=False).agg(
            portfolio_return_original=("Returns", "sum"),
            portfolio_return_final=("Returns_final", "sum"),
            scale_to_target_vol_raw=("scale_to_target_vol_raw", "first"),
            scale_to_target_vol=("scale_to_target_vol", "first"),
            portfolio_90d_vol=("portfolio_90d_vol", "first"),
            gross_exposure_original=("Exposer", lambda x: np.sum(np.abs(x))),
            gross_exposure_invvol=("exposure_invvol", lambda x: np.sum(np.abs(x))),
            gross_exposure_final=("exposure_final", lambda x: np.sum(np.abs(x))),
        ).sort_values("date")

        scen_df.to_excel(cfg.broker_grid_dir / f"inst_broker_{name}.xlsx", index=False)
        daily.to_excel(cfg.broker_grid_dir / f"daily_broker_{name}.xlsx", index=False)

        if name == cfg.production_scenario:
            production_df = scen_df

    print(f"[Step D] Saved scenario grid to: {cfg.broker_grid_dir}")
    return production_df


# ======================================================================
# STEP E — Attach instrument-level ex-ante data onto trade-level rows  ("S - 8")
# ======================================================================

def merge_instrument_to_trade(cfg: Config) -> pd.DataFrame:
    print("[Step E] Merging instrument-level ex-ante data onto trade-level rows...")

    files = sorted(cfg.mtm_output_dir.glob("*/TradeDaily_All.csv"))
    if not files:
        raise FileNotFoundError(f"No TradeDaily_All.csv found under {cfg.mtm_output_dir}")

    dfs = []
    for f in files:
        d = pd.read_csv(f, parse_dates=["date", "Opening Date", "Closing Date"])
        d["YearFolder"] = f.parent.name
        dfs.append(d)
    trade_all = pd.concat(dfs, ignore_index=True)

    inst = pd.read_excel(cfg.broker_grid_dir / f"inst_broker_{cfg.production_scenario}.xlsx")
    for col in ("Year", "Month", "Day"):
        inst[col] = inst[col].astype(int)

    df = trade_all.copy()
    df["Year"], df["Month"], df["Day"] = df["date"].dt.year, df["date"].dt.month, df["date"].dt.day

    merge_cols = ["Year", "Month", "Day", "Instruments", "Exposer", "Returns", "exposure_final", "Returns_final"]
    df = df.merge(
        inst[merge_cols], left_on=["Year", "Month", "Day", "Currency"],
        right_on=["Year", "Month", "Day", "Instruments"], how="left",
    )

    mask_valid = df["Exposer"].notna()
    df["n_combos_inst_date"] = 0
    df.loc[mask_valid, "n_combos_inst_date"] = df[mask_valid].groupby(["date", "Currency"])["Combination"].transform("nunique")

    df["n_trades_combo_date"] = 0
    df.loc[mask_valid, "n_trades_combo_date"] = df[mask_valid].groupby(["date", "Currency", "Combination"])["TradeID"].transform("count")

    valid_alloc = mask_valid & (df["n_combos_inst_date"] > 0) & (df["n_trades_combo_date"] > 0)
    denom = df["n_combos_inst_date"].replace(0, np.nan) * df["n_trades_combo_date"].replace(0, np.nan)
    alloc_factor = np.where(valid_alloc, 1.0 / denom, np.nan)

    for col in ["Exposer", "Returns", "exposure_final", "Returns_final"]:
        df[f"{col}_per_trade"] = df[col].astype(float) * alloc_factor
        df.loc[~valid_alloc, f"{col}_per_trade"] = np.nan

    cfg.trade_level_dir.mkdir(parents=True, exist_ok=True)
    # CSV, not xlsx: at equity scale (hundreds of instruments) this table runs to millions of
    # trade-day rows, well past Excel's ~1.05M row sheet limit. It's an intermediate (Step F gets
    # the frame in memory, doesn't re-read it), so the format is free to change; data is identical.
    out_file = cfg.trade_level_dir / f"{cfg.label} - TradeDaily_With_Broker.csv"
    df.to_csv(out_file, index=False)
    print(f"[Step E] Saved: {out_file} ({len(df):,} rows)")
    return df


# ======================================================================
# STEP F — Split trades into OPEN/ACTIVE/CLOSE/EXIT legs as exposure changes  ("S - 9")
# ======================================================================

def split_trades_by_vol_change(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    print("[Step F] Splitting trades into carry/delta/exit legs...")

    EXPOSURE_COL, RETURN_COL = "exposure_final_per_trade", "Returns_final_per_trade"
    group_cols = ["Combination", "Opening Date", "Closing Date"]
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

    df["is_open_day"] = df["date"] == df["Opening Date"]
    df["is_close_day"] = df["date"] == df["Closing Date"]
    df["is_active_day"] = (df["date"] > df["Opening Date"]) & (df["date"] < df["Closing Date"])

    def day_status(row):
        if row["is_open_day"] and row["is_close_day"]:
            return "OPEN_CLOSE"
        if row["is_open_day"]:
            return "OPEN"
        if row["is_close_day"]:
            return "CLOSE"
        if row["is_active_day"]:
            return "ACTIVE"
        return "OTHER"

    df["DayStatus"] = df.apply(day_status, axis=1)
    df["prev_exposure_same_trade"] = df.groupby(group_cols)[EXPOSURE_COL].shift(1)
    df.loc[df["is_open_day"], "prev_exposure_same_trade"] = 0.0

    new_rows = []
    for _, row in df.iterrows():
        base = row.to_dict()
        E_cur, R_total, E_prev = float(row[EXPOSURE_COL]), float(row[RETURN_COL]), float(row["prev_exposure_same_trade"])
        is_open, is_active, is_close = bool(row["is_open_day"]), bool(row["is_active_day"]), bool(row["is_close_day"])
        day_status_val = row["DayStatus"]

        should_split = (
            (is_active or (is_close and not is_open))
            and abs(E_cur - E_prev) > 1e-15 and E_prev != 0.0 and E_cur != 0.0
        )

        if should_split:
            carry_exp, delta_exp = E_prev, E_cur - E_prev
            w_carry, w_delta = abs(carry_exp), abs(delta_exp)
            total_w = w_carry + w_delta
            if total_w == 0:
                base["DayStatus"], base["TradeRole"] = day_status_val, "CARRY_POSITION"
                new_rows.append(base)
            else:
                ratio_carry = w_carry / total_w
                R_carry = R_total * ratio_carry
                R_delta = R_total - R_carry
                label = "ACTIVE" if is_active else ("CLOSE" if is_close else day_status_val)

                carry_row = base.copy()
                carry_row[EXPOSURE_COL], carry_row[RETURN_COL] = carry_exp, R_carry
                carry_row["DayStatus"], carry_row["TradeRole"] = label, "CARRY_POSITION"
                new_rows.append(carry_row)

                delta_row = base.copy()
                delta_row[EXPOSURE_COL], delta_row[RETURN_COL] = delta_exp, R_delta
                delta_row["DayStatus"], delta_row["TradeRole"] = label, "DELTA_TRADE"
                new_rows.append(delta_row)
        else:
            base["DayStatus"] = day_status_val
            base["TradeRole"] = {
                "OPEN": "OPEN_POSITION", "ACTIVE": "CARRY_POSITION",
                "CLOSE": "CARRY_POSITION", "OPEN_CLOSE": "OPEN_POSITION",
            }.get(day_status_val, "OTHER")
            new_rows.append(base)

        if is_close:
            exit_row = base.copy()
            exit_row[EXPOSURE_COL], exit_row[RETURN_COL] = -E_cur, 0.0
            exit_row["DayStatus"], exit_row["TradeRole"] = "CLOSE", "EXIT_TRADE"
            new_rows.append(exit_row)

    out_df = pd.DataFrame(new_rows)
    out_df = out_df.drop(columns=[c for c in ["is_open_day", "is_close_day", "is_active_day", "prev_exposure_same_trade"] if c in out_df.columns])

    orig_pnl, new_pnl = df[RETURN_COL].sum(), out_df[RETURN_COL].sum()
    print(f"[Step F] P&L preserved: original={orig_pnl:.6f}, split={new_pnl:.6f}, diff={new_pnl - orig_pnl:.2e}")

    out_file = cfg.trade_level_dir / f"{cfg.label} - TradeDaily_With_Broker_transactions.csv"
    out_df.to_csv(out_file, index=False)
    print(f"[Step F] Saved: {out_file} ({len(out_df):,} rows)")
    return out_df


# ======================================================================
# STEP G+H+I — Recursive equity curve, costs, and slippage  ("S - 10/11" + slippage)
# ======================================================================
#
# equity_start_of_day must drive position sizing (amount_invested_per_trade),
# but it must ALSO reflect cost+financing (Step H) and slippage (Step I) drag
# from prior days - those are computed AFTER sizing depends on equity. That
# circularity means this can't be three independent vectorized passes like
# the old G/H/I split (which compounded equity off gross returns only, so the
# saved equity curve silently diverged from the cost+slippage performance
# numbers reported downstream). Instead this walks day-by-day: each day's
# costs/slippage are computed off that day's starting equity, the resulting
# net-of-cost-and-slippage return then compounds into the NEXT day's equity.

_SLIPPAGE_LEG_MULTIPLIER = {"OPEN": 1, "CLOSE": 1, "OPEN_CLOSE": 2, "ACTIVE": 0}


def assign_equity_costs_slippage(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    print("[Step G+H+I] Building recursive equity curve (position sizing net of prior-day cost+slippage)...")

    df = df.dropna(subset=["exposure_final_per_trade"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    slip = cfg.slippage_pips * 0.0001
    equity = cfg.initial_equity
    day_chunks = []

    for _, day_df in df.groupby("date", sort=True):
        day_df = day_df.copy()
        day_df["equity_start_of_day"] = equity
        day_df["amount_invested_per_trade"] = equity * day_df["exposure_final_per_trade"].abs()

        day_df["Delta (Close flag)"] = np.where((day_df["exposure_final_per_trade"] * cfg.total_portfolio_capital) < 0, 1, 0)
        day_df["Quantity"] = day_df["amount_invested_per_trade"] / day_df["price"]
        day_df["Transection Cost all"] = np.where(day_df["TradeRole"] == "EXIT_TRADE", 0, day_df["Quantity"] * cfg.txn_cost_per_unit)
        day_df["Transection Cost"] = np.where(
            (day_df["TradeRole"] == "DELTA_TRADE") & (day_df["Delta (Close flag)"] == 1), 0, day_df["Transection Cost all"]
        )
        day_df["financing cost"] = np.where(day_df["DayStatus"] == "CLOSE", 0, cfg.financing_cost_rate)
        day_df["financing amount"] = day_df["amount_invested_per_trade"] * day_df["financing cost"] * day_df["Direction"]
        day_df["Total Cost"] = day_df["Transection Cost"] + day_df["financing amount"]
        day_df["Total cost %"] = day_df["Total Cost"] / day_df["equity_start_of_day"]
        day_df["PL With cost"] = day_df["Returns_final_per_trade"] - day_df["Total cost %"]

        leg_mult = day_df["DayStatus"].map(_SLIPPAGE_LEG_MULTIPLIER).fillna(0)
        day_df["slippage_impact"] = slip * leg_mult * day_df["amount_invested_per_trade"] / day_df["equity_start_of_day"]
        day_df["PL_with_slippage"] = day_df["PL With cost"] - day_df["slippage_impact"]

        equity *= (1 + day_df["PL_with_slippage"].sum())
        day_chunks.append(day_df)

    df = pd.concat(day_chunks, ignore_index=True)

    # Single write. The original saved this frame twice under two names (_with_equity and _cost)
    # with identical content; at equity scale that's an extra ~11 GB on disk and minutes of write
    # for a duplicate nothing re-reads. Keep the _cost name (the one referenced below / downstream).
    cost_file = cfg.trade_level_dir / f"{cfg.label} - TradeDaily_With_Broker_transactions_cost.csv"
    df.to_csv(cost_file, index=False)

    print(f"[Step G+H+I] Saved: {cost_file} ({len(df):,} rows)")
    print(f"[Step G+H+I] Final equity (net of cost+slippage, recursively sized)={equity:,.0f}")
    print(f"[Step G+H+I] Total PL with cost only={df['PL With cost'].sum():.4f}, "
          f"with cost+slippage={df['PL_with_slippage'].sum():.4f}")
    return df


# ======================================================================
# STEP J — Summary tables + final Trade Unit deliverables  ("S - 12")
# ======================================================================

FINAL_TRADE_COLUMNS = [
    "date", "price", "TradeID", "Strategy", "Currency", "Timeframe", "Condition", "Combination",
    "Direction", "Opening Date", "Closing Date", "Year", "Month", "Day",
    "exposure_final_per_trade", "Returns_final_per_trade", "DayStatus", "TradeRole",
    "equity_start_of_day", "amount_invested_per_trade", "Delta (Close flag)", "Quantity",
    "Transection Cost all", "Transection Cost", "financing cost", "financing amount",
    "Total Cost", "Total cost %", "PL With cost", "slippage_impact", "PL_with_slippage",
]


def build_summary_and_final_outputs(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    print("[Step J] Building summary tables and final Trade Unit deliverables...")

    cfg.summary_dir.mkdir(parents=True, exist_ok=True)
    df.groupby(["Year", "DayStatus"])["Combination"].count().reset_index().to_csv(cfg.summary_dir / "year trades.csv", index=False)
    df.groupby(["Year", "Month", "Day", "DayStatus"])["Combination"].count().reset_index().to_csv(cfg.summary_dir / "Month trades.csv", index=False)
    df.groupby(["Year", "Month", "Day"])["exposure_final_per_trade"].sum().reset_index().to_csv(cfg.summary_dir / "Daily exposer.csv", index=False)
    df.groupby(["Year", "Month", "Day", "Combination"])[["Returns_final_per_trade", "Total cost %", "PL With cost"]].sum().reset_index().to_csv(cfg.summary_dir / "Daily PL.csv", index=False)
    df.groupby(["Year", "Month", "Day", "Combination"])["exposure_final_per_trade"].sum().reset_index().to_csv(cfg.summary_dir / "Daily exposer across comb.csv", index=False)

    df = df[FINAL_TRADE_COLUMNS]

    cfg.final_dir.mkdir(parents=True, exist_ok=True)
    trade_level_out = cfg.final_dir / f"{cfg.label}_TradeDaily_with_equity_and_cost.csv"
    df.to_csv(trade_level_out, index=False)

    day_level_data = df.groupby(["Year", "Month", "Day"])[
        ["exposure_final_per_trade", "Returns_final_per_trade", "PL With cost", "PL_with_slippage"]
    ].sum().reset_index()
    day_level_data["date"] = pd.to_datetime(day_level_data[["Year", "Month", "Day"]])
    day_level_data = day_level_data.sort_values("date").reset_index(drop=True)

    daily_pl_out = cfg.final_dir / f"{cfg.label}_daily_data_for_portfolio.xlsx"
    day_level_data.to_excel(daily_pl_out, index=False)

    print(f"[Step J] Final deliverables:\n  {trade_level_out}\n  {daily_pl_out}")
    return day_level_data


def print_performance_summary(day_level_data: pd.DataFrame, label: str) -> None:
    for col, tag in [("PL With cost", "cost only, no slippage"), ("PL_with_slippage", "cost + slippage")]:
        returns = day_level_data.set_index("date")[col].dropna()
        if returns.empty:
            continue

        equity = (1 + returns).cumprod()
        total_days = len(returns)
        total_return = equity.iloc[-1] - 1
        years = total_days / 252
        cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = cagr / ann_vol if ann_vol else np.nan
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()

        print(f"\n=== {label} performance summary ({tag}) ===")
        print(f"  Period          : {returns.index.min().date()} -> {returns.index.max().date()} ({total_days} trading days)")
        print(f"  Total Return    : {total_return:.2%}")
        print(f"  CAGR            : {cagr:.2%}")
        print(f"  Ann. Volatility : {ann_vol:.2%}")
        print(f"  Sharpe Ratio    : {sharpe:.3f}")
        print(f"  Max Drawdown    : {max_dd:.2%}")


# ======================================================================
# STEP K — Final quant-stat report suite
# ======================================================================

PL_COL = "PL_with_slippage"   # the return series the report is built on (cost + slippage)


def _performance_metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    total_days = len(returns)
    years = total_days / 252
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if not downside.empty else np.nan
    sortino = cagr / downside_vol if downside_vol else np.nan

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    win_days, loss_days = (returns > 0).sum(), (returns < 0).sum()
    win_rate_days = win_days / total_days if total_days else np.nan
    avg_win_day = returns[returns > 0].mean() if win_days else np.nan
    avg_loss_day = returns[returns < 0].mean() if loss_days else np.nan
    profit_factor_days = (
        returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if loss_days and returns[returns < 0].sum() != 0 else np.nan
    )

    return {
        "Start Date": returns.index.min(), "End Date": returns.index.max(),
        "Trading Days": total_days, "Years": round(years, 2),
        "Total Return": total_return, "CAGR": cagr,
        "Best Day": returns.max(), "Worst Day": returns.min(),
        "Ann. Volatility": ann_vol, "Sharpe Ratio": sharpe, "Sortino Ratio": sortino,
        "Max Drawdown": max_dd, "Calmar Ratio": calmar,
        "Win Days": int(win_days), "Loss Days": int(loss_days), "Win Rate (Days)": win_rate_days,
        "Avg Win Day": avg_win_day, "Avg Loss Day": avg_loss_day, "Profit Factor (Days)": profit_factor_days,
        "_equity": equity, "_drawdown": drawdown,
    }


def _lookup_gross_cap(production_scenario: str) -> Optional[float]:
    for name, _, gross_cap in VOL_SCENARIOS:
        if name == production_scenario:
            return gross_cap
    return None


def _exposure_diagnostics(trade_df: pd.DataFrame, gross_cap: Optional[float]) -> dict:
    """Daily gross exposure, reconstructed correctly from trade-level legs, with a
    check against the scenario's actual enforced cap (the gross_cap of
    cfg.production_scenario in VOL_SCENARIOS - NOT cfg.max_gross_exposure, which
    only bounds the earlier static percent_invested allocation in Step B, not
    the vol-targeted exposure Step D actually caps). `exposure_final_per_trade`
    is split across OPEN/CARRY/DELTA/EXIT rows per trade per day (Step F) - the
    EXIT_TRADE row's exposure is the negative of everything else for that trade
    that day (it unwinds the position to flat), so it must be excluded; CARRY
    and DELTA rows must be SUMMED, not read independently, since CARRY holds
    the stale pre-rebalance exposure and DELTA is only the incremental
    adjustment. Using CARRY alone (ignoring DELTA) overstates exposure on any
    day a position is being scaled down - this caught a false "cap breach" in
    ad hoc analysis before it was fixed here."""
    held = trade_df[trade_df["TradeRole"] != "EXIT_TRADE"]
    net_exposure = held.groupby(["date", "TradeID"])["exposure_final_per_trade"].sum()
    gross_pct = net_exposure.abs().groupby("date").sum() * 100

    cap_pct = gross_cap * 100 if gross_cap is not None else None
    breaches = gross_pct[gross_pct > cap_pct + 0.01] if cap_pct is not None else gross_pct.iloc[0:0]
    return {
        "Mean Daily Gross Exposure": gross_pct.mean(), "Median Daily Gross Exposure": gross_pct.median(),
        "Max Daily Gross Exposure": gross_pct.max(), "Min Daily Gross Exposure": gross_pct.min(),
        "Configured Cap": cap_pct, "Days Exceeding Cap": int(len(breaches)),
        "Worst Breach": breaches.max() if not breaches.empty else np.nan,
    }


def _statistical_tests(returns: pd.Series, n_trials: int = 1) -> dict:
    """Statistical diagnostics on the daily return series, beyond the descriptive
    metrics in _performance_metrics: distributional shape, serial correlation,
    stationarity, tail risk, and significance of the Sharpe ratio - including a
    multiple-testing adjustment (Deflated Sharpe Ratio) for when this config was
    chosen as the best of several trials (set [statistics] n_trials in config.ini)."""
    returns = returns.dropna()
    n = len(returns)
    mean_ret, std_ret = returns.mean(), returns.std()
    sharpe_daily = mean_ret / std_ret if std_ret else np.nan
    skew = returns.skew()
    kurt_excess = returns.kurtosis()  # pandas: Fisher (excess) kurtosis, normal = 0

    out = {
        "N": n, "Mean Daily Return": mean_ret, "Std Daily Return": std_ret,
        "Skewness": skew, "Excess Kurtosis": kurt_excess,
        "Sharpe Ratio (annualized)": sharpe_daily * math.sqrt(252) if pd.notna(sharpe_daily) else np.nan,
    }

    if HAS_SCIPY and n > 1:
        # Sharpe significance: H0: true daily Sharpe = 0 (iid-normal approximation)
        t_stat = sharpe_daily * math.sqrt(n)
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(t_stat)))
        out["Sharpe t-stat"] = t_stat
        out["Sharpe p-value"] = p_value

        jb_stat, jb_p = scipy_stats.jarque_bera(returns)
        out["Jarque-Bera Stat"], out["Jarque-Bera p-value"] = jb_stat, jb_p
        out["Returns Normal? (p>0.05)"] = jb_p > 0.05

        # Probabilistic Sharpe Ratio vs benchmark SR*=0 (Bailey & Lopez de Prado, 2012).
        # kurt_full is non-excess kurtosis (=3 for a normal distribution).
        kurt_full = kurt_excess + 3
        denom = math.sqrt(max(1 - skew * sharpe_daily + (kurt_full - 1) / 4 * sharpe_daily ** 2, 1e-12))
        psr_z = sharpe_daily * math.sqrt(n - 1) / denom
        out["Probabilistic Sharpe Ratio (vs 0)"] = scipy_stats.norm.cdf(psr_z)

        # Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): raises the benchmark
        # to the Sharpe one would expect from the BEST of n_trials trials by chance
        # alone, then re-applies PSR against that raised benchmark. sigma_SR0 (the
        # estimation std error of a single Sharpe estimate) stands in for the
        # cross-trial Sharpe dispersion, the standard simplification when individual
        # trial Sharpes aren't supplied.
        out["N Trials (for DSR)"] = n_trials
        if n_trials > 1:
            euler_gamma = 0.5772156649
            sigma_sr0 = denom / math.sqrt(n - 1)
            z1 = scipy_stats.norm.ppf(1 - 1 / n_trials)
            z2 = scipy_stats.norm.ppf(1 - 1 / (n_trials * math.e))
            sr_benchmark = sigma_sr0 * ((1 - euler_gamma) * z1 + euler_gamma * z2)
            dsr_z = (sharpe_daily - sr_benchmark) * math.sqrt(n - 1) / denom
            out["Deflated Sharpe Ratio"] = scipy_stats.norm.cdf(dsr_z)
        else:
            out["Deflated Sharpe Ratio"] = np.nan

        for level in (0.95, 0.99):
            var = returns.quantile(1 - level)
            out[f"VaR {int(level * 100)}% (daily)"] = var
            out[f"CVaR {int(level * 100)}% (daily)"] = returns[returns <= var].mean()

    if HAS_STATSMODELS and n > 10:
        try:
            lb = acorr_ljungbox(returns, lags=[10], return_df=True)
            out["Ljung-Box Stat (lag 10)"] = lb["lb_stat"].iloc[0]
            out["Ljung-Box p-value (lag 10)"] = lb["lb_pvalue"].iloc[0]
            out["Returns Autocorrelated? (p<0.05)"] = lb["lb_pvalue"].iloc[0] < 0.05
        except Exception:
            pass
        try:
            adf_stat, adf_p = adfuller(returns)[:2]
            out["ADF Stat"], out["ADF p-value"] = adf_stat, adf_p
            out["Returns Stationary? (p<0.05)"] = adf_p < 0.05
        except Exception:
            pass

    return out


def _write_statistical_tests(path, stats: dict, exposure: dict, strategy_label: str):
    rows = [("Metric", "Value"), ("Strategy", strategy_label),
            ("--- RETURN DISTRIBUTION ---", None),
            ("N (trading days)", stats["N"]),
            ("Mean Daily Return", f"{stats['Mean Daily Return']:.4%}"),
            ("Std Daily Return", f"{stats['Std Daily Return']:.4%}"),
            ("Skewness", round(stats["Skewness"], 3)),
            ("Excess Kurtosis", round(stats["Excess Kurtosis"], 3))]

    if "Jarque-Bera Stat" in stats:
        rows += [("Jarque-Bera Stat", round(stats["Jarque-Bera Stat"], 2)),
                 ("Jarque-Bera p-value", round(stats["Jarque-Bera p-value"], 4)),
                 ("Returns Normal? (p>0.05)", stats["Returns Normal? (p>0.05)"])]

    rows += [("--- SERIAL CORRELATION / STATIONARITY ---", None)]
    if "Ljung-Box Stat (lag 10)" in stats:
        rows += [("Ljung-Box Stat (lag 10)", round(stats["Ljung-Box Stat (lag 10)"], 2)),
                 ("Ljung-Box p-value (lag 10)", round(stats["Ljung-Box p-value (lag 10)"], 4)),
                 ("Returns Autocorrelated? (p<0.05)", stats["Returns Autocorrelated? (p<0.05)"])]
    if "ADF Stat" in stats:
        rows += [("ADF Stat", round(stats["ADF Stat"], 3)),
                 ("ADF p-value", round(stats["ADF p-value"], 4)),
                 ("Returns Stationary? (p<0.05)", stats["Returns Stationary? (p<0.05)"])]

    rows += [("--- SHARPE RATIO SIGNIFICANCE ---", None),
              ("Sharpe Ratio (annualized)", round(stats["Sharpe Ratio (annualized)"], 3))]
    if "Sharpe t-stat" in stats:
        dsr = stats["Deflated Sharpe Ratio"]
        rows += [("Sharpe t-stat (H0: Sharpe=0)", round(stats["Sharpe t-stat"], 2)),
                 ("Sharpe p-value", round(stats["Sharpe p-value"], 6)),
                 ("Probabilistic Sharpe Ratio (vs 0)", f"{stats['Probabilistic Sharpe Ratio (vs 0)']:.2%}"),
                 ("N Trials (for DSR)", stats["N Trials (for DSR)"]),
                 ("Deflated Sharpe Ratio", f"{dsr:.2%}" if pd.notna(dsr) else "N/A (n_trials=1)")]

    if "VaR 95% (daily)" in stats:
        rows += [("--- TAIL RISK ---", None),
                 ("VaR 95% (daily)", f"{stats['VaR 95% (daily)']:.3%}"),
                 ("CVaR 95% (daily)", f"{stats['CVaR 95% (daily)']:.3%}"),
                 ("VaR 99% (daily)", f"{stats['VaR 99% (daily)']:.3%}"),
                 ("CVaR 99% (daily)", f"{stats['CVaR 99% (daily)']:.3%}")]

    rows += [("--- PORTFOLIO EXPOSURE ---", None),
              ("Mean Daily Gross Exposure", f"{exposure['Mean Daily Gross Exposure']:.2f}%"),
              ("Median Daily Gross Exposure", f"{exposure['Median Daily Gross Exposure']:.2f}%"),
              ("Max Daily Gross Exposure", f"{exposure['Max Daily Gross Exposure']:.2f}%"),
              ("Configured Cap", f"{exposure['Configured Cap']:.2f}%" if exposure["Configured Cap"] is not None else "None (nocap scenario)"),
              ("Days Exceeding Cap", exposure["Days Exceeding Cap"]),
              ("Worst Breach", f"{exposure['Worst Breach']:.2f}%" if pd.notna(exposure["Worst Breach"]) else "None")]

    pd.DataFrame(rows[1:], columns=rows[0]).to_excel(path, index=False)


def _trade_level_table(trade_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse split legs (OPEN/ACTIVE/CLOSE/EXIT rows) back to one row per trade."""
    group_cols = ["Combination", "Opening Date", "Closing Date"]
    trades = (
        trade_df.groupby(group_cols, as_index=False)
        .agg(
            Strategy=("Strategy", "first"),
            Currency=("Currency", "first"),
            Condition=("Condition", "first"),
            Direction=("Direction", "first"),
            Timeframe=("Timeframe", "first"),
            PL=(PL_COL, "sum"),
        )
    )
    trades["Year"] = trades["Opening Date"].dt.year
    trades["Quarter"] = trades["Opening Date"].dt.quarter
    return trades


def _trade_stats_block(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"Total Trades (closed)": 0, "Trade Win Rate": np.nan,
                "Avg Winning Trade PL": np.nan, "Avg Losing Trade PL": np.nan, "Trade Profit Factor": np.nan}

    wins, losses = trades[trades["PL"] > 0], trades[trades["PL"] < 0]
    profit_factor = wins["PL"].sum() / abs(losses["PL"].sum()) if not losses.empty and losses["PL"].sum() != 0 else np.nan
    return {
        "Total Trades (closed)": len(trades),
        "Trade Win Rate": len(wins) / len(trades),
        "Avg Winning Trade PL": wins["PL"].mean() if not wins.empty else np.nan,
        "Avg Losing Trade PL": losses["PL"].mean() if not losses.empty else np.nan,
        "Trade Profit Factor": profit_factor,
    }


def _drawdown_periods(equity: pd.Series, top_n: int = 15) -> pd.DataFrame:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    in_dd = drawdown < 0

    periods = []
    start = None
    for date, flag in in_dd.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            window = drawdown.loc[start:date]
            periods.append((start, window.idxmin(), date, window.min()))
            start = None
    if start is not None:
        window = drawdown.loc[start:]
        periods.append((start, window.idxmin(), drawdown.index[-1], window.min()))

    out = pd.DataFrame(periods, columns=["Start", "Trough", "End", "Magnitude"])
    if out.empty:
        return out
    out["Duration (days)"] = (out["End"] - out["Start"]).dt.days
    return out.sort_values("Magnitude").head(top_n).reset_index(drop=True)


def _quarter_bounds(dates) -> tuple[int, int, str]:
    last = pd.Series(dates).max()
    q = (last.month - 1) // 3 + 1
    return last.year, q, f"{last.year}_Q{q}"


def _in_quarter(dates, year: int, q: int) -> np.ndarray:
    """Boolean numpy array (not a Series) so it can mask any frame positionally, regardless of index."""
    idx = pd.DatetimeIndex(pd.Series(dates).values)
    return (idx.year == year) & (((idx.month - 1) // 3 + 1) == q)


def _write_summary_xlsx(path, metrics: dict, trade_stats: dict, strategy_label: str):
    rows = [("Metric", "Value"),
            ("Strategy", strategy_label),
            ("Start Date", metrics["Start Date"].date().isoformat()),
            ("End Date", metrics["End Date"].date().isoformat()),
            ("Trading Days", metrics["Trading Days"]),
            ("Years", metrics["Years"]),
            ("--- RETURN METRICS ---", None),
            ("Total Return", f"{metrics['Total Return']:.2%}"),
            ("CAGR", f"{metrics['CAGR']:.2%}"),
            ("Best Day", f"{metrics['Best Day']:.2%}"),
            ("Worst Day", f"{metrics['Worst Day']:.2%}"),
            ("--- RISK METRICS ---", None),
            ("Ann. Volatility", f"{metrics['Ann. Volatility']:.2%}"),
            ("Sharpe Ratio", round(metrics["Sharpe Ratio"], 3)),
            ("Sortino Ratio", round(metrics["Sortino Ratio"], 3) if pd.notna(metrics["Sortino Ratio"]) else None),
            ("Max Drawdown", f"{metrics['Max Drawdown']:.2%}"),
            ("Calmar Ratio", round(metrics["Calmar Ratio"], 3) if pd.notna(metrics["Calmar Ratio"]) else None),
            ("--- DAY-LEVEL STATS ---", None),
            ("Win Days", metrics["Win Days"]),
            ("Loss Days", metrics["Loss Days"]),
            ("Win Rate (Days)", f"{metrics['Win Rate (Days)']:.2%}"),
            ("Avg Win Day", f"{metrics['Avg Win Day']:.2%}"),
            ("Avg Loss Day", f"{metrics['Avg Loss Day']:.2%}"),
            ("Profit Factor (Days)", round(metrics["Profit Factor (Days)"], 2) if pd.notna(metrics["Profit Factor (Days)"]) else None),
            ("--- TRADE-LEVEL STATS ---", None),
            ("Total Trades (closed)", trade_stats["Total Trades (closed)"]),
            ("Trade Win Rate", f"{trade_stats['Trade Win Rate']:.2%}" if pd.notna(trade_stats["Trade Win Rate"]) else None),
            ("Avg Winning Trade PL", trade_stats["Avg Winning Trade PL"]),
            ("Avg Losing Trade PL", trade_stats["Avg Losing Trade PL"]),
            ("Trade Profit Factor", round(trade_stats["Trade Profit Factor"], 3) if pd.notna(trade_stats["Trade Profit Factor"]) else None)]

    pd.DataFrame(rows[1:], columns=rows[0]).to_excel(path, index=False)


def _write_yearly_performance(path, returns: pd.Series):
    rows = []
    for year, grp in returns.groupby(returns.index.year):
        equity = (1 + grp).cumprod()
        ann_vol = grp.std() * np.sqrt(252)
        yearly_return = equity.iloc[-1] - 1
        sharpe = yearly_return / ann_vol if ann_vol else np.nan
        dd = (equity / equity.cummax() - 1).min()
        win_rate = (grp > 0).mean()
        rows.append({"Year": year, "Return": yearly_return, "Ann. Volatility": ann_vol,
                     "Sharpe": sharpe, "Max Drawdown": dd, "Win Rate": win_rate, "Trading Days": len(grp)})
    pd.DataFrame(rows).to_excel(path, index=False)


def _write_monthly_performance(path, returns: pd.Series):
    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(lambda x: (1 + x).prod() - 1)
    monthly.index.names = ["Year", "Month"]
    pivot = monthly.unstack("Month").sort_index()
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
    pivot.to_excel(path)


def _write_drawdown_analysis(path, equity: pd.Series):
    dd = _drawdown_periods(equity)
    dd.to_excel(path, index=False)
    return dd


def _write_trade_statistics(path, trades: pd.DataFrame):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for group_col, sheet in [("Currency", "By_Instrument"), ("Direction", "By_Direction"),
                                  ("Condition", "By_Condition"), ("Year", "By_Year")]:
            tbl = trades.groupby(group_col).agg(
                Trades=("PL", "count"), Total_PL=("PL", "sum"),
                Win_Rate=("PL", lambda x: (x > 0).mean()),
            ).reset_index()
            tbl.to_excel(writer, sheet_name=sheet, index=False)


def _write_instrument_performance(path, trades_overall: pd.DataFrame, trades_quarter: pd.DataFrame, quarter_label: str):
    def summarize(trades):
        if trades.empty:
            return pd.DataFrame(columns=["Currency", "Trades", "Total_PL", "Avg_PL", "Win_Rate"])
        tbl = trades.groupby("Currency").agg(
            Trades=("PL", "count"), Total_PL=("PL", "sum"), Avg_PL=("PL", "mean"),
            Win_Rate=("PL", lambda x: (x > 0).mean()),
        ).reset_index().sort_values("Total_PL", ascending=False)
        return tbl

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summarize(trades_overall).to_excel(writer, sheet_name="Overall", index=False)
        summarize(trades_quarter).to_excel(writer, sheet_name=quarter_label[:31], index=False)


def _write_rolling_metrics(path, returns: pd.Series, window: int = 63):
    rolling_return = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_return * (252 / window)) / rolling_vol
    out = pd.DataFrame({
        "date": returns.index, f"Rolling_{window}D_Return": rolling_return.values,
        f"Rolling_{window}D_Ann_Vol": rolling_vol.values, f"Rolling_{window}D_Sharpe": rolling_sharpe.values,
    }).dropna()
    out.to_excel(path, index=False)
    return out


def _save_overall_charts(path, metrics: dict, returns: pd.Series, rolling: pd.DataFrame):
    equity, drawdown = metrics["_equity"], metrics["_drawdown"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(equity.index, equity.values)
    axes[0, 0].set_title("Equity Curve")

    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.5)
    axes[0, 1].set_title("Drawdown")

    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(lambda x: (1 + x).prod() - 1)
    monthly.index.names = ["Year", "Month"]
    heat = monthly.unstack("Month").sort_index()
    im = axes[1, 0].imshow(heat.values, aspect="auto", cmap="RdYlGn")
    axes[1, 0].set_yticks(range(len(heat.index)))
    axes[1, 0].set_yticklabels(heat.index)
    axes[1, 0].set_title("Monthly Return Heatmap")
    fig.colorbar(im, ax=axes[1, 0])

    if not rolling.empty:
        axes[1, 1].plot(rolling["date"], rolling[[c for c in rolling.columns if "Sharpe" in c][0]])
    axes[1, 1].set_title("Rolling 63D Sharpe")

    if not rolling.empty:
        axes[2, 0].plot(rolling["date"], rolling[[c for c in rolling.columns if "Return" in c][0]])
    axes[2, 0].set_title("Rolling 63D Return")

    axes[2, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_quarter_charts(path, returns_q: pd.Series):
    equity_q = (1 + returns_q).cumprod()
    drawdown_q = equity_q / equity_q.cummax() - 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(equity_q.index, equity_q.values)
    axes[0].set_title("Equity Curve")
    axes[1].bar(returns_q.index, returns_q.values)
    axes[1].set_title("Daily Returns")
    axes[2].fill_between(drawdown_q.index, drawdown_q.values, 0, color="red", alpha=0.5)
    axes[2].set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_instrument_charts(path, trades: pd.DataFrame):
    if trades.empty:
        return
    tbl = trades.groupby("Currency").agg(
        Total_PL=("PL", "sum"), Win_Rate=("PL", lambda x: (x > 0).mean()),
        Sharpe=("PL", lambda x: x.mean() / x.std() if x.std() else np.nan), Trades=("PL", "count"),
    ).sort_values("Total_PL", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    tbl["Total_PL"].plot(kind="bar", ax=axes[0, 0], title="Total PL")
    tbl["Win_Rate"].plot(kind="bar", ax=axes[0, 1], title="Win Rate")
    tbl["Sharpe"].plot(kind="bar", ax=axes[1, 0], title="Sharpe (per-trade)")
    tbl["Trades"].plot(kind="bar", ax=axes[1, 1], title="Trade Count")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_quantstats_html(path, returns: pd.Series, title: str):
    if not HAS_QUANTSTATS:
        return
    try:
        qs.reports.html(returns, benchmark=None, output=str(path), title=title)
    except Exception as e:
        print(f"  [warn] QuantStats HTML skipped ({title}): {e!r}")


def _write_quarter_html(path, label: str, metrics_q: dict, trade_stats_q: dict, instrument_tbl: pd.DataFrame,
                         chart_rel: str, inst_chart_rel: str):
    rows_html = "".join(
        f"<tr><td>{r.Currency}</td><td>{r.Trades}</td><td>{r.Total_PL:.4f}</td><td>{r.Win_Rate:.1%}</td></tr>"
        for r in instrument_tbl.itertuples()
    )
    html = f"""<html><head><title>{label} Report</title></head><body>
<h1>{label} Performance Report</h1>
<h2>Summary</h2>
<ul>
<li>Total Return: {metrics_q['Total Return']:.2%}</li>
<li>Sharpe Ratio: {metrics_q['Sharpe Ratio']:.3f}</li>
<li>Max Drawdown: {metrics_q['Max Drawdown']:.2%}</li>
<li>Total Trades: {trade_stats_q['Total Trades (closed)']}</li>
<li>Trade Win Rate: {trade_stats_q['Trade Win Rate']:.2%}</li>
</ul>
<h2>Charts</h2>
<img src="{chart_rel}" style="max-width:100%"><br>
<img src="{inst_chart_rel}" style="max-width:100%">
<h2>Instrument Breakdown</h2>
<table border="1" cellpadding="4"><tr><th>Instrument</th><th>Trades</th><th>Total PL</th><th>Win Rate</th></tr>
{rows_html}
</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def build_full_report(cfg: Config, day_level_data: pd.DataFrame, trade_df: pd.DataFrame) -> None:
    print("\n[Step K] Building final quant-stat report suite...")
    out_dir = cfg.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.label

    returns = day_level_data.set_index("date")[PL_COL].dropna().sort_index()
    metrics = _performance_metrics(returns)
    trades = _trade_level_table(trade_df)
    trade_stats = _trade_stats_block(trades)
    strategy_label = f"{tag} Strategy - Broker Vol Target ({cfg.production_scenario}), {cfg.slippage_pips}-pip slippage"

    _write_summary_xlsx(out_dir / f"01_{tag}_Overall_Summary.xlsx", metrics, trade_stats, strategy_label)
    _write_yearly_performance(out_dir / f"02_{tag}_Yearly_Performance.xlsx", returns)
    _write_monthly_performance(out_dir / f"03_{tag}_Monthly_Performance.xlsx", returns)
    _write_drawdown_analysis(out_dir / f"04_{tag}_Drawdown_Analysis.xlsx", metrics["_equity"])

    _write_trade_statistics(out_dir / f"05_{tag}_Trade_Statistics.xlsx", trades)

    stat_tests = _statistical_tests(returns, n_trials=cfg.n_trials)
    gross_cap = _lookup_gross_cap(cfg.production_scenario)
    exposure_stats = _exposure_diagnostics(trade_df, gross_cap)
    if exposure_stats["Days Exceeding Cap"]:
        print(f"  [WARNING] {exposure_stats['Days Exceeding Cap']} day(s) exceed the {exposure_stats['Configured Cap']:.1f}% "
              f"gross exposure cap - worst breach {exposure_stats['Worst Breach']:.1f}%")
    _write_statistical_tests(out_dir / f"08_{tag}_Statistical_Tests.xlsx", stat_tests, exposure_stats, strategy_label)

    q_year, q_num, q_label = _quarter_bounds(day_level_data["date"])
    q_mask = _in_quarter(returns.index, q_year, q_num)
    trades_quarter = trades[_in_quarter(trades["Opening Date"], q_year, q_num)]
    _write_instrument_performance(out_dir / f"05b_{tag}_Instrument_Performance.xlsx", trades, trades_quarter, q_label)

    rolling = _write_rolling_metrics(out_dir / f"06_{tag}_Rolling_Metrics.xlsx", returns)

    returns_q = returns[q_mask]
    metrics_q = _performance_metrics(returns_q) if not returns_q.empty else metrics
    trade_stats_q = _trade_stats_block(trades_quarter)
    _write_summary_xlsx(out_dir / f"07_{tag}_{q_label}_Performance.xlsx", metrics_q, trade_stats_q,
                         f"{strategy_label} ({q_label})")

    _save_overall_charts(out_dir / f"{tag}_Overall_Charts.png", metrics, returns, rolling)
    quarter_chart = out_dir / f"{tag}_{q_label}_Report_charts.png"
    _save_quarter_charts(quarter_chart, returns_q)
    inst_chart_overall = out_dir / f"{tag}_Instrument_Charts_Overall.png"
    _save_instrument_charts(inst_chart_overall, trades)
    inst_chart_quarter = out_dir / f"{tag}_Instrument_Charts_{q_label}.png"
    _save_instrument_charts(inst_chart_quarter, trades_quarter)

    _write_quantstats_html(out_dir / f"{tag}_QuantStats_Overall.html", returns, f"{strategy_label} - Overall")
    if not returns_q.empty:
        _write_quantstats_html(out_dir / f"{tag}_QuantStats_{q_label}.html", returns_q, f"{strategy_label} - {q_label}")

    inst_tbl_q = trades_quarter.groupby("Currency").agg(Trades=("PL", "count"), Total_PL=("PL", "sum"),
                                                         Win_Rate=("PL", lambda x: (x > 0).mean())).reset_index() \
        if not trades_quarter.empty else pd.DataFrame(columns=["Currency", "Trades", "Total_PL", "Win_Rate"])
    _write_quarter_html(out_dir / f"{tag}_{q_label}_Report.html", f"{tag} {q_label}", metrics_q, trade_stats_q,
                         inst_tbl_q, quarter_chart.name, inst_chart_quarter.name)

    print(f"[Step K] Saved {len(list(out_dir.glob('*')))} files to: {out_dir}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    t_start = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.ini",
                         help="Path to config.ini (default: config.ini next to this script)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.years = cfg.years or discover_years(cfg)

    print(f"Config       : {args.config}")
    print(f"Raw data dir : {cfg.raw_data_dir}")
    print(f"MTM data dir : {cfg.mtm_data_dir}")
    print(f"Output dir   : {cfg.output_dir}")
    print(f"Years        : {cfg.years}")
    print(f"Scenario     : {cfg.production_scenario}\n")

    build_vol_wall(cfg)
    run_mtm_all_years(cfg)
    instrument_master = build_instrument_level_data(cfg)
    build_broker_vol_portfolio(cfg, instrument_master)
    trade_df = merge_instrument_to_trade(cfg)
    split_df = split_trades_by_vol_change(cfg, trade_df)
    cost_df = assign_equity_costs_slippage(cfg, split_df)
    day_level_data = build_summary_and_final_outputs(cfg, cost_df)

    print_performance_summary(day_level_data, cfg.label)
    build_full_report(cfg, day_level_data, cost_df[FINAL_TRADE_COLUMNS])

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes). Final outputs in: {cfg.final_dir}")


if __name__ == "__main__":
    main()
