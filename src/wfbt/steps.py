"""The five pipeline steps.

1. monthly_rollup        trades -> monthly return per iteration (by exit month)
2. walk_forward_select   rolling recency-weighted Sharpe -> best parameter set per combination per month
3. forward_trades        the chosen iteration's trades *entered* in the following month (forward test)
4. monthly_summary       one-open-trade-at-a-time filter + per (combination, month) statistics
5. daily_mtm             forward-test trades marked to market daily -> portfolio returns

A "combination" is (strategy, instrument, side). An "iteration" is one
combination with one parameter set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

COMBO = ["strategy", "instrument", "side"]

# recency weights over a 60-month window, newest months weigh most (normalised to sum to 1)
WEIGHT_CONFIG = {"recent_3m": 0.20, "rest_year1": 0.20, "year2": 0.25, "year3": 0.12, "year4": 0.12,
                 "year5": 0.12}


def recency_weights(window: int = 60, config: dict = WEIGHT_CONFIG) -> np.ndarray:
    """Weights oldest -> newest for the rolling window."""
    buckets = [(3, "recent_3m"), (12, "rest_year1"), (24, "year2"), (36, "year3"), (48, "year4"), (60, "year5")]
    w = np.zeros(window)
    lo = 0
    for hi, key in buckets:
        hi = min(hi, window)
        if hi > lo:
            w[window - hi: window - lo] = config[key] / (hi - lo)
        lo = hi
    return w / w.sum()


# --------------------------------------------------------------------- step 1
def monthly_rollup(trades: pd.DataFrame, capital_base: float = 100_000.0) -> pd.DataFrame:
    """Monthly return and trade count per iteration, bucketed by *exit* month."""
    m = trades.assign(month=trades["exit_date"].dt.to_period("M"))
    out = (m.groupby(["iteration_id", *COMBO, "month"], sort=True)
           .agg(monthly_pnl=("pnl", "sum"), monthly_trades=("pnl", "size")).reset_index())
    out["monthly_return"] = out["monthly_pnl"] / capital_base
    return out


# --------------------------------------------------------------------- step 2
def walk_forward_scores(monthly: pd.DataFrame, months: pd.PeriodIndex, window: int = 60,
                        weights: np.ndarray | None = None) -> pd.DataFrame:
    """Rolling weighted Sharpe for every iteration at every eval month.

    Every iteration is scored on the same global calendar (silent months are
    zero). The original only scored an iteration between its own first and
    *last* trade month, which quietly used future knowledge: a parameter set
    that would never trade again could not be selected.
    """
    w = recency_weights(window) if weights is None else weights
    ret = monthly.pivot_table(index="iteration_id", columns="month", values="monthly_return",
                              aggfunc="sum").reindex(columns=months, fill_value=0.0).fillna(0.0)
    cnt = monthly.pivot_table(index="iteration_id", columns="month", values="monthly_trades",
                              aggfunc="sum").reindex(columns=months, fill_value=0).fillna(0)
    R = sliding_window_view(ret.to_numpy(float), window, axis=1)      # (iters, n-w+1, w)
    C = sliding_window_view(cnt.to_numpy(float), window, axis=1)
    mean = (R * w).sum(axis=2)
    std = np.sqrt((((R - mean[..., None]) ** 2) * w).sum(axis=2))
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, mean / std * np.sqrt(12), np.nan)
    eval_months = months[window - 1:]
    out = pd.DataFrame({
        "iteration_id": np.repeat(ret.index.to_numpy(), len(eval_months)),
        "eval_month": np.tile(eval_months, len(ret)),
        "wf_pnl": mean.ravel(), "wf_sharpe": sr.ravel(), "wf_trades": C.sum(axis=2).ravel(),
    })
    meta = monthly.drop_duplicates("iteration_id").set_index("iteration_id")[COMBO]
    out = out.join(meta, on="iteration_id")
    out["trade_month"] = out["eval_month"] + 1
    return out


def walk_forward_select(scores: pd.DataFrame, min_sharpe: float = 1.0, min_trades: int = 0) -> pd.DataFrame:
    """Best iteration per combination per eval month, kept if wf_sharpe >= min_sharpe.

    `min_trades` drops iterations with too few trades in the window. Zero-filled
    silent months shrink the standard deviation, so a sparse set with a few
    lucky winners can post a high Sharpe. The original had no such guard.
    """
    s = scores.dropna(subset=["wf_sharpe"])
    s = s[s["wf_trades"] >= min_trades]
    s = s.sort_values([*COMBO, "eval_month", "wf_sharpe", "wf_pnl", "iteration_id"],
                      ascending=[True, True, True, True, False, False, True])
    best = s.groupby([*COMBO, "eval_month"], sort=False).head(1)
    return best[best["wf_sharpe"] >= min_sharpe].reset_index(drop=True)


# --------------------------------------------------------------------- step 3
def forward_trades(selection: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Trades of the selected iteration whose entry month is the selection's trade month."""
    t = trades.assign(entry_month=trades["entry_date"].dt.to_period("M"))
    sel = selection[["iteration_id", "trade_month", "wf_sharpe"]]
    ft = t.merge(sel, left_on=["iteration_id", "entry_month"], right_on=["iteration_id", "trade_month"])
    return ft.drop(columns=["entry_month"]).sort_values(["entry_date", "iteration_id"]).reset_index(drop=True)


def train_set(selection_row: pd.Series, trades: pd.DataFrame, purge: bool = True) -> pd.DataFrame:
    """Training trades for a model that decides at the start of `trade_month`.

    The original kept every trade *entered* before the month, including trades
    still open at the decision time, whose outcome was not yet known (label
    leakage). With `purge=True`, a trade must have *exited* before the month starts.
    """
    start = selection_row["trade_month"].start_time
    t = trades[trades["iteration_id"] == selection_row["iteration_id"]]
    col = "exit_date" if purge else "entry_date"
    return t[t[col] < start]


# --------------------------------------------------------------------- step 4
def non_overlapping(ft: pd.DataFrame) -> pd.DataFrame:
    """Per combination keep a trade only if it enters after the last kept trade exited.

    Switching parameter sets at a month boundary can otherwise leave two
    positions open in the same combination.
    """
    keep = []
    for _, g in ft.sort_values("entry_date").groupby(COMBO, sort=False):
        last_exit = pd.Timestamp.min
        for idx, e, x in zip(g.index, g["entry_date"], g["exit_date"]):
            if e > last_exit:
                keep.append(idx)
                last_exit = x
    return ft.loc[sorted(keep)]


def monthly_summary(ft: pd.DataFrame) -> pd.DataFrame:
    """Per (combination, month): trades, wins, win %, total and average P&L, notional."""
    f = ft.assign(month=ft["entry_date"].dt.to_period("M"), win=ft["pnl"] > 0)
    return (f.groupby([*COMBO, "month"]).agg(
        total_trades=("pnl", "size"), winning_trades=("win", "sum"),
        win_pct=("win", lambda x: 100.0 * x.mean()), total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"), total_notional=("notional", "sum")).reset_index())


# --------------------------------------------------------------------- step 5
def daily_mtm(ft: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily mark-to-market P&L for every forward-test trade.

    Day d of a trade earns direction * notional * (P_d - P_{d-1}) / P_entry. The
    sum over the holding period equals the trade's gross P&L. Costs are booked on
    the entry day (half) and the exit day (half).
    """
    px = prices.pivot(index="date", columns="instrument", values="close").sort_index()
    rows = []
    for tr in ft.itertuples(index=False):
        s = px.loc[tr.entry_date:tr.exit_date, tr.instrument]
        pnl = tr.direction * tr.notional * s.diff().fillna(0.0).to_numpy() / tr.entry_price
        pnl[0] -= tr.cost / 2
        pnl[-1] -= tr.cost / 2
        rows.append(pd.DataFrame({"date": s.index, "iteration_id": tr.iteration_id,
                                  "trade_idx": getattr(tr, "trade_idx", -1), "pnl": pnl}))
    if not rows:
        return pd.DataFrame(columns=["date", "iteration_id", "trade_idx", "pnl"])
    return pd.concat(rows, ignore_index=True)


def portfolio_returns(mtm: pd.DataFrame, calendar: pd.DatetimeIndex, capital: float) -> pd.Series:
    """Daily portfolio return on a fixed capital base (P&L / capital, not compounded)."""
    r = mtm.groupby("date")["pnl"].sum() / capital if len(mtm) else pd.Series(dtype=float)
    return r.reindex(calendar, fill_value=0.0)
