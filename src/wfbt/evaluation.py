"""Run the five steps, then judge the selection threshold honestly.

The original research swept the selection threshold (`MIN_SHARPE`) and
measured each value on the *same* forward-test period it then reported. The
forward-test Sharpe rose with the threshold, and the best threshold was adopted.
That makes the forward test in-sample for the threshold. Here the threshold is
treated as a hyper-parameter:

* **naive**: the threshold with the best Sharpe over the whole forward test
  (what the original did). Reported only for contrast, with a deflated Sharpe.
* **nested walk-forward**: for each year, use the threshold that did best on
  the forward-test years *before* it.
* **hold-out**: the last `holdout_years` are untouched. The threshold is chosen
  on the forward-test years before them, then scored once.
* **selection decay**: the average in-sample Sharpe of the picks against what
  the same picks earned the next month.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import deflated_sharpe, sharpe, summarize
from .steps import (COMBO, daily_mtm, forward_trades, monthly_rollup, monthly_summary, non_overlapping,
                    portfolio_returns, walk_forward_scores, walk_forward_select)

THRESHOLDS = [0.5, 1.0, 1.5, 2.0]


@dataclass
class PipelineRun:
    monthly: pd.DataFrame
    scores: pd.DataFrame
    picks: pd.DataFrame          # best iteration per combination-month, no threshold applied
    ft: pd.DataFrame             # forward-test trades of every pick (with the pick's wf_sharpe)
    mtm: pd.DataFrame            # daily P&L per forward-test trade (column trade_idx)
    calendar: pd.DatetimeIndex   # forward-test business days
    capital: float


def run_pipeline(trades: pd.DataFrame, prices: pd.DataFrame, window: int = 60, min_trades: int = 20,
                 notional: float = 100_000.0) -> PipelineRun:
    months = pd.period_range(trades["entry_date"].min(), trades["exit_date"].max(), freq="M")
    monthly = monthly_rollup(trades, capital_base=notional)
    scores = walk_forward_scores(monthly, months, window)
    picks = walk_forward_select(scores, min_sharpe=-np.inf, min_trades=min_trades)
    ft = forward_trades(picks, trades).reset_index(drop=True)
    ft["trade_idx"] = ft.index
    mtm = daily_mtm(ft, prices)
    first = picks["trade_month"].min().start_time
    calendar = pd.DatetimeIndex(sorted(prices.loc[prices["date"] >= first, "date"].unique()))
    capital = notional * trades.groupby(COMBO).ngroups
    return PipelineRun(monthly, scores, picks, ft, mtm, calendar, capital)


def returns_for_threshold(run: PipelineRun, t: float) -> tuple[pd.Series, pd.DataFrame]:
    """Portfolio daily returns and kept trades when only picks with wf_sharpe >= t are traded."""
    kept = non_overlapping(run.ft[run.ft["wf_sharpe"] >= t])
    m = run.mtm[run.mtm["trade_idx"].isin(kept["trade_idx"])]
    return portfolio_returns(m, run.calendar, run.capital), kept


def _sr_in(r: pd.Series, years: list[int]) -> float:
    return sharpe(r[r.index.year.isin(years)])


def threshold_study(run: PipelineRun, thresholds: list[float] = THRESHOLDS, holdout_years: int = 2) -> dict:
    rets = {t: returns_for_threshold(run, t)[0] for t in thresholds}
    years = sorted(set(run.calendar.year))
    research, holdout = years[:-holdout_years], years[-holdout_years:]

    per_t = {t: {"ft_sharpe_all_years": sharpe(r), "research_sharpe": _sr_in(r, research),
                 "holdout_sharpe": _sr_in(r, holdout)} for t, r in rets.items()}

    naive_t = max(thresholds, key=lambda t: (per_t[t]["ft_sharpe_all_years"], -t))
    naive = {"threshold": naive_t, **summarize(rets[naive_t]),
             "deflated_sharpe_prob": deflated_sharpe(rets[naive_t], [v["ft_sharpe_all_years"] for v in per_t.values()])}

    nested_parts, nested_choice = [], {}
    for i, y in enumerate(research[1:], start=1):
        past = research[:i]
        t = max(thresholds, key=lambda t: (_sr_in(rets[t], past), -t))
        nested_choice[y] = t
        nested_parts.append(rets[t][rets[t].index.year == y])
    nested = pd.concat(nested_parts) if nested_parts else pd.Series(dtype=float)

    hold_t = max(thresholds, key=lambda t: (per_t[t]["research_sharpe"], -t))
    r_h = rets[hold_t][rets[hold_t].index.year.isin(holdout)]
    return {
        "research_years": [research[0], research[-1]], "holdout_years": [holdout[0], holdout[-1]],
        "per_threshold": per_t,
        "naive": naive,
        "nested": {"choices": nested_choice, **summarize(nested)},
        "holdout": {"threshold": hold_t, **summarize(r_h)},
    }


def selection_decay(run: PipelineRun, t: float = 1.0, notional: float = 100_000.0) -> dict:
    """In-sample Sharpe of the picks vs the Sharpe of their next-month returns (single-combination scale)."""
    picks = run.picks[run.picks["wf_sharpe"] >= t]
    nxt = (run.ft.groupby(["iteration_id", "trade_month"])["pnl"].sum() / notional).rename("ft_return")
    p = picks.join(nxt, on=["iteration_id", "trade_month"]).fillna({"ft_return": 0.0})
    p = p[p["trade_month"] <= run.ft["trade_month"].max()]
    realised = float(p["ft_return"].mean() / p["ft_return"].std(ddof=1) * np.sqrt(12)) if len(p) > 1 else 0.0
    return {"n_picks": int(len(p)), "mean_in_sample_sharpe": float(p["wf_sharpe"].mean()),
            "realised_next_month_sharpe": realised}


def step_outputs(run: PipelineRun, t: float = 1.0) -> dict[str, pd.DataFrame]:
    """The tables the original pipeline wrote, for the default threshold."""
    _, kept = returns_for_threshold(run, t)
    return {
        "step1_monthly_rollup": run.monthly,
        "step2_selection": run.picks[run.picks["wf_sharpe"] >= t],
        "step3_forward_trades": run.ft[run.ft["wf_sharpe"] >= t],
        "step4_monthly_summary": monthly_summary(kept),
    }
