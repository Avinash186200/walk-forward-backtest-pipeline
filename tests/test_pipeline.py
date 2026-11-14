import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wfbt.evaluation import returns_for_threshold, run_pipeline, selection_decay, threshold_study  # noqa: E402
from wfbt.metrics import sharpe  # noqa: E402
from wfbt.steps import (WEIGHT_CONFIG, daily_mtm, forward_trades, monthly_rollup, non_overlapping,  # noqa: E402
                        recency_weights, train_set, walk_forward_scores, walk_forward_select)
from wfbt.synthetic import SyntheticConfig, make_backtest  # noqa: E402


@pytest.fixture(scope="module")
def planted():
    trades, prices = make_backtest(SyntheticConfig(signal_strength=1.0))
    return trades, prices, run_pipeline(trades, prices)


@pytest.fixture(scope="module")
def noise():
    trades, prices = make_backtest(SyntheticConfig(signal_strength=0.0))
    return trades, prices, run_pipeline(trades, prices)


def _months(trades):
    return pd.period_range(trades["entry_date"].min(), trades["exit_date"].max(), freq="M")


# ------------------------------------------------------------------ step 1 / 2 mechanics
def test_recency_weights_match_original_construction():
    n, w = 60, np.zeros(60)
    c = WEIGHT_CONFIG
    for i in range(n):  # the original loop, oldest -> newest
        m = (n - 1) - i
        w[i] = (c["recent_3m"] / 3 if m <= 2 else c["rest_year1"] / 9 if m <= 11 else c["year2"] / 12 if m <= 23
                else c["year3"] / 12 if m <= 35 else c["year4"] / 12 if m <= 47 else c["year5"] / 12)
    assert np.allclose(recency_weights(60), w / w.sum())


def test_rollup_buckets_by_exit_month(planted):
    trades, _, run = planted
    it = trades["iteration_id"].iloc[0]
    t = trades[trades["iteration_id"] == it]
    m = run.monthly[run.monthly["iteration_id"] == it]
    assert m["monthly_trades"].sum() == len(t)
    assert np.isclose(m["monthly_pnl"].sum(), t["pnl"].sum())


def test_vectorised_scores_match_reference_loop(planted):
    trades, _, run = planted
    it = sorted(trades["iteration_id"].unique())[5]
    w = recency_weights(60)
    months = _months(trades)
    ser = run.monthly[run.monthly["iteration_id"] == it].set_index("month")["monthly_return"]
    ser = ser.reindex(months, fill_value=0.0).to_numpy()
    got = run.scores[run.scores["iteration_id"] == it].reset_index(drop=True)
    for k in (0, 17, len(got) - 1):
        win = ser[k:k + 60]
        mu = np.sum(w * win)
        sd = np.sqrt(np.sum(w * (win - mu) ** 2))
        assert np.isclose(got.loc[k, "wf_sharpe"], mu / sd * np.sqrt(12))


def test_scores_use_only_the_past(planted):
    trades, _, run = planted
    months = _months(trades)
    cut = months[80]
    shocked = trades.copy()
    late = shocked["exit_date"].dt.to_period("M") > cut
    shocked.loc[late, "pnl"] *= -5.0
    s2 = walk_forward_scores(monthly_rollup(shocked), months)
    a = run.scores[run.scores["eval_month"] <= cut].sort_values(["iteration_id", "eval_month"])
    b = s2[s2["eval_month"] <= cut].sort_values(["iteration_id", "eval_month"])
    assert np.allclose(a["wf_sharpe"].to_numpy(), b["wf_sharpe"].to_numpy(), equal_nan=True)


def test_silent_iterations_are_still_scored(planted):
    trades, _, _ = planted
    it = trades["iteration_id"].iloc[0]
    cut = trades["exit_date"].quantile(0.6)
    t = trades[(trades["iteration_id"] != it) | (trades["exit_date"] <= cut)]  # `it` goes silent
    s = walk_forward_scores(monthly_rollup(t), _months(trades))
    last = s[s["iteration_id"] == it]["eval_month"].max()
    assert last == s["eval_month"].max()  # no knowledge that it will never trade again


def test_min_trades_guard_blocks_sparse_lucky_sets():
    months = pd.period_range("2015-01", "2020-12", freq="M")
    rows = [("LUCKY", "S", "I", "Long", m, 1000.0, 1, 0.01) for m in months[-3:]]  # 3 winning trades
    monthly = pd.DataFrame(rows, columns=["iteration_id", "strategy", "instrument", "side", "month",
                                          "monthly_pnl", "monthly_trades", "monthly_return"])
    s = walk_forward_scores(monthly, months)
    assert s["wf_sharpe"].iloc[-1] > 1.0                       # passes the original filter
    assert walk_forward_select(s, 1.0, min_trades=0).shape[0] > 0
    assert walk_forward_select(s, 1.0, min_trades=20).empty   # blocked by the guard


# ------------------------------------------------------------------ step 3 / 4 / 5
def test_forward_trades_are_after_the_decision(planted):
    _, _, run = planted
    ft = run.ft
    assert (ft["entry_date"].dt.to_period("M") == ft["trade_month"]).all()
    decision = ft["trade_month"].apply(lambda p: p.start_time)
    assert (ft["entry_date"] >= decision).all()
    picks = run.picks
    assert (picks["trade_month"] == picks["eval_month"] + 1).all()


def test_train_set_is_purged(planted):
    trades, _, run = planted
    pick = run.picks.iloc[len(run.picks) // 2]
    start = pick["trade_month"].start_time
    leaky = train_set(pick, trades, purge=False)
    clean = train_set(pick, trades, purge=True)
    assert (clean["exit_date"] < start).all()
    assert len(clean) <= len(leaky)


def test_non_overlapping_per_combination():
    d = pd.Timestamp
    ft = pd.DataFrame({"strategy": "S", "instrument": "I", "side": "Long",
                       "entry_date": [d("2020-01-30"), d("2020-02-03"), d("2020-02-10")],
                       "exit_date": [d("2020-02-05"), d("2020-02-06"), d("2020-02-12")]})
    kept = non_overlapping(ft)
    assert kept["entry_date"].tolist() == [d("2020-01-30"), d("2020-02-10")]


def test_mtm_adds_up_to_trade_pnl(planted):
    _, prices, run = planted
    ft = run.ft.head(50)
    m = daily_mtm(ft, prices)
    got = m.groupby("trade_idx")["pnl"].sum()
    assert np.allclose(got.to_numpy(), ft.set_index("trade_idx").loc[got.index, "pnl"].to_numpy())


# ------------------------------------------------------------------ selection bias
def test_threshold_choice_is_nested(planted):
    _, _, run = planted
    st = threshold_study(run)
    rets = {t: returns_for_threshold(run, t)[0] for t in st["per_threshold"]}
    years = list(range(st["research_years"][0], st["research_years"][1] + 1))
    for y, chosen in st["nested"]["choices"].items():
        past = [x for x in years if x < y]
        best = max(rets, key=lambda t: (sharpe(rets[t][rets[t].index.year.isin(past)]), -t))
        assert chosen == best


def test_noise_selection_does_not_invent_an_edge(noise):
    _, _, run = noise
    d = selection_decay(run)
    assert d["mean_in_sample_sharpe"] > 1.0          # the picks always look good in-sample
    assert d["realised_next_month_sharpe"] < 0.2     # ...and do not deliver
    st = threshold_study(run)
    assert st["naive"]["deflated_sharpe_prob"] < 0.5
    assert st["nested"]["sharpe"] < 0.5


def test_planted_edge_survives_out_of_sample(planted):
    _, _, run = planted
    d = selection_decay(run)
    assert 0 < d["realised_next_month_sharpe"] < d["mean_in_sample_sharpe"]
    assert threshold_study(run)["nested"]["sharpe"] > 0.5


def test_forward_test_uses_selected_iterations_only(planted):
    trades, _, run = planted
    pairs = set(zip(run.picks["iteration_id"], run.picks["trade_month"]))
    assert all((i, m) in pairs for i, m in zip(run.ft["iteration_id"], run.ft["trade_month"]))
    assert len(forward_trades(run.picks.head(0), trades)) == 0
