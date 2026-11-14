# Methodology and bias audit

## Problem

A strategy library produces thousands of backtests: each strategy runs on each instrument, long and short, over a grid of parameters. Some of those backtests look excellent by chance. The question is which parameter set to trade next month, and how much it will really earn.

## Data contract

| Table | Columns |
|---|---|
| Trades (one row per backtest trade) | `iteration_id, strategy, instrument, side, param_id, entry_date, exit_date, entry_price, exit_price, direction, notional, gross_pnl, cost, pnl` |
| Prices | `date, instrument, close` |

`pnl` is after costs. The synthetic generator (`wfbt.synthetic`) builds both tables.

**How the effect is planted.** Each instrument has a hidden AR(1) state `z` that moves the next day's return a little. About half of the combinations get a base skill between 0 and 1, the rest get 0. Each parameter set adds noise to its combination's base skill (clipped to 0-0.95) to get its own k, sees `k * z + noise`, trades when that observation crosses its threshold, and holds for 1-8 days. Trades within one iteration never overlap. `signal_strength = 0` turns the effect off.

## The five steps

1. **Monthly rollup.** P&L per iteration, bucketed by exit month and divided by a fixed capital base.
2. **Walk-forward selection.** For each iteration and each month-end E, compute a recency-weighted mean and standard deviation of monthly returns over months E-59..E, and annualise: `sharpe = mean / std * sqrt(12)`. For each combination, take the iteration with the highest score (ties go to higher weighted P&L, then id). It trades month E+1 if `sharpe >= T` and it had at least `min_trades` trades in the window.
3. **Forward test.** Trades of the picked iteration whose *entry* is in month E+1. Every such trade starts after the decision.
4. **Monthly summary.** When the pick changes at a month boundary, two positions could be open in the same combination. A trade is kept only if it enters after the last kept trade has exited. The summary per (combination, month) gives trades, wins, win %, total and average P&L, and notional.
5. **Daily mark-to-market.** Each kept trade earns `direction * notional * (P_d - P_{d-1}) / P_entry` per day, with half the cost booked on entry and half on exit. The portfolio return is the daily P&L sum divided by fixed capital (one notional per combination). It is not compounded.

## What was wrong in the original, and the fixes

### 1. The threshold was tuned on the forward test

The pipeline's forward test is honest *for a fixed procedure*. The procedure itself was then tuned on that forward test. The threshold `T` was swept, the forward-test Sharpe rose with it, and the best `T` was recommended. A "best horizon per asset class" combination was also chosen by forward-test performance. A curated list of combinations was used too; how it was chosen isn't documented. Each choice turns part of the forward test into in-sample data.

**Fix.** `T` is a hyper-parameter and is chosen in a nested way:

- **Nested walk-forward.** For year y, use the `T` with the best forward-test Sharpe over the years before y.
- **Hold-out.** The last 2 years are untouched. `T` is chosen on the years before them and scored once.
- **Deflated Sharpe** for the naive choice, counting every threshold tried.

### 2. Best-of-N every month

Picking the top score of N noisy estimates guarantees an optimistic score. The selection-decay report shows the size of this: on synthetic data with a real edge, picks average an in-sample Sharpe of 1.35 against 0.36 the following month. With no edge, it is 1.22 against -0.35. The forward test already pays this price. The problem only arises if the in-sample `wf_sharpe` is used as a performance estimate, so it never is.

### 3. Existence look-ahead in scoring

The original expanded each iteration's months only from its first to its *last* trade month in the whole sample. At month E, an iteration whose trading had permanently stopped had no score, so it could not be picked. That uses knowledge of the future. Now every iteration is scored on one global calendar, and silent months count as zero.

### 4. Sparse sets pass the Sharpe filter

Silent months are zero, which shrinks the standard deviation. An iteration with three winning trades in the last three months scores above 1. **Fix:** at least `min_trades` trades (default 20) in the 60-month window. On the synthetic library the guard changes few picks, because the parameter sets trade often. The test shows the mechanism.

### 5. Label leakage in model training sets

The training file for each month kept trades *entered* before the month. Some of those trades were still open at the decision time, so their outcome was not yet known. **Fix:** a trade is used for training only if it *exited* before the month starts (`train_set(..., purge=True)`).

### 6. Multiple testing understated

The reports' deflated Sharpe counted only the handful of final configurations compared (and defaulted to 1). The threshold study counts its whole grid.

### 7. Costs

Early headline figures were cost-only. Adding realistic slippage lowered the Sharpe materially. Every synthetic trade pays `--cost-bps` per side.

### 8. Survivorship (not fixable here)

The equity book's universe came from current index members and recent additions. Names that dropped out were missing. Any result on that universe is biased upward. This needs a point-in-time membership list.

## Checks in CI

- The recency weights equal the original loop's weights.
- The rollup conserves trades and P&L.
- The vectorised walk-forward Sharpe equals a per-window reference loop.
- Shocking every trade that exits after month E leaves all scores up to E unchanged.
- An iteration that goes silent is still scored at the last month.
- A sparse 3-trade set passes `T = 1` without the guard and is blocked with it.
- Forward-test trades are entered in the trade month, after the decision. `trade_month = eval_month + 1`.
- Purged training sets contain only trades closed before the month.
- The overlap filter keeps one open trade per combination.
- Daily MTM sums back to each trade's P&L.
- The nested threshold for each year equals the argmax over earlier years only.
- **Noise test:** with no planted edge, the picks look good in-sample (> 1.0) and deliver less than 0.2 the next month. The naive deflated Sharpe probability is below 0.5, and the nested Sharpe is below 0.5.
- **Planted edge:** next-month Sharpe is positive but below in-sample, and the nested Sharpe is above 0.5.

## Dropped from the original

- **The strategy library**: about 190 notebooks, each a strategy with a vectorised indicator loop (long and short versions), plus a run-all script. This repo starts from their output.
- **The downstream portfolio engine**: volatility wall, ex-ante inverse-vol sizing, trade splitting into carry/delta legs, financing costs and report suites. It is replaced here by a fixed-capital daily MTM.
- **A broker-specific margin table and asset-class lists** used in step 5.
- **Parallel runners, memory-budget schedulers and worker benchmarks.** These were important on the full data (hundreds of strategies, millions of trade files), but they don't change the method.
- **Supporting notebooks**:
  - a yearly backtest/forward-test selection (S1-S5) that the monthly walk-forward replaced. Its BT-vs-FT comparison survives as the selection-decay report.
  - parameter sweeps for single strategies
  - trailing stop-loss studies
  - a price-momentum study
  - a model-prediction notebook
  - a data-download notebook
  - an implied-volatility calculation from options files
  - an index hedge-level calculator

  The last two are unrelated to this pipeline.
