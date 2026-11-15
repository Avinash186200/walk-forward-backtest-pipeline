# Walk-Forward Backtest Pipeline

[![CI](https://github.com/Avinash186200/walk-forward-backtest-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/walk-forward-backtest-pipeline/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A five-step pipeline that turns a large library of strategy backtests (many strategies x instruments x long/short x parameter grids) into a monthly, walk-forward selection that is traded and measured only on data it has not seen. It also asks the question that matters most in this kind of work: **how much of the result comes from choosing the best of many?**

I built the original as notebooks and scripts for books of FX, commodity and large-cap equity strategies. That data is proprietary, so this repo runs on a **synthetic backtest library with the same structure** (iterations, trades, daily prices), and every number below can be reproduced from a clone.

## At a glance

| | |
|---|---|
| **Input** | Trade-level backtest output per iteration (`iteration_id`, strategy, instrument, side, entry/exit dates and prices, P&L after costs) plus daily closes |
| **Method** | 1) monthly rollup, 2) 60-month recency-weighted walk-forward Sharpe, picking the best parameter set per combination each month, 3) forward-test trades of the picks, 4) monthly summary with one open trade per combination, 5) daily mark-to-market portfolio |
| **Output** | Step tables (CSV), forward-test portfolio returns, a threshold study (naive vs nested walk-forward vs hold-out), and a selection-decay report |
| **Leakage controls** | Scores use only months that have closed, every iteration is scored on one calendar, a minimum trade count, purged training sets, the threshold chosen by nested walk-forward, a final hold-out, a deflated Sharpe, and a noise test in CI |
| **Status** | Method complete and tested. The numbers below are synthetic. Real-data sweep figures were optimistic because the threshold was tuned on the forward test (see the audit), so I don't publish them |

## How it works

```
iteration trades ─► 1 monthly rollup ─► 2 walk-forward score ─► best set per combination ─► 3 next-month trades
 (backtest)          by exit month        60m weighted Sharpe     wf_sharpe >= T               (forward test)
                                                                                                   │
           threshold study ◄── 5 daily MTM portfolio ◄── 4 one-open-trade filter + summary ◄───────┘
   naive / nested / hold-out
```

- A **combination** is (strategy, instrument, side). An **iteration** is one combination with one parameter set.
- **Step 2** scores every iteration at every month-end with a recency-weighted Sharpe over the last 60 months. The weights are 20% on the last 3 months, 20% on months 4-12, 25% on year 2, and about 12% on each of years 3-5. For each combination it picks the best iteration, which trades the *next* month if its score is at least `T`.
- **Step 3** takes only trades *entered* in that next month. Those trades are the forward test.
- **Step 5** marks every forward-test trade to market each day on fixed capital (one notional slot per combination). Costs are booked on the entry and exit days.

## Results on synthetic data

`python scripts/run_pipeline.py`: 3 strategies x 5 instruments x 2 sides x 25 parameter sets = 750 iterations, 2013-2025, 2 bps per side. The forward test starts in 2018. 2018-2023 is the research period and 2024-2025 is the hold-out.

| Threshold T | Forward-test Sharpe, all years | Research years | Hold-out years |
|---|---|---|---|
| 0.5 | 1.94 | 1.48 | 3.34 |
| 1.0 | 1.59 | 1.25 | 2.54 |
| 1.5 | 0.74 | 0.58 | 1.22 |
| 2.0 | 0.15 | -0.14 | 0.94 |

| How T is chosen | Sharpe |
|---|---|
| **Naive**: best T on the whole forward test (what the original did) | 1.94 |
| **Nested walk-forward**: each year uses the T that was best on earlier years | 1.27 |
| **Hold-out**: T chosen on 2018-2023, scored once on 2024-2025 | 3.34 (T = 0.5) |

**Selection decay.** The picks score an average in-sample Sharpe of **1.35**. The same picks earn **0.36** the next month. That gap is the price of choosing the best of 25 parameter sets every month. The forward test pays it. A report built on the in-sample scores would not.

**The same run with no planted edge** (`--signal-strength 0`): the picks still look good in-sample (Sharpe 1.22) and then lose money the next month (-0.35). The naive best threshold shows a positive Sharpe (0.23), but its deflated Sharpe probability is only 0.15. Nested walk-forward ends up not trading at all (Sharpe 0.00). The hold-out shows +0.54 on a tiny return (+0.08% a year), which is what luck looks like on two years.

A two-year hold-out is noisy: here it happens to be the best stretch of the sample. The nested walk-forward figure (1.27) is the more honest summary of what the procedure would have earned.

> **About the real-data results.** On the equity book, the forward-test Sharpe rose steadily as the selection threshold went up, and a higher threshold was then recommended from that same sweep. A combined book took the better-performing holding horizon for each asset class. Both are choices made on the forward test, which is exactly the effect the threshold study above measures. The FX and commodity book also used a hand-curated list of combinations; how that list was chosen isn't documented, so I can't rule the same effect in or out. I don't publish those figures. With the nested threshold choice, a hold-out and costs on every trade, I would expect them to come down materially.

## Bias and leakage audit

| Issue | Effect | Fix |
|---|---|---|
| Selection threshold (`MIN_SHARPE`) swept on the same forward-test period that was reported, and a higher value recommended from that sweep. The same applied to the "best horizon per asset class" choice, and possibly to a curated combination list whose selection method isn't documented | The forward test became in-sample for these choices, so the headline Sharpe was optimistic | Threshold treated as a hyper-parameter: nested walk-forward by year, a final hold-out scored once, and a deflated Sharpe over the threshold grid |
| Best of N parameter sets picked every month. Its window Sharpe is the maximum of N noisy estimates | The in-sample score overstates what the pick earns (1.35 vs 0.36 here, 1.22 vs -0.35 on noise) | Only forward results are reported, and a selection-decay report shows the gap |
| Each iteration was scored only between its own first and *last* trade month | Existence look-ahead: an iteration that would never trade again could not be picked | Every iteration is scored on one global calendar, with silent months counted as zero (tested) |
| Silent months zero-filled, with no minimum trade count | A sparse set with a few recent winners passes a Sharpe of 1 (shown in a test with 3 trades) | `min_trades` in the window (default 20) |
| Training sets for the ML filter kept trades *entered* before the month, including ones still open | Label leakage: outcomes unknown at decision time | Purged: a trade must have exited before the month starts |
| The deflated Sharpe in the reports counted only the final configurations compared (7), and default reports used 1 | Multiple testing understated | The threshold grid is counted. Per-month best-of-N is handled by forward testing, not by an adjustment |
| Early headline figures were cost-only. Adding 1 pip per leg of slippage lowered the Sharpe noticeably | Overstated net performance | Costs on every trade (`--cost-bps`, per side) |
| The equity book's universe was built from current index members and additions | Survivorship bias | Not fixable on synthetic data. Stated as a limitation |

The details are in [docs/methodology.md](docs/methodology.md).

## Repo layout

```
src/wfbt/   steps.py (the five steps) · evaluation.py (threshold study, decay) · metrics.py · synthetic.py
scripts/    run_pipeline.py
tests/      weights vs original, rollup, vectorised = reference loop, future-shock leakage test, silent iterations,
            min-trades guard, forward-test timing, purged train set, overlap filter, MTM = trade P&L,
            nested threshold choice, noise test, planted-edge test
docs/       methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_pipeline.py                        # planted edge
python scripts/run_pipeline.py --signal-strength 0    # no edge: selection should not invent one
python scripts/run_pipeline.py --min-trades 0 --cost-bps 0
```

## Original research code

[`original/`](original/README.md) holds the code as I first wrote it for this work, sanitised and nothing else:
client, company and person names replaced, credentials and local paths removed, data files and notebook outputs
dropped. It is kept so the real work can be read alongside the maintained version. Some of it contains the issues
described above; comparing the two shows each fix.

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).
