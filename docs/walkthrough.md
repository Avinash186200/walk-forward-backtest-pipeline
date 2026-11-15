# Walkthrough: a first run

This is a guided first run. It assumes you know Python but not trading. Read the "Key concepts" list in the
[README](../README.md#key-concepts) first if a term is new. All numbers below come from running the code on the
synthetic backtest library.

## 1. Set up and run the tests

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

You should see `14 passed`. Read `tests/test_pipeline.py` from the top. Two tests are worth reading first:

- The future-shock test changes every trade that exits after month E and checks that no score up to E changes.
  That is the whole walk-forward promise in one check.
- The min-trades test builds a parameter set with only 3 winning trades and shows that it passes a Sharpe
  filter of 1.0 without the guard. Zero-filled quiet months make its volatility look tiny.

## 2. Run with the planted edge

```bash
python scripts/run_pipeline.py
```

The first lines describe the input:

```
backtest: 750 iterations, 133,657 trades, 30 combinations (seed 7, signal_strength 1.0)
step 2: 2,910 combination-months with a pick; step 3: 2,917 forward-test trades
```

750 iterations are 30 combinations x 25 parameter sets. Step 2 makes one pick per combination per month (before any
threshold). Step 3 keeps only the trades those picks enter in the following month.

Then the threshold table. Each row is one value of T, the minimum score a pick needs to be traded:

```
threshold   FT Sharpe (all years)   research years   hold-out years
   0.5       +1.94                   +1.48            +3.34
   1.0       +1.59                   +1.25            +2.54
   1.5       +0.74                   +0.58            +1.22
   2.0       +0.15                   -0.14            +0.94
```

Then three ways of choosing T:

- **naive** picks the best row of the first column: T = 0.5, Sharpe 1.94. This is what the original did. It uses
  the forward test to choose, so it is not a forward result any more.
- **nested walk-forward** chooses T for each year from the years before it: 1.0 for 2019-2020, 0.5 after.
  Sharpe 1.27. This is the honest summary.
- **hold-out** chooses T on 2018-2023 and scores 2024-2025 once: Sharpe 3.34. Two years is a short, noisy sample,
  and here it happens to be the best stretch.

The last block is selection decay: the picks score an average in-sample Sharpe of 1.35, and the same picks earn
0.36 the next month.

## 3. Run with no edge

```bash
python scripts/run_pipeline.py --signal-strength 0
```

Now nothing has an edge. Look at what still looks good:

- Selection decay: in-sample 1.22, next month -0.35. The picks look skilful when they are picked and lose money
  afterwards.
- The naive threshold is 1.5, with a Sharpe of 0.23. Its deflated Sharpe probability is 0.15, so it does not pass.
- The nested walk-forward chooses T = 2.0 every year. At 2.0 nothing clears the bar, so it never trades:
  Sharpe 0.00. Doing nothing is the right answer on noise.

## 4. Look at the files

The script writes `results/`:

- `step1_monthly_rollup.csv`: monthly return and trade count per iteration, by exit month.
- `step2_selection.csv`: the picks with a score of at least 1.0, with `eval_month` (when it was scored) and
  `trade_month` (always the month after).
- `step3_forward_trades.csv`: the picks' trades in their trade month. Every `entry_date` is in `trade_month`.
- `step4_monthly_summary.csv`: per (combination, month), after the one-open-trade filter.
- `summary.json`: the threshold study and selection decay.

## 5. Try changing one thing

```bash
python scripts/run_pipeline.py --min-trades 0 --cost-bps 0
```

With no guard and no costs, the naive Sharpe rises from 1.94 to 2.46 and the nested one from 1.27 to 1.93. Costs
are most of the difference. Other flags to try are `--window` (months of history per score), `--holdout-years`
and `--seed`. Add `--out somewhere_else` to keep `results/` as it is.

## 6. Where to go next

- [methodology.md](methodology.md) has the full method and the bias audit.
- `src/wfbt/steps.py` holds the five steps in about 170 lines.
- `threshold_study` in `src/wfbt/evaluation.py` shows how the nested and hold-out choices are made.
