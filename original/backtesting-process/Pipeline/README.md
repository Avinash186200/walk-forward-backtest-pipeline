# Pipeline — Portfolio Construction & Reporting

`run_pipeline.py` replaces the old manual notebook chain (Vol calc → S6 → S7 →
S8 → S9 → S10 → S11 → S12 → ad hoc report) with one deterministic script,
driven entirely by `config.ini`. It runs 12 steps (A–K): builds the
volatility wall, runs the MTM engine per year, builds the ex-ante (1/vol)
targeted portfolio, splits trades into carry/delta/exit legs, builds the
equity curve with costs and slippage, and generates the full 16-file
QuantStats-based report suite.

## Running it

```
cd Pipeline
python run_pipeline.py                  # uses config.ini in this folder
python run_pipeline.py --config path/to/other_config.ini
```

Edit `config.ini` (paths, tag, years, costs, slippage, target vol scenario)
— nothing in `run_pipeline.py` itself needs to change for a new dataset.

## 2026-06-26 changes

### 1. Equity curve now correctly reflects cost + slippage (Steps G+H+I merged)

**The bug:** `equity_start_of_day` was compounded from gross (pre-cost)
returns in isolation (old Step G), then costs (Step H) and slippage
(Step I) were calculated afterward as separate columns in the same file —
they never fed back into the equity curve itself. The result: the
`equity_start_of_day` column silently understated cost/slippage drag
relative to the `PL With cost` / `PL_with_slippage` columns sitting right
next to it in the same row, and relative to the headline Total
Return/Sharpe figures the QuantStats report was built from.

**The fix:** Steps G, H, and I are now one function
(`assign_equity_costs_slippage`) that walks day-by-day in chronological
order. Each day's position sizing (`amount_invested_per_trade`, and
therefore `Quantity`, transaction cost, financing cost, slippage) is based
on the previous day's *actual* net-of-cost-and-slippage equity, and that
day's resulting `PL_with_slippage` compounds into the next day's starting
equity. `equity_start_of_day` now lands exactly where the cost+slippage
Total Return implies it should (verified: $1,000,000 compounded at the
reported cost+slippage return equals the column's final value, to the
dollar).

**What did *not* change:** the headline Total Return / CAGR / Sharpe /
Max Drawdown numbers in every report already generated are unaffected.
Cost and slippage are defined as a percentage of equity, and position size
is also sized as a percentage of equity, so the equity term cancels out of
the daily `PL_with_slippage` return series algebraically — only the
dollar-denominated columns (`equity_start_of_day`, `amount_invested_per_trade`,
`Quantity`, `Total Cost`, `financing amount`) were wrong before. No
previously-shared report needs to be reissued with different metrics; the
trade-level CSV's dollar columns do need to be regenerated, and `run_pipeline.py`
now does this correctly by default — no config change needed.

### 2. Cross-horizon FX/Comm combination (`build_cross_horizon_combo.py`)

The existing "Combined" MTM folders only combine FX and Commodities at the
*same* holding-period horizon (T+3 with T+3, T+15 with T+15).
`build_cross_horizon_combo.py` builds a combination across *different*
horizons (e.g. FX at T+3 + Commodities at T+15, picking whichever horizon
performs best per asset class), using the exact same reclassification rule
the existing same-horizon Combined files use — verified against every year
of both existing T+3 and T+15 Combined outputs before being applied here:

- **Commodities side:** kept in full, unfiltered.
- **FX side:** any combination whose `Currency` contains `XAU`, `XAG`, or
  `XBR` is dropped — those are metals/oil quoted in FX-style pairs that are
  already booked under Commodities, so they're excluded from the FX side to
  avoid double-counting the same underlying instrument in both legs of the
  portfolio.

Edit the `BASE`, `FX_DIR`, `COMM_DIR` constants at the top of the script to
point at your two source MTM folders, run it to produce the combined
MTM Data folder, then point a `run_pipeline.py` config at that output folder
as you would any other MTM Data source.

`generate_configs_FXComm_June2026.py` is the one-off script used to generate
the 7 `config_*.ini` files for the June 2026 FX/Comm run (FX, Comm, Combined
× T+3/T+15, plus the new cross-horizon FX-T3+Comm-T15 combo) — not part of
the reusable pipeline, just a record of how those configs were built.

### 3. New standard report: `08_{label}_Statistical_Tests.xlsx`

Every run now produces a 16th report file with statistical diagnostics
beyond the descriptive metrics in `01_*_Overall_Summary.xlsx`:

- **Return distribution** — skewness, excess kurtosis, Jarque-Bera
  normality test.
- **Serial correlation / stationarity** — Ljung-Box test (lag 10) on daily
  returns, Augmented Dickey-Fuller test.
- **Sharpe ratio significance** — t-stat/p-value (H0: true Sharpe = 0),
  Probabilistic Sharpe Ratio, and **Deflated Sharpe Ratio**. The DSR raises
  the bar for "is this Sharpe real" based on how many candidate
  configs/variants were compared before landing on this one — set
  `[statistics] n_trials` in `config.ini` to that count (e.g. `7` for the
  June 2026 FX/Comm run, since FX, Comm, and Combined × T+3/T+15 plus the
  cross-horizon combo were all compared before picking a winner). Leave at
  the default `1` if this config wasn't chosen by comparing against others
  — that disables the DSR adjustment (PSR vs 0 is still reported).
- **Tail risk** — historical VaR/CVaR at 95% and 99%.
- **Portfolio exposure** — daily gross exposure reconstructed from
  trade-level legs, checked against the scenario's actual enforced cap
  (`gross_cap` from `VOL_SCENARIOS` for `production_scenario` — *not*
  `max_gross_exposure`, which only bounds the earlier static allocation in
  Step B). Flags any day that exceeds the cap.

  This exists because an early ad hoc check (outside the pipeline) computed
  gross exposure by reading only the `CARRY_POSITION` leg per trade per
  day and seemed to show breaches above 180% — Step F splits a
  rebalanced position into a stale `CARRY_POSITION` row plus a separate
  `DELTA_TRADE` adjustment row, and the two must be **summed** to get the
  true net exposure; reading `CARRY_POSITION` alone overstates it whenever
  a position is being scaled down. The cap was never actually breached —
  confirmed against the engine's own `gross_exposure_final` series in
  `_intermediate/mtm_output/.../daily_broker_<scenario>.xlsx`. This report
  uses the corrected (summed) reconstruction so that mistake can't recur.
