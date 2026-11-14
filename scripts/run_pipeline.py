"""Run the five-step pipeline on a synthetic backtest and judge the selection threshold honestly.

    python scripts/run_pipeline.py                        # planted edge
    python scripts/run_pipeline.py --signal-strength 0    # no edge anywhere: selection should not invent one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfbt.evaluation import THRESHOLDS, run_pipeline, selection_decay, step_outputs, threshold_study  # noqa: E402
from wfbt.synthetic import SyntheticConfig, make_backtest  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--signal-strength", type=float, default=1.0)
    ap.add_argument("--cost-bps", type=float, default=2.0, help="cost per side, basis points")
    ap.add_argument("--window", type=int, default=60, help="walk-forward window, months")
    ap.add_argument("--min-trades", type=int, default=20, help="minimum trades in the window to be selectable")
    ap.add_argument("--holdout-years", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)

    cfg = SyntheticConfig(seed=a.seed, signal_strength=a.signal_strength, cost_bps=a.cost_bps)
    trades, prices = make_backtest(cfg)
    run = run_pipeline(trades, prices, window=a.window, min_trades=a.min_trades)
    study = threshold_study(run, THRESHOLDS, a.holdout_years)
    decay = selection_decay(run, 1.0)
    run0 = run_pipeline(trades, prices, window=a.window, min_trades=0)
    decay0 = selection_decay(run0, 1.0)

    print(f"backtest: {trades['iteration_id'].nunique()} iterations, {len(trades):,} trades, "
          f"{trades.groupby(['strategy', 'instrument', 'side']).ngroups} combinations "
          f"(seed {a.seed}, signal_strength {a.signal_strength})")
    print(f"step 2: {len(run.picks):,} combination-months with a pick; step 3: {len(run.ft):,} forward-test trades")
    print(f"\nresearch years {study['research_years']}, hold-out years {study['holdout_years']}")
    print("threshold   FT Sharpe (all years)   research years   hold-out years")
    for t, v in study["per_threshold"].items():
        print(f"  {t:>4}      {v['ft_sharpe_all_years']:+6.2f}                  {v['research_sharpe']:+6.2f}"
              f"           {v['holdout_sharpe']:+6.2f}")
    n = study["naive"]
    print(f"\nnaive: best threshold on the whole forward test = {n['threshold']}  Sharpe {n['sharpe']:+.2f}  "
          f"(deflated Sharpe p = {n['deflated_sharpe_prob']:.2f})")
    print(f"nested walk-forward threshold choice {study['nested']['choices']}  Sharpe {study['nested']['sharpe']:+.2f}")
    h = study["holdout"]
    print(f"hold-out, threshold chosen on research years = {h['threshold']}  Sharpe {h['sharpe']:+.2f}  "
          f"ann. return {h['ann_return']:+.2%}  max DD {h['max_drawdown']:.2%}")
    print(f"\nselection decay (threshold 1.0): picks' in-sample Sharpe {decay['mean_in_sample_sharpe']:.2f} -> "
          f"next-month Sharpe {decay['realised_next_month_sharpe']:+.2f}  ({decay['n_picks']} picks)")
    print(f"  without the min-trades guard: {decay0['mean_in_sample_sharpe']:.2f} -> "
          f"{decay0['realised_next_month_sharpe']:+.2f}  ({decay0['n_picks']} picks)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in step_outputs(run, 1.0).items():
        df.to_csv(out / f"{name}.csv", index=False)
    summary = {"config": vars(a), "threshold_study": study, "selection_decay": decay,
               "selection_decay_no_min_trades": decay0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
