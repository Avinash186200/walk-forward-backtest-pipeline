"""Synthetic backtest output: daily prices plus trade-level results for many parameter sets.

The real input to this pipeline is the finished backtest of a strategy library:
for each (strategy, instrument, side, parameter set) "iteration", a list of
trades. This module generates the same shape.

How the effect is planted: each instrument's daily return depends a little on
a hidden, persistent state `z` (AR(1)). A strategy/instrument/side combination
has a *skill* k, and each parameter set sees `k * z + noise`. It goes long when
that observation is high (short when low) and holds for h days. Skilful
combinations therefore have a small, persistent edge. Most parameter sets of a
skill-less combination are pure noise. With `signal_strength=0`, `z` has no
effect on prices and nothing has an edge, so the tests use it as the no-edge
control.

Trades of one iteration never overlap (a backtester holds one position per
iteration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NOTIONAL = 100_000.0
HOLDS = np.array([1, 2, 3, 5, 8])


@dataclass(frozen=True)
class SyntheticConfig:
    n_strategies: int = 3
    n_instruments: int = 5
    n_params: int = 25
    start: str = "2013-01-01"
    end: str = "2025-12-31"
    seed: int = 7
    signal_strength: float = 1.0
    daily_vol: float = 0.006
    edge_per_z: float = 0.0008      # daily drift per unit of z when signal_strength = 1
    cost_bps: float = 2.0           # per side


def make_backtest(cfg: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trades, prices).

    trades: iteration_id, strategy, instrument, side, param_id, entry_date, exit_date,
            entry_price, exit_price, direction, notional, gross_pnl, cost, pnl
    prices: date, instrument, close
    """
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, cfg.end)
    T = len(dates)
    instruments = [f"INST{i:02d}" for i in range(cfg.n_instruments)]
    strategies = [f"STRAT_{chr(65 + s)}" for s in range(cfg.n_strategies)]

    closes, zs = {}, {}
    for inst in instruments:
        z = np.empty(T)
        z[0] = rng.normal()
        eps = rng.normal(0, np.sqrt(1 - 0.9 ** 2), T)
        for t in range(1, T):
            z[t] = 0.9 * z[t - 1] + eps[t]
        ret = np.r_[0.0, cfg.signal_strength * cfg.edge_per_z * z[:-1]] + rng.normal(0, cfg.daily_vol, T)
        ret[0] = 0.0
        closes[inst] = 100 * np.exp(np.cumsum(ret))
        zs[inst] = z
    prices = pd.DataFrame([(d, inst, closes[inst][t]) for inst in instruments for t, d in enumerate(dates)],
                          columns=["date", "instrument", "close"])

    cost = cfg.cost_bps / 1e4
    rows = []
    for strat in strategies:
        for inst in instruments:
            for side, direction in (("Long", 1), ("Short", -1)):
                base_skill = rng.uniform(0, 1) if rng.random() < 0.5 else 0.0
                for p in range(cfg.n_params):
                    k = float(np.clip(base_skill + rng.normal(0, 0.25), 0, 0.95))
                    thr = rng.uniform(1.0, 2.0)
                    hold = int(rng.choice(HOLDS))
                    obs = k * zs[inst] + np.sqrt(1 - k ** 2) * rng.normal(size=T)
                    signal = obs > thr if direction == 1 else obs < -thr
                    px = closes[inst]
                    t = 0
                    it_id = f"{inst}_{strat}_P{p:03d}_{side}"
                    while t < T - hold:
                        if signal[t]:
                            e, x = t, t + hold
                            gross = direction * (px[x] / px[e] - 1.0) * NOTIONAL
                            c = 2 * cost * NOTIONAL
                            rows.append((it_id, strat, inst, side, p, dates[e], dates[x], px[e], px[x],
                                         direction, NOTIONAL, gross, c, gross - c))
                            t = x + 1
                        else:
                            t += 1
    trades = pd.DataFrame(rows, columns=[
        "iteration_id", "strategy", "instrument", "side", "param_id", "entry_date", "exit_date",
        "entry_price", "exit_price", "direction", "notional", "gross_pnl", "cost", "pnl"])
    return trades, prices
