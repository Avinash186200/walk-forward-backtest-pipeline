"""Return statistics for fixed-capital (non-compounded) daily return series."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

ANN = 252


def sharpe(r: pd.Series, periods: int = ANN) -> float:
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative (summed) return, in return units."""
    eq = r.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def summarize(r: pd.Series) -> dict:
    years = max(len(r) / ANN, 1e-9)
    return {
        "total_return": float(r.sum()),
        "ann_return": float(r.sum() / years),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(ANN)) if len(r) > 1 else 0.0,
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
    }


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float]) -> float:
    """Probability the true Sharpe beats the best of len(trial_sharpes) noise trials.

    Bailey and Lopez de Prado (2014). `trial_sharpes` are annualised.
    """
    T, n = len(r), len(trial_sharpes)
    sd = r.std(ddof=1)
    if T < 3 or sd == 0 or n < 1:
        return float("nan")
    sr = r.mean() / sd
    nd = NormalDist()
    if n > 1:
        v = float(np.var(np.asarray(trial_sharpes) / math.sqrt(ANN), ddof=1))
        g = 0.5772156649
        sr0 = math.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e)))
    else:
        sr0 = 0.0
    z = (r - r.mean()) / sd
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(nd.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
