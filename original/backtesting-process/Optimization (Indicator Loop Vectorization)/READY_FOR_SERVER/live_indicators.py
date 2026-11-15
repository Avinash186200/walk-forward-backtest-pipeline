"""
Vectorized replacements for the "live-replication" indicator pattern used
across the strategy notebooks: every rolling indicator is computed using the
true historical close/high/low for all past days, but for *today* (the last
row) it must use the 5-minutes-before-close proxy (close_m05/high_m05/low_m05
etc.), because live trading has to decide before the actual close prints.

The original notebooks replicate this by looping over every row and rebuilding
the full historical array from scratch just to recompute the indicator and
keep its last value - O(n^2) per (symbol, parameter) combination. Every
function here produces numerically identical output (validated against the
loop-based originals in validate_vectorization.py) with no Python-level loop,
turning that section from O(n^2) into O(n).

General idea per indicator family:
  - fixed-window rolling aggregates (mean, std, max/min): algebraic split into
    "rolling agg over the w-1 historical points" (shifted by 1) combined with
    today's proxy value.
  - distance/lookback indicators (ROC): already a one-line elementwise formula.
  - recursive/exponential indicators (Wilder RSI): compute the indicator once
    on the pure historical series (vectorized), then replay a single
    incremental Wilder-EMA update using today's proxy value.
"""
import numpy as np
import pandas as pd

# The notebooks were written against an older numpy where
# np.where(cond, "Long", 0) silently produced an object array (mixed
# str/int branches). numpy>=2.x's stricter type promotion (NEP 50) raises
# DTypePromotionError for that same call - an environment difference
# unrelated to the optimization, but every notebook here imports
# live_indicators, so patching it once here makes every notebook work
# regardless of which numpy version the server has, without changing any
# actual output (object-array np.where behaves identically either way).
_np_where = np.where


def _object_safe_where(cond, x=None, y=None):
    if x is None and y is None:
        return _np_where(cond)
    try:
        return _np_where(cond, x, y)
    except TypeError:
        cond_arr = np.asarray(cond)
        x_arr = np.full(cond_arr.shape, x, dtype=object) if np.isscalar(x) else np.asarray(x, dtype=object)
        y_arr = np.full(cond_arr.shape, y, dtype=object) if np.isscalar(y) else np.asarray(y, dtype=object)
        return _np_where(cond_arr, x_arr, y_arr)


np.where = _object_safe_where


def sma_with_today(hist_close: pd.Series, today_proxy: pd.Series, window: int, fillna: bool = False) -> pd.Series:
    """SMA(window) where the most recent point uses today_proxy instead of
    hist_close. fillna=True matches ta's _sma(min_periods=0): for the first
    window-1 rows, falls back to an expanding mean over whatever points are
    available instead of NaN."""
    hist_sum = hist_close.rolling(window=window - 1).sum().shift(1)
    full_ma = (hist_sum + today_proxy) / window
    if not fillna:
        return full_ma

    cumsum_excl = hist_close.cumsum().shift(1).fillna(0.0)
    n = pd.Series(np.arange(1, len(hist_close) + 1), index=hist_close.index)
    expanding_ma = (cumsum_excl + today_proxy) / n

    pos = np.arange(len(hist_close))
    is_ramp = pos < (window - 1)
    return pd.Series(np.where(is_ramp, expanding_ma, full_ma), index=hist_close.index)


def roc_with_today(hist_close: pd.Series, today_proxy: pd.Series, window: int) -> pd.Series:
    """Rate of change vs. `window` days ago, where today uses today_proxy."""
    base = hist_close.shift(window)
    return (today_proxy - base) / base * 100


def rolling_max_with_today(hist_high: pd.Series, today_proxy: pd.Series, window: int) -> pd.Series:
    hist_max = hist_high.rolling(window=window - 1).max().shift(1)
    return np.maximum(hist_max, today_proxy)


def rolling_min_with_today(hist_low: pd.Series, today_proxy: pd.Series, window: int) -> pd.Series:
    hist_min = hist_low.rolling(window=window - 1).min().shift(1)
    return np.minimum(hist_min, today_proxy)


def bollinger_with_today(hist_close: pd.Series, today_proxy: pd.Series, window: int, num_std: float, fillna_ma: bool = True):
    """Returns (MA, Upper, Lower). Matches `ta.trend.sma_indicator(..., fillna=True)`
    for MA plus `series.rolling(window).std()` (ddof=1, no fillna) for the bands."""
    MA = sma_with_today(hist_close, today_proxy, window, fillna=fillna_ma)

    hist_sumsq = (hist_close ** 2).rolling(window=window - 1).sum().shift(1)
    sumsq = hist_sumsq + today_proxy ** 2
    var = (sumsq - window * MA ** 2) / (window - 1)
    std = np.sqrt(var.clip(lower=0))

    Upper = MA + num_std * std
    Lower = MA - num_std * std
    return MA, Upper, Lower


def bollinger_class_with_today(hist_close: pd.Series, today_proxy: pd.Series, window: int, num_std: float, fillna: bool = False):
    """Matches the ta.volatility.BollingerBands *class* exactly (different
    from bollinger_with_today, which matches the manual
    sma_indicator+plain-.rolling().std() pattern used elsewhere): population
    std (ddof=0), and fillna=True applies the expanding-ramp to *both* mean
    and std (min_periods=0 for both), not just the mean."""
    w = window
    hist_sum = hist_close.rolling(window=w - 1).sum().shift(1)
    hist_sumsq = (hist_close ** 2).rolling(window=w - 1).sum().shift(1)
    full_ma = (hist_sum + today_proxy) / w
    full_var = (hist_sumsq + today_proxy ** 2) / w - full_ma ** 2
    full_std = np.sqrt(full_var.clip(lower=0))

    if fillna:
        cumsum_excl = hist_close.cumsum().shift(1).fillna(0.0)
        cumsumsq_excl = (hist_close ** 2).cumsum().shift(1).fillna(0.0)
        n = pd.Series(np.arange(1, len(hist_close) + 1), index=hist_close.index)
        ramp_ma = (cumsum_excl + today_proxy) / n
        ramp_var = (cumsumsq_excl + today_proxy ** 2) / n - ramp_ma ** 2
        ramp_std = np.sqrt(ramp_var.clip(lower=0))
        pos = np.arange(len(hist_close))
        is_ramp = pos < (w - 1)
        MA = pd.Series(np.where(is_ramp, ramp_ma, full_ma), index=hist_close.index)
        STD = pd.Series(np.where(is_ramp, ramp_std, full_std), index=hist_close.index)
    else:
        pos = np.arange(len(hist_close))
        MA = full_ma.mask(pos < (w - 1))
        STD = full_std.mask(pos < (w - 1))

    Upper = MA + num_std * STD
    Lower = MA - num_std * STD
    return MA, Upper, Lower


def atr_with_today(hist_high: pd.Series, hist_low: pd.Series, high_proxy: pd.Series, low_proxy: pd.Series, window: int) -> pd.Series:
    """Mean of (high-low) range over `window` days, today using the m05 proxy range."""
    rng_hist_sum = (hist_high - hist_low).rolling(window=window - 1).sum().shift(1)
    rng_today = high_proxy - low_proxy
    return (rng_hist_sum + rng_today) / window


def _wilder_rsi_raw_state(hist_close: pd.Series, window: int):
    """Unmasked (min_periods=0) Wilder avg gain/loss - the true recursive
    value at every position with no NaN blanking, used as internal state to
    splice in today's proxy-based step."""
    diff = hist_close.diff(1)
    up = diff.where(diff > 0, 0.0)
    down = -diff.where(diff < 0, 0.0)
    alpha = 1 / window
    avg_up = up.ewm(alpha=alpha, min_periods=0, adjust=False).mean()
    avg_dn = down.ewm(alpha=alpha, min_periods=0, adjust=False).mean()
    return avg_up, avg_dn


def rsi_with_today(hist_close: pd.Series, today_proxy: pd.Series, window: int) -> pd.Series:
    """Wilder RSI where today's diff is (today_proxy - yesterday's real close).
    Matches `ta.momentum.rsi(..., fillna=False)`: NaN until `window`
    observations have been seen."""
    avg_up_hist, avg_dn_hist = _wilder_rsi_raw_state(hist_close, window)
    alpha = 1 / window
    diff_today = today_proxy - hist_close.shift(1)
    up_today = diff_today.clip(lower=0.0)
    dn_today = (-diff_today).clip(lower=0.0)

    avg_up_adj = (1 - alpha) * avg_up_hist.shift(1).fillna(0.0) + alpha * up_today
    avg_dn_adj = (1 - alpha) * avg_dn_hist.shift(1).fillna(0.0) + alpha * dn_today

    rs = avg_up_adj / avg_dn_adj
    rsi = pd.Series(np.where(avg_dn_adj == 0, 100, 100 - (100 / (1 + rs))), index=hist_close.index)

    pos = np.arange(len(hist_close))
    rsi = rsi.mask(pos < (window - 1))
    return rsi


def rolling_sum_with_today(hist: pd.Series, today_proxy: pd.Series, window: int, fillna: bool = False) -> pd.Series:
    """fillna=True matches pandas .rolling(window, min_periods=0).sum(): for
    the first window-1 rows, sums whatever points are available instead of
    NaN (same ramp idea as sma_with_today, without the division)."""
    hist_sum = hist.rolling(window=window - 1).sum().shift(1)
    full_sum = hist_sum + today_proxy
    if not fillna:
        return full_sum
    cumsum_excl = hist.cumsum().shift(1).fillna(0.0)
    ramp_sum = cumsum_excl + today_proxy
    pos = np.arange(len(hist))
    is_ramp = pos < (window - 1)
    return pd.Series(np.where(is_ramp, ramp_sum, full_sum), index=hist.index)


def rolling_std_with_today(hist: pd.Series, today_proxy: pd.Series, window: int) -> pd.Series:
    """Matches plain `series.rolling(window).std()` (ddof=1, no fillna)."""
    hist_sum = hist.rolling(window=window - 1).sum().shift(1)
    hist_sumsq = (hist ** 2).rolling(window=window - 1).sum().shift(1)
    MA = (hist_sum + today_proxy) / window
    sumsq = hist_sumsq + today_proxy ** 2
    var = (sumsq - window * MA ** 2) / (window - 1)
    return np.sqrt(var.clip(lower=0))


def _ema_raw_state(hist: pd.Series, window: int):
    """Unmasked (min_periods=0) EMA - the true recursive value at every
    position, alpha=2/(window+1) like ta's _ema/ema_indicator."""
    alpha = 2 / (window + 1)
    return hist.ewm(alpha=alpha, min_periods=0, adjust=False).mean()


def ema_with_today(hist: pd.Series, today_proxy: pd.Series, window: int, fillna: bool = False) -> pd.Series:
    """EMA where today's point uses today_proxy. Matches ta.trend.ema_indicator:
    fillna=True simply means no min_periods masking (the recursion is the
    same either way - ta's _ema uses min_periods=0 internally when fillna=True,
    not the generic ffill/0 _check_fillna path)."""
    alpha = 2 / (window + 1)
    ema_hist_raw = _ema_raw_state(hist, window)
    ema_adj = (1 - alpha) * ema_hist_raw.shift(1) + alpha * today_proxy
    ema_adj.iloc[0] = today_proxy.iloc[0]  # ewm seeds on the very first observation

    if fillna:
        return ema_adj
    pos = np.arange(len(hist))
    return ema_adj.mask(pos < (window - 1))


def ema_with_today_from_hist_today_series(hist_pure_series: pd.Series, today_adj_series: pd.Series, window: int) -> pd.Series:
    """Like ema_with_today, but for composing EMA-of-an-already-adjusted-series
    (e.g. MACD's signal line = EMA(macd_line)): hist_pure_series is the fully
    historical version of that intermediate series (computed from real OHLC
    throughout, no proxy anywhere), today_adj_series is its proxy-adjusted
    value at every row (only the value *at* each row's own evaluation matters
    - this is what `ema_with_today` already does generically)."""
    return ema_with_today(hist_pure_series, today_adj_series, window, fillna=True)


def macd_with_today(hist_close: pd.Series, today_proxy: pd.Series, window_fast: int, window_slow: int, window_sign: int, fillna: bool = False):
    """Returns (macd_line, signal_line, macd_diff), matching ta.trend.MACD."""
    ema_fast_hist = _ema_raw_state(hist_close, window_fast)
    ema_slow_hist = _ema_raw_state(hist_close, window_slow)
    macd_hist_pure = ema_fast_hist - ema_slow_hist  # fully historical macd line, no proxy

    ema_fast_adj = ema_with_today(hist_close, today_proxy, window_fast, fillna=True)
    ema_slow_adj = ema_with_today(hist_close, today_proxy, window_slow, fillna=True)
    macd_adj = ema_fast_adj - ema_slow_adj

    pos = np.arange(len(hist_close))
    macd_boundary = max(window_fast, window_slow) - 1

    # ta's ema(macd_line, window_sign, fillna=False) sees a NaN-prefixed macd
    # input when fillna=False - pandas ewm re-seeds at the first valid value
    # rather than continuing a recursion that started at position 0, so the
    # signal's *numeric* values (not just masking) depend on this. Reproduce
    # that by feeding the same NaN-prefixed series in for that case.
    macd_hist_for_signal = macd_hist_pure if fillna else macd_hist_pure.mask(pos < macd_boundary)
    signal_adj = ema_with_today_from_hist_today_series(macd_hist_for_signal, macd_adj, window_sign)
    diff_adj = macd_adj - signal_adj

    if fillna:
        return macd_adj, signal_adj, diff_adj
    # ema(macd_line, window_sign, fillna=False) counts valid (non-NaN)
    # observations of the already NaN-prefixed macd line, so the signal only
    # becomes valid window_sign-1 rows after the macd line itself does.
    signal_boundary = macd_boundary + window_sign - 1
    macd_masked = macd_adj.mask(pos < macd_boundary)
    signal_masked = signal_adj.mask(pos < signal_boundary)
    diff_masked = diff_adj.mask(pos < signal_boundary)
    return macd_masked, signal_masked, diff_masked


def wma_with_today(hist: pd.Series, today_proxy: pd.Series, window: int, fillna: bool = False) -> pd.Series:
    """Linearly-weighted moving average (heaviest weight on the most recent
    point), matching ta.trend.wma_indicator/WMAIndicator. Closed-form via the
    recursion Q(t) = w*x[t] + Q_{w-1}(x)[t-1] (weighted sum, un-normalized),
    where Q_{w-1} is the same construct evaluated with window-1 on the pure
    historical series - itself derived from prefix sums (no python loop):
        Q_w(t) = (w - t)*P_w(t) + CumIX(t) - CumIX(t-w)
    with P_w = rolling sum over window w, CumIX = cumsum(i * x[i])."""

    def _Q(x: pd.Series, w: int) -> pd.Series:
        idx = pd.Series(np.arange(len(x)), index=x.index)
        cum_ix = (idx * x).cumsum()
        cum_ix_shift = cum_ix.shift(w).fillna(0.0)
        P = x.rolling(window=w).sum()
        return (w - idx) * P + cum_ix - cum_ix_shift

    norm = 2 / (window * (window + 1))
    R = _Q(hist, window - 1).shift(1)  # weighted sum over the w-1 historical points
    wma_adj = (window * today_proxy + R) * norm

    if fillna:
        return wma_adj.fillna(0.0)
    pos = np.arange(len(hist))
    return wma_adj.mask(pos < (window - 1))


def stochastic_k_with_today(hist_high: pd.Series, hist_low: pd.Series, hist_close: pd.Series,
                             high_proxy: pd.Series, low_proxy: pd.Series, close_proxy: pd.Series, window: int) -> pd.Series:
    """%K = (close - lowest_low) / (highest_high - lowest_low) * 100,
    matching ta.momentum.stoch / StochasticOscillator (smooth_window=1)."""
    lowest = rolling_min_with_today(hist_low, low_proxy, window)
    highest = rolling_max_with_today(hist_high, high_proxy, window)
    return (close_proxy - lowest) / (highest - lowest) * 100


def stochastic_k_pure_historical(hist_high: pd.Series, hist_low: pd.Series, hist_close: pd.Series, window: int) -> pd.Series:
    """%K computed using only real historical OHLC throughout (no proxy
    anywhere) - the building block for %D's "history" argument."""
    lowest = hist_low.rolling(window=window).min()
    highest = hist_high.rolling(window=window).max()
    return (hist_close - lowest) / (highest - lowest) * 100


def stochastic_d_with_today(k_pure_historical: pd.Series, k_adj_today: pd.Series, d_window: int) -> pd.Series:
    """%D = SMA(%K, d_window). k_pure_historical/k_adj_today as returned by
    stochastic_k_pure_historical / stochastic_k_with_today."""
    return sma_with_today(k_pure_historical, k_adj_today, d_window, fillna=False)


def wilder_smoothed_with_today(hist_value: pd.Series, today_value_proxy: pd.Series, window: int, seed_window: int = None):
    """General Wilder-style smoothing matching ta's hand-rolled recursive
    loops (AverageTrueRange, ADXIndicator's internal _trs/_dip/_din):
        seed = mean of the first `window` values
        state[i] = state[i-1] - state[i-1]/window + value[i]   (sum form)
    or the ATR mean form: state[i] = (state[i-1]*(window-1) + value[i]) / window

    Computes the *mean-form* (ATR-style) recursion's pure-historical pass
    once (single O(n) forward pass - this still replaces an O(n) loop that
    used to run on every one of n rows, so the O(n^2) is gone), then splices
    in one incremental step using today_value_proxy for the very last row.
    seed_window defaults to `window`.
    """
    seed_window = seed_window or window
    n = len(hist_value)
    vals = hist_value.to_numpy(dtype=float)
    state = np.full(n, np.nan)
    if n >= seed_window:
        state[seed_window - 1] = vals[:seed_window].mean()
        for i in range(seed_window, n):
            state[i] = (state[i - 1] * (window - 1) + vals[i]) / window
    state_hist = pd.Series(state, index=hist_value.index)

    today_vals = today_value_proxy.to_numpy(dtype=float)
    state_prev = state_hist.shift(1).to_numpy(dtype=float)

    adj = (state_prev * (window - 1) + today_vals) / window
    # at the exact seed row (pos == seed_window-1), the rule is the seed mean,
    # using historical values for all but the last point
    pos = np.arange(n)
    seed_row = pos == (seed_window - 1)
    if seed_row.any():
        hist_part_sum = vals[: seed_window - 1].sum() if seed_window >= 1 else 0.0
        adj[seed_row] = (hist_part_sum + today_vals[seed_row]) / seed_window
    adj[pos < (seed_window - 1)] = np.nan
    return pd.Series(adj, index=hist_value.index)


def mass_index_with_today(hist_high: pd.Series, hist_low: pd.Series, high_proxy: pd.Series, low_proxy: pd.Series,
                           window_fast: int, window_slow: int, fillna: bool = False) -> pd.Series:
    """Matches ta.trend.MassIndex / mass_index: EMA-of-EMA of the high-low
    range, then a rolling sum. Composed entirely from ema_with_today and
    rolling_sum_with_today. When fillna=False, each stage's NaN-prefix gets
    fed forward into the next (same re-seeding behavior as macd_with_today's
    signal line), since that changes the *values*, not just the masking."""
    pos = np.arange(len(hist_high))
    amp_hist = hist_high - hist_low
    amp_today = high_proxy - low_proxy

    ema1_hist_pure = _ema_raw_state(amp_hist, window_fast)
    ema1_adj = ema_with_today(amp_hist, amp_today, window_fast, fillna=True)

    ema1_boundary = window_fast - 1
    ema1_hist_for_ema2 = ema1_hist_pure if fillna else ema1_hist_pure.mask(pos < ema1_boundary)
    ema1_adj_for_ema2 = ema1_adj if fillna else ema1_adj.mask(pos < ema1_boundary)

    ema2_hist_pure = _ema_raw_state(ema1_hist_for_ema2, window_fast)
    ema2_adj = ema_with_today_from_hist_today_series(ema1_hist_for_ema2, ema1_adj_for_ema2, window_fast)

    mass_hist_pure = ema1_hist_pure / ema2_hist_pure
    mass_adj = ema1_adj / ema2_adj

    ema2_boundary = ema1_boundary + window_fast - 1
    mass_hist_for_sum = mass_hist_pure if fillna else mass_hist_pure.mask(pos < ema2_boundary)
    mass_adj_for_sum = mass_adj if fillna else mass_adj.mask(pos < ema2_boundary)

    result = rolling_sum_with_today(mass_hist_for_sum, mass_adj_for_sum, window_slow, fillna=fillna)
    if fillna:
        return result.fillna(0.0)
    sum_boundary = ema2_boundary + window_slow - 1
    return result.mask(pos < sum_boundary)


def wilder_sum_form_with_today(hist_value: pd.Series, today_value_proxy: pd.Series, window: int):
    """Wilder smoothing in ta's ADXIndicator-internal *sum* form (different
    from AverageTrueRange's mean form):
        seed = sum of the first `window` values
        state[i] = state[i-1] - state[i-1]/window + value[i]
    Pure-historical pass computed once via the exact recursion (single O(n)
    forward pass mirrors ta's own internal loop byte-for-byte), then one
    incremental splice step for today's row."""
    n = len(hist_value)
    vals = hist_value.to_numpy(dtype=float)
    state = np.full(n, np.nan)
    if n >= window:
        state[window - 1] = vals[:window].sum()
        for i in range(window, n):
            state[i] = state[i - 1] - state[i - 1] / window + vals[i]
    state_hist = pd.Series(state, index=hist_value.index)

    today_vals = today_value_proxy.to_numpy(dtype=float)
    state_prev = state_hist.shift(1).to_numpy(dtype=float)
    adj = state_prev - state_prev / window + today_vals

    pos = np.arange(n)
    seed_row = pos == (window - 1)
    if seed_row.any():
        hist_part_sum = vals[: window - 1].sum()
        adj[seed_row] = hist_part_sum + today_vals[seed_row]
    adj[pos < (window - 1)] = np.nan
    return pd.Series(adj, index=hist_value.index)


def exit_positions_for(open_positions: np.ndarray, flag: pd.Series, max_lookahead: int = 3, inclusive: bool = False):
    """Vectorized replacement for the per-trade O(n) scan
    `df[(df['index'] > open_idx) & (df['Exit_Signal'] == 1)]` (or, for a
    handful of strategies, `df['index'] >= open_idx` - pass inclusive=True
    to allow the entry bar itself to also satisfy the exit condition), run
    once per open trade in the original loop. One searchsorted call replaces
    all of those scans at once.

    Reproduces the original's three cases exactly:
      1. no Exit_Signal==1 row after open_idx at all -> "not reached", use
         the very last row of df (== current_price_data in the original).
      2. nearest Exit_Signal==1 row is within max_lookahead bars -> "reached",
         use that exact row.
      3. nearest Exit_Signal==1 row is >max_lookahead bars away -> forced
         exit at exactly open_idx+max_lookahead, *unless* that position is
         past the end of the data, in which case it also falls back to case 1.

    Returns (exit_pos, reached) arrays aligned with open_positions:
      - exit_pos: row position to read exit prices/date from
      - reached: True for case 2/3 ("RSI_exit_reached"), False for case 1
        ("RSI_exit_not_reached")
    """
    exit_candidates = np.flatnonzero(flag.values)
    n = len(flag)
    if len(exit_candidates) == 0:
        return np.full(len(open_positions), n - 1), np.zeros(len(open_positions), dtype=bool)
    search_from = open_positions if inclusive else open_positions + 1
    insert_at = np.searchsorted(exit_candidates, search_from)
    has_match = insert_at < len(exit_candidates)
    matched_pos = np.where(has_match, exit_candidates[np.clip(insert_at, 0, len(exit_candidates) - 1)], -1)

    near_enough = has_match & (matched_pos - open_positions <= max_lookahead)
    forced_in_bounds = has_match & ~near_enough & (open_positions + max_lookahead <= n - 1)

    reached = near_enough | forced_in_bounds
    exit_pos = np.where(
        near_enough,
        matched_pos,
        np.where(forced_in_bounds, open_positions + max_lookahead, n - 1),
    )
    return exit_pos, reached
