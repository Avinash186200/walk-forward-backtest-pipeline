# Threshold sweep — does raising MIN_SHARPE lift the Russell Sharpe?

**Yes — strongly and monotonically.** Rebuilt the full vol-targeted portfolio at each cut-off
(filter trades by their selection `wf_sharpe >= T`, re-run Step 5 + the vol-targeted portfolio).
Sharpe is the pre-cost vol-targeted portfolio Sharpe (T=1.0 = 1.74 reproduces the baseline; cost-only
on the full run was 1.69, net-of-slippage 1.31).

| MIN_SHARPE (T) | Trades | Combos/yr | **Sharpe** | CAGR | Vol |
|---|---|---|---|---|---|
| 1.0 (current) | 2,653,066 | 183,289 | **1.74** | 4.61% | 2.60% |
| 1.25 | 1,447,555 | 107,415 | **1.97** | 5.63% | 2.80% |
| 1.50 | 692,586 | 54,186 | **2.13** | 6.54% | 2.99% |
| **2.0** | 106,578 | 9,256 | **2.44** | 8.31% | 3.29% |

## Takeaways
- **At MIN_SHARPE = 2.0 the Russell book reaches Sharpe 2.44** — essentially your usual ~2.5, on the
  same Russell data, with NO ML and NO code change. Just a more selective threshold.
- **It's a win-win:** Sharpe AND CAGR both rise (4.6% → 8.3%) as the threshold rises. The marginal
  combos `MIN_SHARPE=1.0` lets in are not just low-Sharpe, they're low-return too — they dilute on
  both axes. (Vol rises slightly because the book is less diversified, but return rises faster.)
- This matches the per-trade predictiveness result: `wf_sharpe` 2.0–3.0 trades have ~2.5× the per-trade
  Sharpe of the 1.0–1.25 bucket, and that weak bottom bucket is 45% of all trades.
- It also matches the FX+Comm book (138 curated combos → Sharpe 3.5): **concentration is the lever.**

## Recommendation
Set `MIN_SHARPE` in `Walk-Forward Selection (Step 2)/walk_forward_selection.py` to ~**1.75–2.0**
(start at 2.0 → Sharpe ~2.44; 1.75 is a middle ground with more breadth). Then the ML layer on top of
that concentrated, high-quality book should push Sharpe further still.

Trade-off to weigh: T=2.0 keeps only ~9,256 combos/yr (vs 183k). That's far fewer positions — confirm
it still meets your capacity/diversification needs. T=1.5 (Sharpe 2.13, 54k combos) is a more
diversified compromise.
