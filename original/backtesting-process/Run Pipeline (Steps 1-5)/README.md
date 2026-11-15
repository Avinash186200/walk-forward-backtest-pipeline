# Run Pipeline — Steps 1 → 5 (one command)

Runs the whole selection pipeline end-to-end. You point it at the **finished backtest**
and the **raw price data**; it produces everything from the monthly roll-up through the
MTM dataset automatically.

```bash
cd "Run Pipeline (Steps 1-5)"
# edit pipeline_config.ini (just the two input paths), then:
python run_all_steps.py
```

## What it runs

| Step | Script (in its own folder) | Input | Output |
|------|----------------------------|-------|--------|
| **1 Monthly Rollup** | `monthly_rollup_parallel.py` | **BT results** | `Monthly Rolling/<strategy>/monthly_stats_all.csv` |
| **2 Walk-Forward Selection** | `walk_forward_selection_within.py` | Monthly Rolling | `walk_forward_selection_*.csv` |
| **3 FT Trade Level** | `ft_trade_level_parallel.py` | Monthly Rolling + **BT results** | `selected_trade_level_all_months.csv` |
| **4 Monthly Summary** | `month_summary.py` | Monthly Rolling | `Monthly_Combination_Summary.xlsx` |
| **5 MTM Data** | `step5_optimized.py` | Monthly Rolling | `<year> all selected comb mtm data Test.csv` |

Everything after Step 1 flows through the shared **Monthly Rolling** folder, so you only
ever supply the two inputs — the intermediates are created and chained for you.

## The two inputs (edit `pipeline_config.ini`)

```ini
[paths]
bt_results_dir = ...\BT results     # finished backtest: one folder per strategy,
                                    # each with a "Trade Level Data" sub-folder of parquets
raw_data_dir   = ...\Raw Data       # raw price data
```

The intermediate/output folders default to the standard project layout and are created
automatically; change them only if you want the results elsewhere.

## Levers you might change

All optional — sensible defaults are set.

```ini
[run]
steps         = 1,2,3,4,5   # run a subset to resume, e.g. 3,4,5
stop_on_error = true        # abort if a step fails

[workers]
step1_workers = 8 ;  step1_threads_per_worker = 4
step2_workers = 12
step3_workers = 8 ;  step3_mem_budget_gb = 40   # Step 3 is the RAM-heavy one
step4_workers = 16
step5_workers = 12
```

> **Memory note:** Step 3 caches each strategy's trade parquets per worker. On a 128 GB box,
> **8 workers / 40 GB budget** was stable; higher concurrency was measured to exhaust RAM on
> the big folders. Lower these on a smaller machine.

## How it works (and why it's safe)

- **Each step runs as an isolated subprocess.** A step's memory is fully released before the
  next begins — which matters because Step 3 is memory-hungry. It also means one step crashing
  can't corrupt another's process state.
- **Paths are injected via environment variables** (`PIPE_BT_RESULTS`, `PIPE_MONTHLY_ROLLING`,
  `PIPE_SUMMARY_OUT`, `PIPE_MTM_OUT`). Each step reads these with a **fallback to its own
  hardcoded default**, so every step still runs standalone exactly as before. Environment
  variables (unlike runtime-set globals) are inherited by the `multiprocessing` workers each
  step spawns on Windows — the reason this approach is used instead of patching configs.
- **The step scripts are unchanged in logic** — this driver only chooses paths and worker
  counts. Output is identical to running each step by hand (verified: Step 5 driven through
  the pipeline is byte-for-byte identical to running `step5_optimized.py` directly).
- **Fully resumable.** Every step skips work it already finished, so re-running after an
  interruption (or with `--steps 3,4,5`) picks up where it left off.

## CLI

```bash
python run_all_steps.py                 # use pipeline_config.ini
python run_all_steps.py --config x.ini   # a different config
python run_all_steps.py --steps 3,4,5    # only some steps
python run_all_steps.py --dry-run        # print the plan + validate paths, run nothing
```

At the end it prints a per-step OK/FAIL + timing summary; exit code is non-zero if any step
failed.

## Rough timings (94-strategy Russell-1000 set, 128 GB box)

Step 1 ~tens of min · Step 2 ~1–2 h · Step 3 ~3 h (the bottleneck, I/O-bound) ·
Step 4 ~10 min · Step 5 ~1.5 min. Plan for a **half-day** end-to-end run; Steps 4–5 are quick.

## Scope

This driver covers **selection Steps 1→5** (BT results → MTM dataset). The separate
`Pipeline/` folder is the **downstream** chain (MTM dataset + raw data → vol-targeted
portfolio + quant-stat report) and is run after this one.
