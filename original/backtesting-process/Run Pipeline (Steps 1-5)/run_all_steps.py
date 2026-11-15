"""
One-shot driver for the selection pipeline, Step 1 -> Step 5.

You give it two inputs (in pipeline_config.ini):
  * bt_results_dir  - the finished backtest ("BT results"): one folder per strategy,
                      each containing a "Trade Level Data" sub-folder of iteration parquets.
  * raw_data_dir    - the raw price data.

It then runs, in order and automatically:
  Step 1  Monthly Rollup          BT results            -> Monthly Rolling (monthly_stats_all.csv)
  Step 2  Walk-Forward Selection  Monthly Rolling       -> walk_forward_selection_*.csv
  Step 3  FT Trade Level          Monthly Rolling + BT  -> selected_trade_level_all_months.csv
  Step 4  Monthly Summary         Monthly Rolling       -> Monthly_Combination_Summary.xlsx
  Step 5  MTM Data                Monthly Rolling       -> <year> all selected comb mtm data Test.csv

Each step is the SAME optimized script that lives in its own folder in this repo; we run it
as an isolated subprocess (so its RAM is fully released before the next step starts) and pass
the user's paths in through environment variables, which the steps read with a fallback to
their own defaults. Nothing about a step's logic or output changes when driven from here.

Usage:
    python run_all_steps.py                       # uses pipeline_config.ini next to this file
    python run_all_steps.py --config other.ini
    python run_all_steps.py --steps 3,4,5         # run only some steps (resume / re-run)
    python run_all_steps.py --dry-run             # print the plan and validate paths, run nothing

Every step is resumable on its own (each skips work it already finished), so a re-run after an
interruption picks up where it left off.
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


# Each step: (number, label, folder name in repo, script file, builder for the argv tail).
# The path env vars are set for every step (harmless if a step ignores one); per-step worker
# levers are passed both as argv (for the step's main) and, for Step 1, as NW/TPW env vars.
STEP_FOLDERS = {
    1: "Monthly Rollup (Step 1)",
    2: "Walk-Forward Selection (Step 2)",
    3: "FT Trade Level Data (Step 3)",
    4: "Monthly Summary (Step 4)",
    5: "MTM Data (Step 5)",
}


def load_config(path: Path) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(path):
        raise FileNotFoundError("Config file not found: %s" % path)

    def p(section, key, fallback=None):
        return cp.get(section, key, fallback=fallback)

    def i(section, key, fallback):
        return cp.getint(section, key, fallback=fallback)

    cfg = {
        "bt_results_dir": p("paths", "bt_results_dir"),
        "raw_data_dir": p("paths", "raw_data_dir", ""),
        "monthly_rolling_dir": p("paths", "monthly_rolling_dir"),
        "summary_output_dir": p("paths", "summary_output_dir"),
        "mtm_output_dir": p("paths", "mtm_output_dir"),
        "steps": p("run", "steps", "1,2,3,4,5"),
        "stop_on_error": cp.getboolean("run", "stop_on_error", fallback=True),
        "step1_workers": i("workers", "step1_workers", 8),
        "step1_threads_per_worker": i("workers", "step1_threads_per_worker", 4),
        "step2_workers": i("workers", "step2_workers", 12),
        "step3_workers": i("workers", "step3_workers", 8),
        "step3_mem_budget_gb": i("workers", "step3_mem_budget_gb", 40),
        "step4_workers": i("workers", "step4_workers", 16),
        "step5_workers": i("workers", "step5_workers", 12),
    }
    for k in ("bt_results_dir", "monthly_rolling_dir", "summary_output_dir", "mtm_output_dir"):
        if not cfg[k]:
            raise ValueError("Missing required [paths] setting: %s" % k)
    return cfg


def build_steps(cfg: dict) -> dict:
    mr = cfg["monthly_rolling_dir"]
    return {
        1: (["monthly_rollup_parallel.py"]),
        2: (["walk_forward_selection_within.py", mr, str(cfg["step2_workers"])]),
        3: (["ft_trade_level_parallel.py", mr, str(cfg["step3_mem_budget_gb"]), str(cfg["step3_workers"])]),
        4: (["month_summary.py", mr, str(cfg["step4_workers"])]),
        5: (["step5_optimized.py", mr, cfg["mtm_output_dir"], str(cfg["step5_workers"])]),
    }


def step_env(cfg: dict) -> dict:
    env = os.environ.copy()
    env["PIPE_BT_RESULTS"] = cfg["bt_results_dir"]
    env["PIPE_MONTHLY_ROLLING"] = cfg["monthly_rolling_dir"]
    env["PIPE_SUMMARY_OUT"] = cfg["summary_output_dir"]
    env["PIPE_MTM_OUT"] = cfg["mtm_output_dir"]
    if cfg["raw_data_dir"]:
        env["PIPE_RAW_DATA"] = cfg["raw_data_dir"]
    env["NW"] = str(cfg["step1_workers"])              # Step 1 reads worker count from NW
    env["TPW"] = str(cfg["step1_threads_per_worker"])  # ... and threads-per-worker from TPW
    env["PYTHONUNBUFFERED"] = "1"                        # stream child output live
    return env


def validate_paths(cfg: dict, steps_to_run: list) -> None:
    problems = []
    # Step 1 is the only one that needs BT results to already exist; later steps consume
    # intermediates this run produces, so we don't hard-fail on those up front.
    if 1 in steps_to_run and not Path(cfg["bt_results_dir"]).is_dir():
        problems.append("bt_results_dir does not exist: %s" % cfg["bt_results_dir"])
    if 1 not in steps_to_run and not Path(cfg["monthly_rolling_dir"]).is_dir():
        problems.append("monthly_rolling_dir must already exist when Step 1 is skipped: %s"
                        % cfg["monthly_rolling_dir"])
    for folder in STEP_FOLDERS.values():
        if not (REPO_ROOT / folder).is_dir():
            problems.append("missing step folder in repo: %s" % folder)
    if problems:
        raise SystemExit("Config problems:\n  - " + "\n  - ".join(problems))
    # make sure output parents exist
    for k in ("monthly_rolling_dir", "summary_output_dir", "mtm_output_dir"):
        Path(cfg[k]).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Run the selection pipeline Step 1 -> Step 5.")
    ap.add_argument("--config", default=str(HERE / "pipeline_config.ini"))
    ap.add_argument("--steps", default=None, help="comma list, e.g. 3,4,5 (overrides config)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    steps_spec = args.steps if args.steps is not None else cfg["steps"]
    steps_to_run = [int(s.strip()) for s in steps_spec.split(",") if s.strip()]
    steps_to_run = [s for s in steps_to_run if s in STEP_FOLDERS]

    import subprocess

    cmds = build_steps(cfg)
    env = step_env(cfg)
    validate_paths(cfg, steps_to_run)

    print("=" * 78)
    print("PIPELINE Step 1 -> Step 5")
    print("  BT results       : %s" % cfg["bt_results_dir"])
    print("  Raw data         : %s" % (cfg["raw_data_dir"] or "(not set)"))
    print("  Monthly Rolling  : %s" % cfg["monthly_rolling_dir"])
    print("  Summary out      : %s" % cfg["summary_output_dir"])
    print("  MTM out          : %s" % cfg["mtm_output_dir"])
    print("  Steps to run     : %s" % steps_to_run)
    print("=" * 78)

    if args.dry_run:
        for s in steps_to_run:
            print("[Step %d] would run: %s  (cwd=%s)" % (s, " ".join(cmds[s]), STEP_FOLDERS[s]))
        print("\nDry run only - nothing executed.")
        return

    results = []
    t_all = time.time()
    for s in steps_to_run:
        cwd = REPO_ROOT / STEP_FOLDERS[s]
        cmd = [sys.executable] + cmds[s]
        print("\n" + "-" * 78)
        print(">>> STEP %d  (%s)" % (s, STEP_FOLDERS[s]))
        print(">>> %s" % " ".join(cmd))
        print("-" * 78, flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(cwd), env=env).returncode
        dt = time.time() - t0
        results.append((s, rc, dt))
        if rc != 0:
            print("\n*** STEP %d FAILED (exit %d) after %.1fs ***" % (s, rc, dt))
            if cfg["stop_on_error"]:
                print("stop_on_error=true -> aborting pipeline.")
                break
        else:
            print("\n>>> STEP %d done in %.1fs" % (s, dt))

    print("\n" + "=" * 78)
    print("PIPELINE SUMMARY (total %.1fs)" % (time.time() - t_all))
    for s, rc, dt in results:
        print("  Step %d: %-8s %8.1fs" % (s, "OK" if rc == 0 else "FAIL(%d)" % rc, dt))
    print("=" * 78)
    if any(rc != 0 for _, rc, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
