"""
benchmark_workers.py

Finds the best worker count (NW) for THIS machine, for future monthly-rollup runs.

Methodology (important — an earlier version got this wrong):
- One strategy is processed by exactly one worker, so a test set must contain MANY MORE
  strategies than the largest NW being tested, otherwise the extra workers sit idle and
  "NW=24" secretly runs as "NW=7". We therefore build ONE shared pool of the smallest
  available strategies (so the strategy COUNT is high for a modest file budget).
- The SAME pool is run at every NW, so differences reflect worker count, not strategy mix.
- A warm-up pass reads the whole pool once before timing, so the 128 GB OS file cache is
  equally warm for every NW (no "later run looks faster because files are cached" bias).
  Note: this measures CPU/parallelism scaling with the disk mostly cached; a fully
  cold-disk run will be lower in absolute terms but the relative ranking holds.

Run AFTER the real rollup has finished:  python benchmark_workers.py
Writes worker_benchmark_results.txt into OUTPUT_ROOT_DIR.
"""
from __future__ import annotations

import os
import time
import shutil
import importlib.util
from pathlib import Path
from multiprocessing import Pool

# Path to the single-process optimized module (lives next to this file).
SCRIPT = str(Path(__file__).with_name("monthly_rollup.py"))
TMP_OUT = None
RESULTS_FILE = None

NW_SETTINGS = [8, 16, 24]
TPW = 2
POOL_FILE_BUDGET = 300_000                   # ~total files in the shared test pool (keeps it quick)
MIN_POOL_STRATEGIES = max(NW_SETTINGS) * 2   # enough strategies to keep every worker busy

_R = None


def _load():
    spec = importlib.util.spec_from_file_location("rollup_opt", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _init(tpw):
    global _R
    import pyarrow as pa
    pa.set_cpu_count(tpw)
    try:
        pa.set_io_thread_count(tpw)
    except Exception:
        pass
    _R = _load()


def _process(args):
    name, out_root = args
    R = _R
    trade_dir = Path(R.STRATEGY_ROOT_DIR) / name / R.TRADE_SUBDIR_NAME
    bt_files = sorted(trade_dir.glob(R.BT_FILE_PATTERN))
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "iteration_summary_all.csv"
    monthly_path = out_dir / "monthly_stats_all.csv"
    for p in (meta_path, monthly_path):
        if p.exists():
            p.unlink()
    meta_buf, monthly_buf = [], []
    total = len(bt_files)
    count = 0
    BATCH = R.BATCH_SIZE
    for bs in range(0, total, BATCH):
        batch = bt_files[bs: bs + BATCH]
        try:
            meta_list, monthly = R.process_batch_fast(batch)
            if meta_list:
                meta_buf.extend(meta_list)
            if monthly is not None and not monthly.empty:
                monthly_buf.append(monthly)
        except Exception:
            for p in batch:
                res = R.process_single_file(p)
                if res is not None:
                    meta, monthly = res
                    meta_buf.append(meta)
                    monthly_buf.append(monthly)
        count += len(batch)
        if count % R.FLUSH_EVERY < BATCH or count == total:
            R.flush_buffers(meta_buf, monthly_buf, meta_path, monthly_path)
    R.flush_buffers(meta_buf, monthly_buf, meta_path, monthly_path)
    return total


def build_pool(mod):
    """Pick the SMALLEST strategies (so we get many of them within the file budget)."""
    strat = []
    for child in sorted(Path(mod.STRATEGY_ROOT_DIR).iterdir()):
        if not child.is_dir():
            continue
        td = child / mod.TRADE_SUBDIR_NAME
        if not td.is_dir():
            continue
        n = sum(1 for _ in td.glob(mod.BT_FILE_PATTERN))
        if n:
            strat.append((child.name, n))
    strat.sort(key=lambda x: x[1])  # smallest first
    pool, tot = [], 0
    for name, n in strat:
        pool.append(name); tot += n
        if len(pool) >= MIN_POOL_STRATEGIES and tot >= POOL_FILE_BUDGET:
            break
    return pool, tot


def run_pool(nw, pool):
    if TMP_OUT.exists():
        shutil.rmtree(TMP_OUT)
    TMP_OUT.mkdir(parents=True)
    t0 = time.time()
    with Pool(nw, initializer=_init, initargs=(TPW,)) as pool_proc:
        counts = pool_proc.map(_process, [(s, str(TMP_OUT)) for s in pool])
    return sum(counts), time.time() - t0


if __name__ == "__main__":
    mod = _load()
    TMP_OUT = Path(mod.OUTPUT_ROOT_DIR) / "_bench_tmp"
    RESULTS_FILE = Path(mod.OUTPUT_ROOT_DIR) / "worker_benchmark_results.txt"

    pool, tot = build_pool(mod)
    print(f"Shared test pool: {len(pool)} strategies, {tot:,} files. Warming cache...", flush=True)
    run_pool(max(NW_SETTINGS), pool)  # warm-up (discard timing)

    lines = [f"=== Worker-count benchmark (TPW={TPW}, {len(pool)} strategies, {tot:,} files, warm cache) ==="]
    best_nw, best_rate = None, -1.0
    for nw in NW_SETTINGS:
        files, dt = run_pool(nw, pool)
        rate = files / max(dt, 1e-9)
        line = f"NW={nw:>2} TPW={TPW}: {files:>8,} files in {dt:6.1f}s -> {rate:5.0f} files/s"
        print(line, flush=True)
        lines.append(line)
        if rate > best_rate:
            best_rate, best_nw = rate, nw

    if TMP_OUT.exists():
        shutil.rmtree(TMP_OUT)
    lines.append("")
    lines.append(f"Best: NW={best_nw} (~{best_rate:.0f} files/s). Use this for future runs.")
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + lines[-1])
    print("Saved:", RESULTS_FILE, flush=True)
