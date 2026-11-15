"""
monthly_rollup_parallel.py

Runs the OPTIMIZED rollup across MANY strategies at once, one strategy per worker
process. Each strategy is fully independent (own input folder, own output folder),
so this parallelises cleanly with no shared state between workers.

Reuses the exact, already-verified functions from
    "S - 1 monthly_rollup_only_optimized.py"
(process_batch_fast / process_single_file / flush_buffers), so per-strategy output
is byte-for-byte identical to the single-process run - this only changes HOW MANY
strategies run concurrently.

Thread oversubscription guard: each worker caps pyarrow's CPU thread pool to
THREADS_PER_WORKER, so N_WORKERS * THREADS_PER_WORKER stays at/below the physical
core count instead of every worker spawning a full-width thread pool.

Config via env vars:  NW = number of worker processes, TPW = threads per worker.
"""
from __future__ import annotations

import os
import sys
import time
import importlib.util
from pathlib import Path
from multiprocessing import Pool, Value
import threading

# Path to the single-process optimized module (lives next to this file).
SCRIPT = str(Path(__file__).with_name("monthly_rollup.py"))

# Worker config. Override per-run with env vars, e.g.  NW=16 TPW=2 python monthly_rollup_parallel.py
# Defaults below are the production-proven config (~690 files/s on a 24-core/SATA-SSD box).
# Guidance: keep NW <= physical cores, and NW * TPW <= logical cores (avoids thread oversubscription).
# Run benchmark_workers.py to find the best NW for your hardware.
N_WORKERS          = int(os.environ.get("NW", "8"))
THREADS_PER_WORKER = int(os.environ.get("TPW", "4"))

_counter = None   # multiprocessing.Value, shared progress (files processed)
R = None          # the loaded optimized module


def _load_module():
    spec = importlib.util.spec_from_file_location("rollup_opt", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _init(counter):
    """Runs once per worker process (Windows spawn)."""
    global _counter, R
    _counter = counter
    import pyarrow as pa
    pa.set_cpu_count(THREADS_PER_WORKER)
    try:
        pa.set_io_thread_count(THREADS_PER_WORKER)
    except Exception:
        pass
    R = _load_module()


def process_strategy(name):
    """Process ONE whole strategy. Mirrors optimized.run_rollup but bumps the
    shared counter per batch so the parent can report aggregate throughput."""
    trade_dir = Path(R.STRATEGY_ROOT_DIR) / name / R.TRADE_SUBDIR_NAME
    bt_files = sorted(trade_dir.glob(R.BT_FILE_PATTERN))
    if R.MAX_FILES_PER_STRATEGY is not None:
        bt_files = bt_files[:R.MAX_FILES_PER_STRATEGY]

    out_dir = Path(R.OUTPUT_ROOT_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path    = out_dir / "iteration_summary_all.csv"
    monthly_path = out_dir / "monthly_stats_all.csv"
    for p in (meta_path, monthly_path):
        if p.exists():
            p.unlink()

    meta_buf, monthly_buf = [], []
    total = len(bt_files)
    count = 0
    BATCH = R.BATCH_SIZE
    t0 = time.time()

    for bs in range(0, total, BATCH):
        batch = bt_files[bs: bs + BATCH]
        try:
            meta_list, monthly = R.process_batch_fast(batch)
            if meta_list:
                meta_buf.extend(meta_list)
            if monthly is not None and not monthly.empty:
                monthly_buf.append(monthly)
        except Exception as e:
            for p in batch:
                res = R.process_single_file(p)
                if res is not None:
                    meta, monthly = res
                    meta_buf.append(meta)
                    monthly_buf.append(monthly)
        count += len(batch)
        with _counter.get_lock():
            _counter.value += len(batch)
        if count % R.FLUSH_EVERY < BATCH or count == total:
            R.flush_buffers(meta_buf, monthly_buf, meta_path, monthly_path)

    R.flush_buffers(meta_buf, monthly_buf, meta_path, monthly_path)
    return name, total, time.time() - t0


def _monitor(counter, t_start, stop_evt):
    last_t, last_c = t_start, 0
    while not stop_evt.wait(30):
        now = time.time()
        c = counter.value
        inst = (c - last_c) / max(now - last_t, 1e-9)
        overall = c / max(now - t_start, 1e-9)
        print(f"  [agg] {c:,} files done | inst {inst:.0f} f/s | "
              f"overall {overall:.0f} f/s | {(now - t_start)/60:.1f} min",
              flush=True)
        last_t, last_c = now, c


if __name__ == "__main__":
    mod = _load_module()
    done = set(os.listdir(mod.OUTPUT_ROOT_DIR))
    strategies = []
    for child in sorted(Path(mod.STRATEGY_ROOT_DIR).iterdir()):
        if not child.is_dir() or child.name in done:
            continue
        td = child / mod.TRADE_SUBDIR_NAME
        if not td.is_dir():
            continue
        if not any(td.glob(mod.BT_FILE_PATTERN)):
            continue
        strategies.append(child.name)

    print(f"Parallel rollup: {len(strategies)} strategies | "
          f"{N_WORKERS} workers x {THREADS_PER_WORKER} threads", flush=True)

    counter = Value("Q", 0)
    t_start = time.time()
    stop_evt = threading.Event()
    mon = threading.Thread(target=_monitor, args=(counter, t_start, stop_evt), daemon=True)
    mon.start()

    completed = 0
    with Pool(N_WORKERS, initializer=_init, initargs=(counter,)) as pool:
        for name, n, dt in pool.imap_unordered(process_strategy, strategies):
            completed += 1
            print(f"  DONE [{completed}/{len(strategies)}] {name}: "
                  f"{n:,} files in {dt/60:.1f} min", flush=True)

    stop_evt.set()
    print(f"ALL DONE: {len(strategies)} strategies in "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)
