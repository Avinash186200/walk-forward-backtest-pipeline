"""
Parallel runner for Step 2 (Walk-Forward Monthly Selection) — MEMORY-AWARE.

Strategy folders vary enormously in size (a few MB up to several GB of
monthly_stats_all.csv). Naively running N workers OOMs the machine when several
big folders load at once. This scheduler instead allocates concurrency by size
in real time:

  - estimate each folder's RAM need (~MEM_FACTOR x its CSV size),
  - keep submitting folders while the in-flight estimated total stays under a
    memory budget, releasing budget as each finishes.

Net effect: tiny folders run many-at-once; the biggest folders run nearly alone.
Max throughput without ever exceeding RAM. Per-folder output is identical to the
single-process reference (same process_strategy from walk_forward_selection.py).

Usage:
    python walk_forward_selection_parallel.py [ROOT_DIR] [MEM_BUDGET_GB] [MAX_WORKERS]

Resumable: folders that already have walk_forward_selection_sharpe_ge_1_unique.csv
are skipped (SKIP_DONE in walk_forward_selection.py).
"""
import os
import sys
import time
import concurrent.futures as cf

import walk_forward_selection as wfs

# ---- memory model (tune for your box) ----
MEM_BUDGET_GB = 40.0    # cap on concurrent ESTIMATED memory (leave headroom below total RAM)
MEM_FACTOR = 15.0       # estimated peak RAM per folder = CSV size x this (multi-GB inputs blow up to ~15x)
MIN_EST_GB = 2.5        # floor per folder (base pandas + processing overhead)
MAX_WORKERS = None      # hard cap on processes (None -> min(24, cpu-1))


def est_mem_bytes(folder):
    f = os.path.join(folder, wfs.MONTHLY_CSV_PATTERN)
    sz = os.path.getsize(f) if os.path.exists(f) else 0
    return max(MIN_EST_GB * 1e9, sz * MEM_FACTOR)


def _worker(folder):
    try:
        return wfs.process_strategy(folder)
    except Exception as e:
        return folder, "ERR:%s" % e


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else wfs.ROOT_DIR
    budget = (float(sys.argv[2]) if len(sys.argv) > 2 else MEM_BUDGET_GB) * 1e9
    maxw = int(sys.argv[3]) if len(sys.argv) > 3 else (MAX_WORKERS or min(24, max(1, (os.cpu_count() or 2) - 1)))

    folders = wfs.iter_strategy_folders(root)
    # biggest first: the monsters start early (and run nearly alone), small ones fill the gaps
    folders.sort(key=est_mem_bytes, reverse=True)
    n = len(folders)
    print("Step 2 (memory-aware): %d folders | budget %.0f GB | max %d workers\n  root: %s"
          % (n, budget / 1e9, maxw, root), flush=True)

    t0 = time.time()
    done_ct = 0
    with cf.ProcessPoolExecutor(max_workers=maxw) as ex:
        inflight = {}   # future -> (folder, est_bytes)
        used = 0.0
        i = 0
        while i < n or inflight:
            # submit while we have a worker slot AND budget (or nothing is running yet)
            while (i < n and len(inflight) < maxw and
                   (not inflight or used + est_mem_bytes(folders[i]) <= budget)):
                e = est_mem_bytes(folders[i])
                fut = ex.submit(_worker, folders[i])
                inflight[fut] = (folders[i], e)
                used += e
                i += 1
            # wait for at least one to finish, free its budget, then re-evaluate
            done, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                folder, e = inflight.pop(fut)
                used -= e
                done_ct += 1
                try:
                    _, status = fut.result()
                except Exception as ex2:
                    status = "CRASH:%s" % ex2
                print("  [%d/%d] %-11s %-42s | inflight %d, ~%.0fGB"
                      % (done_ct, n, status, os.path.basename(folder), len(inflight), used / 1e9),
                      flush=True)
    print("DONE in %.1fs (%d folders)" % (time.time() - t0, n), flush=True)


if __name__ == "__main__":
    main()
