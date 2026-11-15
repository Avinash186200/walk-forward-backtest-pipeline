"""
Parallel runner for Step 3 — MEMORY-AWARE (same scheduler idea as Step 2).

Per selection folder, Step 3 loads that folder's iteration trade parquets and
filters them. Memory scales with the trade data pulled, which we proxy by the
selection file size (more selected rows -> more/larger iterations). The scheduler
keeps in-flight estimated memory under a budget so big folders don't pile up.

Per-folder logic = step3_core.process_strategy_selection_folder (byte-identical
to the original notebook cells 2 + 7).

Usage: python step3_parallel.py [SELECTION_ROOT] [MEM_BUDGET_GB] [MAX_WORKERS]
"""
import os
import sys
import time
import concurrent.futures as cf

import ft_trade_level as s3

MEM_BUDGET_GB = 40.0
MEM_FACTOR = 30.0     # est RAM ~ selection_csv_size x this (trades dwarf the selection file)
MIN_EST_GB = 2.0
MAX_WORKERS = None    # None -> min(12, cpu-1)


def est_mem_bytes(folder):
    f = os.path.join(folder, s3.SELECTION_FILE_NAME)
    sz = os.path.getsize(f) if os.path.exists(f) else 0
    return max(MIN_EST_GB * 1e9, sz * MEM_FACTOR)


def _worker(folder):
    try:
        return s3.process_strategy_selection_folder(folder)
    except Exception as e:
        return folder, "ERR:%s" % e


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else s3.SELECTION_ROOT
    budget = (float(sys.argv[2]) if len(sys.argv) > 2 else MEM_BUDGET_GB) * 1e9
    maxw = int(sys.argv[3]) if len(sys.argv) > 3 else (MAX_WORKERS or min(12, max(1, (os.cpu_count() or 2) - 1)))

    folders = s3.iter_selection_folders(root)
    # only folders that actually have a selection file are real work
    folders = [f for f in folders if os.path.exists(os.path.join(f, s3.SELECTION_FILE_NAME))]
    folders.sort(key=est_mem_bytes, reverse=True)
    n = len(folders)
    print("Step 3 (memory-aware): %d folders w/ selection | budget %.0f GB | max %d workers"
          % (n, budget / 1e9, maxw), flush=True)

    t0 = time.time()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=maxw) as ex:
        inflight = {}
        used = 0.0
        i = 0
        while i < n or inflight:
            while (i < n and len(inflight) < maxw and
                   (not inflight or used + est_mem_bytes(folders[i]) <= budget)):
                e = est_mem_bytes(folders[i])
                fut = ex.submit(_worker, folders[i]); inflight[fut] = e; used += e; i += 1
            d, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
            for fut in d:
                used -= inflight.pop(fut); done += 1
                try:
                    fld, status = fut.result()
                except Exception as ex2:
                    fld, status = "?", "CRASH:%s" % ex2
                print("  [%d/%d] %-12s %-40s | inflight %d ~%.0fGB"
                      % (done, n, status, os.path.basename(str(fld)), len(inflight), used / 1e9), flush=True)
    print("DONE in %.1fs (%d folders)" % (time.time() - t0, n), flush=True)


if __name__ == "__main__":
    main()
