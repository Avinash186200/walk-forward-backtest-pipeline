"""
Master runner: executes every "* - OPTIMIZED.ipynb" notebook in this folder,
one strategy after another (sequential), each one using its own internal
multiprocessing Pool (config.CPU_COUNT workers) exactly like the original
notebooks did when run by hand in Jupyter - just without having to open
each one manually.

The raw data file is loaded ONCE (via the first notebook's own data-loading
cells) and the resulting PRICE_BY_KEY lookup is reused for every subsequent
strategy, since they all read the same file - this avoids re-loading and
re-grouping a large parquet ~190 times.

Usage (run from THIS folder, so config.py / live_indicators.py / the
tmp_func_*.py pickling workaround all resolve via the current directory):

    cd READY_FOR_SERVER
    python run_all_strategies.py

Edit config.py first to point at your data file and results folder, and to
set how many worker processes to use.

To run a subset (e.g. resuming after a crash, or just a few strategies),
edit NOTEBOOK_GLOB below or pass notebook filenames as command-line args:

    python run_all_strategies.py "Bulls_strategy - Long - OPTIMIZED.ipynb" "Bulls_strategy - Short - OPTIMIZED.ipynb"
"""
import glob
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # the tmp_func_*.py pickling trick and config/live_indicators imports rely on cwd

import config  # noqa: E402

# Every notebook creates its own strategy subfolder with os.mkdir (not
# os.makedirs), which fails if the parent results folder doesn't exist yet -
# create it once up front so that doesn't surprise you on a fresh server.
os.makedirs(config.RESULTS_BASE_DIR, exist_ok=True)

LOG_PATH = os.path.join(HERE, "run_all_strategies.log")

DATA_LOAD_MARKER = "d1 = pd.read_parquet"
GROUPBY_MARKER = "PRICE_BY_KEY = {"
MAGIC_LINE_RE = re.compile(r"^\s*%")


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def strip_magics(src):
    return "\n".join(line for line in src.split("\n") if not MAGIC_LINE_RE.match(line))


def load_notebook_cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    cells = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.strip():
            cells.append(strip_magics(src))
    return cells


TMP_CELL_PATH = os.path.join(HERE, "_cell_tmp.py")


def exec_cell(src, ns, cell_index):
    """Execs one cell's source as if it were a real .py file on disk (not
    the .ipynb path) - inspect.getsource(), used internally by each
    notebook's own multiprocessing pickling trick (parallel_task() writes
    Func's source to tmp_func_*.py), only works if the compiled code
    object's filename points at an actual file containing that exact text.
    Pointing it at the .ipynb (JSON) file instead silently corrupts that
    trick - this is what writing a real temp .py file per cell avoids."""
    cell_path = f"{TMP_CELL_PATH}.{cell_index}.py"
    with open(cell_path, "w", encoding="utf-8") as f:
        f.write(src)
    exec(compile(src, cell_path, "exec"), ns)


def run_strategy(path, shared_data):
    """Executes one notebook's cells in a fresh namespace, reusing
    shared_data['price_data']/['PRICE_BY_KEY'] if already loaded (skips that
    notebook's own data-loading cells in that case)."""
    cells = load_notebook_cells(path)
    ns = {"__name__": "__main__"}  # parallel_task()'s `if __name__=='__main__'` guard needs this

    if shared_data.get("PRICE_BY_KEY") is not None:
        ns["price_data"] = shared_data["price_data"]
        ns["PRICE_BY_KEY"] = shared_data["PRICE_BY_KEY"]

    for i, src in enumerate(cells):
        if shared_data.get("PRICE_BY_KEY") is not None and (DATA_LOAD_MARKER in src or GROUPBY_MARKER in src):
            continue  # already loaded from the first strategy - reuse it
        exec_cell(src, ns, i)
        if DATA_LOAD_MARKER in src or GROUPBY_MARKER in src:
            shared_data["price_data"] = ns.get("price_data")
            shared_data["PRICE_BY_KEY"] = ns.get("PRICE_BY_KEY")

    n_tasks = len(ns.get("iteration_list", []))
    return n_tasks


def main():
    args = sys.argv[1:]
    if args:
        notebooks = [os.path.join(HERE, a) if not os.path.isabs(a) else a for a in args]
    else:
        notebooks = sorted(glob.glob(os.path.join(HERE, "*-OPTIMIZED.ipynb")) + glob.glob(os.path.join(HERE, "* - OPTIMIZED.ipynb")))
        notebooks = sorted(set(notebooks))

    log(f"\n{'='*70}\nStarting run: {len(notebooks)} strategies, {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}")

    shared_data = {}
    results = []
    overall_start = time.perf_counter()

    for i, path in enumerate(notebooks):
        name = os.path.basename(path)
        log(f"\n[{i+1}/{len(notebooks)}] {name}")
        t0 = time.perf_counter()
        try:
            n_tasks = run_strategy(path, shared_data)
            elapsed = time.perf_counter() - t0
            log(f"  done: {n_tasks} tasks in {elapsed:.1f}s ({elapsed/max(n_tasks,1):.4f}s/task)")
            results.append((name, n_tasks, elapsed, None))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log(f"  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            results.append((name, 0, elapsed, str(e)))

    total_elapsed = time.perf_counter() - overall_start
    n_ok = sum(1 for _, _, _, err in results if err is None)
    total_tasks = sum(n for _, n, _, err in results if err is None)

    log(f"\n{'='*70}")
    log(f"FINISHED: {n_ok}/{len(notebooks)} strategies completed, {total_tasks} total tasks, {total_elapsed/3600:.2f} hours")
    log(f"{'='*70}")
    failed = [(n, e) for n, _, _, e in results if e]
    if failed:
        log(f"\n{len(failed)} strategies FAILED:")
        for n, e in failed:
            log(f"  {n}: {e}")


if __name__ == "__main__":
    main()
