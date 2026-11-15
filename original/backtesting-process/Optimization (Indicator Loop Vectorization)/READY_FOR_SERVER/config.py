"""
Single place to set paths/settings for every optimized notebook in this
folder. Edit the four values below for your server, then every notebook
picks them up automatically (each notebook does `import config` and reads
config.DATA_DIR / DATA_FILE / RESULTS_BASE_DIR / CPU_COUNT) - no need to
touch the notebooks themselves.

IMPORTANT: this file must sit in the same folder as the notebooks (or
somewhere on PYTHONPATH) - the notebooks add their own working directory to
sys.path before importing it, so keep config.py + live_indicators.py +
all the "* - OPTIMIZED.ipynb" files together in one folder on the server.
"""

# Folder containing the raw daily OHLC parquet file (was the hardcoded
# "<local-path>" path in the original notebooks).
DATA_DIR = "<local-path>"

# Filename of that parquet file (was hardcoded as
# "CMC_DAILY_OHLC_close_m20_m25_m30_updated.parquet" in every notebook).
DATA_FILE = "CMC_DAILY_OHLC_close_m20_m25_m30_updated.parquet"

# Base folder where each strategy's BT/FT/Trade Level Data/Open_Trades/Error
# subfolders get created (was the hardcoded
# "<local-path>" path). Each notebook appends its
# own strategy_name + "/" to this.
RESULTS_BASE_DIR = "<local-path>"

# Number of parallel worker processes (was hardcoded cpu_count = 55).
CPU_COUNT = 40
