"""One-off generator for the 6 FX/Comm/Combined x T+3/T+15 config.ini files
used for the June 2026 run. Not part of the reusable pipeline - just keeps
the 6 configs consistent instead of hand-editing each one."""
from pathlib import Path

BASE = Path("<local-path>")
RAW_DATA_DIR = BASE / "Raw data"
OUT_DIR = Path(__file__).resolve().parent / "configs_FXComm_June2026"

RUNS = [
    ("T+3", "FX MTM Data", "FX", "FX_T3"),
    ("T+3", "Comm MTM Data", "comm", "COMM_T3"),
    ("T+3", "Combined/MTM Data", "FXComm", "COMBINED_T3"),
    ("T+15", "FX MTM Data", "FX", "FX_T15"),
    ("T+15", "Comm MTM Data", "comm", "COMM_T15"),
    ("T+15", "Combined/MTM Data", "FXComm", "COMBINED_T15"),
]

TEMPLATE = """; Auto-generated for the FX/Comm June 2026 run - see generate_configs_FXComm_June2026.py
[paths]
mtm_data_dir = {mtm_data_dir}
raw_data_dir = {raw_data_dir}
output_dir = {output_dir}

[data]
tag = {tag}
label = {label}
years = 2021,2022,2023,2024,2025,2026

[portfolio]
total_portfolio_capital = 1000000
max_gross_exposure = 0.90
initial_equity = 1000000

[vol_targeting]
vol_lookback_days = 90
production_scenario = tv05_cap100
run_all_scenarios = false

[costs]
txn_cost_per_unit = 0.00002
financing_cost_rate = 0.00005
slippage_pips = 1.0

[run]
force_vol_wall = false
"""

OUT_DIR.mkdir(parents=True, exist_ok=True)
for period, mtm_subdir, tag, label in RUNS:
    mtm_data_dir = BASE / period / mtm_subdir
    output_dir = mtm_data_dir / "Pipeline Output"
    cfg_text = TEMPLATE.format(
        mtm_data_dir=str(mtm_data_dir).replace("\\", "/"),
        raw_data_dir=str(RAW_DATA_DIR).replace("\\", "/"),
        output_dir=str(output_dir).replace("\\", "/"),
        tag=tag,
        label=label,
    )
    out_path = OUT_DIR / f"config_{label}.ini"
    out_path.write_text(cfg_text, encoding="utf-8")
    print(f"wrote {out_path}")
