"""Baseline = original Step 5 cell-5 logic, inlined verbatim (year-outer loop,
strategy-inner loop, re-reading each strategy CSV once per year). Strategy order
forced to sorted() so the output is deterministic and comparable to the optimized
version. For validation only.

Usage: python baseline_step5.py [ROOT] [OUT_DIR] [STRATEGY_LIMIT]
"""
import os, sys, itertools
import numpy as np
import pandas as pd
import step5_data as sd

ROOT = sys.argv[1]
OUT = sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
TRADE_FILE = "selected_trade_level_all_months.csv"
Year_list = [2021, 2022, 2023, 2024, 2025, 2026]

strategies_trade_data_path = ROOT + os.sep
strategy_list = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
if LIMIT:
    strategy_list = strategy_list[:LIMIT]
iter_list = pd.DataFrame(itertools.product(strategy_list), columns=['Strategy'])
os.makedirs(OUT, exist_ok=True)
BROKER_Margin_df = sd.BROKER_Margin_df

for year in Year_list:
    all_stg_MTM_data = pd.DataFrame()
    for row, item in iter_list.iterrows():
        strategy = item['Strategy']
        strategy_trade_data_path = strategies_trade_data_path + strategy + os.sep
        try:
            close_trade_data = pd.read_csv(strategy_trade_data_path + TRADE_FILE)
            close_trade_data['Open close Flag'] = 'Closed'
            close_trade_data = close_trade_data[(close_trade_data['EntryDate Year'] == year)
                                                & (close_trade_data['ClosingDate Year'] == year)]
        except Exception as e:
            print(e); print("here"); continue

        trade_data = close_trade_data
        trade_data = trade_data[trade_data['Timeframe'] == 'D1'].reset_index(drop=True)
        trade_data.rename(columns={'Profit_Loss': 'New P/L_pips'}, inplace=True)
        trade_data['Strategy'] = trade_data['Strategy'].str.replace('77 strategy', '77')
        trade_data['Strategy'] = trade_data['Strategy'].str.replace('RSI-Larry 25', 'RSI 25')
        trade_data_for_MTM = trade_data.copy()
        trade_data_for_MTM['date'] = pd.to_datetime(trade_data_for_MTM['date'])
        trade_data_for_MTM['Closing Date'] = pd.to_datetime(trade_data_for_MTM['Closing Date'])
        trade_data_for_MTM['Year'] = trade_data_for_MTM['date'].dt.year
        trade_data_for_MTM['Month'] = trade_data_for_MTM['date'].dt.month
        trade_data_for_MTM['Day'] = trade_data_for_MTM['date'].dt.day
        data_for_MTM = trade_data_for_MTM

        columns_list = ['Strategy', 'Opening Date', 'Currency', 'Timeframe',
                        'Condition', 'Trading Price Point', 'Stop_Loss', 'TakeProfit_by_pips',
                        'Closing Date', 'Capping Date', 'SL to Feed',
                        'Data_Type', 'New P/L_pips', 'Year', 'Month',
                        'Day', 'Hour', 'Day Level Date', 'Currency Type',
                        'BROKER Margin', 'Open close Flag']

        filters = [data_for_MTM['Currency'].isin(sd.Major_pairs_list),
                   data_for_MTM['Currency'].isin(sd.Exotic_pairs_list),
                   data_for_MTM['Currency'].isin(sd.Commodities_pairs_list),
                   data_for_MTM['Currency'].isin(sd.ETF_pairs_list),
                   data_for_MTM['Currency'].isin(sd.bond),
                   data_for_MTM['Currency'].isin(sd.futures),
                   data_for_MTM['Currency'].isin(sd.Index_CFDs),
                   data_for_MTM['Currency'].isin(sd.NDFs),
                   data_for_MTM['Currency'].isin(sd.Single_Stock_CFDs)]
        values = ["Non-Exotic", "Exotic", "Commodities", "ETF",
                  "bond", "Single Stock CFD", "Index CFD", "NDF", "Single Stock CFD"]
        data_for_MTM['Currency Type'] = np.select(filters, values)

        data_for_MTM.rename(columns={'date': 'Opening Date', 'Train Cut Off': 'Cut Off',
                                     'Model Passing': 'Passed Trades'}, inplace=True)
        data_for_MTM['Capping Date'] = data_for_MTM['Closing Date']
        data_for_MTM['Data_Type'] = 'SL_Updated'
        data_for_MTM['Stop_Loss'] = 0
        data_for_MTM['TakeProfit_by_pips'] = 0
        data_for_MTM['SL to Feed'] = np.where(data_for_MTM['Condition'] == 'Long',
                                              data_for_MTM['Exit_bidclose'], data_for_MTM['Exit_askclose'])
        data_for_MTM['Day Level Date'] = data_for_MTM['Opening Date']
        data_for_MTM['Hour'] = data_for_MTM['Opening Date'].dt.hour

        data_for_MTM_1 = data_for_MTM.merge(BROKER_Margin_df, on=['Currency'], how='left')
        final_data_for_MTM = data_for_MTM_1[columns_list]
        final_data_for_MTM['Closing Type'] = 'No_SL_TP'
        all_stg_MTM_data = pd.concat([all_stg_MTM_data, final_data_for_MTM], ignore_index=True)

    all_stg_MTM_data.to_csv(os.path.join(OUT, "%s all selected comb mtm data Test.csv" % year), index=False)
    print("baseline %s -> %d rows" % (year, len(all_stg_MTM_data)), flush=True)
