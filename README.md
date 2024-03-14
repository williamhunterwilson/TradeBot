import datetime as dt
import mplfinance as mpf
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import numpy as np
import matplotlib.pyplot as plt
from alphaVantageAPI.alphavantage import AlphaVantage
import watchlist
#import requests
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# Insert any API you wish to use
#av_news_key = 
av_intraday_key = 
av_daily_adj_key = 
av_daily_key = 

# Use any specific companies you want to analyse
#tickers
symbol = 
url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}&datatype=csv".format(symbol, av_daily_key)

#ts = TimeSeries(key = av_stock_api, output_format = 'pandas')

#data, meta_data = ts.get_daily(symbol = symbol, outputsize='compact')
def decode(input):
    
    arr = [0]*len(input)
                  
    for str in input:

        line = str.split()
        arr[int(line[0]) - 1] = line[1]

    n=0
    inc = 2
    final_arr = []
    while n < len(input):
        
        final_arr.append(arr[n])
        n += inc
        inc += 1

    print(' '.join(final_arr))

file_path = 'test.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

decode(lines)
import pandas as pd

# Load the data from the CSV file
file_path = 'C:/Users/will_/Code/world-population-by-country-2020.csv'
population_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
population_data.head()


import plotly.express as px

# Creating a 3D scatter plot
fig = px.scatter_3d(population_data, 
                    x='Population 2020', 
                    y='Land Area (Km²)', 
                    z='Med. Age', 
                    color='Density',
                    hover_name='Country (or dependency)',
                    hover_data={'Population 2020': ':,', 'Land Area (Km²)': ':,', 'Med. Age': True, 'Density': ':.2f'},
                    title='World Population, Land Area, and Median Age (2020)')

# Enhancing the plot
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), 
                  scene=dict(
                      xaxis_title='Population',
                      yaxis_title='Land Area (km²)',
                      zaxis_title='Median Age'
                  ))

# Show plot
fig.show()
df = pd.read_csv(url)
df = pd.DataFrame(df)
new_headers = {'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
df.rename(columns=new_headers, inplace=True)

df.set_index = df.Date
# Can use any financial functions listed in pandas_ta library
function_names = ['rsi', 'sma', 'eri', 'ema', 'vwma']

for func_name in function_names:
    func = getattr(df.ta, func_name)

    result = func()

    df = pd.concat([df, result], axis = 1)

print(df)

#percentage split of train/test data, split = % amount of training data
split = 0.8
split_row = int(df.shape[0] * split)

split_date = df['Date'].iloc[split_row]
train_data = df[df['Date'] < split_date]
test_data = df[df['Date'] >= split_date]

TRAIN_START_DATE = df['Date'].iloc[-1]
TRAIN_END_DATE = split_date
TEST_START_DATE = df['Date'].iloc[split_row + 1]
TEST_END_DATE = df['Date'].iloc[0]
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
TRAIN_START_DATE = '2009-04-01'
TRAIN_END_DATE = '2021-01-01'
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2022-06-01'

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TEST_END_DATE,
                     ticker_list = DOW_30_TICKER).fetch_data()

INDICATORS = ['macd',
               'rsi_30',
               'cci_30',
               'dx_30']

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf,0)
print(processed)

stock_dimension = len(processed.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000, 
    "buy_cost_pct": 0.01, 
    "sell_cost_pct": 0.01, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4,
    "print_verbosity":5
    
}
rebalance_window = 63 #63 # rebalance_window is the number of days to retrain the model
validation_window = 63 #63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

ensemble_agent = DRLEnsembleAgent(df=processed,
                 train_period=(TRAIN_START_DATE,TRAIN_END_DATE),
                 val_test_period=(TEST_START_DATE,TEST_END_DATE),
                 rebalance_window=rebalance_window, 
                 validation_window=validation_window, 
                 **env_kwargs)
A2C_model_kwargs = {
                    'n_steps': 5,
                    'ent_coef': 0.005,
                    'learning_rate': 0.0007
                    }

PPO_model_kwargs = {
                    "ent_coef":0.01,
                    "n_steps": 2, #2048
                    "learning_rate": 0.00025,
                    "batch_size": 128
                    }

DDPG_model_kwargs = {
                      #"action_noise":"ornstein_uhlenbeck",
                      "buffer_size": 1, #10_000
                      "learning_rate": 0.0005,
                      "batch_size": 64
                    }

timesteps_dict = {'a2c' : 1, #10_000 each
                 'ppo' : 1, 
                 'ddpg' : 1
                 }
df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                 PPO_model_kwargs,
                                                 DDPG_model_kwargs,
                                                 timesteps_dict)
unique_trade_date = processed[(processed.date > TEST_START_DATE)&(processed.date <= TEST_END_DATE)].date.unique()
import pandas as pd
print(pd.__version__)
df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

df_account_value=pd.DataFrame()

for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
    temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble',i))
    df_account_value = pd.concat([df_account_value, temp], ignore_index=True)
sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
print('Sharpe Ratio: ',sharpe)
df_account_value=df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))
%matplotlib inline
df_account_value.account_value.plot()
print("==============Get Backtest Results===========")
now = dt.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI", 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

stats = backtest_stats(baseline_df, value_col_name = 'close')
print(baseline_df)
df_dji = pd.DataFrame()
df_dji['date'] = df_account_value['date']
df_dji['dji'] = baseline_df['close'] / baseline_df['close'][0] * env_kwargs["initial_amount"]
print("df_dji: ", df_dji)
df_dji.to_csv("df_dji.csv")
df_dji = df_dji.set_index(df_dji.columns[0])
print("df_dji: ", df_dji)
df_dji.to_csv("df_dji+.csv")

df_account_value.to_csv('df_account_value.csv')
# print("==============Compare to DJIA===========")
# %matplotlib inline
# # S&P 500: ^GSPC
# # Dow Jones Index: ^DJI
# # NASDAQ 100: ^NDX
# backtest_plot(df_account_value, 
#               baseline_ticker = '^DJI', 
#               baseline_start = df_account_value.loc[0,'date'],
#               baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])
df.to_csv("df.csv")
df_result_ensemble = pd.DataFrame({'date': df_account_value['date'], 'ensemble': df_account_value['account_value']})
df_result_ensemble = df_result_ensemble.set_index('date')

print("df_result_ensemble.columns: ", df_result_ensemble.columns)

# df_result_ensemble.drop(df_result_ensemble.columns[0], axis = 1)
print("df_trade_date: ", df_trade_date)
# df_result_ensemble['date'] = df_trade_date['datadate']
# df_result_ensemble['account_value'] = df_account_value['account_value']
df_result_ensemble.to_csv("df_result_ensemble.csv")
print("df_result_ensemble: ", df_result_ensemble)
print("==============Compare to DJIA===========")
result = pd.DataFrame()
# result = pd.merge(result, df_result_ensemble, left_index=True, right_index=True)
# result = pd.merge(result, df_dji, left_index=True, right_index=True)
result = pd.merge(df_result_ensemble, df_dji, left_index=True, right_index=True)
print("result: ", result)
result.to_csv("result.csv")
result.columns = ['ensemble', 'dji']

%matplotlib inline
plt.rcParams["figure.figsize"] = (15,5)
plt.figure()
result.plot()
