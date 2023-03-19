import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm

from fredapi import Fred
import pandas as pd

# import settings
import json

import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen, Request


def get_betas_wide(betas_list):
    list_df = []
    for item in betas_list:
        df = get_price_data(item)
        list_df.append(df)

    df_long = pd.concat(list_df, axis=0).reset_index()
    df_wide = df_long.pivot_table(
        index=["index"], columns="ticker", values="pct_change"
    )
    return df_wide


def get_parallel_betas_from_list(ticker_list, betas_list):
    df_wide = get_betas_wide(betas_list)
    from joblib import Parallel, delayed
    import multiprocessing

    inputs = range(10)

    def processInput(i):
        return i * i

    num_cores = multiprocessing.cpu_count()
    all_df = Parallel(n_jobs=num_cores)(
        delayed(get_beta_data)(ticker, df_wide, betas_list) for ticker in ticker_list
    )
    df_betas = pd.concat(all_df)
    return df_betas


def get_beta_data(ticker, df_wide, betas_list):
    list_out = []
    print(ticker)
    try:
        df = get_price_data(ticker).reset_index()
    except:
        print("failed")
        return

    dataset = pd.merge(df, df_wide, how="inner", left_on="index", right_on="index")

    beta_lens = [60, 90, 252]  # window for betas
    dict_lens = {}
    for beta_len in beta_lens:
        betas = {}
        ns = {}
        r2s = {}
        for col in dataset.columns:
            if col in betas_list:
                df = dataset[["pct_change", col]].dropna()
                # try:
                beta, r2, n = get_betas(df["pct_change"], df[col], beta_len)
                # except:
                #    print("get_betas error")
                #    return
                betas[col] = beta
                ns[col] = n
                r2s[col] = r2
        dict_lens[beta_len] = [betas, ns, r2s]

        df1 = pd.DataFrame.from_dict(
            dict_lens[beta_len][2], orient="index"
        ).reset_index()

        try:
            df1.columns = ["BETA", ticker]
        except:
            print("df columns error")
            return
        df1 = df1.T
        headers = df1.iloc[0]
        df1 = pd.DataFrame(df1.values[1:], columns=headers)
        df1["ticker"] = ticker
        df1["beta_days"] = beta_len
        list_out.append(df1)
    df = pd.concat(list_out)
    return df


def get_fred_key():
    key_file_name = f"{settings.ROOT_DIR}/keys.json"
    with open(key_file_name) as json_file:
        keys = json.load(json_file)
    return Fred(keys["fred_key"])


def get_price_data(ticker):
    price_data = si.get_data(ticker, start_date="01/01/2009")
    df = pd.DataFrame(price_data)
    df = df[["adjclose"]]
    df["pct_change"] = df.adjclose.pct_change()
    df["log_return"] = np.log(1 + df["pct_change"].astype(float))
    df["ticker"] = ticker
    return df


def get_betas(x, y, n=0):
    if n > 0:
        x = x.iloc[-n:,]
        y = y.iloc[-n:,]
    res = sm.OLS(y, x).fit()
    beta = res.params[0]
    r2 = res.rsquared
    n = len(x)
    return [beta, r2, n]


def get_fred_data(series_list):
    FRED = get_fred_key()

    df_fred = pd.DataFrame()
    for fred_series in series_list:
        s = pd.DataFrame(
            FRED.get_series(
                fred_series, observation_start="2014-09-02", observation_end="2021-03-1"
            )
        ).reset_index()
        s.columns = ["end_date", fred_series]
        s = calendar_quarter(s, "end_date")
        s = s.groupby("end_date_CALENDAR_QUARTER").last().reset_index()
        if df_fred.shape[0] > 0:
            s = s.drop(columns="end_date")
            df_fred = pd.merge(
                df_fred,
                s,
                how="inner",
                left_on="end_date_CALENDAR_QUARTER",
                right_on="end_date_CALENDAR_QUARTER",
            )
        else:
            df_fred = s
    return df_fred


def append_price_data(df):
    df = calendar_quarter(df, "end_date")

    pd.set_option("mode.chained_assignment", None)
    ticker = df.iloc[0]["ticker"]
    stock_ticker = ticker.split(" ")[0]
    df_prices = get_price_data(stock_ticker)
    df_prices = df_prices.reset_index()
    df_prices = df_prices[["pricing_date", "adjclose"]]
    df_prices.columns = ["end_date", "adjclose"]

    df_prices = calendar_quarter(df_prices, "end_date")
    df_prices = df_prices.sort_values("end_date")
    df_prices = df_prices.groupby("end_date_CALENDAR_QUARTER").last().reset_index()

    # fiscal_quarter
    df = df_filter(df, "period_duration_type", ["fiscal_quarter"])
    dates = list(set(list(df["end_date_CALENDAR_QUARTER"])))
    df_p = df_filter(df_prices, "end_date_CALENDAR_QUARTER", dates)

    df_p.columns = ["end_date_CALENDAR_QUARTER", "end_date", "value"]
    df_p["ticker"] = ticker
    df_p["period"] = ""
    df_p["period_duration_type"] = "fiscal_quarter"
    df_p["category"] = ""
    df_p["type"] = ""
    df_p["row_header"] = "Stock Price"
    df_p["unit"] = "$"
    df_p = df_p.sort_values("end_date")
    df_1 = df_p[
        [
            "ticker",
            "period",
            "period_duration_type",
            "end_date",
            "category",
            "type",
            "row_header",
            "unit",
            "value",
            "end_date_CALENDAR_QUARTER",
        ]
    ]

    df = pd.concat([df, df_1])

    return df


def get_price_data(ticker, col="adjclose"):
    price_data = si.get_data(ticker, start_date="01/01/2009")
    df = pd.DataFrame(price_data).reset_index()
    # df = df[[col]]
    df.columns = [
        "pricing_date",
        "open",
        "high",
        "low",
        "close",
        "adjclose",
        "volume",
        "ticker",
    ]
    return df

    # helper function to return year month


def year_month(df, col, collapse=False, datetime=True):
    # translate a date into sort-able and group-able YYYY-mm format.
    df[col] = pd.to_datetime(df[col])
    df[col + "_YEAR_MONTH"] = df[col].dt.strftime("%Y-%m")
    if collapse == True:
        df[col + "_YEAR_MONTH"] = df[col + "_YEAR_MONTH"].str.replace("-", "")
        df[col + "_YEAR_MONTH"] = df[col + "_YEAR_MONTH"].astype(int)
    return df

    # quarterly returns from yahoo finance


def get_quarterly_returns(ticker):
    df_prices = get_price_data(ticker)[["pricing_date", "adjclose", "ticker"]]
    df_prices = calendar_quarter(df_prices, "pricing_date")
    df_prices.groupby("pricing_date_CALENDAR_QUARTER").last().reset_index()
    df_prices["quarterly_return"] = df_prices["adjclose"].pct_change()
    df_prices = calendar_quarter(df_prices, "pricing_date")
    df_prices = df_prices.groupby("pricing_date_CALENDAR_QUARTER").last().reset_index()
    df_prices["quarterly_return"] = df_prices["adjclose"].pct_change()
    df_prices = df_prices.dropna()
    return df_prices

    # quarterly return vs spy


def get_quarterly_outperformance(ticker):
    df_prices = get_quarterly_returns(ticker)
    df_spy = get_quarterly_returns("SPY")
    df_join = pd.merge(
        df_prices,
        df_spy,
        how="inner",
        left_on="pricing_date_CALENDAR_QUARTER",
        right_on="pricing_date_CALENDAR_QUARTER",
    )
    df_join["excess_return"] = (
        df_join["quarterly_return_x"] - df_join["quarterly_return_y"]
    )
    df_join["outperform"] = np.where(df_join["excess_return"] > 0, 1, 0)
    df_join = df_join[["pricing_date_CALENDAR_QUARTER", "outperform"]]
    return df_join

    # calendar quarter helper functions
