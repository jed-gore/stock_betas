import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si

import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import itertools
from statsmodels.tsa.stattools import coint


class Portfolio:
    def __init__(self, stock_list, log=False):
        self.stock_list = stock_list
        self.get_prices(log)
        self.get_pairs()

    def rolling_beta(self, returns, period):
        cov = returns.iloc[0:, 0].rolling(period).cov(returns.iloc[0:, 1])
        market_var = returns.iloc[0:, 1].rolling(period).var()
        individual_beta = cov / market_var
        return individual_beta

    def get_alphas(self):
        cols = self.dataset.columns
        alpha_cols = ["price_date"]
        df_earnings_list = []
        for col in cols:
            if "date" not in col:
                self.dataset[col + "_pct_change"] = self.dataset[col].pct_change()
                self.dataset[col + "_pct_change_5"] = self.dataset[col].pct_change(5)
                self.dataset[col + "_log_return"] = np.log(
                    1 + self.dataset[col + "_pct_change"]
                )
                cols = [col + "_pct_change", "SPY_pct_change", "price_date"]
                df = self.dataset[cols]
                self.dataset[col + "_60_day_rolling_beta"] = self.rolling_beta(df, 60)
                if "SPY" not in col:
                    self.dataset[col + "_5_day_alpha"] = (
                        self.dataset[col + "_pct_change_5"]
                        - self.dataset["SPY_pct_change_5"]
                        * self.dataset[col + "_60_day_rolling_beta"]
                    )
                    self.dataset[col + "_5_day_alpha_shifted"] = self.dataset[
                        col + "_5_day_alpha"
                    ].shift(-5)
                    alpha_cols.append(col + "_5_day_alpha")
                    alpha_cols.append(col + "_5_day_alpha_shifted")

                    df_earnings = pd.read_csv(f"data/{col}_earnings.csv")
                    df_earnings = df_earnings[
                        ["reportedDate", "fiscalDateEnding", "surprisePercentage"]
                    ]
                    df_earnings.columns = [
                        col + "_rdate",
                        col + "_fdate",
                        col + "_surprisepct",
                    ]
                    df_earnings[col + "_rdate"] = pd.to_datetime(
                        df_earnings[col + "_rdate"]
                    )
                    df_earnings = pd.merge(
                        self.dataset[["price_date", col + "_5_day_alpha_shifted"]],
                        df_earnings,
                        how="left",
                        left_on="price_date",
                        right_on=col + "_rdate",
                    ).dropna()
                    df_earnings.to_csv(f"data/{col}_earnings_5_day_alpha.csv")

                    df_earnings.columns = [
                        "price_date",
                        "5_day_alpha_shifted",
                        "rdate",
                        "fdate",
                        "surprise_pct",
                    ]
                    df_earnings["ticker"] = col
                    df_earnings_list.append(df_earnings)

        self.earnings_alpha = pd.concat(df_earnings_list)
        self.alphas = self.dataset[alpha_cols].dropna()

    def get_pairs(self):
        self.all_pairs = list(itertools.combinations(self.stock_list, 2))
        self.cointegrating_pairs = []
        self.pairs = []

        for item in self.all_pairs:
            pair = self.Pair(
                self.dataset[item[0]],
                self.dataset[item[1]],
                item[0],
                item[1],
                self.dataset["price_date"],
            )
            self.pairs.append(pair)

    class Pair:
        def __init__(self, x, y, x_name, y_name, price_date):
            self.x_name = x_name
            self.y_name = y_name
            self.price_date = price_date
            self.x = x
            self.y = y
            self.cointegrates = False
            self.beta = sm.OLS(x, y).fit().params[0]
            self.spread = y - self.beta * x
            self.normalized_spread_trading = (
                self.spread - self.spread.mean()
            ) / self.spread.std()
            self.coint_t, self.pvalue, self.crit_value = coint(x, y)
            if self.pvalue < 0.05:
                self.cointegrates = True
            return

        def plot_spread(self):
            df = pd.DataFrame()
            df["price_date"] = self.price_date
            df["normalized_spread"] = self.normalized_spread_trading
            df.plot(
                x="price_date",
                y="normalized_spread",
                title=f"{self.x_name} vs {self.y_name}",
            )

    def get_prices(self, log=False):
        list_df = []
        for item in self.stock_list:
            df = self.get_price_data(item)
            list_df.append(df)
        self.df_long = pd.concat(list_df, axis=0).reset_index()
        self.df_wide = self.df_long.pivot_table(
            index=["index"], columns="ticker", values="adjclose"
        ).reset_index()

        self.spy = self.get_price_data("SPY")

        self.spy = (
            self.spy.reset_index()
            .pivot_table(index="index", columns="ticker", values="adjclose")
            .reset_index()
        )

        self.dataset = pd.merge(
            self.spy, self.df_wide, how="inner", left_on="index", right_on="index"
        )
        self.dataset = self.dataset.dropna()
        self.dataset["price_date"] = self.dataset["index"]
        self.dataset.drop(columns="index", inplace=True)
        if log == True:
            for item in self.stock_list:
                self.dataset[item] = np.log(self.dataset[item])
            self.dataset["SPY"] = np.log(self.dataset["SPY"])

    def get_price_data(self, ticker):
        price_data = si.get_data(ticker, start_date="01/01/2012")
        df = pd.DataFrame(price_data)
        df = df[["adjclose"]]
        df["pct_change"] = df.adjclose.pct_change()
        df["log_return"] = np.log(1 + df["pct_change"].astype(float))
        df["ticker"] = ticker
        return df


class StockData:
    def get_cross_correlations(
        self,
        stock_list=[],
        rolling_window=60,
    ):
        i = []
        j = []
        run_list = []
        df_output = pd.DataFrame()
        df_output["index"] = self.dataset["index"]
        df = self.dataset
        for ticker_i in stock_list:
            for ticker_j in stock_list:
                if (
                    ticker_j != ticker_i
                    and ticker_j + ticker_i not in run_list
                    and ticker_i + ticker_j not in run_list
                ):
                    df_output[ticker_j + "_" + ticker_i] = (
                        df[ticker_j].rolling(rolling_window).corr(df[ticker_i])
                    )
                    run_list.append(ticker_j + ticker_i)
                    run_list.append(ticker_i + ticker_j)
        df_output[f"{rolling_window}_corr"] = df_output.drop("index", axis=1).mean(
            axis=1
        )
        self.rolling_window = rolling_window
        self.correlations = df_output
        return df_output

    def plot_cross_correlations(self):
        self.correlations["std"] = self.correlations[
            f"{self.rolling_window}_corr"
        ].std()
        self.correlations["mean"] = self.correlations[
            f"{self.rolling_window}_corr"
        ].mean()
        self.correlations["std_1"] = (
            self.correlations["mean"] + self.correlations["std"]
        )
        self.correlations["std_2"] = (
            self.correlations["mean"] + 2 * self.correlations["std"]
        )
        self.correlations["std_minus_1"] = (
            self.correlations["mean"] - self.correlations["std"]
        )
        self.correlations["std_minus_2"] = (
            self.correlations["mean"] - 2 * self.correlations["std"]
        )
        self.correlations.plot(
            title=f"Cross Correlations for {self.stock_list}",
            x="index",
            y=[
                f"{self.rolling_window}_corr",
                "std_1",
                "std_2",
                "std_minus_1",
                "std_minus_2",
                "mean",
            ],
        ).legend(loc="lower left")

    def get_r2_data(self, betas_list, ticker):
        list_df = []
        for item in betas_list:
            df = self.get_price_data(item)
            list_df.append(df)

        df_long = pd.concat(list_df, axis=0).reset_index()
        df_wide = df_long.pivot_table(
            index=["index"], columns="ticker", values="pct_change"
        )

        df_sq = self.get_price_data(ticker).reset_index()

        dataset = pd.merge(
            df_sq, df_wide, how="inner", left_on="index", right_on="index"
        )

        beta_lens = [60, 90, 252]  # window for betas
        dict_lens = {}
        for beta_len in beta_lens:
            betas = {}
            ns = {}
            r2s = {}
            for col in dataset.columns:
                if col in betas_list:
                    df = dataset[["pct_change", col]].dropna()
                    beta, r2, n = self.get_betas(
                        df["pct_change"], df[col], beta_len, col
                    )
                    betas[col] = beta
                    ns[col] = n
                    r2s[col] = r2
            dict_lens[beta_len] = [betas, ns, r2s]

        df1 = pd.DataFrame.from_dict(dict_lens[60][2], orient="index").reset_index()
        df1.columns = ["TICKER", "R2_60"]
        df2 = pd.DataFrame.from_dict(dict_lens[90][2], orient="index").reset_index()
        df2.columns = ["TICKER", "R2_90"]
        df3 = pd.DataFrame.from_dict(dict_lens[252][2], orient="index").reset_index()
        df3.columns = ["TICKER", "R2_252"]
        df = pd.merge(df1, df2)
        df = pd.merge(df, df3)
        df.sort_values("R2_60")
        self.r2_data = df

    def plot_r2_data(self, ticker):
        self.r2_data.set_index("TICKER")[["R2_60", "R2_90", "R2_252"]].plot(
            title=f"{ticker} Relative R2", kind="bar"
        )

    def get_beta_data(self, stock_list):
        self.stock_list = stock_list
        list_df = []
        for item in stock_list:
            df = self.get_price_data(item)
            list_df.append(df)
        self.df_long = pd.concat(list_df, axis=0).reset_index()
        self.df_wide = self.df_long.pivot_table(
            index=["index"], columns="ticker", values="pct_change"
        )

        self.spy = self.get_price_data("SPY").reset_index()
        self.dataset = pd.merge(
            self.spy, self.df_wide, how="inner", left_on="index", right_on="index"
        )
        betas = {}
        ns = {}
        r2s = {}
        for col in self.dataset.columns:
            if col in self.stock_list:
                df = self.dataset[["pct_change", col]].dropna()
                beta, r2, n = self.get_betas(df["pct_change"], df[col])
                betas[col] = beta
                ns[col] = n
                r2s[col] = r2

        self.betas_dict = {
            k: v for k, v in sorted(betas.items(), key=lambda item: item[1])
        }
        self.dataset_stocks = self.dataset.drop(
            columns=["adjclose", "pct_change", "log_return", "ticker"]
        )
        self.portfolio_return = self.get_portfolio_return(
            self.dataset_stocks, self.stock_list, n=252
        )
        self.spy_return = self.get_portfolio_return(
            self.dataset[["pct_change"]], "pct_change", n=252
        )
        self.spy_return = self.spy_return.drop(
            columns=["pct_change", "pct_change_cmltv_ret"]
        )
        self.spy_return.columns = ["SPY_return"]

    def plot_relative_return(self):
        self.df_plot = pd.concat([self.portfolio_return, self.spy_return], axis=1)
        self.df_plot.set_index("index")[
            ["portfolio_return", "SPY_return", "SQ_cmltv_ret"]
        ].plot(title="Cumulative Return")

    def get_price_data(self, ticker):
        price_data = si.get_data(ticker, start_date="01/01/2009")
        df = pd.DataFrame(price_data)
        df = df[["adjclose"]]
        df["pct_change"] = df.adjclose.pct_change()
        df["log_return"] = np.log(1 + df["pct_change"].astype(float))
        df["ticker"] = ticker
        return df

    def get_portfolio_return(self, df, list_stocks, n):
        df = df.iloc[-n:,].copy()
        for col in df.columns:
            if col in list_stocks:
                df[col + "_cmltv_ret"] = np.exp(np.log1p(df[col]).cumsum()) - 1
        list_cols = []
        for col in df.columns:
            if "cmltv" in col and "SPY" not in col:
                list_cols.append(col)
        df["portfolio_return"] = df[list_cols].mean(axis=1)
        return df

    def get_betas(self, x, y, n=0, col=""):
        if n > 0:
            x = x.iloc[-n:,]
            y = y.iloc[-n:,]
        res = sm.OLS(y, x).fit()
        ticker = col.split("_")[0]
        beta = res.params[0]
        r2 = res.rsquared
        n = len(x)
        return [beta, r2, n]
