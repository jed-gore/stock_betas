import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm


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
        )

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
        df = df.iloc[
            -n:,
        ].copy()
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
            x = x.iloc[
                -n:,
            ]
            y = y.iloc[
                -n:,
            ]
        res = sm.OLS(y, x).fit()
        ticker = col.split("_")[0]
        beta = res.params[0]
        r2 = res.rsquared
        n = len(x)
        return [beta, r2, n]
