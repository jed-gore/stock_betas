from example import db
from example import Company, Portfolio, DaloopaCompany  # , DaloopaData
import pandas as pd

import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

import sqlite3
from sqlite3 import Error

APIKEY = "MgcAvxZBUpXYbi5t0RIpFfnruGYrv4kDTvPdxK7_w7DpozIvQpQ8EQ"

df = pd.read_csv("reference.csv")
focus_list = ["AAPL", "BAC", "NVDA", "TSLA", "WMT"]
df = df.loc[df["symbol"].isin(focus_list)]


# for ticker in focus_list:
#     endpoint = "https://www.daloopa.com"
#     TICKER = ticker
#     response = requests.get(
#         f"{endpoint}/api/v1/export/{TICKER}",
#         auth=HTTPBasicAuth("jedgore1@gmail.com", APIKEY),
#         stream=True,
#     )
#     df = pd.read_csv(response.raw)
#     file_name = f"data/{ticker}_daloopa.csv"
#     df.to_csv(file_name)


for ticker in focus_list:
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey=0JJNLUHA7KK0CIM7"
    r = requests.get(url)
    data = r.json()
    data = pd.DataFrame(data["quarterlyEarnings"])
    data["ticker"] = ticker
    file_name = f"data/{ticker}_earnings.csv"
    data.to_csv(file_name)


db.create_all()

##############

portfolio = Portfolio(name="meta_portfolio")
db.session.add_all([portfolio])
db.session.commit()

companies = []
daloopas = []
for i, row in df.head().iterrows():
    ticker = row["symbol"]
    name = row["name"]
    exchange = row["exchange"]
    daloopa_mapping = DaloopaCompany(daloopa_ticker=ticker)
    company = Company(
        ticker=ticker,
        exchange=exchange,
        name=name,
        portfolio=portfolio,
        daloopa_company=daloopa_mapping,
    )
    companies.append(company)
db.session.add_all(companies)
db.session.commit()

################

import requests


conn = sqlite3.connect("instance/database.db")
c = conn.cursor()

id_len = 0
for ticker in focus_list:
    print(ticker)
    sqlite_table = "daloopa_data"

    file_name = f"data/{ticker}_daloopa.csv"
    df = pd.read_csv(file_name, index_col=0)
    df.drop(columns=["id"], inplace=True)
    df["id"] = range(id_len, id_len + len(df))
    id_len = id_len + len(df)
    df.to_sql(name=sqlite_table, con=conn, if_exists="append", index=False)

    conn.commit()

    sqlite_table = "earnings_data"
    file_name = f"data/{ticker}_earnings.csv"
    df = pd.read_csv(file_name, index_col=0)
    df.to_sql(name=sqlite_table, con=conn, if_exists="append", index=False)

    conn.commit()
