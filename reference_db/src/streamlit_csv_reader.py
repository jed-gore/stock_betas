import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Loading the data
def get_data(ticker):
    return pd.read_csv(os.path.join(os.getcwd(), "data/" + ticker + "_daloopa.csv"))


# configuration of the page
st.set_page_config(layout="wide")
# load dataframes
ticker = "AAPL"
df_dep = get_data(ticker)


st.title(ticker + " KPI Data Viz")
st.markdown(
    """
This app performs simple visualization from the open data
"""
)
# st.write(df_dep)

st.sidebar.header("Select KPI to display")
pol_parties = df_dep["title"].unique().tolist()
pol_party_selected = st.sidebar.selectbox("KPIs", pol_parties)

# creates masks from the sidebar selection widgets
mask_pol_par = df_dep["title"] == pol_party_selected

df_dep_filtered = df_dep[mask_pol_par]
st.write(df_dep_filtered)

chart_data = df_dep_filtered[["filing_date", "value_normalized"]]

st.line_chart(chart_data, x="filing_date", y="value_normalized")


def convert_df(df):
    return df.to_csv().encode("utf-8")


csv = convert_df(df_dep_filtered)


# adding a download button to download csv file

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=ticker + "_df.csv",
    mime="text/csv",
)
