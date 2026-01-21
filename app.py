
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

st.set_page_config(page_title="Silicon Commodity Risk Dashboard", layout="wide")

@st.cache_data
def make_data():
    np.random.seed(0)
    dates = pd.date_range("2016-01-01", "2026-01-01", freq="D")
    price = 2000.0
    prices = []
    for _ in range(len(dates)):
        price *= (1 + np.random.normal(0.0002, 0.01))
        prices.append(price)
    df = pd.DataFrame({"Close": prices}, index=dates)
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df[["Open","Close"]].max(axis=1) * (1 + np.random.rand(len(df))*0.01)
    df["Low"] = df[["Open","Close"]].min(axis=1) * (1 - np.random.rand(len(df))*0.01)
    df["VIX_US"] = 15 + np.random.randn(len(df))*3
    df["DXY"] = 100 + np.random.randn(len(df))*2
    df["Gold"] = 1800 + np.random.randn(len(df))*10
    df["Silver"] = 22 + np.random.randn(len(df))*0.3
    df["Copper"] = 4 + np.random.randn(len(df))*0.05
    return df

df = make_data()

st.title("Silicon Commodity – Risk & Predictive Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Price Action", "Macro Correlation", "Prediction"
])

with tab1:
    st.metric("Latest Silicon Price", f"{df['Close'].iloc[-1]:.2f}")
    st.metric("US VIX", f"{df['VIX_US'].iloc[-1]:.2f}")
    st.metric("DXY", f"{df['DXY'].iloc[-1]:.2f}")
    st.write("This is a **fully offline demo** using synthetic data.")

with tab2:
    st.subheader("10-Year Daily Candlestick")
    fig, ax = plt.subplots(figsize=(10,4))
    dates_num = mdates.date2num(df.index.to_pydatetime())
    for d,o,h,l,c in zip(dates_num, df.Open, df.High, df.Low, df.Close):
        ax.plot([d,d],[l,h], linewidth=0.4)
        color = "green" if c>=o else "red"
        rect = Rectangle((d-0.3, min(o,c)), 0.6, abs(c-o)+0.01, color=color)
        ax.add_patch(rect)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    st.pyplot(fig)

with tab3:
    st.subheader("Rolling Correlation (90D)")
    corr = df["Close"].rolling(90).corr(df["Copper"])
    st.line_chart(corr)

with tab4:
    st.subheader("Simple Trend Prediction (30D)")
    df["ret_30"] = df["Close"].pct_change(30)
    trend = "UP" if df["ret_30"].iloc[-1] > 0 else "DOWN"
    st.write(f"Predicted Trend: **{trend}**")
