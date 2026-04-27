import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="BTC Predictor Research",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_parquet("crypto_dataset/btc_global_index.parquet")
    df["bucket"] = pd.to_datetime(df["bucket"])
    return df

# =====================================================
# APP
# =====================================================

st.title("BTC Predictor Research Dashboard")

st.markdown("""
Analyse der Bitcoin Preisbewegungen für ein 5-Minuten Richtungsmodell.
""")

df = load_data()

# =====================================================
# FILTER
# =====================================================

sample = st.sidebar.selectbox("Sample", ["1 day", "7 days", "30 days"])

if sample == "1 day":
    cutoff = df["bucket"].max() - pd.Timedelta(days=1)
elif sample == "7 days":
    cutoff = df["bucket"].max() - pd.Timedelta(days=7)
else:
    cutoff = df["bucket"].max() - pd.Timedelta(days=30)

df = df[df["bucket"] >= cutoff].copy()

# =====================================================
# FEATURES
# =====================================================

df = df.sort_values("bucket")

df["return_1s"] = df["global_vwap"].pct_change()
df["rolling_vol"] = df["return_1s"].rolling(60).std()
df["rolling_momentum"] = df["global_vwap"].pct_change(30)

# =====================================================
# CHART 1 VWAP
# =====================================================

st.header("1. Global VWAP")
fig = px.line(df, x="bucket", y="global_vwap")
st.plotly_chart(fig, use_container_width=True)

st.markdown("VWAP zeigt den volumengewichteten Bitcoin Preis über alle Exchanges.")

# =====================================================
# CHART 2 MOMENTUM
# =====================================================

st.header("2. Momentum")
fig = px.line(df, x="bucket", y="rolling_momentum")
st.plotly_chart(fig, use_container_width=True)

st.markdown("Momentum zeigt kurzfristige Trendbewegungen.")

# =====================================================
# CHART 3 VOLATILITY
# =====================================================

st.header("3. Volatility")
fig = px.line(df, x="bucket", y="rolling_vol")
st.plotly_chart(fig, use_container_width=True)

st.markdown("Volatilität zeigt Unsicherheit im Markt.")

# =====================================================
# CHART 4 RETURNS
# =====================================================

st.header("4. Return Distribution")
fig = px.histogram(df, x="return_1s", nbins=200)
st.plotly_chart(fig, use_container_width=True)

st.markdown("Verteilung der kurzfristigen Preisbewegungen.")

# =====================================================
# CHART 5 FIXED PROBABILITY (NO INTERVAL BUG)
# =====================================================

st.header("5. Momentum vs Future Direction Probability")

future = df["global_vwap"].shift(-300) / df["global_vwap"] - 1

df["future_up"] = (future > 0).astype(int)

# FIX: no pd.qcut intervals used for plotting

df["momentum_bin"] = pd.cut(
    df["rolling_momentum"].fillna(0),
    bins=10
).astype(str)

prob = df.groupby("momentum_bin")["future_up"].mean().reset_index()
prob.columns = ["momentum_bin", "prob_up"]

fig = px.bar(prob, x="momentum_bin", y="prob_up")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
Zeigt die Wahrscheinlichkeit, dass der Preis in 5 Minuten höher ist,
abhängig vom aktuellen Momentum.
""")

# =====================================================
# CHART 6
# =====================================================

st.header("6. Intraday Volatility Pattern")

df["hour"] = df["bucket"].dt.hour
hourly = df.groupby("hour")["rolling_vol"].mean().reset_index()

fig = px.line(hourly, x="hour", y="rolling_vol")
st.plotly_chart(fig, use_container_width=True)

st.markdown("Zeigt Tageszeitabhängigkeit der Volatilität.")

# =====================================================
# SUMMARY
# =====================================================

st.header("Summary")
st.markdown("""
Wichtige Fragen:
- Gibt es Momentum?
- Gibt es Mean Reversion?
- Ist Volatilität regimeabhängig?

Dieses Dashboard ist die Grundlage für dein 5-Minuten Prognosemodell.
""")

