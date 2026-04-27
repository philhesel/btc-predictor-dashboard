import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# PAGE
# =====================================================

st.set_page_config(layout="wide")

st.title("BTC Predictor Dashboard")

st.markdown("""
Dieses Dashboard analysiert kurzfristige Bitcoin Preisbewegungen.
""")

# =====================================================
# FAKE MARKET DATA
# =====================================================

np.random.seed(42)

n = 5000

timestamps = pd.date_range(
    end=pd.Timestamp.now(),
    periods=n,
    freq="1s"
)

returns = np.random.normal(
    0,
    0.0002,
    n
)

prices = 100000 * (1 + returns).cumprod()

df = pd.DataFrame({
    "timestamp": timestamps,
    "price": prices
})

# =====================================================
# FEATURES
# =====================================================

df["return_1s"] = df["price"].pct_change()

df["momentum_30s"] = (
    df["price"].pct_change(30)
)

df["volatility_60s"] = (
    df["return_1s"]
    .rolling(60)
    .std()
)

future_return = (
    df["price"].shift(-300)
    / df["price"]
    - 1
)

df["future_up"] = (
    future_return > 0
).astype(int)

# =====================================================
# PRICE CHART
# =====================================================

st.header("1. BTC Price")

fig = px.line(
    df,
    x="timestamp",
    y="price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Erklärung

Dieser Chart zeigt die simulierte Preisentwicklung.

Später wird hier dein echter globaler VWAP stehen.
""")

# =====================================================
# MOMENTUM
# =====================================================

st.header("2. Momentum")

fig = px.line(
    df,
    x="timestamp",
    y="momentum_30s"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Erklärung

Momentum misst kurzfristige Trendbewegungen.

Positive Werte bedeuten Aufwärtsbewegung.
""")

# =====================================================
# VOLATILITY
# =====================================================

st.header("3. Volatility")

fig = px.line(
    df,
    x="timestamp",
    y="volatility_60s"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Erklärung

Volatilität misst kurzfristige Unsicherheit.
""")

# =====================================================
# PROBABILITY ANALYSIS
# =====================================================

st.header("4. Probability of Future Up Move")

df["momentum_bucket"] = pd.cut(
    df["momentum_30s"].fillna(0),
    bins=10
).astype(str)

prob = (
    df.groupby("momentum_bucket")["future_up"]
    .mean()
    .reset_index()
)

fig = px.bar(
    prob,
    x="momentum_bucket",
    y="future_up"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Erklärung

Dieser Chart zeigt:

Wie verändert Momentum die Wahrscheinlichkeit,
dass der Preis in 5 Minuten höher ist?
""")

# =====================================================
# SUMMARY
# =====================================================

st.header("Summary")

st.markdown("""
Nächster Schritt:

- echte Daten einladen
- Features erweitern
- probabilistischen Predictor bauen
""")
