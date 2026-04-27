import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    layout="wide",
    page_title="BTC Microstructure Predictor"
)

# =====================================================
# TITLE
# =====================================================

st.title("BTC Microstructure Predictor")

st.markdown("""
## Projektidee

Dieses Projekt untersucht kurzfristige Bitcoin-Marktbewegungen
auf Basis hochfrequenter Marktdaten.

Ziel ist die Schätzung der Wahrscheinlichkeit:

> Wird der BTC-Preis in 5 Minuten höher oder niedriger sein
> als jetzt?

Die Analyse basiert auf:

- VWAP-Dynamik
- Momentum
- kurzfristiger Volatilität
- Mikrostruktur-Signalen

Die Daten stammen von Binance Tickdaten
und werden auf 1-Sekunden-Bars aggregiert.
""")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_parquet(
        "crypto_dataset/binance_btc_1s_30d.parquet"
    )

    return df


df = load_data()

# =====================================================
# DOWNLOAD
# =====================================================

st.header("Dataset Download")

with open(
    "crypto_dataset/binance_btc_1s_30d.parquet",
    "rb"
) as f:

    st.download_button(
        label="Download Parquet Dataset",
        data=f,
        file_name="binance_btc_1s_30d.parquet",
        mime="application/octet-stream"
    )

st.markdown("""
Das Dataset enthält:

- 1s VWAP
- OHLC
- Volumen
- Momentum Features
- Volatilitätsfeatures
- Future Return Labels
""")

# =====================================================
# PRICE
# =====================================================

st.header("1. BTC VWAP")

fig = px.line(
    df.tail(20000),
    x="bucket",
    y="vwap"
)

fig.update_layout(
    xaxis_title="Zeit",
    yaxis_title="BTC Preis (USD)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Interpretation

Der VWAP zeigt den volumen-gewichteten Marktpreis.

Dieser Preis ist robuster als einzelne Trades.
""")

# =====================================================
# MOMENTUM
# =====================================================

st.header("2. 30s Momentum")

fig = px.line(
    df.tail(20000),
    x="bucket",
    y="momentum_30s"
)

fig.update_layout(
    xaxis_title="Zeit",
    yaxis_title="30s Return"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Interpretation

Positives Momentum deutet auf kurzfristigen Trenddruck hin.
""")

# =====================================================
# VOLATILITY
# =====================================================

st.header("3. 60s Volatility")

fig = px.line(
    df.tail(20000),
    x="bucket",
    y="volatility_60s"
)

fig.update_layout(
    xaxis_title="Zeit",
    yaxis_title="Rolling Volatility"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Interpretation

Volatilität misst kurzfristige Unsicherheit
im Markt.
""")

# =====================================================
# PROBABILITY ANALYSIS
# =====================================================

st.header("4. Momentum vs Future Direction")

tmp = df.copy()

tmp["momentum_bucket"] = pd.qcut(
    tmp["momentum_30s"],
    20,
    duplicates="drop"
)

prob = (
    tmp.groupby("momentum_bucket")
    ["future_up_5m"]
    .mean()
    .reset_index()
)

prob["bucket_str"] = (
    prob["momentum_bucket"]
    .astype(str)
)

fig = px.bar(
    prob,
    x="bucket_str",
    y="future_up_5m"
)

fig.update_layout(
    xaxis_title="Momentum Quantile",
    yaxis_title="P(price up in 5m)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Interpretation

Dieser Chart zeigt:

Wie verändert aktuelles Momentum
die Wahrscheinlichkeit eines zukünftigen
Preisanstiegs?
""")

# =====================================================
# FUTURE RETURN DISTRIBUTION
# =====================================================

st.header("5. Future Return Distribution")

fig = px.histogram(
    df,
    x="future_return_5m",
    nbins=100
)

fig.update_layout(
    xaxis_title="5m Future Return",
    yaxis_title="Häufigkeit"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Interpretation

Die Verteilung zeigt:

Wie stark sich Bitcoin typischerweise
innerhalb von 5 Minuten bewegt.
""")

# =====================================================
# VOLUME
# =====================================================

st.header("6. Volume")

fig = px.line(
    df.tail(20000),
    x="bucket",
    y="volume"
)

fig.update_layout(
    xaxis_title="Zeit",
    yaxis_title="Volume"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("""
### Interpretation

Hohes Volumen deutet häufig auf
Informationsereignisse oder Momentumphasen hin.
""")

# =====================================================
# SUMMARY
# =====================================================

st.header("Summary")

st.markdown("""
Nächste Entwicklungsschritte:

- probabilistische ML-Modelle
- Live-Predictor
- Echtzeit-Signalengine
- Regime Detection
- Orderflow Features
""")
