import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

st.title("BTC Predictor Dashboard")

st.write("Deployment funktioniert 🚀")

# Fake data
x = np.arange(100)

df = pd.DataFrame({
    "x": x,
    "y": np.cumsum(np.random.randn(100))
})

fig = px.line(df, x="x", y="y")

st.plotly_chart(fig, use_container_width=True)

st.write("Test chart loaded successfully.")import streamlit as st

st.title("BTC Dashboard")

st.write("It works 🚀")
