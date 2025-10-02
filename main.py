import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stock_predictor.predictor import StockPredictor

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Personal Stock Forecasting Tool")

# Sidebar settings
model_type = st.sidebar.selectbox("Prediction Model", ["random_forest", "linear_regression", "lstm"], index=0)
if model_type == "lstm":
    lstm_seq_len = st.sidebar.slider("LSTM Sequence Length", min_value=5, max_value=30, value=10)
else:
    lstm_seq_len = None

symbol = st.sidebar.text_input("Stock Symbol (e.g. AAPL)", "AAPL")
period = st.sidebar.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Data Interval", ["1d", "1h"], index=0)
period_options = {"1 day": 1, "1 week": 7, "1 month": 30, "3 months": 90}
prediction_period_str = st.sidebar.selectbox("Predict How Far Ahead?", list(period_options.keys()), index=2)
target_days = period_options[prediction_period_str]

# Download and clean data
try:
    df = yf.download(symbol, period=period, interval=interval)
    df = df.dropna(subset=["Close"])  # Only use rows with valid Close
    if df.empty or len(df) < 50:
        st.error("❌ Not enough valid data. Try a longer period, wider interval, or different symbol.")
        st.stop()
    df.reset_index(drop=True, inplace=True)
except Exception as e:
    st.error(f"❌ Data download error: {e}")
    st.stop()

latest = df.iloc[-1]  # This should now always be valid

# Info panel (Yahoo-style)
st.subheader(f"{symbol.upper()} Price & Prediction")
info_cols = st.columns([3, 2])
with info_cols[0]:
    st.metric("Last Close", f"${latest['Close']:.2f}")
    st.metric("Open", f"${latest['Open']:.2f}")
    st.metric("Day High", f"${latest['High']:.2f}")
    st.metric("Day Low", f"${latest['Low']:.2f}")
    st.metric("Volume", f"{latest['Volume']:,}")
    st.metric("52-Week High", f"${df['High'].max():.2f}")
    st.metric("52-Week Low", f"${df['Low'].min():.2f}")

with info_cols[1]:
    st.metric("Change", f"${latest['Close'] - latest['Open']:.2f}", delta=f"{100*(latest['Close']-latest['Open'])/latest['Open']:.2f}%")
    st.metric("Average Volume", f"{int(df['Volume'].mean()):,}")

# Features for prediction
df_feat = df[["Open", "High", "Low", "Close", "Volume"]].copy()
if len(df_feat) < 50:
    st.error("❌ Not enough feature rows. Try a longer period or wider interval.")
    st.stop()

# Model prediction
try:
    if model_type == "lstm":
        predictor = StockPredictor(model_type="lstm", lstm_seq_len=lstm_seq_len)
    else:
        predictor = StockPredictor(model_type=model_type)
    metrics = predictor.train(df_feat, target_days=target_days)
    pred = predictor.predict(df_feat, target_days=target_days)
except Exception as e:
    st.error(f"❌ Model error: {e}")
    st.stop()

# Chart: D is last data day, prediction is D+N
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=pd.date_range(end=pd.Timestamp.today(), periods=len(df)),
    open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price History"
))
fig.add_trace(go.Scatter(
    x=[pd.Timestamp.today() + pd.Timedelta(days=target_days)],
    y=[pred["predicted_price"]],
    mode="markers+text",
    marker=dict(size=12, color="red"),
    text=[f"Predicted (D+{target_days}): ${pred['predicted_price']:.2f}"],
    textposition="top center",
    name="Prediction"
))
fig.update_layout(
    title=f"{symbol.upper()} Stock Price & Prediction",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

trend = "Bull" if pred["predicted_price"] > pred["current_price"] else "Bear"
st.info(f"**Trend prediction for {prediction_period_str}: {trend} market**")
st.success(f"**Current price (D):** ${pred['current_price']:.2f}")
st.success(f"**Predicted price (D+{target_days}):** ${pred['predicted_price']:.2f}")

with st.expander("Model Performance & Details"):
    st.write(f"**Train R2:** {metrics['train_r2']:.3f}")
    st.write(f"**Test R2:** {metrics['test_r2']:.3f}")
    st.write(f"**Train MAE:** {metrics['train_mae']:.2f}")
    st.write(f"**Test MAE:** {metrics['test_mae']:.2f}")
    st.write(f"**Train MSE:** {metrics['train_mse']:.2f}")
    st.write(f"**Test MSE:** {metrics['test_mse']:.2f}")
    st.write("**Model Used:**", model_type)
    st.write("**Forecast Window:**", prediction_period_str)
    st.write("**Sequence Length (LSTM):**", lstm_seq_len if model_type=="lstm" else "-")
