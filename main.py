#!/usr/bin/env python3
"""
Enhanced Stock Forecasting Tool
Features: Watchlist, Live Prices, News, and Predictions
"""

import streamlit as st
import plotly.graph_objects as go
import logging
import os
import datetime
from pathlib import Path
from newsapi import NewsApiClient
from streamlit_autorefresh import st_autorefresh
from stock_predictor.data_collector import StockDataCollector
from stock_predictor.feature_engineer import FeatureEngineer
from stock_predictor.predictor import StockPredictor

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DB_DIR = PROJECT_ROOT / "data" / "models"
LOG_DIR = PROJECT_ROOT / "logs"
DB_FILE = DB_DIR / "stocks.db"
LOG_FILE = LOG_DIR / "app.log"

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="a"
)

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Personal Stock Forecasting Tool")

# Sidebar: Model selection
st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox("Prediction Model", ["random_forest", "linear_regression", "lstm"], index=0)
if model_type == "lstm":
    lstm_seq_len = st.sidebar.slider("LSTM Sequence Length", min_value=5, max_value=30, value=10)
else:
    lstm_seq_len = None

symbol = st.sidebar.text_input("Stock Symbol (e.g. AAPL)", "AAPL")
period = st.sidebar.selectbox("History Period", ["1mo","3mo","6mo","1y","2y"], index=2)
interval = st.sidebar.selectbox("Data Interval", ["1d","1h","30m"], index=0)

# NEW: Prediction period selector
period_options = {
    "1 day": 1,
    "1 week": 7,
    "1 month": 30,
    "3 months": 90,
    "6 months": 180,
    "1 year": 365
}
prediction_period_str = st.sidebar.selectbox("Predict How Far Ahead?", list(period_options.keys()), index=2)
target_days = period_options[prediction_period_str]

# UI columns
col1, col2 = st.columns([3,2])

with col1:
    st.subheader("📊 Price & Prediction")

with col2:
    st.subheader("📰 News")

status_text = st.empty()
progress_bar = st.progress(0)

# Data collection
try:
    status_text.text("🔄 Collecting data...")
    progress_bar.progress(10)
    dc = StockDataCollector()
    df_raw = dc.fetch(symbol, period, interval)
    if df_raw.empty:
        st.error("❌ No data available for this symbol/period!")
        st.stop()
    progress_bar.progress(30)
    status_text.text("🛠 Feature engineering...")
    fe = FeatureEngineer()
    df_feat = fe.assemble_features(df_raw)
    if df_feat.empty:
        raise ValueError("Feature engineering produced no data")
    logging.info(f"Engineered features: {df_feat.shape}")
    progress_bar.progress(50)
except Exception as e:
    logging.error(f"Feature engineering failed: {e}")
    st.error("❌ Error processing features. Try a longer time period.")
    st.stop()

try:
    # Model training and prediction
    status_text.text("🤖 Training model and predicting...")
    progress_bar.progress(75)
    if model_type == "lstm":
        predictor = StockPredictor(model_type=model_type, lstm_seq_len=lstm_seq_len)
    else:
        predictor = StockPredictor(model_type=model_type)
    metrics = predictor.train(df_feat, target_days=target_days)
    pred = predictor.predict(df_feat, target_days=target_days)
    logging.info(f"Model trained, test R2: {metrics['test_r2']:.3f}")
    progress_bar.progress(100)
    status_text.text("✅ Analysis completed!")
except Exception as e:
    logging.error(f"Model training failed: {e}")
    st.error("❌ Error training model. Please try again.")
    st.stop()

# Clear progress indicators
progress_bar.empty()
status_text.empty()

# Price chart with prediction
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_raw.index,
    open=df_raw.Open,
    high=df_raw.High,
    low=df_raw.Low,
    close=df_raw.Close,
    name="Price History"
))
fig.add_trace(go.Scatter(
    x=[df_raw.index[-1] + datetime.timedelta(days=target_days)],
    y=[pred["predicted_price"]],
    mode="markers+text",
    marker=dict(size=15, color="red", symbol="star"),
    text=[f"Predicted ({prediction_period_str}): ${pred['predicted_price']:.2f}"],
    textposition="top center",
    name="Prediction"
))
fig.update_layout(
    title=f"{symbol.upper()} Stock Price & Prediction",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500
)
col1.plotly_chart(fig, use_container_width=True)

# Bear/Bull Trend Analysis
trend = "Bull" if pred["predicted_price"] > pred["current_price"] else "Bear"
col1.write(f"**Trend prediction for {prediction_period_str}: {trend} market**")

with col2:
    st.subheader("🤖 Model Performance")
    st.write(f"**Train R2:** {metrics['train_r2']:.3f}")
    st.write(f"**Test R2:** {metrics['test_r2']:.3f}")
    st.write(f"**Train MAE:** {metrics['train_mae']:.2f}")
    st.write(f"**Test MAE:** {metrics['test_mae']:.2f}")
    st.write(f"**Train MSE:** {metrics['train_mse']:.2f}")
    st.write(f"**Test MSE:** {metrics['test_mse']:.2f}")

    # News section (optional, unchanged)
    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY", ""))
        news = newsapi.get_everything(q=symbol, language="en", sort_by="publishedAt", page_size=5)
        for article in news.get("articles", []):
            st.markdown(f"**[{article['title']}]({article['url']})**  \n{article['description']}")
    except Exception as e:
        st.warning("News not available.")
