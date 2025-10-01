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

# News API (register for free at newsapi.org)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY_HERE")
if NEWS_API_KEY != "YOUR_NEWSAPI_KEY_HERE":
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
else:
    newsapi = None

# Auto-refresh interval (60 seconds)
AUTORELOAD_INTERVAL = 60_000

# Streamlit page config
st.set_page_config(
    page_title="📈 Advanced Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .price-positive {
        color: #4CAF50;
    }
    .price-negative {
        color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh the page for live data
count = st_autorefresh(interval=AUTORELOAD_INTERVAL, key="price_refresh")

# Main title
st.markdown('<h1 class="main-header">📈 Advanced Stock Forecasting Tool</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Predefined global watchlist
watchlist = {
    "🇺🇸 US": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"],
    "🇪🇺 Europe": ["NESN.SW", "SAP.DE", "RDSA.AS", "ASML.AS", "SIE.DE", "OR.PA"],
    "🌏 Asia": ["0700.HK", "7203.T", "BABA", "TSM", "005930.KS", "6758.T"]
}

# Region and symbol selection
region = st.sidebar.selectbox("📍 Select Region", list(watchlist.keys()))
symbols = watchlist[region]

# Multi-select for watchlist
selected_symbols = st.sidebar.multiselect(
    "📋 Your Watchlist",
    options=symbols,
    default=symbols[:3],
    help="Select stocks to monitor"
)

# Display live prices for selected symbols
if selected_symbols:
    st.sidebar.subheader("💰 Live Prices")
    
    price_data = []
    for sym in selected_symbols:
        try:
            collector = StockDataCollector(db_path=str(DB_FILE))
            price_df = collector.get_stock_data(sym, period="5d")
            if not price_df.empty:
                current_price = price_df["Close"].iloc[-1]
                prev_price = price_df["Close"].iloc[-2] if len(price_df) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                
                price_data.append({
                    "Symbol": sym,
                    "Price": f"${current_price:.2f}",
                    "Change": f"{change:+.2f}",
                    "Change %": f"{change_pct:+.1f}%"
                })
            else:
                price_data.append({
                    "Symbol": sym,
                    "Price": "N/A",
                    "Change": "N/A",
                    "Change %": "N/A"
                })
        except Exception as e:
            logging.error(f"Error fetching price for {sym}: {e}")
            price_data.append({
                "Symbol": sym,
                "Price": "Error",
                "Change": "N/A",
                "Change %": "N/A"
            })
    
    # Display price table
    st.sidebar.dataframe(price_data, use_container_width=True)

# Analysis section
st.sidebar.subheader("🔍 Analysis")

# Stock symbol for detailed analysis
symbol = st.sidebar.text_input("Stock Symbol for Analysis", value="AAPL", help="Enter symbol for detailed analysis")

# Time period and model selection
period = st.sidebar.selectbox("📅 Historical Period", ["1y", "2y", "5y"], index=0)
model_type = st.sidebar.selectbox("🤖 Prediction Model", ["random_forest", "linear_regression"])

# Run analysis button
run_button = st.sidebar.button("🔍 Run Detailed Analysis", type="primary")

# Main content area
if run_button and symbol:
    logging.info(f"Starting detailed analysis: {symbol} ({model_type}, {period})")
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Data collection
        status_text.text("📊 Collecting stock data...")
        progress_bar.progress(25)
        
        collector = StockDataCollector(db_path=str(DB_FILE))
        df_raw = collector.get_stock_data(symbol.upper(), period=period)
        
        if df_raw.empty:
            raise ValueError("No data retrieved")
        
        logging.info(f"Collected {len(df_raw)} rows for {symbol}")
        
    except Exception as e:
        logging.error(f"Data collection failed for {symbol}: {e}")
        st.error("❌ Error fetching data. Check symbol or try again later.")
        st.stop()

    try:
        # Feature engineering
        status_text.text("⚙️ Engineering features...")
        progress_bar.progress(50)
        
        fe = FeatureEngineer()
        df_feat = fe.assemble_features(df_raw)
        
        if df_feat.empty:
            raise ValueError("Feature engineering produced no data")
        
        logging.info(f"Engineered features: {df_feat.shape}")
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        st.error("❌ Error processing features. Try a longer time period.")
        st.stop()

    try:
        # Model training and prediction
        status_text.text("🤖 Training model and predicting...")
        progress_bar.progress(75)
        
        predictor = StockPredictor(model_type=model_type)
        metrics = predictor.train(df_feat)
        pred = predictor.predict(df_feat)
        
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

    # Results section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock information
        st.subheader(f"📊 {symbol.upper()} Analysis")
        
        # Current stock metrics
        current_price = df_raw['Close'].iloc[-1]
        prev_price = df_raw['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}")
        with metric_col2:
            st.metric("Change %", f"{price_change_pct:+.2f}%")
        with metric_col3:
            st.metric("Volume", f"{df_raw['Volume'].iloc[-1]:,}")
        with metric_col4:
            st.metric("Data Points", len(df_raw))

        # Price chart with prediction
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df_raw.index,
            open=df_raw.Open,
            high=df_raw.High,
            low=df_raw.Low,
            close=df_raw.Close,
            name="Price History"
        ))
        
        # Prediction marker
        fig.add_trace(go.Scatter(
            x=[df_raw.index[-1]],
            y=[pred["predicted_price"]],
            mode="markers+text",
            marker=dict(size=15, color="red", symbol="star"),
            text=[f"Predicted: ${pred['predicted_price']:.2f}"],
            textposition="top center",
            name="Prediction"
        ))
        
        fig.update_layout(
            title=f"{symbol.upper()} Stock Price & Prediction",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Model performance
        st.subheader("🤖 Model Performance")
        st.metric("Test R² Score", f"{metrics['test_r2']:.3f}")
        st.metric("Test MAE", f"${metrics['test_mae']:.2f}")
        st.metric("Test MSE", f"${metrics['test_mse']:.2f}")
        
        # Prediction details
        st.subheader("🔮 Prediction")
        st.metric("Predicted Price", f"${pred['predicted_price']:.2f}")
        st.metric("Expected Change", f"${pred['price_change']:+.2f}")
        st.metric("Expected Change %", f"{pred['price_change_pct']:+.2f}%")
        
        # Signal interpretation
        if pred['price_change_pct'] > 2:
            st.success("🚀 **Bullish Signal**")
        elif pred['price_change_pct'] < -2:
            st.error("📉 **Bearish Signal**")
        else:
            st.info("➡️ **Neutral Signal**")

    # News section
    st.subheader(f"📰 Latest News: {symbol.upper()}")
    
    if newsapi:
        try:
            # Fetch company news
            articles = newsapi.get_everything(
                q=symbol,
                language="en",
                sort_by="publishedAt",
                page_size=5
            )["articles"]
            
            if articles:
                for i, article in enumerate(articles):
                    with st.expander(f"📄 {article['title'][:80]}..."):
                        published = datetime.datetime.fromisoformat(
                            article["publishedAt"].rstrip("Z")
                        )
                        st.write(f"**Published:** {published:%Y-%m-%d %H:%M}")
                        st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                        st.write(article.get("description", "No description available"))
                        st.markdown(f"[Read full article]({article['url']})")
            else:
                st.info("No recent news found for this symbol.")
                
        except Exception as e:
            logging.error(f"News fetch error for {symbol}: {e}")
            st.warning("Unable to fetch news at this time.")
    else:
        st.info("News feature requires NEWS_API_KEY environment variable.")

    logging.info(f"Analysis completed successfully for {symbol}")

else:
    # Welcome screen
    st.info("👆 Select stocks from your watchlist in the sidebar, then click **'Run Detailed Analysis'** for predictions!")
    
    # Feature overview
    st.subheader("🚀 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📋 Watchlist**
        - Global stock monitoring
        - Live price updates
        - Multi-region support
        - Auto-refresh every minute
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI Predictions**
        - Random Forest & Linear models
        - Feature engineering
        - Performance metrics
        - Signal interpretation
        """)
    
    with col3:
        st.markdown("""
        **📰 Market News**
        - Latest company news
        - Real-time updates
        - Multiple sources
        - Integrated analysis
        """)

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This tool is for educational purposes only. Not financial advice.")
