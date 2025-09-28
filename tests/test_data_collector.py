import sys
import os

# Add parent directory to path so we can import from stock_predictor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported"""
    import yfinance
    import pandas
    import sqlite3
    print("All imports successful")

def test_basic_data_collection():
    """Test basic data collection without the full class"""
    import yfinance as yf
    import pandas as pd
    
    # Simple test - fetch 5 days of AAPL data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="5d")
    
    assert not data.empty, "Should fetch some data"
    assert "Close" in data.columns, "Should have Close column"
    print(f"Fetched {len(data)} rows of AAPL data")
