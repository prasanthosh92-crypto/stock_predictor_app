import sys
import os
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_predictor.data_collector import StockDataCollector

def test_imports():
    """Test that all required modules can be imported"""
    import yfinance
    import pandas
    import sqlite3
    # If we reach here without error, imports are successful

def test_basic_data_collection():
    """Test basic data collection without the full class"""
    import yfinance as yf
    import pandas as pd

    # Simple test - fetch 5 days of AAPL data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="5d")

    assert not data.empty, "Should fetch some data"
    assert "Close" in data.columns, "Should have Close column"

def test_data_collector_init():
    """Test that StockDataCollector can be instantiated"""
    collector = StockDataCollector(db_path=":memory:")  # Use in-memory DB for testing
    assert collector is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
