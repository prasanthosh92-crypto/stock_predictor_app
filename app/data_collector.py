"""
Stock Data Collector Module
Handles data collection from yfinance with caching in SQLite
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Optional

class StockDataCollector:
    def __init__(self, db_path: str = "../data/models/stocks.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._setup_database()

    def _setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
        """)
        conn.commit()
        conn.close()
        self.logger.info("Database initialized at %s", self.db_path)

    def get_stock_data(self, symbol: str, period: str = "1y", use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch stock data with optional caching.
        """
        try:
            if use_cache:
                cached = self._get_cached_data(symbol, period)
                if cached is not None and not cached.empty:
                    self.logger.info("Using cached data for %s", symbol)
                    return cached

            self.logger.info("Fetching %s data for %s", period, symbol)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                raise ValueError(f"No data for {symbol}")

            self._cache_data(symbol, data)
            return data

        except Exception as e:
            self.logger.error("Error fetching data for %s: %s", symbol, e)
            return pd.DataFrame()

    def _get_cached_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        conn = sqlite3.connect(self.db_path)
        end = datetime.now()
        days = {"1y": 365, "2y": 730, "5y": 1825}.get(period, 365)
        start = end - timedelta(days=days)
        query = """
            SELECT date, open, high, low, close, volume
            FROM stock_data
            WHERE symbol=? AND date BETWEEN ? AND ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn,
                               params=[symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")])
        conn.close()
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def _cache_data(self, symbol: str, data: pd.DataFrame):
        conn = sqlite3.connect(self.db_path)
        to_insert = [
            (symbol, dt.strftime("%Y-%m-%d"), float(row["Open"]), float(row["High"]),
             float(row["Low"]), float(row["Close"]), int(row["Volume"]))
            for dt, row in data.iterrows()
        ]
        conn.executemany("""
            INSERT OR REPLACE INTO stock_data
            (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, to_insert)
        conn.commit()
        conn.close()
        self.logger.info("Cached %d records for %s", len(to_insert), symbol)

    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Moving Averages
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        # EMA
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()
        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        # Bollinger Bands
        df["BB_MID"] = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["BB_UP"] = df["BB_MID"] + 2 * std
        df["BB_LOW"] = df["BB_MID"] - 2 * std
        return df

