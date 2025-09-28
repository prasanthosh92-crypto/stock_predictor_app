"""
Feature Engineering Module
Handles data cleaning, preprocessing, and feature creation
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        1. Drop duplicates
        2. Forward-fill missing values
        3. Drop any remaining NaNs
        """
        df = data.copy()
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        df = df.ffill().bfill()
        df = df.dropna()
        self.logger.info("Data cleaned: %d rows remain", len(df))
        return df

    def create_lag_features(self, data: pd.DataFrame, lags: Tuple[int] = (1, 3, 5, 10)) -> pd.DataFrame:
        """
        Create lag features for Close and Volume
        """
        df = data.copy()
        for lag in lags:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        df = df.dropna()
        self.logger.info("Lag features created: %s", lags)
        return df

    def create_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily and multi-day return (%) features
        """
        df = data.copy()
        df['Return_1d'] = df['Close'].pct_change(periods=1)
        df['Return_5d'] = df['Close'].pct_change(periods=5)
        df = df.dropna()
        self.logger.info("Return features created")
        return df

    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling volatility features (standard deviation)
        """
        df = data.copy()
        df['Volatility_10d'] = df['Close'].rolling(window=10).std()
        df['Volatility_30d'] = df['Close'].rolling(window=30).std()
        df = df.dropna()
        self.logger.info("Volatility features created")
        return df

    def assemble_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all steps:
        1. Clean 
        2. Lag 
        3. Returns 
        4. Volatility 
        5. Merge technical indicators already in data
        """
        df = self.clean_data(data)
        df1 = self.create_lag_features(df)
        df2 = self.create_return_features(df1)
        df3 = self.create_volatility_features(df2)
        # Assumes technical indicators already present
        final = df3.dropna()
        self.logger.info("Final feature set assembled: %d rows, %d cols", final.shape[0], final.shape[1])
        return final

