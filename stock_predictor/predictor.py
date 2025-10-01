"""
Stock Predictor Module
Implements ML models for stock price forecasting
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class StockPredictor:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        if model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "linear_regression":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.is_trained = False

    def _prepare(self, data: pd.DataFrame, target_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        df = data.copy()
        # Feature columns: price & volume
        features = ["Open","High","Low","Close","Volume"]
        X = df[features].shift(0)
        y = df["Close"].shift(-target_days)
        # Drop rows with NaN
        mask = X.notnull().all(axis=1) & y.notnull()
        return X[mask], y[mask]

    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        X, y = self._prepare(data)
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        # Fit
        self.model.fit(X_train_s, y_train)
        self.is_trained = True
        # Predict
        train_pred = self.model.predict(X_train_s)
        test_pred  = self.model.predict(X_test_s)
        # Metrics
        return {
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
        }

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X, _ = self._prepare(data)
        last = X.iloc[-1:]
        last_s = self.scaler.transform(last)
        pred = self.model.predict(last_s)[0]
        current = data["Close"].iloc[-1]
        change = pred - current
        pct = (change / current) * 100
        return {
            "predicted_price": pred,
            "current_price": current,
            "price_change": change,
            "price_change_pct": pct,
        }

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        self.logger.info("Model saved to %s", path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        self.is_trained = True
        self.logger.info("Model loaded from %s", path)
