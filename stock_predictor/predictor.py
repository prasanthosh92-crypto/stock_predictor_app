import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class StockPredictor:
    def __init__(self, model_type="random_forest", lstm_seq_len=10):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False
        self.lstm_seq_len = lstm_seq_len
        if model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "linear_regression":
            self.model = LinearRegression()
        elif model_type == "lstm":
            self.model = None
        else:
            raise ValueError("Unsupported model type")

    def _prepare(self, df, target_days=1):
        features = ["Open", "High", "Low", "Close", "Volume"]
        if self.model_type == "lstm":
            arr = df[features].values
            closes = df["Close"].values
            X, y = [], []
            for i in range(len(df) - self.lstm_seq_len - target_days + 1):
                X.append(arr[i:i+self.lstm_seq_len])
                y.append(closes[i+self.lstm_seq_len+target_days-1])
            X = np.array(X)
            y = np.array(y)
            if len(X) == 0:
                return np.empty((0, self.lstm_seq_len, len(features))), np.empty((0,))
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped).reshape(X.shape)
            return X_scaled, y
        else:
            X = df[features]
            y = df["Close"].shift(-target_days)
            mask = X.notnull().all(axis=1) & y.notnull()
            return X[mask], y[mask]

    def train(self, df, test_size=0.2, target_days=1):
        X, y = self._prepare(df, target_days=target_days)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Not enough data for training")
        if self.model_type == "lstm":
            split = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            model = Sequential([
                LSTM(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            model.fit(X_train, y_train, epochs=25, batch_size=8, verbose=0, validation_data=(X_test, y_test))
            self.model = model
            self.is_trained = True
            train_pred = model.predict(X_train).flatten()
            test_pred = model.predict(X_test).flatten()
        else:
            split = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)
            self.model.fit(X_train_s, y_train)
            self.is_trained = True
            train_pred = self.model.predict(X_train_s)
            test_pred = self.model.predict(X_test_s)
        return {
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
        }

    def predict(self, df, target_days=1):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        features = ["Open", "High", "Low", "Close", "Volume"]
        current = df["Close"].iloc[-1]
        if self.model_type == "lstm":
            arr = df[features].values
            if len(arr) < self.lstm_seq_len:
                raise ValueError("Not enough data for LSTM prediction")
            last_seq = arr[-self.lstm_seq_len:]
            last_seq_scaled = self.scaler.transform(last_seq)
            last_seq_scaled = np.expand_dims(last_seq_scaled, axis=0)
            pred = self.model.predict(last_seq_scaled)[0][0]
        else:
            X, _ = self._prepare(df, target_days=target_days)
            last = X.iloc[-1:]
            last_s = self.scaler.transform(last)
            pred = self.model.predict(last_s)[0]
        change = pred - current
        pct = (change / current) * 100
        return {
            "predicted_price": float(pred),
            "current_price": float(current),
            "price_change": float(change),
            "price_change_pct": float(pct),
        }
