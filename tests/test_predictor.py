import sys, os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stock_predictor.predictor import StockPredictor

@pytest.fixture
def simple_data():
    # 10 days linear upward trend
    dates = pd.date_range(end=datetime.today(), periods=10)
    prices = np.linspace(100, 110, 10)
    df = pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 0.5,
        "Low": prices - 1,
        "Close": prices,
        "Volume": np.linspace(1000, 2000, 10).astype(int)
    }, index=dates)
    return df

def test_predictor_init():
    sp = StockPredictor()
    assert sp.model_type == "random_forest"

def test_training_and_metrics(simple_data):
    sp = StockPredictor(model_type="linear_regression")
    metrics = sp.train(simple_data, test_size=0.2)
    # Check all keys exist
    for k in ["train_mse","test_mse","train_mae","test_mae","train_r2","test_r2"]:
        assert k in metrics
        assert isinstance(metrics[k], float)

def test_predict_before_training(simple_data):
    sp = StockPredictor()
    with pytest.raises(RuntimeError):
        sp.predict(simple_data)

def test_predict_after_training(simple_data):
    sp = StockPredictor()
    sp.train(simple_data, test_size=0.2)
    result = sp.predict(simple_data)
    # Validate expected keys
    for k in ["predicted_price","current_price","price_change","price_change_pct"]:
        assert k in result
        assert isinstance(result[k], (float, int))

def test_save_and_load(tmp_path, simple_data):
    sp = StockPredictor()
    sp.train(simple_data, test_size=0.2)
    fpath = tmp_path / "model.pkl"
    sp.save(str(fpath))
    sp2 = StockPredictor()
    sp2.load(str(fpath))
    assert sp2.is_trained
    res1 = sp.predict(simple_data)
    res2 = sp2.predict(simple_data)
    # Predictions should match after load
    assert res1["predicted_price"] == pytest.approx(res2["predicted_price"], rel=1e-8)

