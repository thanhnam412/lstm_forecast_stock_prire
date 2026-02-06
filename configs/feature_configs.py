from typing import TypedDict

feature_build_configs = {
    "enable_returns": True,
    "zscore_window": 28,
    "volatility_window": 28,
    "annualize_volatility": True,
    "periods_per_year": 252,
    "ema_span": 9,
}


class DatasetConfigs(TypedDict):
    features = ["log_returns", "volatility", "zscore"]
    target = "log_returns"
    lookback = 7
    train_ratio = 0.6
