import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(self, dataframe: pd.DataFrame, configuration: dict):
        self.dataframe = dataframe.copy()
        self.configuration = configuration

    # =====================================================
    # Core mathematical utilities
    # =====================================================

    @staticmethod
    def compute_log_returns(price_series: pd.Series) -> pd.Series:
        log_returns = np.log(price_series / price_series.shift(1))
        return log_returns.fillna(0.0)

    @staticmethod
    def compute_simple_returns(price_series: pd.Series) -> pd.Series:
        simple_returns = price_series.pct_change()
        return simple_returns.fillna(0.0)

    @staticmethod
    def compute_rolling_mean(series: pd.Series, window_size: int) -> pd.Series:
        return series.rolling(window_size, min_periods=window_size).mean()

    @staticmethod
    def compute_rolling_standard_deviation(
        series: pd.Series, window_size: int
    ) -> pd.Series:
        return series.rolling(window_size, min_periods=window_size).std(ddof=0)

    @staticmethod
    def compute_exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def compute_rolling_zscore(series: pd.Series, window_size: int) -> pd.Series:
        rolling_mean = FeatureBuilder.compute_rolling_mean(series, window_size)
        rolling_std = FeatureBuilder.compute_rolling_standard_deviation(
            series, window_size
        )

        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (series - rolling_mean) / rolling_std

        return zscore.fillna(0.0)

    # =====================================================
    # Feature creation blocks
    # =====================================================

    def add_return_features(self, price_column: str = "close") -> pd.Series:
        price_series = self.dataframe[price_column]

        log_returns_series = self.compute_log_returns(price_series)
        simple_returns_series = self.compute_simple_returns(price_series)

        self.dataframe["log_returns"] = log_returns_series
        self.dataframe["simple_returns"] = simple_returns_series

        return log_returns_series

    def add_zscore_feature(self, log_returns_series: pd.Series) -> None:
        window_size = self.configuration["zscore_window"]

        column_name = f"zscore_rolling_{window_size}"

        self.dataframe[column_name] = self.compute_rolling_zscore(
            log_returns_series, window_size
        )

    def add_volatility_feature(self, log_returns_series: pd.Series) -> None:
        window_size = self.configuration["volatility_window"]

        rolling_standard_deviation = self.compute_rolling_standard_deviation(
            log_returns_series, window_size
        )

        if self.configuration.get("annualize_volatility", False):
            periods_per_year = self.configuration.get("periods_per_year", 252)
            rolling_standard_deviation *= np.sqrt(periods_per_year)

        self.dataframe[f"volatility_rolling_{window_size}"] = (
            rolling_standard_deviation.fillna(0.0)
        )

    def add_exponential_moving_average_feature(
        self, price_column: str = "close"
    ) -> None:
        span = self.configuration["ema_span"]

        price_series = self.dataframe[price_column]

        column_name = f"ema_{span}"

        self.dataframe[column_name] = self.compute_exponential_moving_average(
            price_series, span
        )

        self.dataframe["ema_21"] = self.compute_exponential_moving_average(
            price_series, 21
        )

    # =====================================================
    # Public pipeline
    # =====================================================

    def build_features(self) -> pd.DataFrame:
        """
        Generate all configured features and return the enriched dataframe.
        """

        log_returns_series = None

        if self.configuration.get("enable_returns", True):
            log_returns_series = self.add_return_features()

        if log_returns_series is not None and "zscore_window" in self.configuration:
            self.add_zscore_feature(log_returns_series)

        if log_returns_series is not None and "volatility_window" in self.configuration:
            self.add_volatility_feature(log_returns_series)

        if "ema_span" in self.configuration:
            self.add_exponential_moving_average_feature()

        return self.dataframe
