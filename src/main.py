from src.featuring.main import FeatureBuilder
from configs.feature_configs import feature_build_configs
import pandas as pd


df = pd.read_csv("data/raw/XAUT_USD_history.csv")
builder = FeatureBuilder(dataframe=df, configuration=feature_build_configs)

processed = builder.build_features()

processed.to_csv("data/processing/processed_v2.csv")
