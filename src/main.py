from src.featuring.main import FeatureBuilder
from configs.feature_configs import feature_build_configs, DatasetConfigs
import pandas as pd


def featuring() -> pd.DataFrame:
    df = pd.read_csv("data/raw/XAUT_USD_history.csv")
    builder = FeatureBuilder(dataframe=df, configuration=feature_build_configs)

    processed = builder.build_features()

    processed.to_csv("data/processing/processed_v2.csv")
    return processed


featuring()

# # ===== build sequences
# data = df[]
# X, y = make_sequences(data, target, lookback)

# split = int(len(X) * train_ratio)

# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]


# # =========================
# # EXPORT CSV
# # =========================
# col_names = [f"{f}_t-{lookback-t}" for t in range(lookback) for f in features]

# df_train = pd.DataFrame(X_train, columns=col_names)
# df_train["target"] = y_train

# df_test = pd.DataFrame(X_test, columns=col_names)
# df_test["target"] = y_test

# df_train.to_csv("../data/datasets/train_sequences.csv", index=False)
# df_test.to_csv("../data/datasets/test_sequences.csv", index=False)


# def make_dataset(d):


# def pipeline():
#     processed = featuring()
