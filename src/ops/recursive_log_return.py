import pandas as pd


def recursive_forecast_log_return(
    model,
    df: pd.DataFrame,
    start_index,
    n_steps,
    lookback,
    scaler_X,
    scaler_y,
    features,
    target,
    rolling_window=28,
    volatility_col="volatility_rolling_28",
):
    model.eval()

    # 1. Init Window
    window = df.loc[start_index - lookback : start_index - 1, features].values.copy()
    logret_preds = []

    # Map index
    target_idx = features.index(target)
    z_idx = features.index("zscore_scaled") if "zscore_scaled" in features else None
    vol_idx = features.index(volatility_col) if volatility_col in features else None
    ema_dist_idx = features.index("ema_dist") if "ema_dist" in features else None

    # 2. Init Rolling Stats (Lấy đúng rolling_window=28)
    # Lấy lịch sử log_returns để tính lại zscore/volatility cho các bước sau
    ret_hist = (
        df["log_returns"].iloc[start_index - rolling_window : start_index].tolist()
    )

    # Init EMA Values
    last_close = df["close"].iloc[start_index - 1]
    last_ema_9 = df["ema_9"].iloc[start_index - 1]
    last_ema_21 = df["ema_21"].iloc[start_index - 1]

    # Hệ số Alpha cho EMA
    alpha_9 = 2 / (9 + 1)
    alpha_21 = 2 / (21 + 1)

    for _ in range(n_steps):
        # A. Encode & Predict
        x_scaled = scaler_X.encode(window)
        x = torch.tensor(x_scaled).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)  # Đảm bảo shape (1, Lookback, Feat)

        with torch.no_grad():
            y_hat_norm = model(x).item()

        # B. Decode Output (Về log_returns thực)
        new_log_ret = float(scaler_y.decode(y_hat_norm))
        logret_preds.append(new_log_ret)

        # C. Update Derived Features
        # C.1 Update Close & EMA
        new_close = last_close * np.exp(new_log_ret)
        new_ema_9 = alpha_9 * new_close + (1 - alpha_9) * last_ema_9
        new_ema_21 = alpha_21 * new_close + (1 - alpha_21) * last_ema_21

        # C.2 Update EMA Dist
        new_ema_dist = (new_ema_9 - new_ema_21) / new_close * 10

        # C.3 Update Rolling Stats
        ret_hist.append(new_log_ret)
        if len(ret_hist) > rolling_window:
            ret_hist = ret_hist[-rolling_window:]  # Trượt cửa sổ, giữ đúng độ dài 28

        # Tính toán thống kê mới
        mu_rolling = float(np.mean(ret_hist))
        sigma_rolling = float(np.std(ret_hist)) + 1e-8

        # --- LOGIC RAW ---
        # Tính Z-score thuần túy: (x - mean) / std
        new_z = (new_log_ret - mu_rolling) / sigma_rolling

        new_vol = sigma_rolling

        # D. Build New Row
        new_row = window[-1].copy()
        new_row[target_idx] = new_log_ret

        if z_idx is not None:
            new_row[z_idx] = new_z  # Gán Raw Z
        if vol_idx is not None:
            new_row[vol_idx] = new_vol
        if ema_dist_idx is not None:
            new_row[ema_dist_idx] = new_ema_dist

        # Update Window & Memory
        window = np.vstack([window[1:], new_row])
        last_close = new_close
        last_ema_9 = new_ema_9
        last_ema_21 = new_ema_21

    return np.array(logret_preds)
