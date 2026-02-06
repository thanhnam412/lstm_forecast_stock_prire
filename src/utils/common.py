import numpy as np
from typing import Tuple, Union, Optional


class StandardScalerCustom:
    """
    Class thực hiện chuẩn hóa Z-score (Standardization) cho dữ liệu.
    Hỗ trợ cả dữ liệu đặc trưng (X) và biến mục tiêu (y).
    """

    def __init__(self) -> None:
        self.mu: Optional[Union[np.ndarray, float]] = None
        self.std: Optional[Union[np.ndarray, float]] = None

    def fit(
        self, data: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> None:
        """
        Tính toán giá trị trung bình và độ lệch chuẩn từ tập dữ liệu huấn luyện.

        Args:
            data: Mảng numpy chứa dữ liệu huấn luyện.
            axis: Trục để tính toán (ví dụ: (0, 1) cho dữ liệu ảnh/chuỗi).
        """
        # keepdims=True để đảm bảo tính toán broadcasting sau này không bị lỗi
        self.mu = (
            np.mean(data, axis=axis, keepdims=True)
            if axis is not None
            else np.mean(data)
        )
        self.std = (
            np.std(data, axis=axis, keepdims=True) if axis is not None else np.std(data)
        )

        # Thêm epsilon để tránh chia cho 0
        self.std += 1e-8

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa dữ liệu (Transform/Encode).

        Args:
            data: Mảng dữ liệu cần chuẩn hóa.
        Returns:
            Dữ liệu đã chuẩn hóa có cùng shape với đầu vào.
        """
        if self.mu is None or self.std is None:
            raise ValueError("Bạn phải gọi hàm .fit() trước khi encode!")

        return (data - self.mu) / self.std

    def decode(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Giải chuẩn hóa dữ liệu về đơn vị gốc (Inverse Transform/Decode).
        Thường dùng cho y_pred để xem kết quả thực tế.

        Args:
            normalized_data: Dữ liệu đã chuẩn hóa.
        """
        if self.mu is None or self.std is None:
            raise ValueError("Bạn phải gọi hàm .fit() trước khi decode!")

        return (normalized_data * self.std) + self.mu


import pandas as pd
import numpy as np


def make_sequences(
    df_feat: pd.DataFrame, target_col: str, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chuyển đổi DataFrame thành các chuỗi (sequences) phục vụ cho bài toán Time Series.
    """
    X: list[np.ndarray] = []
    y: list[float] = []

    for i in range(len(df_feat) - lookback):
        # Lấy lookback dòng dữ liệu và làm phẳng (flatten)
        X.append(df_feat.iloc[i : i + lookback].values.flatten())
        # Lấy giá trị mục tiêu tại thời điểm tiếp theo
        y.append(df_feat.iloc[i + lookback][target_col])

    return np.array(X), np.array(y)
