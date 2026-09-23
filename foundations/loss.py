import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        clipped_preds = np.clip(y_pred, eps, 1 - eps)
        return round(float(-np.mean(y_true * np.log(clipped_preds) + (1 - y_true) * np.log(1 - clipped_preds))), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        clipped_preds = np.clip(y_pred, eps, 1 - eps)
        return round(float(-np.sum(y_true * np.log(clipped_preds)) / len(y_true)), 4)
