import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n_samples, n_features = X.shape
        
        # 1. Ініціалізація: починаємо з нуля
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        
        # 2. Цикл навчання
        for _ in range(epochs):
            # Прямий прохід (Forward pass)
            y_hat = X @ w + b
            
            # Обчислення помилки
            error = y_hat - y
            
            # Обчислення градієнтів (Backward pass)
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            
            # Оновлення параметрів (Step)
            w -= lr * dw
            b -= lr * db
            
        # Rounding strictly according to task instructions
        return (np.round(w, 5), round(b, 5))