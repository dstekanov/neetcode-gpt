import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
# Перетворюємо вхідні дані у NumPy масиви
        x_arr = np.array(x, dtype=np.float64)
        W1_arr = np.array(W1, dtype=np.float64)
        b1_arr = np.array(b1, dtype=np.float64)
        W2_arr = np.array(W2, dtype=np.float64)
        b2_arr = np.array(b2, dtype=np.float64)
        y_true_arr = np.array(y_true, dtype=np.float64)
        
        # --- FORWARD PASS ---
        # 1. Перший лінійний шар: z1 = W1 @ x + b1
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        
        # 2. Активація ReLU: a1 = max(0, z1)
        a1 = np.maximum(0, z1)
        
        # 3. Другий лінійний шар: z2 = W2 @ a1 + b2 (він же y_pred)
        z2 = np.dot(W2_arr, a1) + b2_arr
        
        # 4. Втрати (MSE loss): L = (1/n) * sum((z2 - y_true)^2)
        n = len(y_true_arr)
        loss = np.mean((z2 - y_true_arr) ** 2)
        
        # --- BACKWARD PASS ---
        # 1. Градієнт по відношенню до z2: dL/dz2 = 2/n * (z2 - y_true)
        dz2 = (2.0 / n) * (z2 - y_true_arr)
        
        # 2. Градієнти другого шару:
        # dL/dW2 = dz2 (outer product) a1
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        
        # 3. Поширення градієнта назад через W2 до a1: dL/da1 = W2^T @ dz2
        da1 = np.dot(W2_arr.T, dz2)
        
        # 4. Поширення градієнта через ReLU (маска де z1 > 0)
        dz1 = da1 * (z1 > 0).astype(np.float64)
        
        # 5. Градієнти першого шару:
        # dL/dW1 = dz1 (outer product) x
        dW1 = np.outer(dz1, x_arr)
        db1 = dz1
        
        # Повертаємо результати, заокруглені до 4 знаків після коми
        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
