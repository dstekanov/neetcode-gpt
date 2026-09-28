import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # 1. Forward pass
        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))
        
        # 2. Chain rule: dL/dz = dL/dy_hat * dy_hat/dz
        # dL/dy_hat = y_hat - y_true
        # dy_hat/dz = y_hat * (1 - y_hat)
        delta = (y_hat - y_true) * y_hat * (1.0 - y_hat)
        
        # 3. Gradients w.r.t weights and bias
        dL_dw = delta * x
        dL_db = delta
        
        return (np.round(dL_dw, 5), round(dL_db, 5))
