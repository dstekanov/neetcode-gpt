import numpy as np
from numpy.typing import NDArray

class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # 1. Знаходимо середнє значення вектора x
        mean = np.mean(x)
        
        # 2. Знаходимо дисперсію вектора x
        var = np.var(x)
        
        # 3. Страховка від ділення на 0
        eps = 1e-5
        
        # 4. Центруємо та масштабуємо (нормалізуємо)
        x_hat = (x - mean) / np.sqrt(var + eps)
        
        # 5. Применяємо навчальні параметри gamma та beta
        out = gamma * x_hat + beta
        
        # 6. Округлюємо до 5 знаків після коми
        return np.round(out, 5)