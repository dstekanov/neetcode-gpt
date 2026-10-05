import numpy as np
from typing import Tuple, List

class Solution:
    def batch_norm(
        self, 
        x: List[List[float]], 
        gamma: List[float], 
        beta: List[float], 
        running_mean: List[float], 
        running_var: List[float], 
        momentum: float, 
        eps: float, 
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        
        # Перетворюємо наші списки на масиви NumPy для швидкої та зручної математики
        X = np.array(x, dtype=np.float64)
        g = np.array(gamma, dtype=np.float64)
        b = np.array(beta, dtype=np.float64)
        r_mean = np.array(running_mean, dtype=np.float64)
        r_var = np.array(running_var, dtype=np.float64)

        if training:
            # КРОК 1: Шукаємо "центр мас" нашої групи (середнє по кожному стовпчику/фічі)
            # axis=0 означає: усереднюємо по всіх елементах батчу
            batch_mean = np.mean(X, axis=0)
            
            # КРОК 2: Вимірюємо "буйність" даних (дисперсію по кожній фічі)
            batch_var = np.var(X, axis=0)

            # КРОК 3: Стандартизуємо!
            # Віднімаємо середнє і ділимо на стандартне відхилення.
            # Додаємо eps (крихітну величину), щоб випадково не поділити на нуль, якщо дисперсія = 0!
            x_hat = (X - batch_mean) / np.sqrt(batch_var + eps)

            # КРОК 4: Оновлюємо наш "щоденник спостережень" (накопичувальні статистику)
            # Формула експоненційного згладжування: трохи старого + трохи нового (momentum)
            r_mean = (1 - momentum) * r_mean + momentum * batch_mean
            r_var = (1 - momentum) * r_var + momentum * batch_var

        else:
            # Якщо ми в режимі оцінки (inference), ми НЕ вираховуємо нове середнє батчу!
            # Ми використовуємо наш накопичений досвід з "щоденника".
            x_hat = (X - r_mean) / np.sqrt(r_var + eps)

        # КРОК 5: Афінне перетворення (налаштування від нейромережі)
        # Мережа каже: "Дякую за нормування, а тепер дозволь мені розстягнути це на gamma і зсунути на beta"
        out = g * x_hat + b

        # КРОК 6: Закругляємо все до 4 знаків після коми та повертаємо у початкових типах даних
        out_rounded = np.round(out, 4).tolist()
        r_mean_rounded = np.round(r_mean, 4).tolist()
        r_var_rounded = np.round(r_var, 4).tolist()

        return out_rounded, r_mean_rounded, r_var_rounded