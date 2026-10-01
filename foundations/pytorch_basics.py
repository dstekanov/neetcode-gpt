import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Змінюємо форму тензора на 2 стовпчики (-1 означає підрахунок рядків автоматично)
        tensor = torch.reshape(to_reshape, (-1, 2))
        return torch.round(tensor, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Обчислюємо середнє значення уздовж виміру 0 (по стовпчиках)
        tensor = torch.mean(to_avg, dim=0)
        return torch.round(tensor, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Об'єднуємо два тензори бік-о-бік уздовж dim=1
        tensor = torch.cat((cat_one, cat_two), dim=1)
        return torch.round(tensor, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Обчислюємо середньоквадратичну помилку (MSE)
        tensor = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(tensor, decimals=4)
