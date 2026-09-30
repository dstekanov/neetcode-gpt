import torch
import torch.nn as nn
import math
from typing import List

class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []

        # 1. Скарбниця багу: СПОЧАТКУ генеруємо ВСІ ваги для кожного шару
        for i in range(num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (in_dim + out_dim))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / in_dim)
            else:
                std = 1.0
                
            w = torch.randn(out_dim, in_dim) * std
            weights.append(w)

        # 2. І ТІЛЬКИ ПІСЛЯ ЦЬОГО генерується вхідний вектор x
        x = torch.randn(1, input_dim)

        # 3. Прямий прохід через уже згенеровані ваги
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds