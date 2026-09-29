import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        curr = x
        num_layers = len(weights)
        
        for i in range(num_layers):
            # Множення на ваги та додавання зсуву
            curr = np.dot(curr, weights[i]) + biases[i]
            
            # ReLU застосовується лише до прихованих шарів (тобто не до останнього)
            if i < num_layers - 1:
                curr = np.maximum(0, curr)
                
        # Округлення за вимогою задачі
        return np.round(curr, 5)
