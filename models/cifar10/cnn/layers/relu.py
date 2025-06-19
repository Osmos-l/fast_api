import numpy as np

class ReLU:
    def __init__(self):
        self.input = None 

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)