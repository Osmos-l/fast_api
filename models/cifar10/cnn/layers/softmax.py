import numpy as np

class Softmax:
    def forward(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        self.out = exp_x / sum_exp_x  # save for backward
        return self.out
