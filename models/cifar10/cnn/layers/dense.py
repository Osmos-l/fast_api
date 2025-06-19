import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def load_weights(self, f):
        self.weights = np.load(f)
        self.bias = np.load(f)
        self.ready = True

    def is_ready(self):
        return self.ready