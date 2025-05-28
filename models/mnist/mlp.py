import numpy as np

class MLP:
    def __init__(self):
        self.ready = False

    def relu(self, x):
        return np.maximum(0, x)
        
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.weights_hidden_output) + self.bias_output
        a2 = self.softmax(z2)
        return a2

    def load_model(self, filename):
        data = np.load(filename)

        self.input_size = int(data['input_size'])
        self.hidden_size = int(data['hidden_size'])
        self.output_size = int(data['output_size'])

        self.weights_input_hidden = data['weights_input_hidden']
        self.bias_hidden = data['bias_hidden']
        self.weights_hidden_output = data['weights_hidden_output']
        self.bias_output = data['bias_output']

        self.ready = True