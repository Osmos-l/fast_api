import numpy as np

from models.cifar10.cnn.layers.conv2d import Conv2D
from models.cifar10.cnn.layers.maxpool2d import MaxPool2D
from models.cifar10.cnn.layers.flatten import Flatten
from models.cifar10.cnn.layers.dense import Dense
from models.cifar10.cnn.layers.relu import ReLU
from models.cifar10.cnn.layers.softmax import Softmax

class CNN:
    def __init__(self):
        self.layers = [
            # Convolutional layers followed by ReLU and Max Pooling
            Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Flatten(),

            # Fully connected layers followed by ReLU and Softmax (MLP)
            Dense(input_dim=8*8*16, output_dim=64),
            ReLU(),
            Dense(input_dim=64, output_dim=10),
            Softmax()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'load_weights'):
                    ident = f.readline().decode().strip()
                    expected = f"{i}:{type(layer).__name__}"
                    if ident != expected:
                        raise ValueError(f"Erreur de correspondance des couches : attendu {expected}, trouvé {ident}")
                    layer.load_weights(f)

    def is_ready(self):
        return all(layer.is_ready() for layer in self.layers if hasattr(layer, 'is_ready'))