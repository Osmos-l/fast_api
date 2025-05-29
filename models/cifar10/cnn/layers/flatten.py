class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1) # Reshape to (batch_size, -1) to flatten all dimensions except the batch size