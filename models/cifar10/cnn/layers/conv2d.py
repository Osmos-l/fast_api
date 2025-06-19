import numpy as np

def im2col(x, kernel_size, stride, padding):
    # x: (batch, h_in, w_in, c_in)
    batch, h_in, w_in, c_in = x.shape
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1

    x_padded = np.pad(x, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
    cols = np.zeros((batch, h_out, w_out, kernel_size, kernel_size, c_in))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            cols[:, i, j, :, :, :] = x_padded[:, h_start:h_start+kernel_size, w_start:w_start+kernel_size, :]

    # (batch, h_out, w_out, k, k, c) -> (batch*h_out*w_out, k*k*c)
    cols = cols.reshape(batch * h_out * w_out, -1)
    return cols, h_out, w_out

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding

        # Weights structure initialization
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)

        # Kaiming Initialization
        self.weights = self.weights * np.sqrt(2. / (in_channels * kernel_size * kernel_size))

        self.bias = np.zeros(out_channels)

    def forward(self, x):
        self.input = x  # Save input for backward pass
        cols, h_out, w_out = im2col(x, self.kernel_size, self.stride, self.padding)
        W_col = self.weights.reshape(self.out_channels, -1)  # (out_channels, in_channels*k*k)
        out = cols @ W_col.T + self.bias  # (batch*h_out*w_out, out_channels)
        out = out.reshape(x.shape[0], h_out, w_out, self.out_channels)
        self.cols = cols  # Pour backward
        self.h_out = h_out
        self.w_out = w_out
        return out

    def load_weights(self, f):
        self.weights = np.load(f)
        self.bias = np.load(f)
        self.ready = True

    def is_ready(self):
        return self.ready
