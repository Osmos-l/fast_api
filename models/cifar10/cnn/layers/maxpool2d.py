import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.input = x  # Save input for backward pass
        batch_size, h_in, w_in, channels = x.shape

        # 1. Subtracting kernel size 
        # 2. Dividing by stride
        # 3. Adding 1 for the output dimensions
        h_out = int((h_in - self.kernel_size) / self.stride) + 1
        w_out = int((w_in - self.kernel_size) / self.stride) + 1

        out = np.zeros((batch_size, h_out, w_out, channels))

        # Adding padding to the input images
        x_padded = np.pad(x, ((0, 0), (self.kernel_size // 2, self.kernel_size // 2), (self.kernel_size // 2, self.kernel_size // 2), (0, 0)), mode='constant')

        # For each image in the batch
        for image_index in range(batch_size):
            # For each channel
            for channel in range(channels):
                # For each row of the output
                for i in range(h_out):
                    # For each column of the output
                    for j in range(w_out):
                        h_start = i * self.stride
                        w_start = j * self.stride

                        # Extracting the patch from the input image
                        patch = x_padded[image_index,                           # Selecting the image 
                                        h_start:(h_start + self.kernel_size),   # Selecting the range of rows (height patch) in the image
                                        w_start:(w_start + self.kernel_size),   # Selecting the range of columns (width patch) in the image
                                        channel]                                # Selecting the channel

                        # Max pooling
                        out[image_index, i, j, channel] = np.max(patch)

        return out
