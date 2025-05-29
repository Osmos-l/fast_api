import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding

        self.weights    = None
        self.bias       = None

        self.ready = False

    def forward(self, x):
        self.input = x  # Save input for backward pass
        batch_size, h_in, w_in, c_in = x.shape
        assert c_in == self.in_channels, f"Expected {self.in_channels} channels, got {c_in}"

        # 1. Adding padding to the input dimensions
        # 2. Subtracting kernel size 
        # 3. Dividing by stride
        # 4. Adding 1 for the output dimensions
        h_out = int((h_in + 2 * self.padding - self.kernel_size) / self.stride) + 1
        w_out = int((w_in + 2 * self.padding - self.kernel_size) / self.stride) + 1

        # Adding padding to the input images
        x_padded = np.pad(x, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)), mode='constant')

        # Output structure initialization
        out = np.zeros((batch_size, h_out, w_out, self.out_channels))

        # for each image in the batch
        for image_index in range(batch_size):
            # for each row of the output
            for i in range(h_out):
                # for each column of the output
                for j in range(w_out):
                    # for each output channel
                    for oc in range(self.out_channels):

                        h_start = i * self.stride
                        w_start = j * self.stride

                        patch = x_padded[image_index,                           # Selecting the image 
                                        h_start:(h_start + self.kernel_size),   # Selecting the range of rows (height patch) in the image
                                        w_start:(w_start + self.kernel_size),   # Selecting the range of columns (width patch) in the image
                                        :self.in_channels]                      # Selecting all channels 

                        patch_permuted = np.transpose(patch, (2,0,1))

                        out[image_index, i, j, oc] = np.sum(patch_permuted * self.weights[oc]) + self.bias[oc]

        return out

    def load_weights(self, f):
        self.weights = np.load(f)
        self.bias = np.load(f)
        self.ready = True

    def is_ready(self):
        return self.ready
