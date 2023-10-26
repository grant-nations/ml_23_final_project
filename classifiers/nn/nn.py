import torch
from torch import nn

class BinClassificationNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # squashes output between 0 and 1 for probability
        )

# TODO: add inverted dropout layers
# TODO: add batch normalization layers ?
# TODO: whiten data ?
# TODO: use PCA to reduce dimensionality ?

    def forward(self, x):
        x = self.flatten(x)
        p = self.linear_relu_stack(x)
        return p
    
    