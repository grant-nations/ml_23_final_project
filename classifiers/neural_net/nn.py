import torch
from torch import nn
import numpy as np
import torch.nn.init as init

class BinClassificationNN(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 32), # input dimension is shape (14,)
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Dropout(p=dropout_p),
            nn.Linear(32, 1),
            nn.Sigmoid() # squashes output between 0 and 1 for probability
        )

    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #         if m.bias is not None:
    #             init.constant_(m.bias, 0)

# TODO: add inverted dropout layers
# TODO: add batch normalization layers ?
# TODO: whiten data ?
# TODO: residual network?

    def forward(self, x):
        x = self.flatten(x)
        p = self.linear_relu_stack(x)
        return p
    