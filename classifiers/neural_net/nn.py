from torch import nn
import torch.nn.init as init


class BinClassificationNN(nn.Module):
    def __init__(self,
                 dropout_probs=[0.1],
                 hidden_dims=[32],
                 batch_norm=True
                 ):
        super().__init__()
        self.flatten = nn.Flatten()

        if batch_norm:
            if len(hidden_dims) == 1:
                self.stack = nn.Sequential(
                    nn.Linear(14, hidden_dims[0]),  # input dimension is shape (14,)
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_probs[0]),
                    nn.Linear(hidden_dims[0], 1),
                    nn.Sigmoid()  # squashes output between 0 and 1 for probability
                )
            elif len(hidden_dims) == 2:
                self.stack = nn.Sequential(
                    nn.Linear(14, hidden_dims[0]),  # input dimension is shape (14,)
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_probs[0]),
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.BatchNorm1d(hidden_dims[1]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_probs[1]),
                    nn.Linear(hidden_dims[1], 1),
                    nn.Sigmoid()  # squashes output between 0 and 1 for probability
                )
            else:
                raise ValueError("Only 1 or 2 hidden layers supported.")
        else:
            self.stack = nn.Sequential(
                nn.Linear(14, hidden_dims[0]),  # input dimension is shape (14,)
                nn.ReLU(),
                nn.Dropout(p=dropout_probs[0]),
                # nn.Linear(hidden_dims[0], hidden_dims[1]),
                # nn.ReLU(),
                # nn.Dropout(p=dropout_probs[1]),
                nn.Linear(hidden_dims[0], 1),
                nn.Sigmoid()  # squashes output between 0 and 1 for probability
            )

    def forward(self, x):
        x = self.flatten(x)
        p = self.stack(x)
        return p
