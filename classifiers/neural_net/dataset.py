import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class IncomeDataset(Dataset):
    def __init__(self, input_filepath, labels_filepath=None):
        self.eval = labels_filepath is None

        self.X = pd.read_csv(input_filepath, header=None)

        if not self.eval:
            self.labels = pd.read_csv(labels_filepath, header=None)

    def __getitem__(self, index):

        X = self.X.iloc[index].values.astype(np.float32)

        if self.eval:
            return X

        y = self.labels.iloc[index].values.astype(np.float32)

        return X, y

    def __len__(self):
        return len(self.X)
