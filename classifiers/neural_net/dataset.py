import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class IncomeDataset(Dataset):
    def __init__(self, input_dataframe, labels_dataframe=None):
        self.eval = labels_dataframe is None

        self.X = input_dataframe

        if not self.eval:
            self.labels = labels_dataframe

    def __getitem__(self, index):

        X = self.X.iloc[index].values.astype(np.float32)

        if self.eval:
            return X

        y = self.labels.iloc[index].values.astype(np.float32)

        return X, y

    def __len__(self):
        return len(self.X)
    
    def generate_k_folds(self, k):
        """
        Generates k folds of the dataset for k-fold cross validation.
        """

        # shuffle dataset
        shuffled_indices = np.random.permutation(len(self.X))
        self.X = self.X.iloc[shuffled_indices]
        self.labels = self.labels.iloc[shuffled_indices]

        # split into k folds
        k_folds = []
        fold_size = len(self.X) // k
        for i in range(k):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size
            if i == k - 1:
                fold_end = len(self.X)
            k_folds.append((self.X.iloc[fold_start:fold_end], self.labels.iloc[fold_start:fold_end]))

        return k_folds
