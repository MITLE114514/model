import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class MLDataset(Dataset):
    def __init__(self):
        # load data
        data = pd.read_csv('original_training_data.csv', encoding='utf-8')
        # label's columns name, no need to rewrite
        label_col = ['pm25_min','pm25_max','pm25_median']
        # ================================================================================ #
        # Do any operation on self.train you want with data type "dataframe"(recommanded) in this block.
        # For example, do normalization or dimension Reduction.
        # Some of columns have "nan", need to drop row or fill with value first
        # For example:
        data = data.fillna(0)
        self.label = data[label_col]
        self.train = data.drop(label_col, axis=1)


        # ================================================================================ #

    def __len__(self):
        #  no need to rewrite
        return len(self.train)

    def __getitem__(self, index):
        # transform dataframe to numpy array, no need to rewrite
        x = self.train.iloc[index, :].values
        y = self.label.iloc[index, :].values
        return x, y

