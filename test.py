import torch
from pathlib import Path
import pandas as pd
from dataset import MLDataset
from models import MyModel


def test():
    # load model and use weights we saved before.
    model = MyModel()
    model.load_state_dict(torch.load('mymodel.pth', map_location='cpu'))
    model.eval()
    # load testing data
    data = pd.read_csv('saving.csv', encoding='utf-8')
    label_col = ['pm25_min','pm25_max','pm25_median']

    # ================================================================ #
    # if do some operations with training data,
    # do the same operations to the testing data in this block
    data = data.fillna(0)


    # ================================================================ #
    # convert dataframe to tensor, no need to rewrite
    inputs = data.values
    inputs = torch.tensor(inputs)
    # predict and save the result
    result = pd.DataFrame(columns=label_col)
    outputs = model(inputs.float())
    for i in range(len(outputs)):
        tmp = outputs[i].detach().numpy()
        tmp = pd.DataFrame([tmp], columns=label_col)
        result= pd.concat([result, tmp], ignore_index=True)
    result.to_csv('result.csv', index=False)

if __name__ == '__main__':
    test()
