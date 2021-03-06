import math
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from models import MyModel
from dataset import MLDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# WRMSE
def WRMSE(preds, labels, device):
    weight = torch.tensor([0.33,0.33,0.33]).to(device)
    wrmse = torch.pow(preds-labels, 2)
    wrmse = torch.sum(wrmse * weight)
    return wrmse.item()

# training curve
def visualize(record, title):
    plt.title(title)
    plt.plot(record)
    plt.show()

# learning rate, epoch and batch size. Can change the parameters here.
def train(lr=0.001, epoch=1024, batch_size=16):
    train_loss_curve = []
    train_wrmse_curve = []
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel()
    model = model.to(device)
    model.train()

    # dataset and dataloader
    # can use torch random_split to create the validation dataset
    dataset = MLDataset()
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # loss function and optimizer
    # can change loss function and optimizer you want
    criterion  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start training
    for e in range(epoch):
        train_loss = 0.0
        train_wrmse = 0.0
        best = 100
        print(f'\nEpoch: {e+1}/{epoch}')
        print('-' * len(f'Epoch: {e+1}/{epoch}'))
        # tqdm to disply progress bar
        for inputs, labels in tqdm(train_dataloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)
            # weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss calculate
            train_loss += loss.item()
            train_wrmse += wrmse
        # =================================================================== #
        # If you have created the validation dataset,
        # you can refer to the for loop above and calculate the validation loss



        # =================================================================== #
        # save the best model weights as .pth file
        loss_epoch = train_loss / len(dataset)
        wrmse_epoch = math.sqrt(train_wrmse/len(dataset))
        if wrmse_epoch < best :
            best = wrmse_epoch
            torch.save(model.state_dict(), 'mymodel.pth')
        print(f'Training loss: {loss_epoch:.4f}')
        # print(f'Training WRMSE: {wrmse_epoch:.4f}')
        # save loss and wrmse every epoch
        train_loss_curve.append(loss_epoch)
        train_wrmse_curve.append(wrmse_epoch)
    # generate training curve
    visualize(train_loss_curve, 'Train Loss')
    # visualize(train_wrmse_curve, 'Train WRMSE')

if __name__ == '__main__':
    train()
