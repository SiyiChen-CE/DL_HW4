import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.utils.data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(model, train_loader, loss_func, optimizer, epoch):

    epoch_loss = 0
    epoch_counter = 0
    # switch model to train mode (dropout enabled)
    model.train()


    for batch_idx, (data, target) in enumerate(train_loader):
            # send data to cuda
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.item() * data.shape[0]
        epoch_counter += float(data.shape[0])

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    return epoch_loss, epoch_counter


class Autoencoder_Dataset():
    def __init__(self, dataset,  transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        target, label = self.dataset[idx]

        if self.transform:
            data = self.transform(data)
            target = self.transform(target)
        return data, target

class Denoising_Autoencoder_Dataset():
    def __init__(self, dataset,  transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        # add Gaussian Noise with mean = 0, variance = 0.1
        data = data+(0.1**0.5)*torch.randn(data.size())
        target, label = self.dataset[idx]

        if self.transform:
            data = self.transform(data)
            target = self.transform(target)
        return data, target

def main():
    # seed pytorch random number generator for reproducablity
    torch.manual_seed(2)

    train_dataset = torchvision.datasets.CIFAR10(
        './data', train=True, download=False,
        transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    )


    train_dataset_autoencoder=Autoencoder_Dataset(train_dataset, transform=None)
    train_dataset_denoise_autoencoder = Denoising_Autoencoder_Dataset(train_dataset, transform=None)

    # show train sample

    temp_img1, temp_img2 = train_dataset_autoencoder[0]
    fig = plt.figure(figsize=(21, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(temp_img1.numpy().transpose((1,2,0)))
    plt.axis('off')
    plt.title('Data')
    plt.subplot(1, 2, 2)
    plt.imshow(temp_img2.numpy().transpose((1,2,0)))
    plt.axis('off')
    plt.title('Target')
    plt.show()
    fig.savefig('sample_dataset.png')

    train_loader = torch.utils.data.DataLoader(train_dataset_autoencoder, batch_size=32, shuffle=True)
    # train_loader_denoise = torch.utils.data.DataLoader(train_dataset_denoise_autoencoder, batch_size=32, shuffle=True)

    model = nn.Sequential(
        # input [3, 32, 32]
        # Encoder
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),   # [8, 30, 30]
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0),  # [8, 28, 28]
        nn.ReLU(),
        # Decoder
        nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1, padding=0),  # [8, 30, 30]
        nn.ReLU(),
        nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=0),  # [3, 32, 32]
    )

    # send model parameters to cuda
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_func = torch.nn.MSELoss()

    epochs = 10
    train_loss = [0.1]*epochs

    for epoch in range(epochs):

        loss, epoch_counter = train(model, train_loader, loss_func, optimizer, epoch)
        print('Saving model to Auto_encoder_wo_{}.pt'.format(epoch))
        torch.save(model.state_dict(), 'Auto_encoder_wo_{}.pt'.format(epoch))
        train_loss[epoch]=loss / epoch_counter*1.0
        print(train_loss[epoch])

    fig = plt.figure(figsize=(15, 7))
    plt.plot(train_loss)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Train Loss during Training', fontsize=16)
    plt.xticks(range(epochs))
    plt.grid('on')
    plt.legend(fontsize=14)

    fig.savefig('Auto_encoder_wo.png')
    print('Auto_encoder_wo.png')

if __name__ == "__main__":
    main()
