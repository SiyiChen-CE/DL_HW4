import tensorflow
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchgeometry as tgm
import os
import math

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        # add Gaussian Noise with mean = 0, variance = 0.1
        data = data + (0.1 ** 0.5) * torch.randn(data.size())
        target, label = self.dataset[idx]

        if self.transform:
            data = self.transform(data)
            target = self.transform(target)
        return data, target



def main():

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


    saved_state_dict = torch.load('Auto_encoder_wo_9.pt')
    model.load_state_dict(saved_state_dict)

    # switch model to eval model (dropout becomes pass through)
    model.eval()

    test_dataset = torchvision.datasets.CIFAR10(
        './data', train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    )

    test_dataset_autoencoder = Autoencoder_Dataset(test_dataset, transform=None)
    test_dataset_denoise_autoencoder = Denoising_Autoencoder_Dataset(test_dataset, transform=None)

    test_loader = torch.utils.data.DataLoader(test_dataset_autoencoder, batch_size=10, shuffle=True)
    test_loader_denoise = torch.utils.data.DataLoader(test_dataset_denoise_autoencoder, batch_size=10, shuffle=True)

    data, target = iter(test_loader).next()
    with torch.no_grad():
        output = model(data)

    fig = plt.figure()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0,
                        hspace=0)
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                        std=[1., 1., 1.]),
                                   ])

    inv_target = invTrans(target)
    inv_output = invTrans(output)
    inv_data = invTrans(data)

    for i in range(10):
        plt.subplot(5, 4, 2*i+1)
        plt.imshow(inv_target[i].numpy().transpose((1, 2, 0)))
        plt.axis('off')

        plt.subplot(5, 4, 2*i+2)
        plt.imshow(inv_output[i].numpy().transpose((1, 2, 0)))
        plt.axis('off')

    # for i in range(10):
    #     plt.subplot(5, 6, 3*i+1)
    #     plt.imshow(inv_data[i].numpy().transpose((1, 2, 0)))
    #     plt.axis('off')
    #
    #     plt.subplot(5, 6, 3*i+2)
    #     plt.imshow(inv_target[i].numpy().transpose((1, 2, 0)))
    #     plt.axis('off')
    #
    #     plt.subplot(5, 6, 3*i+3)
    #     plt.imshow(inv_output[i].numpy().transpose((1, 2, 0)))
    #     plt.axis('off')

    plt.show()
    fig.savefig('test_result_autoencoder_wo.png')

    test_loader = torch.utils.data.DataLoader(test_dataset_autoencoder, batch_size=10000, shuffle=False)
    test_loader_denoise = torch.utils.data.DataLoader(test_dataset_denoise_autoencoder, batch_size=10000, shuffle=False)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output=model(data)

    #PSNR

    mse = torch.mean(torch.square(target-output),dim=[1,2,3])
    max_target = torch.max(target,dim=1)
    max_target = torch.max(max_target[0], dim=1)
    max_target = torch.max(max_target[0], dim=1)

    psnr = 20 * (torch.log(max_target[0]/ torch.sqrt(mse))/math.log(10))
    np.savetxt('psnr.txt', psnr, delimiter=' ')

    psnr=tensorflow.unstack(psnr)
    psnr_mean = np.nanmean(psnr)
    print(psnr_mean)

    #SSIM
    ssim = tgm.losses.SSIM(1, max_val=2, reduction='mean')
    ssim_index = 1-2*ssim(target, output)
    print(ssim_index)

if __name__ == "__main__":
    main()
