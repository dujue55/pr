import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class CNN_simple(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 60, 3, padding = 'same', bias = True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(60, 40, 3, padding = 'same', bias = True)
        self.fc1 = nn.Linear(40 * (image_size//4)**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, image_size**2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""
Given batch of images, return noisy images with Gaussian noise with std sigma.
INPUT:
 -images: size (batch,c,n1,n2), where c is the number of channel, n1 and n2 are the height and width of the images.
 -sigma: std of the additive Gaussian noise.
"""

def add_noise(images,sigma):
    return images + torch.randn_like(images)*sigma


"""
Define Gaussian convolution kernel with the same resolution then the image.
"""
def Gaussian_blur(sigma_blur,image_size=32):
    xlin = np.linspace(-image_size/2,image_size/2,image_size)
    XX, YY = np.meshgrid(xlin,xlin)
    kernel = np.exp(-(XX**2+YY**2)/(2*sigma_blur**2))
    kernel /= np.sum(kernel) # normalize kernel
    return kernel

"""
Compute circular convolution between two images of the same size.
"""
def convolution_fft(image1,image2):
    return np.real(np.fft.ifft2( np.fft.fft2(image1)*np.fft.fft2(np.fft.fftshift(image2)) ))
def convolution_fft_torch(image1,image2):
    return torch.fft.ifft2( torch.fft.fft2(image1)*torch.fft.fft2(torch.fft.fftshift(image2)) ).real

def data_generator(batch_size=32):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader



"""
Display torch batch of tensors as an image.
"""
def plot_image(images,num_im=0,num_fig=1,title=None):
    plt.figure(num_fig)
    plt.imshow(images[num_im].detach().cpu().permute(1,2,0))
    if title != None:
        plt.title(title)

"""
Calculate SNR of a batch of true images and their estimations (batch size is supposed to be the first dimension)
"""
def SNR(x_true , x_pred):

        
    x_true = np.reshape(x_true , [np.shape(x_true)[0] , -1])
    x_pred = np.reshape(x_pred , [np.shape(x_pred)[0] , -1])
    
    Noise = x_true - x_pred
    Noise_power = np.sum(np.square(np.abs(Noise)), axis = -1)
    Signal_power = np.sum(np.square(np.abs(x_true)) , axis = -1)
    SNR = 10*np.log10(np.mean(Signal_power/Noise_power))
  
    return SNR