import numpy as np 

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from helper import data_generator, plot_image, add_noise, CNN_simple, SNR, convolution_fft_torch, Gaussian_blur

use_cuda=torch.cuda.is_available()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = "cpu"

print("Device:",device)



if __name__ == '__main__':
    batch_size = 32
    sigma_noise = 0.01
    sigma_blur = 1.5

    trainloader, testloader = data_generator(batch_size)

    # Display one image to see what it looks like
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    image_size = images.shape[2]

    # Display one blur
    kernel = Gaussian_blur(sigma_blur,image_size=image_size)
    kernel_torch = torch.from_numpy(kernel).to(device).reshape(1,1,image_size,image_size).type(torch.float32)
    plt.imshow(kernel)

    # Display one blured image
    im_blur_torch = add_noise(convolution_fft_torch(images.to(device),kernel_torch),sigma_noise)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax[0].imshow(images[0,0].detach().cpu().numpy())
    ax[0].set_title("Clean")
    ax[1].imshow(im_blur_torch[0,0].detach().cpu())
    ax[1].set_title("Blury")


    # Load the neural network    
    net = torch.jit.load("deblurring.pt",map_location=device)
    print('---> Number of trainable parameters of supercnn: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    Ntest = len(testloader.dataset)
    image_test_denoised = np.zeros((Ntest,image_size,image_size))
    image_test_noisy = np.zeros((Ntest,image_size,image_size))
    image_test_clean = np.zeros((Ntest,image_size,image_size))
    SNR_est = np.zeros(Ntest)
    SNR_noise = np.zeros(Ntest)
    cpt = 0
    for i, data in enumerate(testloader, 0):
        img_clean, _ = data
        img_clean = img_clean.to(device)
        img_noisy = add_noise(convolution_fft_torch(img_clean,kernel_torch),sigma_noise)
        outputs = net(img_noisy)

        # save as numpy array
        for k in range(img_clean.shape[0]):
            image_test_denoised[cpt] = outputs[k].detach().cpu().numpy().reshape(image_size,image_size)
            image_test_noisy[cpt] = img_noisy[k,0].detach().cpu().numpy()
            image_test_clean[cpt] = img_clean[k,0].detach().cpu().numpy()
            SNR_est[cpt] = SNR(img_clean[k:k+1,0].detach().cpu().numpy(),outputs[k:k+1].detach().cpu().numpy())
            SNR_noise[cpt] = SNR(img_clean[k:k+1,0].detach().cpu().numpy(),img_noisy[k:k+1,0].detach().cpu().numpy())
            cpt += 1
    print("Average SNR on reconstruction: {} || SNR on noisy images: {}".format(np.mean(SNR_est),np.mean(SNR_noise)))
        
    
    index = np.random.randint(Ntest)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
    ax[0].imshow(image_test_clean[index])
    ax[0].set_title("Image {} -- Clean".format(index))
    ax[1].imshow(image_test_noisy[index])
    ax[1].set_title("Blurry-Noisy")
    ax[2].imshow(image_test_denoised[index])
    ax[2].set_title("Denoised-Deblurred")
    plt.show()