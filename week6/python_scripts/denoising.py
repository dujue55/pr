import numpy as np 

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from helper import data_generator, plot_image, add_noise, CNN_simple, SNR

use_cuda=torch.cuda.is_available()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = "cpu"

print("Device:",device)



if __name__ == '__main__':
    batch_size = 32
    sigma = 0.5 

    trainloader, testloader = data_generator(batch_size)

    # Display one image to see what it looks like
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    image_size = images.shape[2]
    plot_image(images,num_fig=1)
    plot_image(add_noise(images,sigma),num_fig=2)
    

    net = CNN_simple(image_size).to(device)
    print('---> Number of trainable parameters of supercnn: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    net.load_state_dict(torch.load("denoising.pt"))

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
        img_noisy = add_noise(img_clean,sigma)
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
    ax[1].set_title("Noisy")
    ax[2].imshow(image_test_denoised[index])
    ax[2].set_title("Denoised")
    plt.show()