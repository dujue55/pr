import torch
import torch.nn as nn
import torch.optim as optim 
from helper import data_generator, add_noise, CNN_simple
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1. setting device
device = 'cpu'


# 2. uploding dataset and setting huperparameter
if __name__ == '__main__':
    batch_size = 64
    sigma = 0.5
    lr = 0.001
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    best_model_path = "denoising.pt"
    num_epochs = 50
    print(f"Device set to : {device}")

    print(f"Loading data with batch size : {batch_size}")
    trainloader, valloader = data_generator(batch_size) #helper.py下载MNIST，分成trainloader和testloader

    # 3. abtain images size and initionalize net
    dataiter = iter(trainloader) # 将这个迭代器赋值给变量 dataiter
    images, _ = next(dataiter)
    image_size = images.shape[2]

    #4. initionalize net, optimizer, loss function
    net = CNN_simple(image_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr = lr)
    criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) # 每5个epoch学习率减为十分之一

    print(f"Network initialized. Using Loss: {criterion.__class__.__name__}, Optimizer: {optimizer.__class__.__name__}")

    #=======train model=========
    print("Starting training...")

    for epoch in range(num_epochs):
        net.train() # 设置为训练模式
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            #1.clean image
            img_clean, _ = data
            img_clean = img_clean.to(device)

            #2. add noise
            img_noisy = add_noise(img_clean, sigma).to(device)

            #3. 梯度归0， 防止累计
            optimizer.zero_grad()

            #4. 向前传播
            outputs = net(img_noisy)

            #5. compute Loss
            target = img_clean.view(img_clean.size(0), -1)
            loss = criterion(outputs, target)

            #6. 反向传播，优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每隔 2000 个批次打印一次统计信息
            if i % 2000 == 1999:  
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.6f}')
                running_loss = 0.0


    #=====Validation Phase=====
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                img_clean, _ = data
                img_clean = img_clean.to(device)
                img_noisy = add_noise(img_clean, sigma).to(device)

                outputs = net(img_noisy)
                target = img_clean.view(img_clean.size(0), -1)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                    
        avg_val_loss = val_loss / len(valloader)
        print(f'Epoch {epoch + 1} finished. Average Validation Loss: {avg_val_loss:.6f}')

    #===== C. 学习率调度和早停逻辑 =====
        scheduler.step(avg_val_loss) 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(net.state_dict(), best_model_path)
            print(f"*** New best model saved to {best_model_path} ***")
        else:
            epochs_no_improve += 1
                
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            early_stop = True
            break
                
    if not early_stop:
        print('Finished Training (Max Epochs Reached).')
    else:
        print('Training finished due to Early Stopping.')

    # --- 5. 确保最终保存的是最好的模型（如果触发了早停）---
    # 如果是早停，denoising.pt已经是最好的模型。
    # 如果是跑完所有epoch，我们也要确保使用最好的模型来评估。
    # 最好在训练结束后加载一遍 best_model_path 以确保最终net是最好的版本
    net.load_state_dict(torch.load(best_model_path))
    print("Final model state loaded (best checkpoint).")
