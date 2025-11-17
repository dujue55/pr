import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from helper import data_generator, add_noise, SNR, convolution_fft_torch, Gaussian_blur
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from helper import plot_image

#===== 1. 定义 deblur net architecture (CNN_simple + 2 Upsample) =====
class DeblurringNet(nn.Module):
    def __init__(self, image_size=28):
        super().__init__()

        # ====== CNN_simple Encoder ======
        self.conv1 = nn.Conv2d(1, 60, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)      # 28 -> 14
        self.conv2 = nn.Conv2d(60, 40, kernel_size=3, padding=1)
        # pool again: 14 -> 7

        # 全连接层部分（扁平化 40×7×7 → FC）
        self.fc1 = nn.Linear(40 * (image_size//4)**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, image_size * image_size)  # 输出 flatten 的 28×28 图像

        # ====== 两层上采样======
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 28->56
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 56->112

        # 最终卷积层（恢复到 28×28）
        self.final_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):

        # ======== CNN_simple ========
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7

        x = torch.flatten(x, 1)                # (B, 40*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                        # 到 (B, 28*28)

        x = x.view(-1, 1, 28, 28)              # reshape 回 28×28

        # ======== 两层上采样 ========
        x = self.up1(x)                        # 28 -> 56
        x = self.up2(x)                        # 56 -> 112

        # ======== 缩回 28×28 ========
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)

        # ======== 最终卷积修复 ========
        x = self.final_conv(x)

        return x


# =====2. 训练与验证=====
if __name__ == '__main__' :
    # 2.1参数设置
    device = 'cpu'
    batch_size = 64
    sigma_noise = 0.005  # 噪声标准差
    sigma_blur = 1.5    # 模糊标准差
    lr = 0.001
    num_epochs = 50
    patience = 7        
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "deblurring.pt"

    print(f"Device set to : {device}")
    print(f"Loading data with batch size : {batch_size}")
    trainloader, valloader = data_generator(batch_size) # helper.py 中的数据加载

    # --- 2.2 确定图像尺寸和模糊核 ---
    dataiter = iter(trainloader)
    images, _ = next(dataiter)  # 这里的 'images' 包含了第一批干净图像
    image_size = images.shape[2] # 默认为 28

    # 计算模糊核 (在 CPU/GPU 上)
    kernel_numpy = Gaussian_blur(sigma_blur, image_size=image_size)
    kernel_torch = torch.from_numpy(kernel_numpy).to(device).reshape(1, 1, image_size, image_size).type(torch.float32)

    # --- 2.3 初始化网络、优化器、损失函数 ---
    net = DeblurringNet().to(device)
    # 打印参数量，作为文件大小（<15MB）的参考
    print('---> Network parameters: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # MSELoss 是去模糊任务最常用的损失函数
    criterion = nn.MSELoss() 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    print(f"Network initialized. Loss: {criterion.__class__.__name__}, Optimizer: {optimizer.__class__.__name__}")

    # --- 2.4 训练循环 ---
    print("Starting training...")

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            img_clean, _ = data
            img_clean = img_clean.to(device)

            # 1. 生成带模糊和噪声的输入图像 (y = h * u + eta)
            img_blur = convolution_fft_torch(img_clean, kernel_torch) # 模糊 (h * u)
            img_noisy = add_noise(img_blur, sigma_noise).to(device)    # 加噪声 (y)

            optimizer.zero_grad()

            # 2. 前向传播：网络尝试从 img_noisy 恢复 img_clean
            outputs = net(img_noisy)

            # 3. 计算损失：输出 (outputs) vs 目标 (img_clean)
            # 注意: 网络的输出是一个 (B, 1, H, W) 的图像张量，可以直接与 img_clean 计算损失
            loss = criterion(outputs, img_clean) 

            # 4. 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999: # 调整打印频率
                print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss / 1000:.6f}')
                running_loss = 0.0

        # --- 2.5 验证阶段 ---
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                img_clean, _ = data
                img_clean = img_clean.to(device)
                
                # 生成验证输入
                img_blur = convolution_fft_torch(img_clean, kernel_torch)
                img_noisy = add_noise(img_blur, sigma_noise).to(device)

                outputs = net(img_noisy)
                
                loss = criterion(outputs, img_clean)
                val_loss += loss.item()       

        avg_val_loss = val_loss / len(valloader)
        print(f'Epoch {epoch + 1} finished. Avg Val Loss: {avg_val_loss:.6f}')

        # --- 2.6 学习率调度和早停逻辑 ---
        scheduler.step(avg_val_loss) 
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 练习 8 的模型不需要 JIT，但评测脚本要求加载 JIT 模型。
            # 练习 8 要求保存的模型是 torch.jit.script 格式
            # 但为了简单起见，我们先保存 state_dict，然后用单独的脚本转换为 JIT。
            # 
            # ********** 练习 8 要求的模型保存方式 **********
            # net 是 nn.Module 实例
            net_scripted = torch.jit.script(net) 
            net_scripted.save(best_model_path)
            # **********************************************
            
            print(f"*** New best model saved as JIT script to {best_model_path} ***")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
                
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
            
    print('Training finished.')