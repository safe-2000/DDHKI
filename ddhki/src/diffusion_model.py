import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(encoded1)
        decoded1 = self.decoder1(encoded2)
        decoded2 = self.decoder2(decoded1 + encoded1)  # 跳跃连接
        return decoded2



# 定义UNet的卷积层块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


# 定义UNet模型
class UNet1(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(UNet, self).__init__()
        
        # 输入通道为1，输出通道为1，处理一维序列
        self.encoder1 = ConvBlock(1, hidden_dim)
        self.encoder2 = ConvBlock(hidden_dim, hidden_dim * 2)
        self.encoder3 = ConvBlock(hidden_dim * 2, hidden_dim * 4)

        self.decoder1 = ConvBlock(hidden_dim * 4, hidden_dim * 2)
        self.decoder2 = ConvBlock(hidden_dim * 2, hidden_dim)
        self.decoder3 = ConvBlock(hidden_dim, 1)

        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # 输入是 [batch_size, 128]，我们需要将其转换成 [batch_size, 1, 128] 以适应1D卷积
        x = x.unsqueeze(1)
        
        # 编码路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        # 解码路径
        d1 = self.up(e3)
        d1 = self.decoder1(d1)

        d2 = self.up(d1 + e2)  # 跳跃连接
        d2 = self.decoder2(d2)

        d3 = self.up(d2 + e1)  # 跳跃连接
        output = self.decoder3(d3)

        # 调整尺寸以匹配输入尺寸
        output = F.interpolate(output, size=x.size(2))

        # 输出是 [batch_size, 1, 128]，我们去掉第2维
        return output.squeeze(1)





class DiffusionModel(nn.Module):
    def __init__(self, input_dim, noise_steps=100, beta_start=0.0001, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # 定义beta schedule
        self.betas = torch.linspace(beta_start, beta_end, noise_steps).cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # U-Net模型
        self.unet = UNet(input_dim)
        
    def forward_diffusion(self, x_0, t):
        """前向扩散过程"""
        noise = torch.randn_like(x_0).cuda()
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    

    def reverse_diffusion(self, x_t, t):
        """逆向扩散过程"""
        return self.unet(x_t)
    










