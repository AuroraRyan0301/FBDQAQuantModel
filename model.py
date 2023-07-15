import torch.nn as nn
import torch

import math
import torch.nn.functional as F
import torchvision.models as models

import os
import re

import numpy as np

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Res50Transformer(nn.Module):
    def __init__(self, num_classes):
        super(Res50Transformer, self).__init__()
        self.num_classes = num_classes
        self.resnet50 = models.resnet50()
        self.layer5 = nn.Sequential(
            ResNetBlock(2048, 2048, stride=1),
        )
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.transformer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, channel=1, seq_len=100, feature=32)
        # x = x.squeeze(dim=1) # (batch_size, seq_len, feature)
        # x = x.permute(0, 2, 1) # (batch_size, feature, seq_len)
        x = self.conv1(x) # (batch_size, 64, seq_len/4, feature/4)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x) # (batch_size, 64, seq_len/8, feature/8)

        # ResNet blocks
        x = self.resnet50.layer1(x) # (batch_size, 256, seq_len/8, feature/8)
        x = self.resnet50.layer2(x) # (batch_size, 512, seq_len/16, feature/16)
        x = self.resnet50.layer3(x) # (batch_size, 1024, seq_len/32, feature/32)
        x = self.resnet50.layer4(x) # (batch_size, 2048, seq_len/64, feature/64)
        
        x = x.squeeze(-1)  # (batch_size, 2048, seq_len/64)
        
        # Apply Transformer layer
        x = x.permute(2, 0, 1)  # (seq_len/64, batch_size, 2048)
        x = self.transformer(x) # (seq_len/64, batch_size, 2048)
        x = x.permute(1, 2, 0)  # (batch_size, 2048, seq_len/64)
        
        x = x.unsqueeze(-1) #(batch_size, 2048, seq_len/64, 1)
        x = self.layer5(x) # (batch_size, 256, seq_len/8, feature/8)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, output_size=(1)) # (batch_size, 2048, 1, 1)
        x = x.view(x.size(0),-1) # (batch_size, 2048)
        # Fully connected layers
        x = self.fc1(x) # (batch_size, 512)
        x = F.relu(x)
        x = self.fc2(x) # (batch_size, output_dim)
        
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.resnet50 = models.resnet50()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        
        
    def forward(self, x):
        # x shape: (batch_size, channel=1, seq_len=100, feature=32)
        # x = x.squeeze(dim=1) # (batch_size, seq_len, feature)
        # x = x.permute(0, 2, 1) # (batch_size, feature, seq_len)
        x = self.conv1(x) # (batch_size, 64, seq_len/4, feature/4)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x) # (batch_size, 64, seq_len/8, feature/8)

        # ResNet blocks
        x = self.resnet50.layer1(x) # (batch_size, 256, seq_len/8, feature/8)
        x = self.resnet50.layer2(x) # (batch_size, 512, seq_len/16, feature/16)
        x = self.resnet50.layer3(x) # (batch_size, 1024, seq_len/32, feature/32)
        x = self.resnet50.layer4(x) # (batch_size, 2048, seq_len/64, feature/64)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1) # (batch_size, 2048)
        
        # Fully connected layers
        x = self.fc1(x) # (batch_size, 512)
        x = F.relu(x)
        x = self.fc2(x) # (batch_size, output_dim)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_table[:, 0::2] = torch.sin(position * div_term)
        pos_table[:, 1::2] = torch.cos(position * div_term)
        self.pos_table = pos_table.unsqueeze(0).to(torch.device("cuda"))  # 将pos_table张量转换为CUDA张量

    def forward(self, enc_inputs):
        enc_inputs += self.pos_table[:, :enc_inputs.size(1), :]
        return self.dropout(enc_inputs)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=3, d_model=128, nhead=8, num_layers=1):
        super(StockTransformer, self).__init__()
        self.src_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEncoding(d_model).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.src_emb(x)
        # x shape: (batch_size, seq_len, d_model)
        x = self.pos_emb(x)
        # x shape: (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)
        # x shape: (batch_size, seq_len, d_model)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        # x shape: (batch_size, d_model)
        x = self.fc(x)
        # x shape: (batch_size, output_dim)
        x = torch.softmax(x, dim=1)  # 使用softmax函数对输出进行归一化，以便输出概率分布
        return x

class deeplob(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
#             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,8)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
       
        # lstm layers
        self.fc = nn.Sequential(nn.Linear(384, 64),nn.Linear(64, self.num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.reshape(-1,48*8)
        x = self.fc(x)

        forecast_y = torch.softmax(x, dim=1)

        return forecast_y


