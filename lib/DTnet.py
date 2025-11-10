import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np


class IncResBlock(nn.Module):
    def __init__(self, inplanes, planes, convstr=1, convsize=15, convpadding=7):
        super(IncResBlock, self).__init__()
        self.Inputconv1x1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=convsize, stride=convstr, padding=convpadding),
            nn.BatchNorm1d(planes//4))
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride=convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(planes//4, planes//4, kernel_size=convsize+2, stride=convstr, padding=convpadding+1),
            nn.BatchNorm1d(planes//4))
        self.conv1_3 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride=convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(planes//4, planes//4, kernel_size=convsize+4, stride=convstr, padding=convpadding+2),
            nn.BatchNorm1d(planes//4))
        self.conv1_4 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride=convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(planes//4, planes//4, kernel_size=convsize+6, stride=convstr, padding=convpadding+3),
            nn.BatchNorm1d(planes//4))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.Inputconv1x1(x)
        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)
        out = torch.cat([c1, c2, c3, c4], 1)
        out += residual
        out = self.relu(out)
        return out


class SmallIncUNet(nn.Module):
    """Smaller version of IncUNet with fewer layers and reduced channels"""
    def __init__(self, in_shape):
        super(SmallIncUNet, self).__init__()
        in_channels, height, width = in_shape
        
        # Encoder - 5 levels for better depth
        self.e1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            IncResBlock(32, 32))
        
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            IncResBlock(64, 64))
        
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            IncResBlock(128, 128))
        
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256, 256))
        
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256))
        
        # Decoder - 5 levels with matching dimensions
        self.d1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256, 256))
        
        self.d2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            IncResBlock(128, 128))
        
        self.d3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            IncResBlock(64, 64))
        
        self.d4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32))
        
        self.out_l = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(1, 1, kernel_size=9, padding=4)
            )


    def forward(self, x):
        x = x.transpose(1, 2)
        
        # Encoder
        en1 = self.e1(x)      # 32 channels
        en2 = self.e2(en1)    # 64 channels
        en3 = self.e3(en2)    # 128 channels
        en4 = self.e4(en3)    # 256 channels
        en5 = self.e5(en4)    # 256 channels (bottleneck)
        
        # Decoder with skip connections
        de1_ = self.d1(en5)                    # 256 channels
        de1 = torch.cat([en4, de1_], 1)        # 256 + 256 = 512 channels
        
        de2_ = self.d2(de1)                    # 128 channels        
        de2 = torch.cat([en3, de2_], 1)        # 128 + 128 = 256 channels
        
        de3_ = self.d3(de2)                    # 64 channels
        de3 = torch.cat([en2, de3_], 1)        # 64 + 64 = 128 channels
        
        de4_ = self.d4(de3)                    # 32 channels
        de4 = torch.cat([en1, de4_], 1)        # 32 + 32 = 64 channels
        
        out = self.out_l(de4)                  # 1 channel (output)
        
        return out