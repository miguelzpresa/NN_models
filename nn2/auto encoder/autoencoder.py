import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tv
import PIL
import random
from delegation import delegates
from Layers import *
import fastai.vision.all as fv

class Autoencoder(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x):
        self.z = self.encoder(x) # guarda la última cosa encodeada
        return self.z
    
    def forward(self, x):
        z = self.encode(x)
        
        if self.training:
            z = z + 0.1*torch.randn_like(z)

        return self.decoder(z)

def create_autoencoder():
    encoder = nn.Sequential(
            nn.BatchNorm2d(3),
            ResBlock(3,32),
            ResBlock(32),
            *cab(32,64, s = 2, k = 2),
            ResBlock(64),
            nn.BatchNorm2d(64),
            ResBlock(64),
            ResBlock(64),
            *cab(64,128, s = 2, k = 2),
            nn.BatchNorm2d(128),
            ResBlock(128),
            ResBlock(128),
            SelfAttention(128),
            ResBlock(128),
            nn.BatchNorm2d(128),
            ResBlock(128,64),
            ResBlock(64,3),
            nn.BatchNorm2d(3)
    )
    
    decoder = nn.Sequential(
            ResBlock(3,64),
            SelfAttention(64),
            ResBlock(64),
            nn.BatchNorm2d(64),
            ResBlock(64),
            fv.PixelShuffle_ICNR(64,32),
            ResBlock(32,64),
            ResBlock(64),
            ResBlock(64),
            nn.BatchNorm2d(64),
            ResBlock(64,128),
            fv.PixelShuffle_ICNR(128,64),
            ResBlock(64,128),
            ResBlock(128),
            nn.BatchNorm2d(128),
            ResBlock(128),
            ResBlock(128),
            conv2d(128,3)
    )
    
    return Autoencoder(encoder, decoder)
