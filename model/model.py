import os
import torch
from layers import *
import torch.nn as nn
from torch import optim
from torchvision import utils
from tqdm import trange, tqdm
import torch.nn.functional as F
import torchvision.transforms as TF

class UNetLevel(nn.Module):
    def __init__(self,
                 *channels: list[int],
                 next_level = None,
                 num_attention_head=None, 
                 positional_encoding_type=None):
        assert len(channels) >= 2, 'UNetLevel requires at least input and output channels'
        super().__init__()
        self.encoder = nn.Sequential(
            *[ResConvBlock(c_in, c_out, c_out) for c_in, c_out in zip(channels[:-1], channels[1:])]
        )

        if next_level is not None:
            self.next_level = nn.Sequential(
                nn.Conv2d(channels[-1], channels[-1], 3, stride=2, padding=1),
                next_level,
                nn.ConvTranspose2d(channels[-1], channels[-1], 3, stride=2, padding=1, output_padding=1)
            )
        else:
            self.next_level = None

        if positional_encoding_type == FourierPositionalEncoding2d:
            self.encoder.add_module('position encoder', FourierPositionalEncoding2d(channels[-1]))
        elif positional_encoding_type == PositionalEncoding2d:
            self.encoder.add_module('position encoder', PositionalEncoding2d())
        else:
            self.positional_encoding = None

        if num_attention_head is not None:
            self.encoder.add_module('self attention', SelfAttention(channels[-1], num_attention_head))
        
    
        if self.next_level is not None:
            channels = list(channels) # for modify, it can't be tuple
            channels[-1] *= 2 # concatenate skip connection
        self.decoder = nn.Sequential(
            *[ResConvBlock(c_in, c_in, c_out) for c_out, c_in in zip(reversed(channels[:-1]), reversed(channels[1:]))]
        )
    
    def forward(self, x):
        if self.next_level is not None:
            x_enc = self.encoder(x)
            x_next = self.next_level(x_enc)
            x_dec = self.decoder(torch.cat([x_enc, x_next], dim=1))
            return x_dec
        else:
            x_enc = self.encoder(x)
            x_dec = self.decoder(x_enc)
            return x_dec

class UNet(nn.Module):
    def __init__(self,
                *channels: list[int],
                num_blocks: int = 2,
                num_attention_head: int = 4,
                positional_encoding_type = FourierPositionalEncoding2d):
        assert  len(channels) >= 2, 'UNet at least have 2 levels'
        super().__init__()
        unet = UNetLevel(*([channels[-2]] + [channels[-1]] * num_blocks), num_attention_head=num_attention_head, positional_encoding_type=positional_encoding_type, next_level=
                           UNetLevel(*([channels[-1]] + [channels[-1]] * num_blocks), num_attention_head=num_attention_head, positional_encoding_type=positional_encoding_type, next_level=None))
        for level in range(len(channels) - 2, 0, -1):
            unet = UNetLevel(*([channels[level-1]] + [channels[level]] * num_blocks), next_level=unet)
        self.unet = unet
    
    def forward(self, x):
        return self.unet(x)

if __name__ == '__main__':
    model = UNet(3, 64, 128, 256, num_blocks=2, num_attention_head=4, positional_encoding_type=FourierPositionalEncoding2d)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(y.shape)