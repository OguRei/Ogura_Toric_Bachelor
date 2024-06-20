import math 
import os 
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.append(os.path.abspath("../"))
from activations import Activation
from misc import PositionalEncoding
from utils import init_weight

class Conv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 stride: int = 1,
                 bias: bool = True,
                 padding: str = "zeros",
                 activation: str = "leaky_relu",
                 activation_args: Dict = {},
                 batch_norm: bool = False,
                 dropout: float = None,
                 weight_init: str = "normal"):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.use_batch_norm = batch_norm
        self.use_dropout = dropout is not None
        padding_size = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=False if self.use_batch_norm else bias)
        init_weight(self.conv.weight, weight_init)
        padding_size = (math.ceil(padding_size / 2), math.floor(padding_size / 2))
        self.padding_size = (*padding_size, *padding_size)
        self.padding_mode = padding if padding != "zeros" else "constant"

        self.activation = Activation(activation,
                                     channels=out_channels,
                                     **activation_args)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if dropout is not None:
            self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, input):
        x = F.pad(input, self.padding_size, mode=self.padding_mode, value=0.)
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x
        
class ResBlockKxK(nn.Module):
    def __init__(self,
                 channels: int,
                 size: int = 2,
                 padding: str = "zeros",
                 activation: str = "leaky_relu",
                 activation_args: Dict = {},
                 weight_init: str = "normal",
                 dp: float = None,
                 pre_activation: bool = None):
        '''
        ResBlock using conv(k, 1) and conv(1, k) instead of conv(k, k)
        '''
        super(ResBlockKxK, self).__init__()
        assert activation != "glu", "not supported activation: glu"
        self.pre_activation = pre_activation if pre_activation is not None else dp is not None

        self.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                padding=size - 1,
                padding_mode=padding,
                kernel_size=(size, size),
                bias=False)
        init_weight(self.conv1.weight, weight_init)
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation1 = Activation(activation, channels=channels, **activation_args)
        self.conv2 = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(size, size),
                bias=self.pre_activation)
        init_weight(self.conv2.weight, weight_init)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation2 = Activation(activation, channels=channels, **activation_args)

        assert dp is None or pre_activation, \
               "if use dropout, must be pre_activation"
        self.dp = dp
        if dp is not None:
            self.dropout = nn.Dropout2d(p=dp)

    def forward(self, input):
        if self.pre_activation:
            x = self.bn1(input)
            x = self.activation1(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.activation2(x)
            if self.dp is not None:
                x = self.dropout(x)
            x = self.conv2(x)
            return input + x
        else:
            x = self.conv1(input)
            x = self.bn1(x)
            x = self.activation1(x)
            if self.dp is not None:
                x = self.dropout(x)
            x = self.conv2(x)
            x = self.bn2(x)
            return self.activation2(input + x)
    
class SelfAttention(nn.Module): #層厚くすればいいから今回使ってない
    def __init__(self, size: int, nhead: int, channels: int, auto: bool):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.pe = PositionalEncoding(pe_size=size, ndim=channels, auto=auto)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=nhead)

    def forward(self, input):
        if self.pe is not None:
            input = self.pe(input)
        x = input.flatten(start_dim=2).permute(2, 0, 1)
        return self.multihead_attn(x, x, x)[0].permute(1, 2, 0).reshape(*input.shape)
    

class CNN(nn.Module):
    def __init__(self,
                 size: int,
                 kernel_size: int,
                 resnet_layers: int,
                 channels: int,
                 padding: str = "zeros",
                 activation: str = "leaky_relu",
                 activation_args: Dict = {},
                 batch_norm: bool = False,
                 dropout: Optional[float] = None,
                 nhead: int = 1, 
                 pe_auto: bool = True,
                 weight_init: str = "normal"):
        super(CNN, self).__init__()
        # Backbone 
        self.features = nn.Sequential(
            Conv(in_channels=1,
                 out_channels=channels,
                 kernel_size=kernel_size,
                 padding=padding,
                 activation=activation,
                 activation_args=activation_args,
                 batch_norm=True,
                 dropout=None,
                 weight_init=weight_init),
            ResBlockKxK(
                size=2,
                channels=channels,
                padding=padding,
                activation=activation,
                activation_args=activation_args,
                pre_activation=True,
                dp=dropout,
                weight_init=weight_init),
            *[
                ResBlockKxK(
                    size=2,
                    channels=channels,
                    padding="circular",
                    activation=activation,
                    activation_args=activation_args,
                    pre_activation=True,
                    dp=dropout,
                    weight_init=weight_init)
                for _ in range(resnet_layers)])
        
        self.cnn = nn.Sequential( #Tail
            Conv(in_channels=channels,
                 out_channels=128,
                 kernel_size=kernel_size,
                 activation=activation,
                 activation_args=activation_args,
                 padding=padding,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 weight_init=weight_init),
            ResBlockKxK(
                size=2,
                channels=128,
                padding=padding,
                activation=activation,
                activation_args=activation_args,
                pre_activation=True,
                dp=dropout,
                weight_init=weight_init),
            Conv(in_channels=128,
                 out_channels=32,
                 kernel_size=kernel_size,
                 activation=activation,
                 activation_args=activation_args,
                 padding=padding,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 weight_init=weight_init),
            ResBlockKxK(
                size=2,
                channels=32,
                padding=padding,
                activation=activation,
                activation_args=activation_args,
                pre_activation=True,
                dp=dropout,
                weight_init=weight_init),
            Conv(in_channels=32,
                 out_channels=8,
                 kernel_size=kernel_size,
                 activation=activation,
                 activation_args=activation_args,
                 padding=padding,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 weight_init=weight_init),
            ResBlockKxK(
                size=2,
                channels=8,
                padding=padding,
                activation=activation,
                activation_args=activation_args,
                pre_activation=True,
                dp=dropout,
                weight_init=weight_init),
            Conv(in_channels=8,
                 out_channels=2,
                 kernel_size=kernel_size,
                 padding=padding,
                 activation="none",
                 batch_norm=True,
                 dropout=None,
                 weight_init=weight_init),
        )
        self.norm=F.normalize
    def forward(self, input, use_sigmoid: bool = True):
        input = input.to(torch.float32)
        x = self.features(input)
        used_cnn = self.cnn(x)
        normalized = self.norm(used_cnn)
        if use_sigmoid:
            return torch.sigmoid(normalized)
        else:
            return self.cnn(x)

