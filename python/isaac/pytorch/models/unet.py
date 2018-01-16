import torch
import torch.nn as nn
import isaac.pytorch as sc
from .common import *

class UNet(nn.Module):

    def __init__(self, out_num=3, filters=[1,24,72,216,648], relu_slope=0.005):
        super(UNet, self).__init__()
        dim = 3

        # Attributes
        self.relu_slope = relu_slope
        self.filters = filters
        self.depth = len(filters) - 1
        self.out_num = out_num

        # Downward convolutions
        DownBlock = lambda in_num, out_num, pool: VggBlock(in_num, out_num, (3, 3, 3), (1, 2, 2), True, 'relu', relu_slope, pool, return_conv = True)
        self.down_conv = nn.ModuleList([DownBlock(filters[x], filters[x+1], x < self.depth - 1) for x in range(self.depth)])

        # Upward convolution
        UpBlock = lambda in_num, out_num: UpVggCropCatBlock(in_num, out_num, (3, 3, 3), (1, 2, 2), True, 'relu', relu_slope, 'cat')
        self.up_conv = nn.ModuleList([UpBlock(filters[x], filters[x-1]) for x in range(self.depth, 1, -1)])

        # Final layer
        self.final = sc.ConvType[dim](filters[1], out_num, (1, 1, 1), bias=True, activation='sigmoid', alpha=0)

    def forward(self, x):
        z = [None]*self.depth
        for i in range(self.depth):
            z[i], x = self.down_conv[i](x)
        for i in range(self.depth - 1):
            x = self.up_conv[i](x, z[self.depth - 2 - i])
        x = self.final(x)
        return x


    def quantize(self, x, approximate = True):
        quantizer = sc.Quantizer(approximate)
        for module in self.modules():
            if hasattr(module, 'set_quantizer'):
                module.set_quantizer(quantizer)
        self.forward(x)
        return self
