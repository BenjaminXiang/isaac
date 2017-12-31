import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import isaac.pytorch
from time import time
import timeit


class UNetBuilder(nn.Module):

    def __init__(self, out_num=3, filters=[1,24,72,216,648], relu_slope=0.005, with_isaac=False):
        super(UNetBuilder, self).__init__()
        # Attributes
        self.relu_slope = relu_slope
        self.filters = filters
        self.depth = len(filters) - 1
        self.out_num = out_num

        # Downward convolutions
        self.down_conv = nn.ModuleList([isaac.pytorch.VggBlock(filters[x], filters[x+1], 3, 'relu', relu_slope, x < self.depth - 1, with_isaac)
                                        for x in range(self.depth)])
        # Upward convolution
        self.up_conv = nn.ModuleList([isaac.pytorch.UpVggCropCatBlock(filters[x], filters[x-1], 3, 'relu', relu_slope, with_isaac)
                                        for x in range(self.depth, 1, -1)])
        # Final layer
        self.final = isaac.pytorch.ConvBiasActivation(filters[1], out_num, kernel_size=1, activation = 'sigmoid', alpha = 0, with_isaac = with_isaac)

    def forward(self, x):
        z = [None]*self.depth
        for i in range(self.depth):
            z[i], x = self.down_conv[i](x)
        for i in range(self.depth - 1):
            x = self.up_conv[i](x, z[self.depth - 2 - i])
        x = self.final(x)
        torch.cuda.synchronize()
        return x


class UNet(UNetBuilder):
    def __init__(self, out_num=3, filters=[1,24,72,216,648],relu_slope=0.005):
        super(UNet, self).__init__(out_num, filters, relu_slope, False)


class UNetInference(UNetBuilder):
    def copy(self, x, y):
        x.weight.data = y.weight.data.permute(1, 2, 3, 4, 0).clone()
        x.bias.data = y.bias.data

    def __init__(self, base):
        super(UNetInference, self).__init__(base.out_num, base.filters, base.relu_slope, True)

        # ISAAC only work on GPUs for now
        self.cuda()

        # Copy weights
        for (x, y) in zip(self.down_conv, base.down_conv):
            self.copy(x.conv1, y.conv1[0])
            self.copy(x.conv2, y.conv2[0])
        for (x, y) in zip(self.up_conv, base.up_conv):
            self.copy(x.upsample, y.upsample.conv_bias_relu[0])
            self.copy(x.conv1, y.conv1[0])
            self.copy(x.conv2, y.conv2[0])
        self.copy(self.final, base.final[0])

    def quantize(self, x):
        history = dict()
        for module in self.down_conv:
            module.arm_quantization(history)
        for module in self.up_conv:
            module.arm_quantization(history)
        self.final.arm_quantization(history)
        self.forward(x)


if __name__ == '__main__':
    torch.manual_seed(0)
    X = Variable(torch.Tensor(1, 1, 31, 204, 204).uniform_(0, 1)).cuda()

    # Build models
    unet_ref = UNet().cuda()
    unet_sc = UNetInference(unet_ref).cuda()
    unet_sc.quantize(X)

    # Test correctness
    y_ref = unet_ref(X)
    y_sc = unet_sc(X)
    error = torch.norm(y_ref - y_sc)/torch.norm(y_ref)
    print('Error: {}'.format(error.data[0]))

    # Benchmark
    t_sc = [int(x*1e3) for x in timeit.repeat(lambda: unet_sc(X), repeat=1, number=1)]
    t_ref = [int(x*1e3) for x in timeit.repeat(lambda: unet_ref(X), repeat=1, number=1)]
    print('Time: {}ms (Isaac) ; {}ms (PyTorch)'.format(t_sc[0], t_ref[0]))




