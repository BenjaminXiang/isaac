import torch
import torch.nn as nn
import isaac.pytorch as sc

# Building Block

class VggBlock(nn.Module):

    def __init__(self, in_num, out_num, kernel_size, window_size, bias, activation, alpha, pool, return_conv):
        super(VggBlock, self).__init__()
        dim = len(kernel_size)
        self.conv1 = sc.ConvType[dim](in_num, out_num, kernel_size, bias = bias, activation = activation, alpha = alpha)
        self.conv2 = sc.ConvType[dim](out_num, out_num, kernel_size, bias = bias, activation = activation, alpha = alpha)
        self.pool = sc.PoolType[dim](window_size, window_size) if pool else None
        self.return_conv = return_conv

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z_tmp, z_out = z if isinstance(z, tuple) else (z, z)
        y = self.pool(z_tmp) if self.pool is not None else z_out
        return (z_out, y) if self.return_conv else y


class UpVggCropCatBlock(nn.Module):

    def __init__(self, in_num, out_num, kernel_size, window_size, bias, activation, alpha, residual):
        super(UpVggCropCatBlock, self).__init__()
        dim = len(kernel_size)
        self.upsample = sc.ConvType[dim](in_num, out_num, (1,1,1), upsample = window_size, activation = 'linear', bias = bias, residual = residual)
        self.conv1 = sc.ConvType[dim](2*out_num, out_num, kernel_size, bias=bias, activation=activation, alpha=alpha)
        self.conv2 = sc.ConvType[dim](out_num, out_num, kernel_size, bias=bias, activation=activation, alpha=alpha)

    def forward(self, x, z):
        x = self.upsample(x, z)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
