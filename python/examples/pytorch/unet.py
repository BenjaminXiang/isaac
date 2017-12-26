import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import isaac.pytorch


def ConvBiasActivation(in_num, out_num, kernel_size, function, alpha):
    conv = nn.Conv3d(in_num, out_num, kernel_size=kernel_size, padding=0, stride=1, bias=True)
    if function == 'relu':
        act = nn.LeakyReLU(alpha)
    if function == 'sigmoid':
        act = nn.Sigmoid()
    if function == 'linear':
        return nn.Sequential(conv)
    return nn.Sequential(conv, act)

class UpConvCropCat(nn.Module):
    def __init__(self, strides, in_num, out_num):
        super(UpConvCropCat, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_num, in_num, strides, strides, groups=in_num, bias=False)
        self.upsample.weight.data.fill_(1.0)
        torch.manual_seed(0)
        self.conv_bias_relu = ConvBiasActivation(in_num, out_num, (1, 1, 1), function = 'linear', alpha = 1)

        
    def forward(self, x, z):
        x = self.upsample(x)
        x = self.conv_bias_relu(x)
        offset = [(z.size()[i]-x.size()[i])//2 for i in range(2,z.dim())]
        return torch.cat([x, z[:,:,offset[0]:offset[0]+x.size(2),
                                offset[1]:offset[1]+x.size(3),
                                offset[2]:offset[2]+x.size(4)]], 1)


class UNet3D(nn.Module):
    
    def ConvBiasActivation(self, in_num, out_num, kernel_size, function, alpha, with_isaac):
        if with_isaac:
            return isaac.pytorch.Conv3d(in_num, out_num, kernel_size, activation=function, alpha=alpha)
        else:
            return ConvBiasActivation(in_num, out_num, kernel_size, function, alpha)

    def UpConvCropCat(self, strides, in_num, out_num, with_isaac):
        if with_isaac:
            torch.manual_seed(0)
            return isaac.pytorch.Conv3dCropCat(in_num, out_num, (1,1,1), upsample=strides, activation = 'linear', bias = True)
        else:
            return UpConvCropCat((1,2,2), in_num, out_num)

    def MaxPool(self, kernel_size, stride, with_isaac):
        if with_isaac:
            return isaac.pytorch.MaxPool3d(kernel_size, stride)
        else:
            return nn.MaxPool3d(kernel_size, stride)

    def __init__(self, in_num=1, out_num=3, filters=[24,72,216,648],relu_slope=0.005,with_isaac=False):
        super(UNet3D, self).__init__()
        if len(filters) != 4: 
            raise AssertionError 
        filters = [in_num] + filters
        self.depth = len(filters) - 1
        self.with_isaac = with_isaac
        
        # Downward convolutions
        self.down_conv = nn.ModuleList([nn.Sequential(
                self.ConvBiasActivation(filters[x  ], filters[x+1], kernel_size = 3, function = 'relu', alpha = relu_slope, with_isaac = with_isaac),
                self.ConvBiasActivation(filters[x+1], filters[x+1], kernel_size = 3, function = 'relu', alpha = relu_slope, with_isaac = with_isaac))
                   for x in range(0, self.depth)])
        # Pooling
        self.pool = nn.ModuleList([self.MaxPool((1,2,2), (1,2,2), with_isaac = with_isaac)
                   for x in range(self.depth)])
        
        # Upsampling
        self.upsample = nn.ModuleList([self.UpConvCropCat((1,2,2), filters[x], filters[x-1], with_isaac = with_isaac) for x in range(self.depth, 1, -1)])
                   
        # Upward convolution
        self.up_conv = nn.ModuleList([nn.Sequential(
                self.ConvBiasActivation(2*filters[x-1], filters[x-1], kernel_size = 3, function = 'relu', alpha = relu_slope, with_isaac = with_isaac),
                self.ConvBiasActivation(  filters[x-1], filters[x-1], kernel_size = 3, function = 'relu', alpha = relu_slope, with_isaac = with_isaac))
                   for x in range(self.depth, 1, -1)])
                   
        # Final layer
        self.final = self.ConvBiasActivation(filters[1], out_num, kernel_size=1, function = 'sigmoid', alpha = 0, with_isaac = with_isaac)
    
    def forward(self, x):
        z = [None]*self.depth
        for i in range(self.depth):
            z[i] = self.down_conv[i](x)
            x = self.pool[i](z[i]) if i < self.depth - 1 else z[i]
        for i in range(self.depth - 1):
            x = self.upsample[i](x, z[self.depth - 2 - i])
            x = self.up_conv[i](x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    X = Variable(torch.Tensor(1, 1, 31, 204, 204).uniform_(0, 1)).cuda()
    torch.manual_seed(0)
    Y1 = UNet3D(with_isaac=False).cuda()(X)
    torch.manual_seed(0)
    Y2 = UNet3D(with_isaac=True).cuda()(X)
    error = torch.norm(Y1 - Y2)/torch.norm(Y1)
    print('Error: {}'.format(error.data[0]))
    
