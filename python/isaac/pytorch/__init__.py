import torch
import torch.nn as nn
from torch.autograd import Function
from ctypes import *
from .c_lib import *
import cffi


class ConvNdFunction(Function):
    def __init__(self, activation, alpha, pad = (0, 0, 0), strides = (1, 1, 1), upsample = (1, 1, 1), crop = (0, 0, 0, 0, 0, 0)):
        self.activation = activation.encode('utf-8')
        self.alpha = float(alpha)
        self.pad = pad
        self.strides = strides
        self.upsample = upsample
        self.crop = crop
        self.ffi = cffi.FFI()

    def forward(self, input, weight, bias, z):
        z = z if z.size() else self.ffi.NULL
        bias = bias if bias.size() else self.ffi.NULL
        output = input.new()
        isaac_conv_nd(input, weight, output,
                      self.upsample[0], self.upsample[1], self.upsample[2], # Upsample
                      self.pad[0], self.pad[1], self.pad[2], # Pad
                      self.strides[0], self.strides[1], self.strides[2], # Strides
                      bias, # Bias
                      self.activation, self.alpha, # Activation
                      z, self.crop[0], self.crop[1], self.crop[2], self.crop[3], self.crop[4], self.crop[5], # Crop-cat
                      )
        return output

class MaxPoolNdFunction(Function):
    def __init__(self, kernel_size, pad = (0, 0, 0), strides = (1, 1, 1)):
        self.kernel_size = kernel_size
        self.pad = pad
        self.strides = strides
        self.ffi = cffi.FFI()

    def forward(self, input):
        output = input.new()
        isaac_max_pool_nd(input, output,
                      self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                      self.pad[0], self.pad[1], self.pad[2],
                      self.strides[0], self.strides[1], self.strides[2])
        return output

#############################
###      Convolutions     ###
#############################

# 2D Conv
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0.):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.alpha = alpha
        self.upsample = upsample
        self.weight.data = self.weight.data.permute(1, 2, 3, 0)

    def forward(self, x):
        return ConvNdFunction(self.activation, self.alpha, self.upsample)(x, self.weight, self.bias)

# 3D Conv
class Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0.):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.alpha = alpha
        self.upsample = upsample
        self.weight.data = self.weight.data.permute(1, 2, 3, 4, 0)

    def forward(self, x):
        return ConvNdFunction(self.activation, self.alpha)(x, self.weight, self.bias, torch.autograd.Variable())

# 3D Conv-Crop-Cat
class Conv3dCropCat(Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0.):
        super(Conv3dCropCat, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, upsample, groups, bias, activation, alpha)
        
    def forward(self, x, z):
        offset = [(z.size()[i]-x.size()[i]*self.upsample[i - 2])//2 for i in range(2,z.dim())]
        bias = self.bias if self.bias is not None else torch.autograd.Variable()
        crop = (offset[0], offset[0], offset[1], offset[1], offset[2], offset[2])
        return ConvNdFunction(self.activation, self.alpha, upsample=self.upsample, crop=crop)(x, self.weight, bias, z)

#############################
###      Pooling          ###
#############################

# 3D Pool
class MaxPool3d(nn.MaxPool3d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool3d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        return MaxPoolNdFunction(self.kernel_size, strides=self.stride)(x)
