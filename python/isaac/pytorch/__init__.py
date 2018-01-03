import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple

from torch.autograd import Function
from ctypes import *
from .c_lib import *
import cffi


def pad_left(dim, x, value):
    return (value,)*(dim-len(x)) + x


def PackNd(input, alpha, beta):
    output = torch.Tensor().cuda()
    isaac_pack_nd(input, output, alpha, beta)
    return output


class ConvNdFunction(Function):
    def __init__(self, activation, alpha, scale, pad = (0, 0, 0), strides = (1, 1, 1), upsample = (1, 1, 1), crop = (0, 0, 0, 0, 0, 0), quantized_in = False, quantized_out = False):
        self.activation = activation.encode('utf-8')
        self.alpha = float(alpha)
        self.scale = tuple(map(float, scale))
        self.pad = pad
        self.strides = strides
        self.upsample = upsample
        self.crop = crop
        self.ffi = cffi.FFI()
        self.quantized_in = quantized_in
        self.quantized_out = quantized_out

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
                      self.quantized_in, self.quantized_out, self.scale[0], self.scale[1], self.scale[2], # Quantization
                      z, self.crop[0], self.crop[1], self.crop[2], self.crop[3], self.crop[4], self.crop[5]# Crop-cat
                      )
        return output

class MaxPoolNdFunction(Function):
    def __init__(self, kernel_size, pad = (0, 0, 0), strides = (1, 1, 1), quantized = False):
        self.kernel_size = pad_left(3, kernel_size, 1)
        self.pad = pad_left(3, pad, 1)
        self.strides = pad_left(3, strides, 1)
        self.ffi = cffi.FFI()
        self.quantized = quantized

    def forward(self, input):
        output = input.new()
        isaac_max_pool_nd(input, output,
                      self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                      self.pad[0], self.pad[1], self.pad[2],
                      self.quantized,
                      self.strides[0], self.strides[1], self.strides[2])
        return output

#############################
##       Quantization      ##
#############################
class Quantizer:

    def scale(self, x):
        return 127./torch.max(x)

    def __init__(self, history, weights):
        self.history = history
        if weights is not None:
            self.weights = weights
            self.quantize_in = weights.data.size()[0] % 4 == 0
            self.quantize_out = weights.data.size()[-1] % 4 == 0

    def scales(self, x, y):
        result = [1., 1., 1.]
        if self.quantize_in:
            result[0] = self.history[id(x)]
            result[1] = self.scale(self.weights.data)
        if self.quantize_out:
            result[2] = self.history[id(y)] = self.scale(x.data)
        return result

#############################
###      Convolutions     ###
#############################

def to_chwn_idx(dim):
    return list(range(1, dim)) + [0]

def from_chwn_idx(dim):
    return [dim-1] + list(range(0, dim-1))

class ConvNd(nn.modules.conv._ConvNd):

    def quantize_if_requested(self, x, y):
        if self.quantizer:
            # Update properties
            self.scale = self.quantizer.scales(x, y)
            self.quantized_in = self.quantizer.quantize_in
            self.quantized_out = self.quantizer.quantize_out
            self.quantizer = None
            # Quantize weights
            if self.quantized_in:
                self.weight.data = PackNd(self.weight.data.permute(*from_chwn_idx(self.dim)), self.scale[1], 0.0)
                self.weight.data = self.weight.data.permute(*to_chwn_idx(self.dim))

    def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding, dilation, upsample, groups, bias, activation, alpha, scale):
        super(ConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias)
        self.activation = activation
        self.alpha = alpha
        self.scale = scale
        self.upsample = upsample
        self.quantizer = None
        self.quantized_in = False
        self.quantized_out = False
        self.dim = dim
        self.weight.data = self.weight.data.permute(*to_chwn_idx(self.dim))

    def arm_quantization(self, history):
        self.quantizer = Quantizer(history, self.weight)

    def forward(self, x):
        y = ConvNdFunction(self.activation, self.alpha, self.scale, quantized_in=self.quantized_in, quantized_out = self.quantized_out)\
                          (x, self.weight, self.bias, torch.autograd.Variable())
        self.quantize_if_requested(x, y)
        return y

class ConvNdCropCat(ConvNd):
    def __init__(self, *args):
        super(ConvNdCropCat, self).__init__(*args)

    def forward(self, x, z):
        offset = [(z.size()[i]-x.size()[i]*self.upsample[i - 2])//2 for i in range(2,z.dim())]
        bias = self.bias if self.bias is not None else torch.autograd.Variable()
        crop = (offset[0], offset[0], offset[1], offset[1], offset[2], offset[2])
        y = ConvNdFunction(self.activation, self.alpha, self.scale, upsample=self.upsample, crop=crop, quantized_in = self.quantized_in, quantized_out = self.quantized_out)\
                          (x, self.weight, bias, z)
        self.quantize_if_requested(x, y)
        return y

# 1D Conv
class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., 1.]):
        super(Conv1d, self).__init__(4, in_channels, out_channels, _single(kernel_size), _single(stride), _single(padding), _single(dilation), upsample, groups, bias, activation, alpha, scale)

class Conv1dCropCat(ConvNdCropCat):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., 1.]):
        super(Conv1dCropCat, self).__init__(4, in_channels, out_channels, _single(kernel_size), _single(stride), _single(padding), _single(dilation), upsample, groups, bias, activation, alpha, scale)


# 2D Conv
class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., 1.]):
        super(Conv2d, self).__init__(4, in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), upsample, groups, bias, activation, alpha, scale)

class Conv2dCropCat(ConvNdCropCat):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., 1.]):
        super(Conv2dCropCat, self).__init__(4, in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), upsample, groups, bias, activation, alpha, scale)


# 3D Conv
class Conv3d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., 1.]):
        super(Conv3d, self).__init__(5, in_channels, out_channels, _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation), upsample, groups, bias, activation, alpha, scale)

class Conv3dCropCat(ConvNdCropCat):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., 1.]):
        super(Conv3dCropCat, self).__init__(5, in_channels, out_channels, _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation), upsample, groups, bias, activation, alpha, scale)


#############################
###      Pooling          ###
#############################

class MaxPoolNd(nn.Module):
    def quantize_if_requested(self, x, y):
        if self.quantizer:
            self.quantizer.history[id(y)] = self.quantizer.history[id(x)]
            self.quantizer = None
            self.quantized = True

    def __init__(self, kernel_size, stride):
        super(MaxPoolNd, self).__init__()
        self.quantizer = None
        self.quantized = False
        self.kernel_size = kernel_size
        self.stride = stride

    def arm_quantization(self, history):
        self.quantizer = Quantizer(history, None)

    def forward(self, x):
        y = MaxPoolNdFunction(self.kernel_size, strides=self.stride, quantized=self.quantized)(x)
        self.quantize_if_requested(x, y)
        return y


class MaxPool1d(MaxPoolNd):
    def __init__(self, kernel_size, stride=1):
        super(MaxPool1d, self).__init__(kernel_size, _single(stride))

class MaxPool2d(MaxPoolNd):
    def __init__(self, kernel_size, stride=1):
        super(MaxPool2d, self).__init__(kernel_size, _pair(stride))

class MaxPool3d(MaxPoolNd):
    def __init__(self, kernel_size, stride=1):
        super(MaxPool3d, self).__init__(kernel_size, _triple(stride))


#############################
###     Modules           ###
#############################
def ConvBiasActivation(in_num, out_num, kernel_size, activation, alpha, with_isaac):
    dim = len(kernel_size)
    if with_isaac:
        Type = [Conv1d, Conv2d, Conv3d][dim - 1]
        return Type(in_num, out_num, kernel_size, activation=activation, alpha=alpha)
    else:
        Type = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]
        conv = Type(in_num, out_num, kernel_size=kernel_size, padding=0, stride=1, bias=True)
        if activation == 'relu':
            act = nn.LeakyReLU(alpha)
        if activation == 'sigmoid':
            act = nn.Sigmoid()
        if activation == 'linear':
            return nn.Sequential(conv)
        return nn.Sequential(conv, act)

def MaxPool(kernel_size, stride, with_isaac):
    dim = len(kernel_size)
    if with_isaac:
        return [MaxPool1d, MaxPool2d, MaxPool3d][dim - 1](kernel_size, stride)
    else:
        return [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dim - 1](kernel_size, stride)

class UpConvCropCat(nn.Module):
    def __init__(self, strides, in_num, out_num):
        super(UpConvCropCat, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_num, in_num, strides, strides, groups=in_num, bias=False)
        self.upsample.weight.data.fill_(1.0)
        self.conv_bias_relu = ConvBiasActivation(in_num, out_num, (1, 1, 1), activation = 'linear', alpha = 1, with_isaac = False)


    def forward(self, x, z):
        x = self.upsample(x)
        x = self.conv_bias_relu(x)
        offset = [(z.size()[i]-x.size()[i])//2 for i in range(2,z.dim())]
        return torch.cat([x, z[:,:,offset[0]:offset[0]+x.size(2),
                                offset[1]:offset[1]+x.size(3),
                                offset[2]:offset[2]+x.size(4)]], 1)


class VggBlock(nn.Module):
    def __init__(self, in_num, out_num, kernel_size, window_size, activation, alpha, pool, with_isaac, return_tmp=True):
        super(VggBlock, self).__init__()
        self.conv1 = ConvBiasActivation(in_num, out_num, kernel_size, activation, alpha, with_isaac)
        self.conv2 = ConvBiasActivation(out_num, out_num, kernel_size, activation, alpha, with_isaac)
        self.pool = MaxPool(window_size, window_size, with_isaac = with_isaac) if pool else None
        self.return_tmp = return_tmp

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        y = self.pool(z) if self.pool is not None else z
        return (z, y) if self.return_tmp else y

    def arm_quantization(self, history):
        self.conv1.arm_quantization(history)
        self.conv2.arm_quantization(history)
        if self.pool is not None:
            self.pool.arm_quantization(history)

class UpVggCropCatBlock(nn.Module):
    def UpConvCropCat(self, strides, in_num, out_num, with_isaac):
        if with_isaac:
            return Conv3dCropCat(in_num, out_num, (1,1,1), upsample=strides, activation = 'linear', bias = True)
        else:
            return UpConvCropCat(strides, in_num, out_num)

    def __init__(self, in_num, out_num, kernel_size, window_size, activation, alpha, with_isaac):
        super(UpVggCropCatBlock, self).__init__()
        self.upsample = self.UpConvCropCat(window_size, in_num, out_num, with_isaac = with_isaac)
        self.conv1 = ConvBiasActivation(2*out_num, out_num, kernel_size, activation, alpha, with_isaac)
        self.conv2 = ConvBiasActivation(out_num, out_num, kernel_size, activation, alpha, with_isaac)

    def forward(self, x, z):
        x = self.upsample(x, z)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def arm_quantization(self, history):
        self.upsample.arm_quantization(history)
        self.conv1.arm_quantization(history)
        self.conv2.arm_quantization(history)
