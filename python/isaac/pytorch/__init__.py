import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple

from torch.autograd import Function
from ctypes import *
from .c_lib import *
import cffi
import struct
import numpy as np

def pad_left(dim, x, value):
    return (value,)*(dim-len(x)) + x

def PackNd(input, alpha, beta):
    output = torch.Tensor().type(torch.IntTensor).cuda()
    isaac_pack_nd(input, output, alpha, beta)
    return output


class ConvNdFunction(Function):
    def __init__(self, activation, alpha, scale, pad = (0, 0, 0), strides = (1, 1, 1), upsample = (1, 1, 1), crop = (0, 0, 0, 0, 0, 0), quantized_in = False, quantized_out = False, residual = ''):
        self.activation = activation.encode('utf-8')
        self.residual = '' if residual is None else residual
        self.residual = self.residual.encode('utf-8')
        self.alpha = float(alpha)
        self.scale = (scale[0], scale[1], tuple(map(float, scale[2])), scale[3])
        self.num_outputs = len(self.scale[2])
        self.pad = pad_left(3, pad, 0)
        self.strides = pad_left(3, strides, 1)
        self.upsample = pad_left(3, upsample, 1)
        self.crop = crop
        self.ffi = c_lib._ffi
        self.quantized_in = quantized_in
        self.quantized_out = quantized_out
        self.function = {(False, False): isaac_conv_nd_float_float,
                          (True, False): isaac_conv_nd_int_float,
                          (False, True): isaac_conv_nd_float_int,
                          (True, True): isaac_conv_nd_int_int}[quantized_in, quantized_out]

    def forward(self, input, weight, bias, z):
        z = z if z.size() else self.ffi.NULL
        bias = bias if bias.size() else self.ffi.NULL
        output = tuple(input.new().type(torch.cuda.IntTensor if self.quantized_out else torch.cuda.FloatTensor) for i in range(self.num_outputs))
        T = torch.utils.ffi._torch_to_cffi.get(type(output[0]))
        p_outputs = self.ffi.new(T + '*[]', [self.ffi.cast(T + '*', x._cdata) for x in output])
        p_scales = self.ffi.new('float[]', self.scale[2])
        self.function(input, weight, p_outputs, self.num_outputs,
                      self.upsample[0], self.upsample[1], self.upsample[2], # Upsample
                      self.pad[0], self.pad[1], self.pad[2], # Pad
                      self.strides[0], self.strides[1], self.strides[2], # Strides
                      bias, # Bias
                      self.activation, self.alpha, # Activation
                      self.quantized_in, self.quantized_out, self.scale[0], self.scale[1], p_scales, self.scale[3], # Quantization
                      self.residual, z, self.crop[0], self.crop[1], self.crop[2], self.crop[3], self.crop[4], self.crop[5]# Crop-cat
                      )
        if self.num_outputs == 1:
            return output[0]
        return output

class PoolNdFunction(Function):
    def __init__(self, type, kernel_size, scale, pad = (0, 0, 0), strides = (1, 1, 1), quantized_in = False, quantized_out = False):
        self.kernel_size = pad_left(3, kernel_size, 1)
        self.pad = pad_left(3, pad, 0)
        self.strides = pad_left(3, strides, 1)
        self.scale = scale
        self.ffi = cffi.FFI()
        self.quantized_in = quantized_in
        self.quantized_out = quantized_out
        self.type = type.encode('utf-8')
        self.function = {(False, False): isaac_pool_nd_float_float,
                          (True, False): isaac_pool_nd_int_float,
                          (True, True): isaac_pool_nd_int_int}[quantized_in, quantized_out]

    def forward(self, input):
        output = input.new().type(torch.cuda.IntTensor if self.quantized_out else torch.cuda.FloatTensor)

        self.function(input, output,
                      self.type,
                      self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                      self.pad[0], self.pad[1], self.pad[2],
                      self.quantized_in, self.quantized_out,
                      self.scale[0], self.scale[1],
                      self.strides[0], self.strides[1], self.strides[2])
        return output

class LinearFunction(Function):

    def __init__(self):
        self.function = isaac_linear_float_float
        self.alpha = 1.
        self.beta = 0.
        self.quantized_in = False
        self.quantized_out = False

    def forward(self, input, weight, bias):
        output = input.new()
        self.function(input, weight, output, bias,
                      self.alpha, self.beta,
                      self.quantized_in, self.quantized_out)
        return output

#############################
##       Quantization      ##
#############################
class Quantizer:

    def scale(self, x):
        eps = 1e-5
        return 127. / (torch.max(torch.abs(x)) + eps)
        #idxs = [int(rho * len(sorted)) for rho in np.arange(0, 1e-5, 1e-6)]
        #scales = [127. / sorted[idx] for idx in idxs]
        #clip = lambda x, scale: torch.round(torch.clamp(x * scale, -128, 127)) / scale
        #loss = [torch.norm(x - clip(x, scale)) for scale in scales]
        #return scales[np.argmin(loss)]

    def __init__(self, approximate):
        self.history = dict()
        self.module_of = dict()
        self.approximate = approximate

    def quantize(self, weight, x, y, z, is_first_conv, is_last_conv):
        # Update scales
        scale = [1., 1., [1.], 1.]
        quantized_in, quantized_out = False, False

        if weight.data.size()[0] % 4 == 0 and not is_first_conv:
            quantized_in = True
            scale[0] = self.history[id(x)]
            scale[1] = self.scale(weight.data)
        if weight.data.size()[-1] % 4 == 0 and not is_last_conv:
            quantized_out = True
            scale[2][0] = self.history[id(y)] = self.scale(y.data)

        # Quantize weights
        dim = len(weight.size())
        if quantized_in:
            tmp = weight.data*scale[1]
            weight.data = PackNd(weight.data.permute(*from_chwn_idx(dim)).clone(), scale[1], 0.0)
            weight.data = weight.data.permute(*to_chwn_idx(dim)).clone()

        # Handle skip connections
        if z is not None:
            if self.approximate:
                scale[3] = self.history[id(z)]
            else:
                self.module_of[id(z)].scale[2].append(scale[2][0])
                scale[3] = scale[2][0]

        # Quantization done
        return scale, weight, quantized_in, quantized_out


#############################
###      Convolutions     ###
#############################

def to_chwn_idx(dim):
    return list(range(1, dim)) + [0]

def from_chwn_idx(dim):
    return [dim-1] + list(range(0, dim-1))

class ConvNd(nn.modules.conv._ConvNd):

    def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding, dilation, upsample, groups, bias, activation, alpha, scale, residual):
        super(ConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias)
        self.activation = activation
        self.alpha = alpha
        self.scale = scale
        self.upsample = upsample
        self.quantizer = None
        self.quantized_in = False
        self.quantized_out = False
        self.is_last_conv = False
        self.is_first_conv = False
        self.dim = dim
        self.weight.data = self.weight.data.permute(*to_chwn_idx(self.dim))
        self.residual = residual


    def set_quantizer(self, quantizer):
        self.quantizer = quantizer

    def forward(self, x, z = None):
        # Cropping
        if z is None:
            crop = [1, 1, 1, 1, 1, 1]
        else:
            offset = tuple([(z.size()[i]-x.size()[i]*self.upsample[i - 2])//2 for i in range(2,z.dim())])
            offset = pad_left(3, offset, 0)
            crop = (offset[0], offset[0], offset[1], offset[1], offset[2], offset[2])
        # Bias
        bias = self.bias if self.bias is not None else torch.autograd.Variable()
        # Computation
        y = ConvNdFunction(self.activation, self.alpha, self.scale, pad=self.padding, strides=self.stride, upsample=self.upsample, crop=crop, quantized_in=self.quantized_in, quantized_out = self.quantized_out, residual = self.residual)\
                          (x, self.weight, bias, torch.autograd.Variable() if z is None else z)
        # Quantize if requested
        if self.quantizer:
            self.scale, self.weight, self.quantized_in, self.quantized_out = self.quantizer.quantize(self.weight, x, y, z, self.is_first_conv, self.is_last_conv)
            self.quantizer.module_of.update({id(y): self})
            self.quantizer = None
        return y


# 1D Conv
class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., [1.], 1.], residual = None):
        super(Conv1d, self).__init__(4, in_channels, out_channels, _single(kernel_size), _single(stride), _single(padding), _single(dilation), _single(upsample), groups, bias, activation, alpha, scale, residual)

# 2D Conv
class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., [1.], 1.], residual = None):
        super(Conv2d, self).__init__(4, in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), _pair(upsample), groups, bias, activation, alpha, scale, residual)

# 3D Conv
class Conv3d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., scale = [1., 1., [1.], 1.], residual = None):
        super(Conv3d, self).__init__(5, in_channels, out_channels, _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation), _triple(upsample), groups, bias, activation, alpha, scale, residual)

#############################
###      Pooling          ###
#############################

class PoolNd(nn.Module):

    def __init__(self, kernel_size, type, stride, padding, scale = [1., 1.]):
        super(PoolNd, self).__init__()
        self.quantizer = None
        self.quantized_in = False
        self.quantized_out = False
        self.is_last_conv = False
        self.is_first_conv = False
        self.scale = scale
        self.kernel_size = kernel_size
        self.type = type
        self.stride = stride
        self.padding = padding

    def set_quantizer(self, quantizer):
        self.quantizer = quantizer

    def forward(self, x):
        # Computations
        y = PoolNdFunction(self.type, self.kernel_size, self.scale, pad=self.padding, strides=self.stride, quantized_in=self.quantized_in, quantized_out=self.quantized_out)(x)
        # Quantization if requested
        if self.quantizer:
            self.scale[0] = self.quantizer.history[id(x)]
            self.scale[1] = self.quantizer.history[id(y)] = self.quantizer.history[id(x)]
            self.quantizer = None
            self.quantized_in = not self.is_first_conv
            self.quantized_out = not self.is_last_conv
        return y


# Max
class MaxPool1d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool1d, self).__init__(_single(kernel_size), 'max', _single(stride), _single(padding))

class MaxPool2d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool2d, self).__init__(_pair(kernel_size), 'max', _pair(stride), _pair(padding))

class MaxPool3d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool3d, self).__init__(_triple(kernel_size), 'max', _triple(stride), _triple(padding))


# Average
class AvgPool1d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPool1d, self).__init__(_single(kernel_size), 'avg', _single(stride), _single(padding))

class AvgPool2d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPool2d, self).__init__(_pair(kernel_size), 'avg', _pair(stride), _pair(padding))

class AvgPool3d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPool3d, self).__init__(_triple(kernel_size), 'avg', _triple(stride), _triple(padding))


#############################
###      Linear           ###
#############################

class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        y = LinearFunction()(x, self.weight, self.bias)
        return y



#############################
###  PyTorch Equivalent   ###
#############################

def ConvBiasActivation(in_num, out_num, kernel_size, bias, activation, alpha):
    dim = len(kernel_size)
    Type = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]
    conv = Type(in_num, out_num, kernel_size = kernel_size, padding=0, stride=1, bias = bias)
    if activation == 'relu':
        act = nn.LeakyReLU(alpha)
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    if activation == 'linear':
        return nn.Sequential(conv)
    return nn.Sequential(conv, act)


class UpConvCropCat(nn.Module):

    def __init__(self, strides, in_num, out_num, residual):
        super(UpConvCropCat, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_num, in_num, strides, strides, groups = in_num, bias = False)
        self.upsample.weight.data.fill_(1.0)
        self.upsample.weight.requires_grad = False
        self.conv_bias_relu = ConvBiasActivation(in_num, out_num, (1, 1, 1), bias = True, activation = 'linear', alpha = 1)
        self.residual = residual


    def forward(self, x, z):
        x = self.upsample(x)
        x = self.conv_bias_relu(x)
        offset = [(z.size()[i]-x.size()[i])//2 for i in range(2,z.dim())]
        z_crop = z[:,:,offset[0]:offset[0]+x.size(2),
                       offset[1]:offset[1]+x.size(3),
                       offset[2]:offset[2]+x.size(4)]
        if self.residual == 'cat':
            return torch.cat([x, z_crop], 1)
        if self.residual == 'add':
            return x + z_crop


#############################
###     Helpers           ###
#############################

ConvType = {1:Conv1d, 2:Conv2d, 3:Conv3d}
MaxPoolType = {1:MaxPool1d, 2:MaxPool2d, 3:MaxPool3d}
AvgPoolType = {1:AvgPool1d, 2:AvgPool2d, 3:AvgPool3d}
