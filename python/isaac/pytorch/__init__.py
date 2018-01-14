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
    def __init__(self, activation, alpha, scale, pad = (0, 0, 0), strides = (1, 1, 1), upsample = (1, 1, 1), crop = (0, 0, 0, 0, 0, 0), quantized_in = False, quantized_out = False):
        self.activation = activation.encode('utf-8')
        self.alpha = float(alpha)
        self.scale = (scale[0], scale[1], tuple(map(float, scale[2])), scale[3])
        self.num_outputs = len(self.scale[2])
        self.pad = pad
        self.strides = strides
        self.upsample = upsample
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
                      z, self.crop[0], self.crop[1], self.crop[2], self.crop[3], self.crop[4], self.crop[5]# Crop-cat
                      )
        if self.num_outputs == 1:
            return output[0]
        return output

class MaxPoolNdFunction(Function):
    def __init__(self, kernel_size, pad = (0, 0, 0), strides = (1, 1, 1), quantized = False):
        self.kernel_size = pad_left(3, kernel_size, 1)
        self.pad = pad_left(3, pad, 1)
        self.strides = pad_left(3, strides, 1)
        self.ffi = cffi.FFI()
        self.quantized = quantized
        self.function = {False: isaac_max_pool_nd_float,
                                 True: isaac_max_pool_nd_int}[quantized]

    def forward(self, input):
        output = input.new()
        self.function(input, output,
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
        sorted = x.abs().view(-1).sort(dim = 0, descending = True)[0]
        return 127. / sorted[0]
        #idxs = [int(rho * len(sorted)) for rho in np.arange(0, 1e-5, 1e-6)]
        #scales = [127. / sorted[idx] for idx in idxs]
        #clip = lambda x, scale: torch.round(torch.clamp(x * scale, -128, 127)) / scale
        #loss = [torch.norm(x - clip(x, scale)) for scale in scales]
        #return scales[np.argmin(loss)]

    def __init__(self, approximate):
        self.history = dict()
        self.module_of = dict()
        self.approximate = approximate

    def quantize(self, weight, x, y, z):
        # Update scales
        scale = [1., 1., [1.], 1.]
        quantized_in, quantized_out = False, False

        if weight.data.size()[0] % 4 == 0:
            quantized_in = True
            scale[0] = self.history[id(x)]
            scale[1] = self.scale(weight.data)
        if weight.data.size()[-1] % 4 == 0:
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
        self.dim = dim
        self.weight.data = self.weight.data.permute(*to_chwn_idx(self.dim))
        self.skip = 'cat'

    def set_quantizer(self, quantizer):
        self.quantizer = quantizer

    def forward(self, x, z = None):
        # Cropping
        if z is None:
            crop = [1, 1, 1, 1, 1, 1]
        else:
            offset = [(z.size()[i]-x.size()[i]*self.upsample[i - 2])//2 for i in range(2,z.dim())]
            crop = (offset[0], offset[0], offset[1], offset[1], offset[2], offset[2])
        # Bias
        bias = self.bias if self.bias is not None else torch.autograd.Variable()
        # Computation
        y = ConvNdFunction(self.activation, self.alpha, self.scale, upsample=self.upsample, crop=crop, quantized_in=self.quantized_in, quantized_out = self.quantized_out)\
                          (x, self.weight, bias, torch.autograd.Variable() if z is None else z)
        # Quantize if requested
        if self.quantizer:
            self.scale, self.weight, self.quantized_in, self.quantized_out = self.quantizer.quantize(self.weight, x, y, z)
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

class MaxPoolNd(nn.Module):

    def __init__(self, kernel_size, stride):
        super(MaxPoolNd, self).__init__()
        self.quantizer = None
        self.quantized = False
        self.kernel_size = kernel_size
        self.stride = stride

    def set_quantizer(self, quantizer):
        self.quantizer = quantizer

    def forward(self, x):
        # Computations
        y = MaxPoolNdFunction(self.kernel_size, strides=self.stride, quantized=self.quantized)(x)
        # Quantization if requested
        if self.quantizer:
            self.quantizer.history[id(y)] = self.quantizer.history[id(x)]
            self.quantizer = None
            self.quantized = True
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
        self.upsample.weight.requires_grad = False
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
        z_tmp, z_out = z if isinstance(z, tuple) else (z, z)
        y = self.pool(z_tmp) if self.pool is not None else z_out
        return (z_out, y) if self.return_tmp else y


class UpVggCropCatBlock(nn.Module):
    def UpConvCropCat(self, strides, in_num, out_num, with_isaac):
        if with_isaac:
            return Conv3d(in_num, out_num, (1,1,1), upsample=strides, activation = 'linear', bias = True, residual = 'cat')
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


class UNet(nn.Module):

    def __init__(self, out_num=3, filters=[1,24,72,216,648], relu_slope=0.005, with_isaac=False):
        super(UNet, self).__init__()
        # Attributes
        self.relu_slope = relu_slope
        self.filters = filters
        self.depth = len(filters) - 1
        self.out_num = out_num

        # Downward convolutions
        self.down_conv = nn.ModuleList([VggBlock(filters[x], filters[x+1], (3, 3, 3), (1, 2, 2), 'relu', relu_slope, x < self.depth - 1, with_isaac)
                                        for x in range(self.depth)])
        # Upward convolution
        self.up_conv = nn.ModuleList([UpVggCropCatBlock(filters[x], filters[x-1], (3, 3, 3), (1, 2, 2), 'relu', relu_slope, with_isaac)
                                        for x in range(self.depth, 1, -1)])
        # Final layer
        self.final = ConvBiasActivation(filters[1], out_num, kernel_size=(1, 1, 1), activation = 'sigmoid', alpha = 0, with_isaac = with_isaac)

    def forward(self, x):
        z = [None]*self.depth
        for i in range(self.depth):
            z[i], x = self.down_conv[i](x)
        for i in range(self.depth - 1):
            x = self.up_conv[i](x, z[self.depth - 2 - i])
        x = self.final(x)
        return x

    def fuse(self):
        result = UNet(self.out_num, self.filters, self.relu_slope, True).cuda()
        parameters = [x for x in self.parameters() if x.requires_grad]
        for x, y in zip(result.parameters(), parameters):
            if(len(x.data.size()) > 1):
                x.data = y.data.permute(1, 2, 3, 4, 0).clone()
            else:
                x.data = y.data
        return result

    def quantize(self, x, approximate = True):
        quantizer = Quantizer(approximate)
        for module in self.modules():
            if hasattr(module, 'set_quantizer'):
                module.set_quantizer(quantizer)
        self.forward(x)
        return self
