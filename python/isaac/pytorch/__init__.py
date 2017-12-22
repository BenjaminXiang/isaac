import torch
import torch.nn as nn
from torch.autograd import Function
from .c_lib import *
from ctypes import *

class ConvNdFunction(Function):
    def __init__(self, activation, alpha):
        self.activation = activation.encode('utf-8')
        self.alpha = float(alpha)

    def forward(self, input, weight, bias):
        output = input.new()
        isaac_conv_nd(input.cuda(), weight.cuda(), bias.cuda(), output.cuda(), self.activation, self.alpha, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
        return output

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation = 'linear', alpha = 0.):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.alpha = alpha

    def forward(self, input):
        return ConvNdFunction(self.activation, self.alpha)(input, self.weight, self.bias)
