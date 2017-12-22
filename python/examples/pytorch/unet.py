import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

def mergeCrop(x1, x2):
    # x1 left, x2 right
    offset = [(x1.size()[x]-x2.size()[x])//2 for x in range(2,x1.dim())] 
    return torch.cat([x2, x1[:,:,offset[0]:offset[0]+x2.size(2),
                     offset[1]:offset[1]+x2.size(3),offset[2]:offset[2]+x2.size(4)]], 1)

def ConvBiasActivation(in_num, out_num, kernel_size, function, alpha):
        conv = nn.Conv3d(in_num, out_num, kernel_size=kernel_size, padding=0, stride=1, bias=True)
        if function == 'relu':
            act = nn.LeakyReLU(alpha)
        if function == 'sigmoid':
            act = nn.Sigmoid()
        if function == 'linear':
            return conv
        return nn.Sequential(conv, act)

class UpConvCropCat(nn.Module):
    def __init__(self, kernel_size, strides_size, in_num, out_num):
        super(UpConvCropCat, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_num, in_num, kernel_size, strides_size, groups=in_num, bias=False)
        self.conv_bias_relu = ConvBiasActivation(in_num, out_num, (1, 1, 1), function = 'linear', alpha = 1)
        
    def forward(self, x, z):
        x = self.upsample(x)
        x = self.conv_bias_relu(x)
        offset = [(z.size()[i]-x.size()[i])//2 for i in range(2,z.dim())]
        return torch.cat([x, z[:,:,offset[0]:offset[0]+x.size(2),
                                offset[1]:offset[1]+x.size(3),
                                offset[2]:offset[2]+x.size(4)]], 1)

class UNet3D(nn.Module):
    def __init__(self, in_num=1, out_num=3, filters=[24,72,216,648],relu_slope=0.005):
        super(UNet3D, self).__init__()
        if len(filters) != 4: 
            raise AssertionError 
        filters = [in_num] + filters
        self.depth = len(filters) - 1
        
        # Downward convolutions
        self.down_conv = nn.ModuleList([nn.Sequential(
                ConvBiasActivation(filters[x  ], filters[x+1], kernel_size = 3, function = 'relu', alpha = relu_slope),
                ConvBiasActivation(filters[x+1], filters[x+1], kernel_size = 3, function = 'relu', alpha = relu_slope)) 
                   for x in range(0, self.depth)])
        # Pooling
        self.pool = nn.ModuleList([nn.MaxPool3d((1,2,2), (1,2,2))
                   for x in range(self.depth)])
        
        # Upsampling
        self.upsample = nn.ModuleList([UpConvCropCat((1,2,2), (1,2,2), filters[x], filters[x-1])
                   for x in range(self.depth, 1, -1)])
                   
        # Upward convolution
        self.up_conv = nn.ModuleList([nn.Sequential(
                ConvBiasActivation(2*filters[x-1], filters[x-1], kernel_size = 3, function = 'relu', alpha = relu_slope),
                ConvBiasActivation(  filters[x-1], filters[x-1], kernel_size = 3, function = 'relu', alpha = relu_slope)) 
                   for x in range(self.depth, 1, -1)])
                   
        # Final layer
        self.final = ConvBiasActivation(filters[1], out_num, kernel_size=1, function = 'sigmoid', alpha = 0)
    
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


class unet3D_m1(nn.Module): # deployed model-1
    def __init__(self, in_num=1, out_num=3, filters=[24,72,216,648],relu_slope=0.005):
        super(unet3D_m1, self).__init__()
        if len(filters) != 4: raise AssertionError 
        filters_in = [in_num] + filters[:-1]
        self.depth = len(filters)-1
        self.seq_num = self.depth*3+2

        self.downC = nn.ModuleList([nn.Sequential(
                nn.Conv3d(filters_in[x], filters_in[x+1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope),
                nn.Conv3d(filters_in[x+1], filters_in[x+1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope))
            for x in range(self.depth)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.depth)]) 
        self.center = nn.Sequential(
                nn.Conv3d(filters[-2], filters[-1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope),
                nn.Conv3d(filters[-1], filters[-1], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope))
        self.upS = nn.ModuleList([nn.Sequential(
                nn.ConvTranspose3d(filters[3-x], filters[3-x], (1,2,2), (1,2,2), groups=filters[3-x], bias=False),
                nn.Conv3d(filters[3-x], filters[2-x], kernel_size=1, stride=1, bias=True))
            for x in range(self.depth)])
        # double input channels: merge-crop
        self.upC = nn.ModuleList([nn.Sequential(
                nn.Conv3d(2*filters[2-x], filters[2-x], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope),
                nn.Conv3d(filters[2-x], filters[2-x], kernel_size=3, stride=1, bias=True),
                nn.LeakyReLU(relu_slope))
            for x in range(self.depth)]) 

        self.final = nn.Sequential(nn.Conv3d(filters[0], out_num, kernel_size=1, stride=1, bias=True))

    def forward(self, x):
        down_u = [None]*self.depth
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])
        x = self.center(x)
        for i in range(self.depth):
            x = mergeCrop(down_u[self.depth-1-i], self.upS[i](x))
            x = self.upC[i](x)
        return F.sigmoid(self.final(x))

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    unet = UNet3D().cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    reference = unet3D_m1().cuda()
    X = Variable(torch.Tensor(1, 1, 31, 204, 204).cuda().uniform_(0, 1))
    Y1 = unet(X)
    Y2 = reference(X)
    print(Y1 - Y2)
    
    
