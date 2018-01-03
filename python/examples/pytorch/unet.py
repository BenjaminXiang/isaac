import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import isaac.pytorch
from time import time
import timeit


def mergeCrop(x1, x2):
    # x1 left, x2 right
    offset = [(x1.size()[x]-x2.size()[x])//2 for x in range(2,x1.dim())]
    return torch.cat([x2, x1[:,:,offset[0]:offset[0]+x2.size(2),
                     offset[1]:offset[1]+x2.size(3),offset[2]:offset[2]+x2.size(4)]], 1)

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

def convert(legacy):
    result = isaac.pytorch.UNet().cuda()

    # Reorder indices because new state dict has upsample-upconv interleaved
    depth = legacy.depth
    ndown = 4*(depth + 1)
    reorder = list(range(ndown))
    for i in range(depth):
        upsamples = list(range(ndown + i*3, ndown + i*3 + 3))
        upconvs = list(range(ndown + depth*3 + i*4, ndown + depth*3 + i*4 + 4))
        reorder +=  upsamples + upconvs
    reorder += [ndown + 7*depth, ndown + 7*depth + 1]

    # Copy in proper order
    legacy_keys = list(legacy.state_dict().keys())
    result_keys = list(result.state_dict().keys())
    legacy_dict = legacy.state_dict()
    result_dict = result.state_dict()
    for i, j in enumerate(reorder):
        result_dict[result_keys[i]] = legacy_dict[legacy_keys[j]].clone()
    result.load_state_dict(result_dict)

    return result


if __name__ == '__main__':
    torch.manual_seed(0)
    X = Variable(torch.Tensor(1, 1, 31, 204, 204).uniform_(0, 1)).cuda()

    # Build models
    unet_ref = unet3D_m1().cuda()
    unet_sc = convert(unet_ref).fuse().quantize(X)

    # Test correctness
    y_ref = unet_ref(X)
    y_sc = unet_sc(X)
    error = torch.norm(y_ref - y_sc)/torch.norm(y_ref)
    print('Error: {}'.format(error.data[0]))

    # Benchmark
    t_sc = [int(x*1e3) for x in timeit.repeat(lambda: (unet_sc(X), torch.cuda.synchronize()), repeat=1, number=1)]
    t_ref = [int(x*1e3) for x in timeit.repeat(lambda: (unet_ref(X), torch.cuda.synchronize()), repeat=1, number=1)]
    print('Time: {}ms (Isaac) ; {}ms (PyTorch)'.format(t_sc[0], t_ref[0]))




