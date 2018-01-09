import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import isaac.pytorch
from time import time
import timeit
import h5py


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
    # Load data
    T = np.array(h5py.File('./im_uint8_half.h5','r')['main']).astype(np.float32)/255
    I, J, K = 31, 204, 204

    # Build models
    unet_legacy = unet3D_m1().cuda()
    unet_legacy.load_state_dict(torch.load('./net_iter_100000_m1.pth')['state_dict'])
    unet_ref = convert(unet_legacy)

    # Quantize
    X = T[:I, :J, :K].reshape(1, 1, I, J, K)
    X = Variable(torch.from_numpy(X)).cuda()
    unet_sc = unet_ref.fuse().quantize(X)

    # Evaluate errors
    N = 10
    errors = np.zeros(N)
    np.random.seed(0)
    i_range = np.random.randint(I, T.shape[0] - I, N)
    j_range = np.random.randint(J, T.shape[1] - J, N)
    k_range = np.random.randint(K, T.shape[2] - K, N)
    for n, (i, j, k) in enumerate(zip(i_range, j_range, k_range)):
        X = T[i : i+I, j : j+J, k : k+K].reshape(1, 1, I, J, K)
        X = Variable(torch.from_numpy(X)).cuda()
        y_ref = unet_ref(X)
        y_sc = unet_sc(X)
        errors[n] = torch.norm(y_ref - y_sc).data[0]/torch.norm(y_ref).data[0]
    print('Error: {:.4f} [+- {:.4f}]'.format(np.mean(errors), np.std(errors)))

    # Benchmark
    t_sc = [int(x*1e3) for x in timeit.repeat(lambda: (unet_sc(X), torch.cuda.synchronize()), repeat=1, number=1)]
    t_ref = [int(x*1e3) for x in timeit.repeat(lambda: (unet_ref(X), torch.cuda.synchronize()), repeat=1, number=1)]
    print('Time: {}ms (Isaac) ; {}ms (PyTorch)'.format(t_sc[0], t_ref[0]))
